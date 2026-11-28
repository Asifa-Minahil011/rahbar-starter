"""Turns a question into an AskResponse: the query pipeline (section 3.2).

    1 validate (FastAPI)   2 cache check   3-4 embed + retrieve   5 augment (in the provider)
    6 generate             7 validate, retry once, fall back      8 store the turn / cache

Honest abstention happens BEFORE the model is called: if nothing retrieved clears the
relevance floor, the answer is a fixed "not covered" and no LLM call is made. Deterministic
abstention is what lets the CI gate demand 100% on must-abstain questions.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from pydantic import ValidationError

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..providers.llm.errors import GenerationError
from ..providers.llm.prompt import build_messages
from ..retrieval.base import Retriever
from ..schemas import Answer, AskResponse, Passage, RetrievedRef
from . import telemetry
from .grounding import ABSTAIN_TEXT, FALLBACK_TEXT, split_citations
from .guardrails import apply_rail, scan_question
from .trace import Trace, prompt_hash

log = logging.getLogger("rahbar.answer")

MAX_ATTEMPTS = 2   # one try plus one retry


class SemanticCache(Protocol):
    """Implemented by the platform side (Redis). Keys must include the index_id from the
    Chroma collection metadata, so a re-index invalidates everything."""

    def lookup(self, question: str) -> AskResponse | None: ...

    def store(self, question: str, response: AskResponse) -> None: ...


def _refs(passages: list[Passage]) -> list[RetrievedRef]:
    return [RetrievedRef(chunk_id=p.chunk_id, source=p.source, section=p.section,
                         page=p.page, score=p.score) for p in passages]


def _response(provider: LLMProvider, passages: list[Passage], text: str, grounded: bool,
              confidence: str = "low", citations: list | None = None) -> AskResponse:
    return AskResponse(
        request_id=request_id_var.get(), answer=text, citations=citations or [],
        answered_from_context=grounded, confidence=confidence,  # type: ignore[arg-type]
        retrieved=_refs(passages), provider=provider.name,
    )


def _generate_validated(question: str, usable: list[Passage], retrieved: list[Passage],
                        provider: LLMProvider) -> Answer | None:
    """Call the model; accept only an Answer that parses AND cites what was retrieved.
    Retry once, then give up (the caller falls back)."""
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            telemetry.GENERATION_RETRIES.inc()
        try:
            answer = provider.generate(question, usable)
        except (GenerationError, ValidationError) as exc:
            log.warning("generation failed", extra={"attempt": attempt, "error": str(exc)[:300]})
            continue
        if not answer.answered_from_context:
            return answer
        valid, invalid = split_citations(answer.citations, retrieved)
        if invalid:
            telemetry.INVALID_CITATIONS.inc(len(invalid))
            log.warning("fabricated citations dropped",
                        extra={"attempt": attempt, "dropped": [c.model_dump() for c in invalid]})
        if not valid:      # a factual answer with no real citation is not grounded
            continue
        return answer.model_copy(update={"citations": valid})
    return None


def _retrieval_detail(passages: list[Passage], info: dict) -> dict:  # type: ignore[type-arg]
    final = info.get("final_scores", {})
    return {
        "filters": info.get("filters"), "exact_terms": info.get("exact_terms"),
        "candidates_fetched": info.get("candidates_fetched"),
        "order_by_similarity": info.get("order_by_similarity"),
        "reranked_order": info.get("reranked_order", [p.chunk_id for p in passages]),
        "retrieved": [{"chunk_id": p.chunk_id, "document_id": p.source, "section": p.section,
                       "page": p.page, "doc_version": p.doc_version, "superseded": p.superseded,
                       "similarity": round(p.score, 4), "final_score": final.get(p.chunk_id)}
                      for p in passages],
    }


def answer_question(
    question: str, retriever: Retriever, provider: LLMProvider, settings: Settings,
    cache: SemanticCache | None = None, follow_up: bool = False,
) -> AskResponse:
    trace = Trace(question, settings)
    trace.guardrails(scan_question(question))
    for action in trace.rec["guardrail_actions"]:
        telemetry.INJECTION_FLAGS.labels(where=str(action["where"])).inc()

    if cache is not None and not follow_up:
        t = time.perf_counter()
        hit = cache.lookup(question)
        trace.stage("cache", time.perf_counter() - t)
        if hit is not None:
            telemetry.CACHE_HITS.inc()
            trace.set(cache="hit")
            trace.set(retrieval={"retrieved": [r.model_dump() for r in hit.retrieved],
                                 "note": "served from the semantic cache; no retrieval ran"})
            return trace.finish(hit.model_copy(update={"request_id": request_id_var.get()}),
                                "cache_hit")
        telemetry.CACHE_MISSES.inc()
        trace.set(cache="miss")

    started = time.perf_counter()
    info: dict = {}  # type: ignore[type-arg]
    try:
        with_info = getattr(retriever, "retrieve_with_info", None)
        if with_info is not None:
            passages, info = with_info(question, k=settings.top_k)
        else:
            passages = retriever.retrieve(question, k=settings.top_k)
    except Exception as exc:   # Chroma or the embedder is down: a safe answer, never a 500
        log.error("retrieval failed", extra={"error": repr(exc)[:300]})
        telemetry.FALLBACKS.labels(reason="retrieval").inc()
        trace.stage("retrieval", time.perf_counter() - started)
        trace.set(retrieval={"error": repr(exc)[:300]})
        return trace.finish(_response(provider, [], FALLBACK_TEXT, False), "fallback_retrieval")
    finally:
        telemetry.RETRIEVAL_SECONDS.observe(time.perf_counter() - started)
    trace.stage("retrieval", time.perf_counter() - started)
    trace.set(retrieval=_retrieval_detail(passages, info))
    meta = info.get("collection_metadata") or {}
    if meta:
        trace.rec["collection"].update({"version": meta.get("index_id"), "metadata": meta})
        trace.rec["embedding"].update({"index_model": meta.get("embed_model"),
                                       "dimension": meta.get("dimension")})
    if passages:
        telemetry.TOP1_SIMILARITY.observe(max(p.score for p in passages))

    above_floor = [p for p in passages if p.score >= settings.relevance_floor]
    rail = apply_rail(above_floor, settings.injection_rail)
    trace.guardrails(rail.actions)
    for action in rail.actions:
        if action["action"] in ("flagged", "dropped", "redacted"):
            telemetry.INJECTION_FLAGS.labels(where="passage").inc()
    usable = rail.usable
    base_log = {"retriever": retriever.name, "provider": provider.name,
                "top_k": settings.top_k, "chunk_ids": [p.chunk_id for p in passages],
                "scores": [round(p.score, 4) for p in passages],
                "relevance_floor": settings.relevance_floor}

    if not usable:
        reason = "below_floor" if not above_floor else "guardrail"
        telemetry.ABSTENTIONS.labels(reason=reason).inc()
        log.info("answered", extra={**base_log, "answered_from_context": False,
                                    "abstention": reason})
        return trace.finish(_response(provider, passages, ABSTAIN_TEXT, False),
                            f"abstain_{reason}")

    messages = build_messages(question, usable)
    trace.set(prompt=messages, prompt_hash=prompt_hash(messages),
              prompt_chunk_ids=[p.chunk_id for p in usable])
    t = time.perf_counter()
    answer = _generate_validated(question, usable, passages, provider)
    trace.stage("generate_validate", time.perf_counter() - t)
    if answer is None:
        telemetry.FALLBACKS.labels(reason="validation").inc()
        log.info("answered", extra={**base_log, "answered_from_context": False,
                                    "fallback": "validation"})
        return trace.finish(_response(provider, passages, FALLBACK_TEXT, False),
                            "fallback_validation")
    if not answer.answered_from_context:
        telemetry.ABSTENTIONS.labels(reason="model").inc()
        log.info("answered", extra={**base_log, "answered_from_context": False,
                                    "abstention": "model"})
        return trace.finish(_response(provider, passages, ABSTAIN_TEXT, False), "abstain_model")

    response = _response(provider, passages, answer.answer, True,
                         confidence=answer.confidence, citations=answer.citations)
    log.info("answered", extra={**base_log, "answered_from_context": True,
                                "citations": len(answer.citations)})
    if cache is not None and not follow_up:   # only grounded, validated answers are cached
        cache.store(question, response)
    return trace.finish(response, "answered")
