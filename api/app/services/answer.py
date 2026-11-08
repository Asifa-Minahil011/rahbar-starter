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
from ..retrieval.base import Retriever
from ..schemas import Answer, AskResponse, Passage, RetrievedRef
from . import telemetry
from .grounding import ABSTAIN_TEXT, FALLBACK_TEXT, split_citations

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


def answer_question(
    question: str, retriever: Retriever, provider: LLMProvider, settings: Settings,
    cache: SemanticCache | None = None, follow_up: bool = False,
) -> AskResponse:
    if cache is not None and not follow_up:
        hit = cache.lookup(question)
        if hit is not None:
            telemetry.CACHE_HITS.inc()
            return hit.model_copy(update={"request_id": request_id_var.get()})
        telemetry.CACHE_MISSES.inc()

    started = time.perf_counter()
    try:
        passages = retriever.retrieve(question, k=settings.top_k)
    except Exception as exc:   # Chroma or the embedder is down: a safe answer, never a 500
        log.error("retrieval failed", extra={"error": repr(exc)[:300]})
        telemetry.FALLBACKS.labels(reason="retrieval").inc()
        return _response(provider, [], FALLBACK_TEXT, False)
    finally:
        telemetry.RETRIEVAL_SECONDS.observe(time.perf_counter() - started)

    usable = [p for p in passages if p.score >= settings.relevance_floor]
    base_log = {"retriever": retriever.name, "provider": provider.name,
                "top_k": settings.top_k, "chunk_ids": [p.chunk_id for p in passages],
                "scores": [round(p.score, 4) for p in passages],
                "relevance_floor": settings.relevance_floor}

    if not usable:
        telemetry.ABSTENTIONS.labels(reason="below_floor").inc()
        log.info("answered", extra={**base_log, "answered_from_context": False,
                                    "abstention": "below_floor"})
        return _response(provider, passages, ABSTAIN_TEXT, False)

    answer = _generate_validated(question, usable, passages, provider)
    if answer is None:
        telemetry.FALLBACKS.labels(reason="validation").inc()
        log.info("answered", extra={**base_log, "answered_from_context": False,
                                    "fallback": "validation"})
        return _response(provider, passages, FALLBACK_TEXT, False)
    if not answer.answered_from_context:
        telemetry.ABSTENTIONS.labels(reason="model").inc()
        log.info("answered", extra={**base_log, "answered_from_context": False,
                                    "abstention": "model"})
        return _response(provider, passages, ABSTAIN_TEXT, False)

    response = _response(provider, passages, answer.answer, True,
                         confidence=answer.confidence, citations=answer.citations)
    log.info("answered", extra={**base_log, "answered_from_context": True,
                                "citations": len(answer.citations)})
    if cache is not None and not follow_up:   # only grounded, validated answers are cached
        cache.store(question, response)
    return response
