"""One JSON line per query: the trace record (Assignment 3, D1; paper section 3.2.3 and Table 2).

Written to stdout (so `kubectl logs` shows it) and, when TRACE_FILE is set, appended to a
file on a volume (so scripts/trace_metrics.py can summarise a time window). The trace id
is the request id: the same value the user gets back in `AskResponse.request_id` and in
the X-Request-ID header.

Tracing must never break an answer: every failure here is swallowed and logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from ..logjson import request_id_var
from ..schemas import AskResponse

log = logging.getLogger("rahbar.trace")
_write_lock = threading.Lock()


def prompt_hash(messages: list[dict[str, str]]) -> str:
    raw = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Trace:
    def __init__(self, question: str, settings: Settings) -> None:
        self._settings = settings
        self._t0 = time.perf_counter()
        self.rec: dict[str, Any] = {
            "trace_id": request_id_var.get(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "query": question,
            "rewritten_query": None,          # RAHBAR does not rewrite queries
            "embedding": {"model": settings.embed_model},
            "collection": {"name": settings.collection},
            "retrieval": {},
            "prompt": None,
            "prompt_hash": None,
            "model": {"provider": settings.llm_provider, "name": settings.llm_model,
                      "temperature": settings.temperature, "top_k": settings.top_k,
                      "relevance_floor": settings.relevance_floor,
                      "hybrid_boost": settings.hybrid_boost},
            "response": None,
            "citations": [],
            "guardrail_actions": [],
            "latency_ms": {},
            "cache": "skipped",              # hit | miss | skipped (follow-up questions)
            "outcome": None,
        }

    def set(self, **fields: Any) -> None:
        self.rec.update(fields)

    def stage(self, name: str, seconds: float) -> None:
        self.rec["latency_ms"][name] = round(seconds * 1000, 1)

    def guardrails(self, actions: list[dict[str, object]]) -> None:
        self.rec["guardrail_actions"].extend(actions)

    def finish(self, response: AskResponse, outcome: str) -> AskResponse:
        """Fill in the answer, write the record, and hand the response back unchanged."""
        self.rec["outcome"] = outcome
        self.rec["response"] = {
            "answer": response.answer, "answered_from_context": response.answered_from_context,
            "confidence": response.confidence, "provider": response.provider,
        }
        self.rec["citations"] = [c.model_dump() for c in response.citations]
        self.stage("total", time.perf_counter() - self._t0)
        self._emit()
        return response

    def _emit(self) -> None:
        try:
            line = json.dumps(self.rec, ensure_ascii=False, default=str)
            log.info("trace", extra={"trace": self.rec})
            path = self._settings.trace_file
            if path:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                with _write_lock, target.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:   # a trace problem is never the user's problem
            log.warning("trace not written", extra={"error": repr(exc)[:200]})
