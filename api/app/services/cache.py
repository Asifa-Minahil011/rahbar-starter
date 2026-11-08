"""Semantic cache in Redis (section 3.3, ADR 0002).

A new question is embedded with the pinned model and compared (cosine) with the cached
questions of the CURRENT index. At or above the threshold it is a hit and the stored answer
is returned without retrieval or generation.

Design points, each one a failure mode the assignment names:
- Invalidate on re-index: every key contains the collection's `index_id` (written by the
  indexer, `indexer/pipeline/store.py`). A new index has a new id, so old entries are simply
  never looked at again and expire by TTL. The indexer never talks to Redis (NetworkPolicy:
  redis from api only).
- Never cache a mistake: only grounded, cited, validated answers are stored (the caller
  already guarantees it; `store` checks again). Abstentions, fallbacks and follow-ups are
  never stored.
- The cache must never break answering: any Redis or embedding error is a miss.
- Threshold from the ConfigMap: CACHE_THRESHOLD, chosen with `scripts/cache_threshold.py`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable
from itertools import islice
from typing import Any

from ..schemas import AskResponse

log = logging.getLogger("rahbar.cache")

PREFIX = "rahbar:cache"
MAX_SCAN = 1000           # entries compared per lookup; the cache is small by design
INDEX_ID_TTL = 5.0        # seconds an index_id is remembered: re-index takes effect within this


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two UNIT vectors."""
    return sum(x * y for x, y in zip(a, b))


class RedisSemanticCache:
    def __init__(self, redis_client: Any, embed: Callable[[str], list[float]],
                 index_id: Callable[[], str | None], threshold: float, ttl: int = 86400) -> None:
        self.r = redis_client
        self.embed = embed
        self.index_id = index_id
        self.threshold = threshold
        self.ttl = ttl

    def lookup(self, question: str) -> AskResponse | None:
        try:
            idx = self.index_id()
            if not idx:
                return None
            vec = _unit(self.embed(question))
            keys = list(islice(self.r.scan_iter(match=f"{PREFIX}:{idx}:*", count=200), MAX_SCAN))
            if not keys:
                return None
            best_sim, best = -1.0, None
            for raw in self.r.mget(keys):
                if raw is None:
                    continue
                item = json.loads(raw)
                sim = cosine(vec, item["vector"])
                if sim > best_sim:
                    best_sim, best = sim, item
            if best is not None and best_sim >= self.threshold:
                log.info("cache hit", extra={"similarity": round(best_sim, 4),
                                             "cached_question": best["question"][:120]})
                return AskResponse.model_validate(best["response"])
            return None
        except Exception as exc:   # a broken cache is a slow service, not a broken one
            log.warning("cache lookup failed", extra={"error": repr(exc)[:200]})
            return None

    def store(self, question: str, response: AskResponse) -> None:
        if not response.answered_from_context or not response.citations:
            return    # never cache abstentions, fallbacks or anything unsupported
        try:
            idx = self.index_id()
            if not idx:
                return
            item = {"question": question, "vector": _unit(self.embed(question)),
                    "response": response.model_dump(mode="json")}
            self.r.set(f"{PREFIX}:{idx}:{uuid.uuid4().hex}", json.dumps(item), ex=self.ttl)
        except Exception as exc:
            log.warning("cache store failed", extra={"error": repr(exc)[:200]})


def get_cache(state: Any) -> RedisSemanticCache | None:
    """The cache for this process, built on first use. None when it cannot be built (stub
    retriever, Redis down); retried every 30 seconds. Uses `state.cache` if main.py set one."""
    existing = getattr(state, "cache", None)
    if existing is not None:
        return existing  # type: ignore[no-any-return]
    now = time.monotonic()
    if now - getattr(state, "_cache_tried", -1e9) < 30:
        return None
    state._cache_tried = now
    settings, retriever = state.settings, state.retriever
    if not hasattr(retriever, "collection"):
        return None                      # stub retriever: nothing to key the cache on
    try:
        import redis

        client = redis.Redis(host=getattr(settings, "redis_host", os.environ.get("REDIS_HOST", "redis")),
                             port=int(getattr(settings, "redis_port", os.environ.get("REDIS_PORT", 6379))),
                             socket_timeout=1.0, socket_connect_timeout=1.0)
        client.ping()
    except Exception as exc:
        log.warning("semantic cache disabled: redis unavailable", extra={"error": repr(exc)[:200]})
        return None

    from ..retrieval.embedding import embed_query

    remembered: dict[str, Any] = {"at": -1e9, "id": None}

    def current_index_id() -> str | None:
        if time.monotonic() - remembered["at"] > INDEX_ID_TTL:
            meta = retriever.collection().metadata or {}
            remembered.update(at=time.monotonic(), id=meta.get("index_id"))
        return remembered["id"]  # type: ignore[no-any-return]

    state.cache = RedisSemanticCache(
        client, lambda q: embed_query(q, settings), current_index_id,
        threshold=float(os.environ.get("CACHE_THRESHOLD", "0.95")),
        ttl=int(os.environ.get("CACHE_TTL_SECONDS", "86400")))
    log.info("semantic cache enabled", extra={"threshold": state.cache.threshold})
    return state.cache  # type: ignore[no-any-return]
