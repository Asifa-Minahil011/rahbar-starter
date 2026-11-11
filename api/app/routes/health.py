"""Liveness and readiness (section 4.3). They answer different questions.

/healthz: is the process alive? It touches nothing else, so a slow Chroma, Redis or LLM
can never make Kubernetes restart the pod.

/readyz: can this pod take traffic now? Chroma must be reachable, the collection must
exist and hold documents, the collection must have been embedded with the model this API
is configured for (section 4.4), and Redis must be reachable. Otherwise it answers 503
and names what failed, and Kubernetes withholds the pod from the Service.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, Response

from ..config import Settings

log = logging.getLogger("rahbar.api")
router = APIRouter()

_chroma_cache: dict[str, Any] = {}
_last_failure: dict[str, str] = {}


def _short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


def _chroma_client(settings: Settings) -> Any:
    """One shared HTTP client. Built lazily, because building it contacts Chroma."""
    client = _chroma_cache.get("client")
    if client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _chroma_cache["client"] = client
    return client


def _redis_client(settings: Settings) -> Any:
    import redis

    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        socket_connect_timeout=1,
        socket_timeout=1,
    )


def _pin_conflict(settings: Settings, collection: Any) -> str | None:
    """Return a sentence describing the conflict, or None when the models agree."""
    found = (collection.metadata or {}).get("embed_model")
    if found == settings.embed_model:
        return None
    return (
        f"embedding model mismatch: collection '{settings.collection}' was indexed with "
        f"{found!r} but this API is configured with EMBED_MODEL={settings.embed_model!r}"
    )


def check_embedding_pin(settings: Settings) -> tuple[str, str]:
    """Startup check (section 4.4). Returns (status, detail) where status is
    'ok', 'mismatch' or 'unavailable'. It never raises: an index that does not exist
    yet is reported by /readyz, it must not stop the process from starting."""
    try:
        collection = _chroma_client(settings).get_collection(settings.collection)
    except Exception as exc:
        return "unavailable", f"index not readable yet: {_short(exc)}"
    conflict = _pin_conflict(settings, collection)
    if conflict:
        return "mismatch", conflict
    return "ok", f"embedding model {settings.embed_model!r} matches the index"


def run_checks(settings: Settings) -> dict[str, dict[str, Any]]:
    """Evaluate every readiness condition. The Chroma checks depend on each other, so
    they stop at the first failure; Redis is independent and always checked."""
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, ok: bool, detail: str) -> bool:
        checks[name] = {"ok": ok, "detail": detail}
        return ok

    try:
        client = _chroma_client(settings)
        client.heartbeat()
        chroma_ok = record("chroma", True, f"{settings.chroma_host}:{settings.chroma_port}")
    except Exception as exc:
        chroma_ok = record("chroma", False, _short(exc))

    if chroma_ok:
        try:
            collection = client.get_collection(settings.collection)
            exists = record("collection", True, settings.collection)
        except Exception as exc:
            exists = record("collection", False,
                            f"'{settings.collection}' not found: {_short(exc)}")
        if exists:
            count = int(collection.count())
            if record("documents", count > 0, f"{count} chunks"):
                conflict = _pin_conflict(settings, collection)
                record("embed_model", conflict is None,
                       conflict or f"index and API both use {settings.embed_model!r}")

    try:
        _redis_client(settings).ping()
        record("redis", True, f"{settings.redis_host}:{settings.redis_port}")
    except Exception as exc:
        record("redis", False, _short(exc))

    return checks


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    # run_checks() also verifies that collection.count() > 0: an empty index is not ready.
    checks = run_checks(request.app.state.settings)
    failed = [name for name, result in checks.items() if not result["ok"]]
    if len(failed) == 0:
        _last_failure.clear()
        return {"status": "ready", "checks": checks}

    response.status_code = 503
    signature = "|".join(f"{name}:{checks[name]['detail']}" for name in failed)
    if _last_failure.get("signature") != signature:   # log a change, not every probe
        _last_failure["signature"] = signature
        log.error("not ready", extra={"failed": failed,
                                      "conflicts": {n: checks[n]["detail"] for n in failed}})
    return {"status": "not ready", "failed": failed, "checks": checks}
