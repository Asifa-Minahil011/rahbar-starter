"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import json
import time

import httpx

from .config import IndexerSettings

BATCH = 16
DOC_PREFIX = "search_document: "   # nomic-embed-text needs task prefixes; see api/app/retrieval/embedding.py


def _prefix(model: str) -> str:
    return DOC_PREFIX if model.startswith("nomic-embed") else ""


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Embed with settings.embed_model and nothing else (Ollama /api/embed, batched).

    The query side (the API) must use the matching "search_query: " prefix: nomic's
    document and query vectors are trained as a pair.
    """
    prefix = _prefix(settings.embed_model)
    url = f"{settings.ollama_url.rstrip('/')}/api/embed"
    vectors: list[list[float]] = []
    with httpx.Client(timeout=300.0) as client:
        for start in range(0, len(texts), BATCH):
            batch = [prefix + t for t in texts[start:start + BATCH]]
            for attempt in range(3):
                try:
                    r = client.post(url, json={"model": settings.embed_model, "input": batch})
                    r.raise_for_status()
                    got = r.json()["embeddings"]
                    if len(got) != len(batch):
                        raise ValueError(f"asked for {len(batch)} vectors, got {len(got)}")
                    vectors.extend(got)
                    break
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    if attempt == 2:
                        raise RuntimeError(
                            f"embedding failed at batch {start // BATCH} "
                            f"(model {settings.embed_model} at {url}): {exc!r}") from exc
                    time.sleep(2 ** attempt)
            print(json.dumps({"logger": "rahbar.indexer.embed", "msg": "embedded",
                              "done": len(vectors), "total": len(texts)}), flush=True)
    dims = {len(v) for v in vectors}
    if len(dims) > 1:
        raise RuntimeError(f"model returned vectors of different sizes: {sorted(dims)}")
    return vectors
