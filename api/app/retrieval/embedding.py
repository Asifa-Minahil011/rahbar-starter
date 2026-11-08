"""Embed a question with the pinned model. Shared by the retriever and the semantic cache."""

from __future__ import annotations

import httpx

from ..config import Settings

# nomic-embed-text is trained with task prefixes. The indexer adds "search_document: "
# to chunks; questions must carry "search_query: " or similarity scores drift down.
QUERY_PREFIX = "search_query: "


def embed_query(text: str, settings: Settings, client: httpx.Client | None = None) -> list[float]:
    prefix = QUERY_PREFIX if settings.embed_model.startswith("nomic-embed") else ""
    url = f"{settings.ollama_url.rstrip('/')}/api/embed"
    payload = {"model": settings.embed_model, "input": [prefix + text]}
    if client is not None:
        r = client.post(url, json=payload)
    else:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(url, json=payload)
    r.raise_for_status()
    return [float(x) for x in r.json()["embeddings"][0]]
