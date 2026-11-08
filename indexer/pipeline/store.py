"""Stage 5: Store. The only code in the whole system that writes to Chroma.

Re-index decision (write it into ADR 0001): the new index is built in a scratch collection
(`<name>__building`) while the API keeps serving the old one, verified by count, and only
then swapped in: delete the old collection, rename the scratch one. The window with no
collection is milliseconds, and /readyz sees it as one failed probe, not a restart.
The API must look the collection up by name on every query (not hold a handle), or it
would keep pointing at the deleted one.

Collection metadata (the pin and the cache key):
    embed_model, dimension   section 4.4: the API compares these with its own EMBED_MODEL
    index_id                 random id per build. The semantic cache keys on it, so a
                             re-index invalidates every cached answer without the indexer
                             ever talking to Redis (NetworkPolicy: redis from api only).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .config import IndexerSettings
from .models import Chunk

ADD_BATCH = 128


def get_client(settings: IndexerSettings) -> Any:
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                               settings=ChromaSettings(anonymized_telemetry=False))


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings,
          client: Any | None = None) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count."""
    if not chunks or len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    ids = [c.id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk ids")
    client = client or get_client(settings)
    building = f"{settings.collection}__building"
    try:
        client.delete_collection(building)        # leftover from a crashed run
    except Exception:
        pass
    collection = client.create_collection(
        name=building,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": settings.embed_model,
            "dimension": len(vectors[0]),
            "index_id": uuid.uuid4().hex,
            "chunk_count": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    for i in range(0, len(chunks), ADD_BATCH):
        part = slice(i, i + ADD_BATCH)
        collection.add(
            ids=ids[part],
            documents=[c.text for c in chunks[part]],
            embeddings=vectors[part],
            metadatas=[c.metadata() for c in chunks[part]],
        )
    if collection.count() != len(chunks):
        raise RuntimeError(f"wrote {len(chunks)} chunks but the collection holds "
                           f"{collection.count()}; not swapping it in")
    try:
        client.delete_collection(settings.collection)
    except Exception:  # first ever index: nothing to replace
        pass
    collection.modify(name=settings.collection)
    return int(collection.count())
