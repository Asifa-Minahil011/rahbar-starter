"""The real retriever (sections 3.2 and 4.4).

Steps: embed the question with the pinned model, ask Chroma for more candidates than k,
filter superseded documents (unless the question is about an older version), give exact
matches (clause numbers, course codes) a small rank bonus, return the top k.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from ..config import Settings
from ..schemas import Passage
from .embedding import embed_query

CANDIDATES = 4   # fetch k * CANDIDATES from Chroma, then re-rank and cut to k

# Exact terms embeddings blur: clause numbers (1.21, 5.4.1) and course codes (CS-2001, MT 1003)
EXACT = re.compile(r"\b\d+(?:\.\d+)+\b|\b[A-Z]{2,4}[- ]?\d{3,4}\b")
OLDER = re.compile(
    r"\b(old|older|previous|prior|earlier|former|superseded|repealed|replaced|"
    r"before|back then|used to)\b", re.I)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def exact_terms(question: str) -> list[str]:
    return [m.group(0).lower() for m in EXACT.finditer(question)]


def wants_older_version(question: str, known_old_versions: set[str] | None = None) -> bool:
    """True when the user explicitly asks about an older edition: words like 'previous'
    or 'old', or a year that belongs to a superseded document."""
    if OLDER.search(question):
        return True
    if known_old_versions:
        return any(y in v for y in YEAR.findall(question) for v in known_old_versions)
    return False


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings, client: Any | None = None,
                 embed: Callable[[str], list[float]] | None = None) -> None:
        self.settings = settings
        if client is None:
            import chromadb  # the thin client, from chromadb-client in requirements.txt
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                                         settings=ChromaSettings(anonymized_telemetry=False))
        self.client = client
        self._embed = embed or (lambda q: embed_query(q, settings))
        self._old: tuple[float, set[str]] = (0.0, set())

    def collection(self) -> Any:
        # Looked up on every query on purpose: the indexer swaps the collection on
        # re-index, and a cached handle would point at the deleted one.
        return self.client.get_collection(self.settings.collection)

    def retrieve(self, question: str, k: int) -> list[Passage]:
        vector = self._embed(question)
        where = None if wants_older_version(question, self._old_versions()) \
            else {"superseded": False}
        res = self.collection().query(
            query_embeddings=[vector], n_results=k * CANDIDATES, where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids, docs = res["ids"][0], res["documents"][0]
        metas, dists = res["metadatas"][0], res["distances"][0]
        terms = exact_terms(question)
        ranked: list[tuple[float, Passage]] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            sim = 1.0 - float(dist)             # Chroma returns cosine distance
            hit = (sum(t in doc.lower() for t in terms) / len(terms)) if terms else 0.0
            passage = Passage(
                chunk_id=cid, text=doc, source=str(meta["source"]),
                section=str(meta["section"]), page=int(meta.get("page") or 0) or None,
                score=sim, doc_version=str(meta.get("doc_version", "")),
                superseded=bool(meta.get("superseded", False)),
            )
            ranked.append((sim + self.settings.hybrid_boost * hit, passage))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [p for _, p in ranked[:k]]

    def _old_versions(self) -> set[str]:
        """doc_version strings of superseded documents, read from the collection and
        cached for a minute (a re-index is rare; a query is not)."""
        fetched_at, versions = self._old
        if time.monotonic() - fetched_at < 60:
            return versions
        try:
            got = self.collection().get(where={"superseded": True}, include=["metadatas"])
            versions = {str(m["doc_version"]) for m in got["metadatas"]}
        except Exception:
            versions = set()
        self._old = (time.monotonic(), versions)
        return versions
