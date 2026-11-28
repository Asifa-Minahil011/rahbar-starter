"""A2 tests: incremental update, no orphans, archive and restore.

Each test runs twice: against a small in-memory fake of Chroma's collection API (always), and
against real Chroma 1.0.12 in-process when the full `chromadb` package is installed (skipped in
CI, where only chromadb-client is). No server, model or network is needed either way.
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path
from typing import Any

import pytest

from pipeline import manage
from pipeline.chunk import chunk
from pipeline.clean import clean
from pipeline.config import IndexerSettings, get_settings
from pipeline.extract import extract
from pipeline.incremental import ACTIVE, ARCHIVED, read_state, sync
from pipeline.run import discover_entries

VEC = [0.1, 0.2, 0.3]


# ---- a fake of the parts of Chroma's collection API that the indexer uses -------------------

def _match(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, clause) for clause in where["$and"])
    return all(meta.get(k) == v for k, v in where.items())


class FakeCollection:
    def __init__(self, name: str, metadata: dict[str, Any]) -> None:
        self.name, self.metadata = name, dict(metadata)
        self.rows: dict[str, list[Any]] = {}          # id -> [text, vector, metadata]

    def count(self) -> int:
        return len(self.rows)

    def get(self, ids: list[str] | None = None, where: dict[str, Any] | None = None,
            include: list[str] | None = None, limit: int | None = None,
            offset: int = 0) -> dict[str, Any]:
        keys = [i for i in self.rows if (ids is None or i in ids)
                and _match(self.rows[i][2], where)]
        keys = keys[offset:] if limit is None else keys[offset:offset + limit]
        return {"ids": keys, "metadatas": [dict(self.rows[i][2]) for i in keys]}

    def upsert(self, ids: list[str], documents: list[str], embeddings: list[list[float]],
               metadatas: list[dict[str, Any]]) -> None:
        for i, t, v, m in zip(ids, documents, embeddings, metadatas):
            merged = {**self.rows[i][2], **m} if i in self.rows else dict(m)   # Chroma merges
            self.rows[i] = [t, v, merged]

    def update(self, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        for i, m in zip(ids, metadatas):
            self.rows[i][2] = {**self.rows[i][2], **m}

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        for i in [i for i in self.rows if (ids is None or i in ids)
                  and _match(self.rows[i][2], where)]:
            del self.rows[i]

    def modify(self, metadata: dict[str, Any] | None = None) -> None:
        if metadata is not None:
            self.metadata = dict(metadata)            # Chroma replaces, it does not merge


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def create_collection(self, name: str, metadata: dict[str, Any]) -> FakeCollection:
        self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]

    def get_collection(self, name: str) -> FakeCollection:
        return self.collections[name]


def _real_client() -> Any:
    chromadb = pytest.importorskip("chromadb")
    try:
        return chromadb.EphemeralClient()
    except RuntimeError:       # chromadb-client (CI) is the http-only build: no in-process Chroma
        pytest.skip("full chromadb package not installed")


# ---- fixtures and helpers --------------------------------------------------------------------

class Spy:
    def __init__(self) -> None:
        self.extracted: list[str] = []
        self.embedded = 0

    def extract(self, path: str, version: str, superseded: bool) -> Any:
        self.extracted.append(Path(path).name)
        return extract(path, version, superseded)

    def embed(self, texts: list[str], settings: IndexerSettings) -> list[list[float]]:
        self.embedded += len(texts)
        return [VEC] * len(texts)


def write_doc(folder: Path, name: str, body: str, **front: str) -> Path:
    lines = ["---"] + [f"{k}: {v}" for k, v in front.items()] + ["---", ""]
    path = folder / name
    path.write_text("\n".join(lines) + body, encoding="utf-8")
    return path


BODY_A = ("# 1. Attendance\n\nStudents must attend at least 80 percent of lectures in each course.\n\n"
          "# 2. Warning\n\nA student whose CGPA falls below 2.00 receives an academic warning.\n")
BODY_B = ("# 1. Fees\n\nThe tuition fee is payable at the start of every semester by the due date.\n")


@pytest.fixture(params=["fake", "real"])
def env(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Any, IndexerSettings, Path, Spy]:
    client = FakeClient() if request.param == "fake" else _real_client()
    settings = dataclasses.replace(get_settings(), corpus_dir=str(tmp_path),
                                   collection=f"t{uuid.uuid4().hex[:10]}",
                                   chunk_size=60, chunk_overlap=10)
    client.create_collection(
        name=settings.collection,
        metadata={"hnsw:space": "cosine", "embed_model": settings.embed_model, "dimension": 3,
                  "index_id": "first", "chunk_size": settings.chunk_size,
                  "chunk_overlap": settings.chunk_overlap})
    return client, settings, tmp_path, Spy()


def run(env: tuple[Any, IndexerSettings, Path, Spy]) -> dict[str, Any] | None:
    client, settings, folder, spy = env
    spy.extracted.clear()
    spy.embedded = 0
    return sync(settings, client, discover_entries(str(folder)), extract_fn=spy.extract,
                clean_fn=clean, chunk_fn=chunk, embed_fn=spy.embed)


def col(env: tuple[Any, IndexerSettings, Path, Spy]) -> Any:
    return env[0].get_collection(env[1].collection)


def count_where(collection: Any, **where: Any) -> int:
    clauses = [{k: v} for k, v in where.items()]
    flt = clauses[0] if len(clauses) == 1 else {"$and": clauses}
    return len(collection.get(where=flt, include=[])["ids"])


# ---- tests -----------------------------------------------------------------------------------

def test_first_run_embeds_everything_and_records_a2_metadata(env: Any) -> None:
    _, _, folder, spy = env
    write_doc(folder, "rules-2026.md", BODY_A, doc_version="2026", doc_id="rules")
    write_doc(folder, "fees.md", BODY_B, doc_version="2026")
    summary = run(env)
    assert summary and summary["new"] == 2 and summary["chunks_embedded"] > 0
    assert spy.embedded == summary["chunks_total"] == col(env).count()
    meta = col(env).get(include=["metadatas"])["metadatas"][0]
    assert {"doc_id", "content_hash", "status", "pinned", "ingested_at"} <= set(meta)
    assert summary["chunks_by_doc_id"].keys() == {"rules", "fees"}


def test_rerun_on_unchanged_corpus_embeds_zero_chunks(env: Any) -> None:
    _, _, folder, spy = env
    write_doc(folder, "rules-2026.md", BODY_A, doc_version="2026")
    write_doc(folder, "fees.md", BODY_B, doc_version="2026")
    first = run(env)
    index_before = col(env).metadata["index_id"]
    second = run(env)
    assert second and second["chunks_embedded"] == 0 and second["unchanged"] == 2
    assert spy.embedded == 0 and spy.extracted == []        # not even re-read
    assert second["chunks_total"] == first["chunks_total"]  # type: ignore[index]
    assert col(env).metadata["index_id"] == index_before    # cache stays valid


def test_changed_document_replaces_only_its_own_chunks(env: Any) -> None:
    _, _, folder, spy = env
    write_doc(folder, "rules-2026.md", BODY_A, doc_version="2026")
    write_doc(folder, "fees.md", BODY_B, doc_version="2026")
    run(env)
    fees_before = set(col(env).get(where={"source": "fees.md"}, include=[])["ids"])
    old_index = col(env).metadata["index_id"]
    write_doc(folder, "rules-2026.md", BODY_A + "\n# 3. Repeat\n\nA course may be repeated "
              "at most twice during the degree.\n", doc_version="2026")
    s = run(env)
    assert s and s["changed"] == 1 and s["new"] == 0 and s["unchanged"] == 1
    assert spy.extracted == ["rules-2026.md"]                       # only the changed file
    assert set(col(env).get(where={"source": "fees.md"}, include=[])["ids"]) == fees_before
    texts = " ".join(m["source"] for m in col(env).get(include=["metadatas"])["metadatas"])
    assert "rules-2026.md" in texts
    assert col(env).metadata["index_id"] != old_index               # cache invalidated


def test_deleted_document_leaves_no_orphan_chunks(env: Any) -> None:
    _, _, folder, _ = env
    write_doc(folder, "rules-2026.md", BODY_A, doc_version="2026", doc_id="rules")
    write_doc(folder, "fees.md", BODY_B, doc_version="2026", doc_id="fees")
    run(env)
    assert count_where(col(env), doc_id="fees") > 0
    (folder / "fees.md").unlink()
    s = run(env)
    assert s and s["deleted"] == 1 and s["chunks_removed"] > 0
    assert count_where(col(env), doc_id="fees") == 0                # the proof: count by doc_id
    assert count_where(col(env), source="fees.md") == 0
    assert s["orphan_sources"] == []


def test_newer_version_archives_the_older_one_without_reembedding(env: Any) -> None:
    _, _, folder, spy = env
    write_doc(folder, "rules-2025.md", BODY_A, doc_version="2025", doc_id="rules")
    run(env)
    assert count_where(col(env), doc_id="rules", status=ACTIVE) > 0
    write_doc(folder, "rules-2026.md", BODY_A.replace("80", "75"), doc_version="2026",
              doc_id="rules")
    s = run(env)
    assert s and s["new"] == 1 and s["relabelled"] == 1
    assert spy.extracted == ["rules-2026.md"]                       # the old one is not re-read
    assert count_where(col(env), source="rules-2025.md", status=ARCHIVED) > 0
    assert count_where(col(env), source="rules-2025.md", status=ACTIVE) == 0
    # what the API's retriever asks for: superseded == False
    live = col(env).get(where={"superseded": False}, include=["metadatas"])["metadatas"]
    assert {m["source"] for m in live} == {"rules-2026.md"}


def test_restore_brings_back_the_old_version_and_survives_the_next_ingest(env: Any) -> None:
    _, _, folder, spy = env
    write_doc(folder, "rules-2025.md", BODY_A, doc_version="2025", doc_id="rules")
    write_doc(folder, "rules-2026.md", BODY_A.replace("80", "75"), doc_version="2026",
              doc_id="rules")
    run(env)
    assert count_where(col(env), source="rules-2025.md", status=ARCHIVED) > 0

    manage.rollback(col(env), "rules", "2025")                      # the rollback demo
    live = col(env).get(where={"superseded": False}, include=["metadatas"])["metadatas"]
    assert {m["source"] for m in live} == {"rules-2025.md"}
    assert count_where(col(env), source="rules-2026.md", status=ARCHIVED) > 0

    s = run(env)                                                    # a later cron run
    assert s and s["relabelled"] == 0 and spy.embedded == 0
    assert count_where(col(env), source="rules-2025.md", status=ACTIVE) > 0   # still restored

    manage.unpin(col(env), "rules")                                 # back to the automatic rule
    s = run(env)
    assert s and s["relabelled"] == 2                               # 2025 archived, 2026 active
    assert count_where(col(env), source="rules-2025.md", status=ARCHIVED) > 0
    assert count_where(col(env), source="rules-2026.md", status=ACTIVE) > 0


def test_flagged_superseded_document_is_archived_on_first_load(env: Any) -> None:
    _, _, folder, _ = env
    write_doc(folder, "rules-2023.md", BODY_A, doc_version="2023", superseded="true")
    run(env)
    assert count_where(col(env), status=ARCHIVED, superseded=True) > 0


def test_embedding_failure_writes_nothing(env: Any) -> None:
    client, settings, folder, _ = env
    write_doc(folder, "rules-2026.md", BODY_A, doc_version="2026")

    def boom(texts: list[str], s: IndexerSettings) -> list[list[float]]:
        raise RuntimeError("ollama is down")

    with pytest.raises(RuntimeError):
        sync(settings, client, discover_entries(str(folder)), extract_fn=extract,
             clean_fn=clean, chunk_fn=chunk, embed_fn=boom)
    assert col(env).count() == 0


def test_other_model_or_pre_a2_index_asks_for_a_full_rebuild(env: Any) -> None:
    client, settings, folder, spy = env
    write_doc(folder, "rules-2026.md", BODY_A, doc_version="2026")
    other = dataclasses.replace(settings, embed_model="some-other-model")
    assert sync(other, client, discover_entries(str(folder)), extract_fn=spy.extract,
                clean_fn=clean, chunk_fn=chunk, embed_fn=spy.embed) is None
    # an index written before A2: chunks have no content_hash / doc_id
    col(env).upsert(ids=["old-0001"], documents=["x"], embeddings=[VEC],
                    metadatas=[{"source": "old.md", "section": "s", "page": 1,
                                "doc_version": "1", "superseded": False}])
    assert read_state(col(env)) is None
    assert run(env) is None


def test_counts_report_chunks_per_document(env: Any) -> None:
    _, _, folder, _ = env
    write_doc(folder, "rules-2026.md", BODY_A, doc_version="2026", doc_id="rules")
    run(env)
    rows = manage.counts(col(env), "rules")
    assert len(rows) == 1 and rows[0]["chunks"] == col(env).count() and rows[0]["pinned"] is False
