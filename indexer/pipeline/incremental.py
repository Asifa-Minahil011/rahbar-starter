"""A2: incremental update of the live collection (assignment 03, part A2).

The old store() rebuilt the whole index on every run (18 minutes of embedding on a laptop).
This module compares the files on disk with what the collection already holds and touches
only what changed. Unit of change = one source file (`source` in the chunk metadata);
`doc_id` groups the versions of one policy.

    unchanged file   -> not even extracted: 0 chunks embedded
    new file         -> extract, chunk, embed, add
    changed file     -> embed the new chunks, upsert them, delete only its own stale chunks
    deleted file     -> delete every chunk with that source (no orphans)
    older version    -> status=archived, superseded=True (metadata only, no re-embedding);
                        the API's existing `superseded=False` filter excludes it
    pinned source    -> an operator ran `manage archive/restore/rollback`; ingestion leaves
                        its status alone until `manage unpin`

Nothing is written until embedding has succeeded, so a failed embed leaves the index as it was.
A full rebuild (store.store(), scratch collection + swap) is still used when the index cannot be
updated in place: first run, a different embedding model or chunk size, or an index built before A2.
Every change gives the collection a new `index_id`, which invalidates the API's semantic cache.

Version order inside one doc_id is plain string order of doc_version ("2026-06" > "2023-08").
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import IndexerSettings
from .models import Chunk

ACTIVE = "active"
ARCHIVED = "archived"
PAGE = 500          # rows per get/update/delete call
WRITE_BATCH = 128   # rows per upsert call


def _log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer.incremental", "msg": msg, **fields}), flush=True)


@dataclass(frozen=True)
class Entry:
    """One document found in the corpus folder."""

    path: Path
    doc_version: str
    superseded: bool      # flagged as replaced in _index.json / front matter
    doc_id: str

    @property
    def source(self) -> str:
        return self.path.name


@dataclass
class SourceState:
    """What the collection currently holds for one source file."""

    source: str
    doc_id: str
    doc_version: str
    content_hash: str
    status: str
    pinned: bool
    ids: list[str] = field(default_factory=list)


@dataclass
class Plan:
    new: list[Entry] = field(default_factory=list)
    changed: list[Entry] = field(default_factory=list)
    unchanged: list[Entry] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)                 # sources gone from disk
    relabel: dict[str, dict[str, Any]] = field(default_factory=dict)  # metadata-only updates
    status: dict[str, str] = field(default_factory=dict)             # effective status per source
    pinned: dict[str, bool] = field(default_factory=dict)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_state(collection: Any) -> dict[str, SourceState] | None:
    """Group the collection's chunks by source. None means the index predates A2 (no
    content_hash/doc_id on its chunks) and has to be rebuilt once."""
    state: dict[str, SourceState] = {}
    offset = 0
    while True:
        got = collection.get(include=["metadatas"], limit=PAGE, offset=offset)
        ids = got["ids"]
        if not ids:
            return state
        for cid, meta in zip(ids, got["metadatas"]):
            if "content_hash" not in meta or "doc_id" not in meta:
                return None
            source = str(meta["source"])
            st = state.get(source)
            if st is None:
                state[source] = SourceState(
                    source=source, doc_id=str(meta["doc_id"]),
                    doc_version=str(meta.get("doc_version", "")),
                    content_hash=str(meta["content_hash"]),
                    status=str(meta.get("status", ACTIVE)),
                    pinned=bool(meta.get("pinned", False)), ids=[cid])
            else:
                st.ids.append(cid)
                if st.content_hash != str(meta["content_hash"]):
                    st.content_hash = ""      # half-written source: treat as changed
        offset += len(ids)


def desired_status(entries: list[Entry]) -> dict[str, str]:
    """Per doc_id, only the newest non-superseded version is active; the rest are archived."""
    groups: dict[str, list[Entry]] = defaultdict(list)
    for e in entries:
        groups[e.doc_id].append(e)
    out: dict[str, str] = {}
    for members in groups.values():
        live = [e for e in members if not e.superseded]
        newest = max((e.doc_version for e in live), default=None)
        for e in members:
            out[e.source] = ACTIVE if (not e.superseded and e.doc_version == newest) else ARCHIVED
    return out


def make_plan(entries: list[Entry], state: dict[str, SourceState],
              hashes: dict[str, str]) -> Plan:
    plan = Plan()
    want = desired_status(entries)
    present = {e.source for e in entries}
    plan.deleted = sorted(s for s in state if s not in present)
    for e in entries:
        st = state.get(e.source)
        pinned = st.pinned if st else False
        status = st.status if (st and st.pinned) else want[e.source]
        plan.status[e.source] = status
        plan.pinned[e.source] = pinned
        if st is None:
            plan.new.append(e)
        elif st.content_hash != hashes[e.source]:
            plan.changed.append(e)
        else:
            plan.unchanged.append(e)
            if (st.status, st.doc_id, st.doc_version) != (status, e.doc_id, e.doc_version):
                plan.relabel[e.source] = {"doc_id": e.doc_id, "doc_version": e.doc_version,
                                          "status": status, "superseded": status != ACTIVE}
    return plan


def compatible(collection: Any, settings: IndexerSettings) -> bool:
    """New vectors can only be mixed into an index built with the same model and chunking."""
    meta = collection.metadata or {}
    try:
        return (meta.get("embed_model") == settings.embed_model
                and int(meta.get("chunk_size", -1)) == settings.chunk_size
                and int(meta.get("chunk_overlap", -1)) == settings.chunk_overlap)
    except (TypeError, ValueError):
        return False


def touch_index(collection: Any) -> str:
    """Give the collection a new index_id (the semantic cache keys on it).

    Chroma's modify(metadata=...) replaces the whole metadata dict, so copy it first.
    """
    meta = {k: v for k, v in (collection.metadata or {}).items() if not k.startswith("hnsw:")}
    meta["index_id"] = uuid.uuid4().hex
    meta["chunk_count"] = collection.count()
    meta["indexed_at"] = now()
    collection.modify(metadata=meta)
    return str(meta["index_id"])


def set_meta(collection: Any, source: str, **fields: Any) -> int:
    """Update metadata on every chunk of one source, without touching text or vectors."""
    got = collection.get(where={"source": source}, include=["metadatas"])
    ids, metas = got["ids"], got["metadatas"]
    for i in range(0, len(ids), PAGE):
        collection.update(ids=ids[i:i + PAGE],
                          metadatas=[{**m, **fields} for m in metas[i:i + PAGE]])
    return len(ids)


def delete_source(collection: Any, source: str) -> int:
    got = collection.get(where={"source": source}, include=[])
    ids = got["ids"]
    for i in range(0, len(ids), PAGE):
        collection.delete(ids=ids[i:i + PAGE])
    return len(ids)


def sync(settings: IndexerSettings, client: Any, entries: list[Entry], *,
         extract_fn: Callable[..., Any], clean_fn: Callable[..., Any],
         chunk_fn: Callable[..., list[Chunk]],
         embed_fn: Callable[..., list[list[float]]]) -> dict[str, Any] | None:
    """Bring the live collection in line with `entries`. Returns a summary, or None when the
    index cannot be updated in place and the caller must do a full rebuild."""
    try:
        collection = client.get_collection(settings.collection)
    except Exception:
        _log("no collection yet; full build needed")
        return None
    if not compatible(collection, settings):
        _log("index was built with a different model or chunking; full rebuild needed")
        return None
    state = read_state(collection)
    if state is None:
        _log("index predates A2 (chunks have no content_hash); full rebuild needed")
        return None

    hashes = {e.source: file_sha256(e.path) for e in entries}
    plan = make_plan(entries, state, hashes)
    stamp = now()

    # 1. extract and chunk only what is new or changed
    fresh: dict[str, list[Chunk]] = {}
    skipped: list[str] = []
    for e in plan.new + plan.changed:
        try:
            doc = clean_fn(extract_fn(str(e.path), e.doc_version, e.superseded))
            pieces = chunk_fn(doc, settings.chunk_size, settings.chunk_overlap)
            if not pieces:
                raise ValueError("no chunks produced")
        except Exception as exc:     # one bad file must not stop the others
            _log("document skipped", source=e.source, error=repr(exc))
            skipped.append(e.source)
            continue
        status = plan.status[e.source]
        for c in pieces:
            c.doc_id, c.content_hash = e.doc_id, hashes[e.source]
            c.status, c.superseded = status, status != ACTIVE
            c.pinned, c.ingested_at = plan.pinned[e.source], stamp
        fresh[e.source] = pieces
        _log("chunked", source=e.source, doc_id=e.doc_id, doc_version=e.doc_version,
             status=status, chunks=len(pieces),
             kind="new" if e in plan.new else "changed")

    # 2. embed everything new in one call; nothing has been written yet
    new_chunks = [c for pieces in fresh.values() for c in pieces]
    vectors = embed_fn([c.text for c in new_chunks], settings) if new_chunks else []
    if vectors:
        want_dim = (collection.metadata or {}).get("dimension")
        if want_dim is not None and int(want_dim) != len(vectors[0]):
            raise RuntimeError(f"model returned {len(vectors[0])}-d vectors but the index "
                               f"holds {want_dim}-d")
    vec_by_id = {c.id: v for c, v in zip(new_chunks, vectors)}

    # 3. write: upsert new chunks first, then delete only this source's stale chunk ids
    removed = 0
    for source, pieces in fresh.items():
        for i in range(0, len(pieces), WRITE_BATCH):
            part = pieces[i:i + WRITE_BATCH]
            collection.upsert(ids=[c.id for c in part], documents=[c.text for c in part],
                              embeddings=[vec_by_id[c.id] for c in part],
                              metadatas=[c.metadata() for c in part])
        if source in state:
            stale = sorted(set(state[source].ids) - {c.id for c in pieces})
            for i in range(0, len(stale), PAGE):
                collection.delete(ids=stale[i:i + PAGE])
            removed += len(stale)
    for source in plan.deleted:
        removed += delete_source(collection, source)
        _log("deleted source", source=source, doc_id=state[source].doc_id)
    for source, fields in plan.relabel.items():
        set_meta(collection, source, **fields)
        _log("relabelled", source=source, **fields)

    changed_index = bool(fresh or plan.deleted or plan.relabel)
    if changed_index:
        touch_index(collection)

    # 4. prove it: every source in the collection is a file that is on disk
    final = read_state(collection) or {}
    on_disk = {e.source for e in entries}
    orphans = sorted(s for s in final if s not in on_disk)
    by_doc: dict[str, int] = defaultdict(int)
    for st in final.values():
        by_doc[st.doc_id] += len(st.ids)
    summary = {
        "mode": "incremental", "sources_on_disk": len(entries),
        "unchanged": len(plan.unchanged), "new": len(plan.new), "changed": len(plan.changed),
        "deleted": len(plan.deleted), "relabelled": len(plan.relabel), "skipped": skipped,
        "chunks_embedded": len(new_chunks), "chunks_removed": removed,
        "chunks_total": int(collection.count()), "orphan_sources": orphans,
        "chunks_by_doc_id": dict(sorted(by_doc.items())), "index_changed": changed_index,
    }
    return summary


def write_report(settings: IndexerSettings, summary: dict[str, Any]) -> str | None:
    """Save the run summary next to the verification report, when REPORT_DIR is set."""
    if not settings.report_dir:
        return None
    out = Path(settings.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ingest-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)
