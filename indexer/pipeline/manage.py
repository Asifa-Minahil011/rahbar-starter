"""Operator commands for the live collection (A2: archive, restore, rollback, count).

    python -m pipeline.manage counts [--doc-id ID]
    python -m pipeline.manage archive  <source-file | doc_id>
    python -m pipeline.manage restore  <source-file | doc_id>
    python -m pipeline.manage rollback <doc_id> --to <doc_version>
    python -m pipeline.manage unpin    <source-file | doc_id>

archive/restore/rollback only change metadata (status, superseded, pinned): the text and the
vectors stay, so a restore is instant and needs no embedding. They pin the source, so the
next ingestion run does not undo the decision; `unpin` hands control back to the automatic rule.
Every change gives the collection a new index_id, so cached answers are dropped.

This writes to the vector store, so on the cluster it runs as an indexer pod (the only pod the
NetworkPolicy lets reach Chroma): see scripts/manage.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import get_settings
from .incremental import ACTIVE, ARCHIVED, read_state, set_meta, touch_index


def select(collection: Any, selector: str) -> dict[str, dict[str, str]]:
    """Resolve a source file name, else a doc_id, to {source: {doc_id, doc_version}}."""
    found: dict[str, dict[str, str]] = {}
    for key in ("source", "doc_id"):
        got = collection.get(where={key: selector}, include=["metadatas"])
        for meta in got["metadatas"]:
            found[str(meta["source"])] = {"doc_id": str(meta["doc_id"]),
                                          "doc_version": str(meta.get("doc_version", ""))}
        if found:
            break
    return found


def _apply(collection: Any, sources: list[str], status: str | None, pinned: bool) -> int:
    changed = 0
    for source in sources:
        fields: dict[str, Any] = {"pinned": pinned}
        if status is not None:
            fields.update(status=status, superseded=status != ACTIVE)
        changed += set_meta(collection, source, **fields)
    if sources:
        touch_index(collection)
    return changed


def archive(collection: Any, selector: str) -> int:
    return _apply(collection, sorted(select(collection, selector)), ARCHIVED, True)


def restore(collection: Any, selector: str) -> int:
    return _apply(collection, sorted(select(collection, selector)), ACTIVE, True)


def unpin(collection: Any, selector: str) -> int:
    return _apply(collection, sorted(select(collection, selector)), None, False)


def rollback(collection: Any, doc_id: str, to_version: str) -> int:
    """Make `to_version` the active one of a doc_id and archive every other version."""
    members = select(collection, doc_id)
    if not members or not all(m["doc_id"] == doc_id for m in members.values()):
        raise SystemExit(f"no document with doc_id {doc_id!r}")
    if to_version not in {m["doc_version"] for m in members.values()}:
        have = sorted({m["doc_version"] for m in members.values()})
        raise SystemExit(f"{doc_id} has no version {to_version!r}; it has {have}")
    changed = 0
    for source, m in sorted(members.items()):
        changed += set_meta(collection, source, pinned=True,
                            status=ACTIVE if m["doc_version"] == to_version else ARCHIVED,
                            superseded=m["doc_version"] != to_version)
    touch_index(collection)
    return changed


def counts(collection: Any, doc_id: str | None = None) -> list[dict[str, Any]]:
    state = read_state(collection)
    if state is None:
        raise SystemExit("this index has no doc_id/content_hash yet; run a full ingestion first")
    rows = [{"doc_id": s.doc_id, "source": s.source, "doc_version": s.doc_version,
             "status": s.status, "pinned": s.pinned, "chunks": len(s.ids)}
            for s in state.values() if doc_id is None or s.doc_id == doc_id]
    return sorted(rows, key=lambda r: (r["doc_id"], r["doc_version"], r["source"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operate on the live collection.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("counts", help="chunks per document, status and pin")
    c.add_argument("--doc-id", default=None)
    for name in ("archive", "restore", "unpin"):
        p = sub.add_parser(name)
        p.add_argument("selector", help="a source file name or a doc_id")
    r = sub.add_parser("rollback", help="activate one version of a doc_id, archive the rest")
    r.add_argument("doc_id")
    r.add_argument("--to", required=True, dest="to_version")
    args = parser.parse_args(argv)

    from .store import get_client
    settings = get_settings()
    collection = get_client(settings).get_collection(settings.collection)

    if args.cmd == "counts":
        rows = counts(collection, args.doc_id)
        for row in rows:
            print(json.dumps(row))
        total = sum(r["chunks"] for r in rows)
        label = f"doc_id={args.doc_id}" if args.doc_id else "all documents"
        print(json.dumps({"summary": label, "sources": len(rows), "chunks": total,
                          "collection_total": int(collection.count())}))
        return 0
    if args.cmd == "rollback":
        n = rollback(collection, args.doc_id, args.to_version)
    else:
        n = {"archive": archive, "restore": restore, "unpin": unpin}[args.cmd](
            collection, args.selector)
    print(json.dumps({"command": args.cmd, "chunks_updated": n}))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
