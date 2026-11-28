"""The indexer Job's entry point. It runs once and exits (section 4.2).

    python -m pipeline.run --dummy    # week 1: three fake chunks, no model needed
    python -m pipeline.run            # incremental: only new/changed/deleted files are touched
    python -m pipeline.run --full     # rebuild everything (a new embedding model, say)

Exit code 0 means the index was written; anything else fails the Job, and
Kubernetes retries it up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .chunk import _slug
from .config import IndexerSettings, get_settings
from .incremental import (ACTIVE, Entry, desired_status, file_sha256, now, sync,
                          write_report)
from .models import Chunk


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer", "msg": msg, **fields}), flush=True)


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (used by eval/fixtures)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def discover_entries(corpus_dir: str) -> list[Entry]:
    """Every document to index, with its doc_version, superseded flag and doc_id.

    corpus/fetch.py writes _index.json next to the files it downloads; use it when present.
    Otherwise read Markdown front matter (the CI fixtures and Batch B work this way).
    `doc_id` is optional in both: it defaults to the file name without its extension, and
    two files with the same doc_id are versions of one document (see incremental.py).
    """
    root = Path(corpus_dir)
    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        return [Entry(root / e["file"], str(e.get("doc_version", "unknown")),
                      bool(e.get("superseded", False)),
                      str(e.get("doc_id") or _slug(e["file"]))) for e in entries]
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith(("readme", "_")):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append(Entry(path, meta.get("doc_version", "unknown"),
                           meta.get("superseded", "false").lower() == "true",
                           meta.get("doc_id") or _slug(path.name)))
    return found


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """(path, doc_version, superseded) for every document; kept for older callers."""
    return [(e.path, e.doc_version, e.superseded) for e in discover_entries(corpus_dir)]


def run_pipeline(settings: IndexerSettings, force_full: bool = False) -> int:
    """Incremental update when the live index allows it, otherwise a full rebuild.
    Returns the number of chunks in the collection afterwards."""
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import get_client, store

    entries = discover_entries(settings.corpus_dir)
    if not entries:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")
    client = get_client(settings)

    if not force_full:
        summary = sync(settings, client, entries, extract_fn=extract, clean_fn=clean,
                       chunk_fn=chunk, embed_fn=embed)
        if summary is not None:
            log("ingest summary", **summary)
            report = write_report(settings, summary)
            if report:
                log("ingest report written", path=report)
            if summary["orphan_sources"]:
                raise SystemExit(f"orphan chunks for deleted sources: {summary['orphan_sources']}")
            return int(summary["chunks_total"])

    return _full_rebuild(settings, client, entries, extract, clean, chunk, embed, store)


def _full_rebuild(settings: IndexerSettings, client: object, entries: list[Entry],
                  extract: object, clean: object, chunk: object, embed: object,
                  store: object) -> int:
    """First build, or the model / chunk size changed, or an index from before A2."""
    status = desired_status(entries)
    stamp = now()
    all_chunks: list[Chunk] = []
    original_words = 0
    skipped: list[str] = []
    for e in entries:
        try:
            doc = clean(extract(str(e.path), e.doc_version, e.superseded))      # type: ignore[operator]
            pieces = chunk(doc, settings.chunk_size, settings.chunk_overlap)    # type: ignore[operator]
        except Exception as exc:   # one bad document must not sink the whole index
            log("document skipped", source=e.source, error=repr(exc))
            skipped.append(e.source)
            continue
        digest = file_sha256(e.path)
        for c in pieces:
            c.doc_id, c.content_hash, c.ingested_at = e.doc_id, digest, stamp
            c.status, c.superseded = status[e.source], status[e.source] != ACTIVE
        words = sum(len(p.text.split()) for p in doc.pages)
        original_words += words
        log("chunked", source=e.source, doc_id=e.doc_id, doc_version=e.doc_version,
            status=status[e.source], chunks=len(pieces), words=words)
        all_chunks.extend(pieces)
    if not all_chunks:
        raise SystemExit("no chunks produced from any document")
    stored_words = sum(len(c.text.split()) for c in all_chunks)
    log("chunking summary", documents=len(entries) - len(skipped), skipped=skipped,
        chunks=len(all_chunks), original_words=original_words, stored_words=stored_words,
        storage_amplification=round(stored_words / max(original_words, 1), 3),
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    vectors = embed([c.text for c in all_chunks], settings)                     # type: ignore[operator]
    count = store(all_chunks, vectors, settings, client)                        # type: ignore[operator]
    summary = {"mode": "full", "sources_on_disk": len(entries), "skipped": skipped,
               "chunks_embedded": len(all_chunks), "chunks_total": count,
               "unchanged": 0, "new": len(entries) - len(skipped), "changed": 0,
               "deleted": 0, "relabelled": 0, "orphan_sources": []}
    log("ingest summary", **summary)
    report = write_report(settings, summary)
    if report:
        log("ingest report written", path=report)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    parser.add_argument("--full", action="store_true",
                        help="ignore the live index and rebuild everything (re-embeds all chunks)")
    args = parser.parse_args()
    settings = get_settings()
    log("indexer starting", dummy=args.dummy, embed_model=settings.embed_model,
        collection=settings.collection, chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        pod=os.environ.get("HOSTNAME", "-"))
    started = time.perf_counter()
    try:
        if args.dummy:
            from .dummy import write_dummy_index
            from .store import get_client

            count = write_dummy_index(settings, get_client(settings))
        else:
            count = run_pipeline(settings, force_full=args.full)
    except NotImplementedError as exc:
        log("indexer stage not implemented yet", error=str(exc))
        return 2
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
