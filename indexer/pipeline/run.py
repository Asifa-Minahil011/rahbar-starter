"""The indexer Job's entry point. It runs once and exits (section 4.2).

    python -m pipeline.run --dummy    # week 1: three fake chunks, no model needed
    python -m pipeline.run            # the real pipeline: extract, clean, chunk, embed, store

Exit code 0 means the index was written; anything else fails the Job, and
Kubernetes retries it up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .config import IndexerSettings, get_settings
from .models import Chunk, Document


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


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index.

    corpus/fetch.py writes _index.json next to the files it downloads; use it when
    present. Otherwise read Markdown front matter (the CI fixtures work this way).
    """
    root = Path(corpus_dir)
    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        return [(root / e["file"], str(e.get("doc_version", "unknown")),
                 bool(e.get("superseded", False))) for e in entries]
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append((path, meta.get("doc_version", "unknown"),
                      meta.get("superseded", "false").lower() == "true"))
    return found


def _verify_stage(docs: list[Document], paths: dict[str, Path], settings: IndexerSettings,
                  embed_fn: Callable[[list[str], IndexerSettings], list[list[float]]]) -> list[Document]:
    """Stage 1b (RAGOps A1): verify the extracted documents, quarantine the failures, write
    the per-run report, and return only the documents that may be chunked."""
    from .verify import Thresholds, manifest_problems, quarantine_files, verify_documents, write_report

    t = Thresholds.from_env()
    reports = verify_documents(docs, t, lambda texts: embed_fn(texts, settings))
    out_dir = os.environ.get("VERIFY_DIR", settings.verify_dir)
    jp, mp = write_report(reports, out_dir, t, manifest_problems(settings.corpus_dir))
    quarantine_files(reports, paths, out_dir)
    for r in reports:
        log("verified", source=r.source, decision=r.decision,
            checks={c.name: c.status for c in r.checks}, reasons=r.reasons)
    log("verification summary", accepted=sum(r.decision == "accept" for r in reports),
        quarantined=sum(r.decision == "quarantine" for r in reports), report=str(jp))
    keep = {r.source for r in reports if r.decision == "accept"}
    return [d for d in docs if d.source in keep]


def run_pipeline(settings: IndexerSettings) -> int:
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import store

    documents = discover(settings.corpus_dir)
    if not documents:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")
    cleaned: list[Document] = []
    paths: dict[str, Path] = {}
    skipped: list[str] = []
    for path, doc_version, superseded in documents:
        try:
            cleaned.append(clean(extract(str(path), doc_version, superseded)))
            paths[path.name] = path
        except Exception as exc:   # one bad document must not sink the whole index
            log("document skipped", source=path.name, error=repr(exc))
            skipped.append(path.name)
    if os.environ.get("VERIFY", "on").lower() != "off":
        cleaned = _verify_stage(cleaned, paths, settings, embed)
    all_chunks: list[Chunk] = []
    original_words = 0
    for doc in cleaned:
        try:
            pieces = chunk(doc, settings.chunk_size, settings.chunk_overlap)
        except Exception as exc:   # same rule as extraction: one bad document must not sink the index
            log("document skipped", source=doc.source, error=repr(exc))
            skipped.append(doc.source)
            continue
        words = sum(len(p.text.split()) for p in doc.pages)
        original_words += words
        log("chunked", source=doc.source, doc_version=doc.doc_version, superseded=doc.superseded,
            chunks=len(pieces), words=words)
        all_chunks.extend(pieces)
    if not all_chunks:
        raise SystemExit("no chunks produced from any document")
    stored_words = sum(len(c.text.split()) for c in all_chunks)
    log("chunking summary", documents=len(cleaned), skipped=skipped,
        chunks=len(all_chunks), original_words=original_words, stored_words=stored_words,
        storage_amplification=round(stored_words / max(original_words, 1), 3),
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    vectors = embed([c.text for c in all_chunks], settings)
    return store(all_chunks, vectors, settings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
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
            count = run_pipeline(settings)
    except NotImplementedError as exc:
        log("indexer stage not implemented yet", error=str(exc))
        return 2
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
