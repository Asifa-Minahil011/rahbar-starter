#!/usr/bin/env python3
"""Measure recall@3 and recall@5 for several (size, overlap) chunking configurations.

Section 3.1 requires three measured configurations, with the numbers in docs/PIPELINE.md.
This script does the whole experiment without touching your real `rahbar` collection:

  1. extract + clean every document ONCE (OCR is the slow part, so it is not repeated)
  2. for each "size:overlap": chunk, embed, store into a scratch collection
     `rahbar_tune_<size>_<overlap>`, then answer every golden question with the real
     retriever (same Chroma query, superseded filter and exact-term boost as the API)
  3. print a Markdown table you can paste into docs/PIPELINE.md, and write eval/tuning.json

    python scripts/tune_chunking.py --corpus corpus/raw --golden eval/golden.yaml \
        --configs 200:30,400:50,600:100 --chroma-port 8001

No LLM is called, so it takes about as long as indexing (embedding dominates).
Write the golden set BEFORE running this; a golden set written after seeing results is
worth nothing (section 5.1).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "indexer"), str(ROOT / "api")]

from app.config import Settings  # noqa: E402
from app.retrieval.chroma import ChromaRetriever  # noqa: E402
from app.retrieval.embedding import embed_query  # noqa: E402
from pipeline.chunk import chunk  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.config import IndexerSettings, get_settings  # noqa: E402
from pipeline.embed import embed  # noqa: E402
from pipeline.extract import extract  # noqa: E402
from pipeline.run import discover  # noqa: E402
from pipeline.store import get_client, store  # noqa: E402


def _load_eval() -> Any:
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)                 # type: ignore[union-attr]
    return module


def parse_configs(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        size, overlap = part.strip().split(":")
        out.append((int(size), int(overlap)))
    return out


def to_markdown(rows: list[dict[str, Any]]) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    lines = ["| Size (words) | Overlap (words) | Chunks stored | Storage amplification "
             "| recall@3 | recall@5 | recall@5 with exact-term boost |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['size']} | {r['overlap']} | {r['chunks']} | {r['amplification']:.3f} "
                     f"| {pct(r['recall_at_3'])} | {pct(r['recall_at_5'])} "
                     f"| {pct(r['recall_at_5_boost'])} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=ROOT / "corpus" / "raw")
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.yaml")
    ap.add_argument("--configs", default="200:30,400:50,600:100",
                    help="comma-separated size:overlap pairs, in words")
    ap.add_argument("--chroma-host", default="localhost")
    ap.add_argument("--chroma-port", type=int, default=8001)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "tuning.json")
    ap.add_argument("--keep", action="store_true", help="keep the scratch collections")
    args = ap.parse_args()

    ev = _load_eval()
    questions = [q for q in ev.load_golden(args.golden) if q.get("relevant")]
    print(f"{len(questions)} golden questions with relevant passages")

    base: IndexerSettings = dataclasses.replace(
        get_settings(), corpus_dir=str(args.corpus), chroma_host=args.chroma_host,
        chroma_port=args.chroma_port, ollama_url=args.ollama_url, embed_model=args.embed_model)
    docs = []
    for path, version, superseded in discover(base.corpus_dir):
        docs.append(clean(extract(str(path), version, superseded)))
    original_words = sum(len(p.text.split()) for d in docs for p in d.pages)

    api_base = Settings(chroma_host=args.chroma_host, chroma_port=args.chroma_port,
                        ollama_url=args.ollama_url, embed_model=args.embed_model, hybrid_boost=0.0)
    query_vectors = {q["question"]: embed_query(q["question"], api_base) for q in questions}
    client = get_client(base)

    results: list[dict[str, Any]] = []
    for size, overlap in parse_configs(args.configs):
        name = f"rahbar_tune_{size}_{overlap}"
        cfg = dataclasses.replace(base, collection=name, chunk_size=size, chunk_overlap=overlap)
        chunks = [c for d in docs for c in chunk(d, size, overlap)]
        print(f"\n[{size}:{overlap}] {len(chunks)} chunks; embedding ...")
        store(chunks, embed([c.text for c in chunks], cfg), cfg, client)
        entry: dict[str, Any] = {"size": size, "overlap": overlap, "chunks": len(chunks),
                                 "amplification": sum(len(c.text.split()) for c in chunks)
                                 / max(original_words, 1)}
        for label, boost in (("", 0.0), ("_boost", 0.15)):
            settings = dataclasses.replace(api_base, collection=name, hybrid_boost=boost)
            retriever = ChromaRetriever(settings, client=client,
                                        embed=lambda q: query_vectors[q])
            r3, r5, by_type = [], [], {}
            for q in questions:
                refs = [{"source": p.source, "section": p.section}
                        for p in retriever.retrieve(q["question"], 5)]
                a3, a5 = ev.recall_at_k(q["relevant"], refs, 3), ev.recall_at_k(q["relevant"], refs, 5)
                if a3 is None or a5 is None:
                    continue
                r3.append(a3)
                r5.append(a5)
                by_type.setdefault(q.get("type", "ordinary"), []).append(a5)
            mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
            entry[f"recall_at_3{label}"] = mean(r3)
            entry[f"recall_at_5{label}"] = mean(r5)
            entry[f"by_type{label}"] = {t: mean(v) for t, v in sorted(by_type.items())}
        results.append(entry)
        print(f"  recall@3 {entry['recall_at_3']}  recall@5 {entry['recall_at_5']}  "
              f"(with exact-term boost {entry['recall_at_5_boost']})")
        if not args.keep:
            client.delete_collection(name)

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + to_markdown(results))
    print(f"\nwritten to {args.out}. Paste the table into docs/PIPELINE.md and justify the "
          "configuration you ship (size, overlap) with these numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
