#!/usr/bin/env python3
"""Print the chunks the indexer makes for one document, so you can READ what a golden
question is supposed to find (E4, Part B review). Runs the real pipeline stages, no cluster.

    .venv/bin/python scripts/show_chunk.py academic-rules-handbook-jun-2026.pdf 4.24
    .venv/bin/python scripts/show_chunk.py grading-page.html Grading --limit 3
    .venv/bin/python scripts/show_chunk.py student-handbook-jun-2026.pdf --grep plagiarism

The second argument matches chunks whose section starts with it; --grep matches chunk text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "indexer"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("document", help="file name in corpus/raw")
    ap.add_argument("section", nargs="?", default="", help="section prefix, e.g. 4.24")
    ap.add_argument("--grep", default="", help="case-insensitive text to find inside chunks")
    ap.add_argument("--corpus", type=Path, default=ROOT / "corpus" / "raw")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()
    from pipeline.chunk import chunk
    from pipeline.clean import clean
    from pipeline.extract import extract

    path = args.corpus / args.document
    if not path.exists():
        print(f"not found: {path}")
        return 2
    chunks = chunk(clean(extract(str(path), "x", False)), args.chunk_size, args.overlap)
    hits = [c for c in chunks if c.section.lower().startswith(args.section.lower())
            and args.grep.lower() in c.text.lower()]
    for c in hits[:args.limit]:
        print(f"--- {c.source} | section: {c.section} | page {c.page}")
        print(" ".join(c.text.split()))
    print(f"\n{len(hits)} matching chunk(s) of {len(chunks)} in {args.document}")
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
