#!/usr/bin/env python3
"""Search the live index for a word or clause number, to write and check golden questions.

    python scripts/find_clause.py withdraw
    python scripts/find_clause.py "6.6" --source handbook
    python scripts/find_clause.py refund --source fee --width 400

Prints source, section (the exact string for golden.yaml) and the chunk text. Read the text:
a golden question is only as good as your check that the answer really is there.
"""

from __future__ import annotations

import argparse
import re


def main() -> int:
    import chromadb

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("term", help="text to look for (case-insensitive; a regex is fine)")
    ap.add_argument("--source", default="", help="only sources whose name contains this")
    ap.add_argument("--width", type=int, default=260, help="characters of text to show")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--collection", default="rahbar")
    args = ap.parse_args()

    col = chromadb.HttpClient(host=args.host, port=args.port).get_collection(args.collection)
    got = col.get(include=["metadatas", "documents"])
    pattern = re.compile(args.term, re.I)
    shown = 0
    for meta, text in zip(got["metadatas"], got["documents"]):
        if args.source and args.source.lower() not in meta["source"].lower():
            continue
        flat = " ".join(text.split())
        if not pattern.search(flat):
            continue
        tag = "SUPERSEDED " if meta.get("superseded") else ""
        print(f"\n{tag}{meta['source']}  |  section: {meta['section']!r}  |  page {meta['page']}")
        print("  " + flat[: args.width])
        shown += 1
        if shown >= args.max:
            break
    print(f"\n{shown} chunk(s) shown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
