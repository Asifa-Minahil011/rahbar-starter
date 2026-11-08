#!/usr/bin/env python3
"""Check the docs against the real index, fix the numbers that went stale, and list unfinished markers.

    python scripts/doc_check.py

1. lists every leftover marker in docs/ (VERIFY, RECORD, WRITE-YOUR-OWN, TUNING-TABLE, STARTER-TODO)
2. recomputes chunks and average words per chunk for every source from the live Chroma index, and
   updates the 'words per chunk' sentence in docs/PIPELINE.md (it quoted the first, older index)
3. counts the chunks that mention 'hostel' per source, to re-confirm the deliberately absent topic
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = re.compile(r"VERIFY|RECORD|WRITE-YOUR-OWN|TUNING-TABLE|STARTER-TODO|<<")


def markers(docs: Path) -> list[str]:
    out = []
    for p in sorted(docs.rglob("*.md")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if MARKERS.search(line):
                out.append(f"{p.relative_to(docs.parent)}:{n}: {line.strip()[:110]}")
    return out


def source_stats(sources: list[str], texts: list[str]) -> dict[str, tuple[int, float]]:
    words: dict[str, list[int]] = defaultdict(list)
    for s, t in zip(sources, texts):
        words[s].append(len(t.split()))
    return {s: (len(w), sum(w) / len(w)) for s, w in words.items()}


def avg_for(stats: dict[str, tuple[int, float]], fragment: str) -> float | None:
    hits = [v for k, v in stats.items() if fragment in k]
    return round(hits[0][1]) if hits else None


def patch_pipeline(text: str, a2023: float | None, a2026: float | None) -> tuple[str, bool]:
    if a2023 is None or a2026 is None:
        return text, False
    pat = re.compile(r"\(the\s+2023\s+edition\s+averages\s+about\s+\d+\s+words\s+per\s+chunk,\s+the\s+2026\s+handbook\s+about\s+\d+\)")
    new = f"(the 2023 edition averages about {a2023:.0f} words per chunk, the 2026 handbook about {a2026:.0f})"
    return pat.subn(new, text, count=1)[0], bool(pat.search(text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    docs = args.root / "docs"

    found = markers(docs)
    print(f"== unfinished markers in docs/: {len(found)}")
    for line in found:
        print("   " + line)

    import chromadb

    col = chromadb.HttpClient(host=args.host, port=args.port).get_collection("rahbar")
    got = col.get(include=["metadatas", "documents"])
    sources = [m["source"] for m in got["metadatas"]]
    stats = source_stats(sources, got["documents"])
    print(f"\n== index: {len(sources)} chunks")
    for s, (n, avg) in sorted(stats.items()):
        print(f"   {n:4d} chunks, {avg:6.1f} words on average  {s}")

    p = args.root / "docs" / "PIPELINE.md"
    text = p.read_text(encoding="utf-8")
    new, matched = patch_pipeline(text, avg_for(stats, "aug-2023"), avg_for(stats, "handbook-jun-2026"))
    if matched and new != text:
        p.write_text(new, encoding="utf-8")
        print("\n== PIPELINE.md: updated the 'words per chunk' sentence to the final index")
    elif matched:
        print("\n== PIPELINE.md: 'words per chunk' sentence already matches the index")
    else:
        print("\n== PIPELINE.md: 'words per chunk' sentence not found (check the Chunking section by hand)")

    hostel: dict[str, int] = defaultdict(int)
    for s, d in zip(sources, got["documents"]):
        if "hostel" in d.lower():
            hostel[s] += 1
    print("\n== chunks mentioning 'hostel' (the deliberately absent topic):",
          dict(hostel) if hostel else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
