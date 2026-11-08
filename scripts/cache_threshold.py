#!/usr/bin/env python3
"""Choose the semantic-cache threshold with data (section 3.3).

The lecture's rule: the LOWEST threshold with ZERO false hits on your test questions.
A false hit = two questions with different answers whose similarity is at or above the
threshold, so the cache would serve the wrong answer.

Put pairs in eval/cache_pairs.yaml:
  same:      pairs that mean the same thing (paraphrases): these SHOULD hit
  different: pairs that look alike but have different answers (clause 4.23 vs 4.24,
             a different fee, 2023 vs 2026): these must NEVER hit

    python scripts/cache_threshold.py --thresholds 0.85,0.90,0.95,0.98

Needs only Ollama (it embeds the questions); no Chroma, no API.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.config import Settings  # noqa: E402
from app.retrieval.embedding import embed_query  # noqa: E402


def cos(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / ((na * nb) or 1.0)


def report(same: list[float], different: list[tuple[float, str]], thresholds: list[float]) -> list[str]:
    lines = ["| Threshold | True hits (of %d paraphrases) | False hits (of %d near-misses) |"
             % (len(same), len(different)), "| --- | --- | --- |"]
    for t in thresholds:
        hits = sum(s >= t for s in same)
        false = [label for s, label in different if s >= t]
        lines.append(f"| {t:.2f} | {hits} | {len(false)}" + (f": {false[0]}" if false else "") + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=ROOT / "eval" / "cache_pairs.yaml")
    ap.add_argument("--thresholds", default="0.85,0.90,0.93,0.95,0.97,0.98")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    args = ap.parse_args()

    settings = Settings(ollama_url=args.ollama_url, embed_model=args.embed_model)
    data = yaml.safe_load(args.pairs.read_text(encoding="utf-8"))
    cache: dict[str, list[float]] = {}

    def vec(q: str) -> list[float]:
        if q not in cache:
            cache[q] = embed_query(q, settings)
        return cache[q]

    same = [cos(vec(a), vec(b)) for a, b in data["same"]]
    different = [(cos(vec(a), vec(b)), f"{a!r} vs {b!r}") for a, b in data["different"]]
    print("similarity of paraphrases     :", " ".join(f"{s:.3f}" for s in same))
    print("similarity of near-misses     :", " ".join(f"{s:.3f}" for s, _ in different))
    print("highest near-miss             :", f"{max(s for s, _ in different):.3f}")
    print()
    print("\n".join(report(same, different, [float(x) for x in args.thresholds.split(",")])))
    print("\nShip the lowest threshold with 0 false hits, and name the false hit that ruled out "
          "the one below it in docs/PIPELINE.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
