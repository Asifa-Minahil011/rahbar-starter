#!/usr/bin/env python3
"""Choose RELEVANCE_FLOOR from data (section 3.4). No LLM needed: it asks the running API's
/api/retrieve for every golden question and compares the best similarity of
  - answerable questions (the floor must sit BELOW the lowest of these), and
  - must-abstain questions (the floor must sit ABOVE the highest of these).

    python scripts/pick_floor.py --api http://localhost:8010
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]


def best_scores(api: str, questions: list[dict]) -> list[tuple[float, dict]]:
    out = []
    with httpx.Client(timeout=120.0) as client:
        for q in questions:
            r = client.post(f"{api.rstrip('/')}/api/retrieve", json={"question": q["question"]})
            r.raise_for_status()
            scores = [x["score"] for x in r.json()["retrieved"]]
            out.append((max(scores) if scores else 0.0, q))
    return out


def suggest(answerable: list[float], abstain: list[float]) -> tuple[float | None, str]:
    """Return (floor, explanation)."""
    lo, hi = min(answerable), max(abstain)
    if hi < lo:
        floor = round((hi + lo) / 2, 2)
        return floor, (f"clean gap: highest abstain score {hi:.3f} < lowest answerable score "
                       f"{lo:.3f}; floor {floor} sits in the middle (margin {(lo - hi) / 2:.3f} each side)")
    return None, (f"OVERLAP: an abstain question scores {hi:.3f}, at or above the lowest answerable "
                  f"question ({lo:.3f}). No floor separates them; the floor alone cannot make abstention "
                  f"100%. Pick a floor just above {hi:.2f} and accept that the lowest answerable question "
                  f"will be declined, or rely on the model's own 'not covered' answer for that topic.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--api", required=True)
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.yaml")
    args = ap.parse_args()
    questions = yaml.safe_load(args.golden.read_text(encoding="utf-8"))["questions"]
    scored = best_scores(args.api, questions)
    answerable = [(s, q) for s, q in scored if q.get("relevant") and not q.get("must_abstain")]
    abstain = [(s, q) for s, q in scored if q.get("must_abstain")]
    if not answerable or not abstain:
        sys.exit("need at least one answerable and one must-abstain question")
    print("best similarity per question (answerable, lowest first):")
    for s, q in sorted(answerable, key=lambda t: t[0]):
        print(f"  {s:.3f}  {q['id']:<5} {q['type']:<9} {q['question'][:70]}")
    print("\nmust-abstain questions (highest first):")
    for s, q in sorted(abstain, key=lambda t: -t[0]):
        print(f"  {s:.3f}  {q['id']:<5} {q['type']:<9} {q['question'][:70]}")
    floor, why = suggest([s for s, _ in answerable], [s for s, _ in abstain])
    print(f"\n{why}")
    if floor is not None:
        print(f"\nSet RELEVANCE_FLOOR={floor} in the ConfigMap, and restart the API with it for the full eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
