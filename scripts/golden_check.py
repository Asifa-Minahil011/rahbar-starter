#!/usr/bin/env python3
"""Check eval/golden.jsonl against the Assignment 3 rules (Part B), and mark reviews.

    python scripts/golden_check.py                       # exit 1 when a rule is not met
    python scripts/golden_check.py --mark-reviewed q01 q02 q03 ...   # after YOU checked them

Rules: >= 25 questions; >= 15 answerable (with a ground-truth document and page), >= 5
unanswerable, >= 5 hard; >= 8 written by hand (label "human"); every question labelled
human or synthetic; >= 8 synthetic questions reviewed by hand (reviewed: true).
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CATEGORIES = {"answerable", "unanswerable", "hard"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids: " + ", ".join(i for i, n in Counter(ids).items() if n > 1))
    cats = Counter(r.get("category") for r in rows)
    labels = Counter(r.get("label") for r in rows)
    for r in rows:
        if r.get("category") not in CATEGORIES:
            problems.append(f"{r.get('id')}: category must be one of {sorted(CATEGORIES)}")
        if r.get("label") not in ("human", "synthetic"):
            problems.append(f"{r.get('id')}: label must be 'human' or 'synthetic'")
        if r.get("category") in ("answerable", "hard"):
            rel = r.get("relevant") or []
            if not rel:
                problems.append(f"{r.get('id')}: no ground-truth document (relevant is empty)")
            for x in rel:
                if not x.get("source") or not x.get("section"):
                    problems.append(f"{r.get('id')}: relevant needs source and section")
                web = str(x.get("source", "")).lower().endswith((".html", ".htm"))
                if x.get("page") in (None, 0) and not web:   # web pages have no page number
                    problems.append(f"{r.get('id')}: ground-truth page missing (run scripts/golden_pages.py)")
        elif not r.get("must_abstain"):
            problems.append(f"{r.get('id')}: unanswerable questions need must_abstain: true")
    if len(rows) < 25:
        problems.append(f"only {len(rows)} questions, need at least 25")
    if cats["answerable"] < 15:
        problems.append(f"only {cats['answerable']} answerable, need at least 15")
    if cats["unanswerable"] < 5:
        problems.append(f"only {cats['unanswerable']} unanswerable, need at least 5")
    if cats["hard"] < 5:
        problems.append(f"only {cats['hard']} hard, need at least 5")
    if labels["human"] < 8:
        problems.append(f"only {labels['human']} human-written questions, need at least 8")
    reviewed = sum(1 for r in rows if r.get("label") == "synthetic" and r.get("reviewed"))
    if reviewed < 8:
        problems.append(f"only {reviewed} synthetic questions reviewed by hand, need at least 8")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--mark-reviewed", nargs="+", metavar="ID")
    args = ap.parse_args()
    rows = load(args.file)
    if args.mark_reviewed:
        known = {r["id"] for r in rows}
        missing = [i for i in args.mark_reviewed if i not in known]
        if missing:
            print("unknown ids:", ", ".join(missing), file=sys.stderr)
            return 2
        for r in rows:
            if r["id"] in args.mark_reviewed:
                r["reviewed"] = True
        args.file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"marked {len(args.mark_reviewed)} question(s) as reviewed by hand")
    cats = Counter(r.get("category") for r in rows)
    labels = Counter(r.get("label") for r in rows)
    reviewed = sum(1 for r in rows if r.get("label") == "synthetic" and r.get("reviewed"))
    print(f"{len(rows)} questions: {dict(cats)}  labels: {dict(labels)}  synthetic reviewed: {reviewed}")
    problems = check(rows)
    for p in problems:
        print("  -", p)
    print("golden set OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
