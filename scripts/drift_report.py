#!/usr/bin/env python3
"""E3: what changed in the results when new data was ingested (paper section 4.2.7).

    python scripts/drift_report.py eval/results/suite-before.json eval/results/suite-after.json
    python scripts/drift_report.py before.json after.json --new-sources batch-b-rules.pdf,batch-b-fees.pdf

Compares two suite results (run before and after ingesting Batch B): every metric that moved,
then the questions whose retrieval changed (recall, rank of the first correct chunk, top-1
chunk) and the questions where the new documents now appear in the retrieved chunks. Those
are the candidates for the "what changed and why" paragraph of the report.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def per_question(res: dict, level: str) -> dict:
    return {q["id"]: q for q in (res.get(level) or {}).get("per_question", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--new-sources", default="", help="comma-separated source names that came with Batch B")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    before, after = json.loads(args.before.read_text()), json.loads(args.after.read_text())
    new_sources = {s.strip() for s in args.new_sources.split(",") if s.strip()}

    print("metrics that moved:")
    moved = {}
    for k in sorted(set(before["flat"]) | set(after["flat"])):
        b, a = before["flat"].get(k), after["flat"].get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) > 1e-9:
            moved[k] = round(a - b, 4)
            print(f"  {k:<32} {b:>10.4f} -> {a:>10.4f}   ({a - b:+.4f})")
    if not moved:
        print("  none")

    changes = []
    cb, ca = per_question(before, "component"), per_question(after, "component")
    for qid in cb:
        if qid not in ca:
            continue
        b, a = cb[qid], ca[qid]
        notes = []
        if b["recall_at_5"] != a["recall_at_5"]:
            notes.append(f"recall@5 {b['recall_at_5']} -> {a['recall_at_5']}")
        if b["first_hit_rank"] != a["first_hit_rank"]:
            notes.append(f"first correct chunk rank {b['first_hit_rank']} -> {a['first_hit_rank']}")
        if (b["top1"] or {}).get("section") != (a["top1"] or {}).get("section") or \
           (b["top1"] or {}).get("source") != (a["top1"] or {}).get("source"):
            notes.append(f"top-1 {(b['top1'] or {}).get('source')}/{(b['top1'] or {}).get('section')} -> "
                         f"{(a['top1'] or {}).get('source')}/{(a['top1'] or {}).get('section')}")
        added = sorted((set(a["sources"]) - set(b["sources"])) & (new_sources or set(a["sources"]) - set(b["sources"])))
        if added:
            notes.append("new sources in the top-k: " + ", ".join(added))
        if notes:
            changes.append({"id": qid, "notes": notes})
    eb, ea = per_question(before, "e2e"), per_question(after, "e2e")
    for qid in eb:
        if qid in ea and eb[qid]["answered_from_context"] != ea[qid]["answered_from_context"]:
            changes.append({"id": qid, "notes": [f"answered_from_context {eb[qid]['answered_from_context']} -> "
                                                 f"{ea[qid]['answered_from_context']}"]})
    print(f"\nquestions whose result changed: {len(changes)}")
    for c in changes:
        print(f"  {c['id']}: " + "; ".join(c["notes"]))
    if args.out:
        args.out.write_text(json.dumps({"before": str(args.before), "after": str(args.after),
                                        "metrics_moved": moved, "questions_changed": changes}, indent=2))
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
