#!/usr/bin/env python3
"""Print eval/results.json as Markdown: recall by question type, for PIPELINE.md and Engineering Notes Q5."""

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "eval" / "results.json")
r = json.loads(path.read_text(encoding="utf-8"))
pct = lambda x: "n/a" if x is None else f"{x:.1%}"  # noqa: E731
print(f"Overall: recall@3 {pct(r.get('recall_at_3'))}, recall@5 {pct(r.get('recall_at_5'))}, citation validity "
      f"{pct(r.get('citation_validity'))} ({r.get('citations_total', 0)} citations), abstention correctness "
      f"{pct(r.get('abstention_correctness'))} ({r.get('abstain_questions', 0)} questions)\n")
print("| Question type | Questions | recall@3 | recall@5 |\n| --- | --- | --- | --- |")
for t, v in r.get("by_type", {}).items():
    print(f"| {t} | {v['n']} | {pct(v['recall_at_3'])} | {pct(v['recall_at_5'])} |")
