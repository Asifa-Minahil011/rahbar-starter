#!/usr/bin/env python3
"""E4 helper: turn uncovered live queries into DRAFT golden questions to review by hand.

    python scripts/golden_draft.py --api http://localhost:8010                  # print drafts, write eval/golden_drafts.jsonl
    python scripts/golden_draft.py --api http://localhost:8010 --append L14,L15,L17   # add the ones you checked

For each uncovered query in the coverage result it asks /api/retrieve and shows the top chunks.
If the best chunk is above the relevance floor the draft is "answerable" with that chunk's
source and section as ground truth; otherwise it is "unanswerable" (must abstain). The draft is
only a guess from the retriever: READ the chunk, fix `relevant` and `expect` if wrong, then
--append. Appended questions are marked label=synthetic, reviewed=false until you run
golden_check.py --mark-reviewed after reading them. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def post(api: str, path: str, body: dict, host: str | None) -> dict:
    req = urllib.request.Request(api.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Host": host} if host else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def draft(row: dict, got: list[dict], floor: float, qid: str) -> dict:
    top = got[0] if got else None
    answerable = bool(top) and top["score"] >= floor
    base = {"id": qid, "label": "synthetic", "reviewed": False, "question": row["query"],
            "from_live_query": row["id"], "forbid": []}
    if answerable:
        return {**base, "category": "answerable", "kind": "ordinary",
                "relevant": [{"source": top["source"], "section": top["section"], "page": top.get("page")}],
                "expect": "", "must_abstain": False, "needs_expect": True}
    return {**base, "category": "unanswerable", "kind": "ordinary", "relevant": [], "expect": "",
            "must_abstain": True, "needs_absence_check": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True)
    ap.add_argument("--host")
    ap.add_argument("--coverage", type=Path, default=Path("eval/results/coverage-before.json"))
    ap.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--drafts", type=Path, default=Path("eval/golden_drafts.jsonl"))
    ap.add_argument("--floor", type=float, default=0.68)
    ap.add_argument("--append", default="", help="comma-separated live-query ids to add to golden.jsonl")
    args = ap.parse_args()

    if args.append:
        wanted = {x.strip() for x in args.append.split(",") if x.strip()}
        drafts = {d["from_live_query"]: d for d in map(json.loads, args.drafts.read_text().splitlines() if args.drafts.exists() else [])}
        missing = wanted - set(drafts)
        if missing:
            print("no draft for:", ", ".join(sorted(missing)), "(run without --append first)")
            return 2
        for d in (drafts[w] for w in sorted(wanted)):
            if d["category"] == "answerable" and not d["expect"].strip():
                print(f"{d['from_live_query']}: fill `expect` in {args.drafts} first (a word the answer must contain)")
                return 2
        with args.golden.open("a", encoding="utf-8") as fh:
            for w in sorted(wanted):
                fh.write(json.dumps({k: v for k, v in drafts[w].items() if k != "needs_expect"}, ensure_ascii=False) + "\n")
        print(f"appended {len(wanted)} question(s) to {args.golden}")
        return 0

    cov = json.loads(args.coverage.read_text())
    rows = [r for r in cov["rows"] if not r["covered"]]
    existing = [json.loads(x)["id"] for x in args.golden.read_text().splitlines() if x.strip()]
    nxt = max(int(i[1:]) for i in existing if i[1:].isdigit()) + 1
    out = []
    for r in rows:
        got = post(args.api, "/api/retrieve", {"question": r["query"]}, args.host)["retrieved"]
        d = draft(r, got, args.floor, f"q{nxt:02d}")
        nxt += 1
        out.append(d)
        top = got[0] if got else None
        print(f"{r['id']} [{d['category']}] {r['query']}")
        for c in got[:3]:
            print(f"      {c['score']:.3f}  {c['source']} / {c['section']}")
        if top and top.get("text"):
            print("      >", " ".join(top["text"].split())[:200])
    args.drafts.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out), encoding="utf-8")
    print(f"\n{len(out)} draft(s) written to {args.drafts}. Read each chunk, fix `expect`, then --append ids.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
