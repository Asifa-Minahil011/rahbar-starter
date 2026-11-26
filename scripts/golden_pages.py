#!/usr/bin/env python3
"""Fill the ground-truth page of every golden question from the live index, and check that
the unanswerable questions really are absent. Needs the API running (retrieval only: no LLM).

    python scripts/golden_pages.py --api http://localhost:8010            # fill pages, report
    python scripts/golden_pages.py --api http://localhost:8010 --dry-run  # report only

For each answerable/hard question it asks /api/retrieve and finds the chunk whose source and
section match `relevant`; that chunk's page becomes the ground-truth page. If the relevant
chunk is not in the retrieved top-k, it says so (the question, or its section, needs a look).
For unanswerable questions it prints the best similarity: if it is high, READ the top chunk
before keeping the question, because the corpus may actually cover it.
Standard library only.
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


def section_matches(golden: str, retrieved: str) -> bool:
    g, r = " ".join(golden.lower().split()), " ".join(retrieved.lower().split())
    return bool(g) and (r == g or r.startswith(g + " ") or r.startswith(g + "."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True)
    ap.add_argument("--host")
    ap.add_argument("--file", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--floor", type=float, default=0.68, help="the API's RELEVANCE_FLOOR")
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.file.read_text(encoding="utf-8").splitlines() if x.strip()]
    changed = problems = 0
    for r in rows:
        got = post(args.api, "/api/retrieve", {"question": r["question"]}, args.host)["retrieved"]
        if r["category"] == "unanswerable":
            top = got[0] if got else None
            sim = top["score"] if top else 0.0
            flag = "  <-- ABOVE the floor: read this chunk, the corpus may cover it" if sim >= args.floor else ""
            print(f"{r['id']:<5} absent? best similarity {sim:.3f} ({top['source'] + ' / ' + top['section'] if top else '-'}){flag}")
            problems += bool(flag)
            continue
        for rel in r["relevant"]:
            hit = next((c for c in got if c["source"] == rel["source"]
                        and section_matches(str(rel["section"]), c["section"])), None)
            if hit is None:
                print(f"{r['id']:<5} NOT FOUND in the top {len(got)}: {rel['source']} / {rel['section']}")
                problems += 1
            else:
                if rel.get("page") != hit.get("page"):
                    rel["page"] = hit.get("page")
                    changed += 1
                print(f"{r['id']:<5} {rel['source']} / {rel['section']} -> page {hit.get('page')}")
    if changed and not args.dry_run:
        args.file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"\nwrote {changed} page(s) to {args.file}")
    print(f"{problems} question(s) need a look" if problems else "all questions check out")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
