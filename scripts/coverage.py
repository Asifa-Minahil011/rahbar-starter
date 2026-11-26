#!/usr/bin/env python3
"""E2 and E4: does the golden set still match real usage? (paper section 4.2.7)

    python scripts/coverage.py --calibrate                       # pick a threshold from data
    python scripts/coverage.py --threshold 0.80                  # coverage report, exit 1 if too low
    python scripts/coverage.py --threshold 0.80 --min-coverage 0.85 --out eval/results/coverage-before.json

For every live query it finds the most similar golden question (cosine of embeddings from
the pipeline's own embedding model). A query is covered when that similarity reaches the
threshold. The script prints the share covered, lists the uncovered queries grouped into
topics, and exits 1 when the share is below --min-coverage (the paper flags a gap when under
85 percent reach the score). Add golden questions for the uncovered topics, run it again,
and the coverage rises.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "eval"))
import ragops_metrics as M  # noqa: E402

QUERY_PREFIX = "search_query: "
GENERIC = {"student", "students", "university", "course", "courses", "does", "what", "how", "can", "get"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def embed_all(args: argparse.Namespace, texts: list[str]) -> list[list[float]]:
    prefix = QUERY_PREFIX if args.embed_model.startswith("nomic-embed") else ""
    r = httpx.post(f"{args.ollama.rstrip('/')}/api/embed", timeout=300.0,
                   json={"model": args.embed_model, "input": [prefix + t for t in texts]})
    r.raise_for_status()
    return [[float(x) for x in v] for v in r.json()["embeddings"]]


def best_match(vec: list[float], golden_vecs: list[list[float]]) -> tuple[int, float]:
    sims = [M.cosine(vec, g) for g in golden_vecs]
    i = max(range(len(sims)), key=sims.__getitem__)
    return i, sims[i]


def topic_clusters(items: list[dict[str, Any]], vecs: dict[str, list[float]], threshold: float) -> list[dict[str, Any]]:
    """Greedy grouping: a query joins the first cluster whose first member it resembles."""
    clusters: list[dict[str, Any]] = []
    for it in items:
        for c in clusters:
            if M.cosine(vecs[it["id"]], vecs[c["members"][0]["id"]]) >= threshold:
                c["members"].append(it)
                break
        else:
            clusters.append({"members": [it]})
    for c in clusters:
        words = Counter(w for m in c["members"] for w in M.content_words(m["query"]) if w not in GENERIC)
        c["topic"] = " / ".join(w for w, _ in words.most_common(3)) or "misc"
        c["queries"] = [m["query"] for m in c["members"]]
        c["ids"] = [m["id"] for m in c["members"]]
        del c["members"]
    return sorted(clusters, key=lambda c: -len(c["ids"]))


def calibrate(args: argparse.Namespace, golden: list[dict[str, Any]]) -> int:
    """Pick the threshold from data: how similar are two questions that MEAN the same thing,
    and how similar are two that do not? Uses the Part B paraphrase and unrelated pairs and
    the paraphrases written for the golden set."""
    pairs = json.loads(Path(args.pairs).read_text())
    by_id = {g["id"]: g for g in golden}
    para = [tuple(p) for p in pairs["paraphrase"]]
    para += [(by_id[g["paraphrase_of"]]["question"], g["question"]) for g in golden if g.get("paraphrase_of") in by_id]
    unrel = [tuple(p) for p in pairs["unrelated"]]
    texts = [t for p in para + unrel for t in p]
    vecs = embed_all(args, texts)
    sims = [M.cosine(vecs[2 * i], vecs[2 * i + 1]) for i in range(len(texts) // 2)]
    ps, us = sims[:len(para)], sims[len(para):]
    p10 = M.percentile(ps, 0.10) or 0.0
    print(f"paraphrase pairs ({len(ps)}): min {min(ps):.3f}  p10 {p10:.3f}  median {M.percentile(ps, .5):.3f}")
    print(f"unrelated  pairs ({len(us)}): median {M.percentile(us, .5):.3f}  p90 {M.percentile(us, .9):.3f}  max {max(us):.3f}")
    print(f"\nSuggested threshold: {p10:.2f}  (the 10th percentile of known paraphrase similarity:")
    print("a live query is 'covered' when it is at least as close to a golden question as 90% of")
    print(f"real paraphrases are; unrelated pairs top out at {max(us):.2f}). Write this reasoning in the report.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    ap.add_argument("--live", type=Path, default=ROOT / "eval" / "live_queries.jsonl")
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.jsonl")
    ap.add_argument("--pairs", type=Path, default=ROOT / "eval" / "embedding_pairs.json")
    ap.add_argument("--threshold", type=float, help="similarity at which a live query counts as covered")
    ap.add_argument("--min-coverage", type=float, default=0.85)
    ap.add_argument("--cluster-threshold", type=float, default=0.72, help="similarity for grouping uncovered queries")
    ap.add_argument("--calibrate", action="store_true", help="print the data behind a threshold and exit")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ollama", default=env("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--embed-model", default=env("EMBED_MODEL", "nomic-embed-text"))
    args = ap.parse_args()
    golden = read_jsonl(args.golden)
    if args.calibrate:
        return calibrate(args, golden)
    if args.threshold is None:
        ap.error("give --threshold (run with --calibrate first to choose it from data)")

    live = read_jsonl(args.live)
    vecs = embed_all(args, [g["question"] for g in golden] + [q["query"] for q in live])
    gvecs, lvecs = vecs[:len(golden)], vecs[len(golden):]
    rows = []
    for q, v in zip(live, lvecs):
        i, sim = best_match(v, gvecs)
        rows.append({**q, "best_golden": golden[i]["id"], "best_question": golden[i]["question"],
                     "similarity": round(sim, 4), "covered": sim >= args.threshold})
    covered = sum(r["covered"] for r in rows)
    share = covered / len(rows)
    by_group = {g: round(sum(r["covered"] for r in rows if r["group"] == g) / max(1, sum(r["group"] == g for r in rows)), 4)
                for g in sorted({r["group"] for r in rows})}
    uncovered = [r for r in rows if not r["covered"]]
    qvec = {r["id"]: lv for r, lv in zip(rows, lvecs)}
    clusters = topic_clusters(uncovered, qvec, args.cluster_threshold)

    print(f"golden questions: {len(golden)}   live queries: {len(rows)}   threshold: {args.threshold}")
    print(f"COVERAGE: {covered}/{len(rows)} = {share:.1%}   by group: {by_group}   (required {args.min_coverage:.0%})\n")
    print(f"uncovered queries grouped into {len(clusters)} topic(s):")
    for c in clusters:
        print(f"  [{c['topic']}]  {len(c['ids'])} quer{'y' if len(c['ids']) == 1 else 'ies'}")
        for r in (x for x in uncovered if x["id"] in c["ids"]):
            print(f"      {r['id']} ({r['group']}) {r['query']}   best {r['best_golden']} @ {r['similarity']}")
    result = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "threshold": args.threshold,
              "min_coverage": args.min_coverage, "golden_questions": len(golden), "live_queries": len(rows),
              "coverage": round(share, 4), "by_group": by_group, "ok": share >= args.min_coverage,
              "uncovered_topics": clusters, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwritten: {args.out}")
    print("\ncoverage OK" if result["ok"] else f"\nCOVERAGE BELOW {args.min_coverage:.0%}: add golden questions for the topics above")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
