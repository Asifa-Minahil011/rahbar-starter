#!/usr/bin/env python3
"""Run the FAILURE-ANALYSIS experiments (section 5.5) and record the measured symptom of each.

Each experiment restarts the API with ONE setting broken, runs the golden set, reads /metrics,
and stops it. You then write the diagnosis and remedy in docs/FAILURE-ANALYSIS.md in your own words;
this script only measures, so the numbers in the doc are real.

    python scripts/failure_experiments.py --stages C,E              # fast: no LLM
    python scripts/failure_experiments.py --stages F                # slow: full eval at temperature 1.2

  C  TOP_K=1                     retrieval-only eval
  E  SUPERSEDED_FILTER=off       retrieval-only eval, plus which editions the version questions retrieved
  F  TEMPERATURE=1.2             FULL eval (needs the LLM; ~1-2 min per question on a CPU)
  (B, chunking, is measured by scripts/tune_chunking.py --configs 100:0: its recall by type is the evidence.)

The API reads the usual environment (CHROMA_HOST, CHROMA_PORT, OLLAMA_URL, EMBED_MODEL, RETRIEVER,
LLM_PROVIDER, ...): set the same variables you use to run it normally.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("rahbar_fallbacks_total", "rahbar_generation_retries_total",
           "rahbar_invalid_citations_total", "rahbar_abstentions_total")
STAGES: dict[str, dict[str, Any]] = {
    "C": {"name": "Retrieval", "env": {"TOP_K": "1"}, "full": False},
    "E": {"name": "Source", "env": {"SUPERSEDED_FILTER": "off"}, "full": False},
    "F": {"name": "Generation", "env": {"TEMPERATURE": "1.2"}, "full": True},
}


def load_eval() -> Any:
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)                 # type: ignore[union-attr]
    return module


def metric_total(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#") and line[len(name):len(name) + 1] in "{ ":
            total += float(line.rsplit(" ", 1)[1])
    return total


def wait_healthy(base: str, proc: subprocess.Popen[bytes], seconds: int = 90) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("the API process exited during start-up")
        try:
            if httpx.get(f"{base}/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"API not healthy after {seconds}s")


def run_one(ev: Any, golden: list[dict], env_over: dict[str, str], full: bool, port: int) -> dict[str, Any]:
    env = {**os.environ, **env_over}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
                            cwd=ROOT / "api", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_healthy(base, proc)
        rows = []
        fetch = ev.ask if full else ev.retrieve_only
        with httpx.Client(timeout=600.0) as client:
            for q in golden:
                rows.append({"id": q["id"], "type": q.get("type", "ordinary"),
                             "must_abstain": bool(q.get("must_abstain")), "relevant": q.get("relevant") or [],
                             "forbid": q.get("forbid") or [], "response": fetch(client, base, q["question"])})
            metrics_text = client.get(f"{base}/metrics").text
        summary = ev.summarise(rows)
        summary["metrics"] = {m: metric_total(metrics_text, m) for m in METRICS}
        summary["version_question_sources"] = {
            r["id"]: sorted({x["source"] for x in r["response"].get("retrieved", [])})
            for r in rows if r["type"] == "version"}
        return summary
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def describe(label: str, s: dict[str, Any]) -> list[str]:
    by_type = ", ".join(f"{t} {v['recall_at_5']:.0%}" for t, v in s["by_type"].items() if v["recall_at_5"] is not None)
    lines = [f"**{label}**: recall@5 {s['recall_at_5']:.1%} (by type: {by_type})"]
    if s.get("citation_validity") is not None:
        lines.append(f"  citation validity {s['citation_validity']:.1%}, abstention "
                     f"{s['abstention_correctness'] if s['abstention_correctness'] is not None else 'n/a'}")
    lines.append("  metrics: " + ", ".join(f"{k.removeprefix('rahbar_')}={v:g}" for k, v in s["metrics"].items()))
    for qid, sources in s["version_question_sources"].items():
        lines.append(f"  version question {qid} retrieved from: {', '.join(sources)}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stages", default="C,E", help="comma-separated: C, E, F")
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.yaml")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "failure-experiments.json")
    args = ap.parse_args()
    ev = load_eval()
    golden = ev.load_golden(args.golden)
    stages = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {sorted(STAGES)}")

    results: dict[str, Any] = {}
    need_full = any(STAGES[s]["full"] for s in stages)
    print("baseline (nothing broken), retrieval-only ...")
    results["baseline"] = run_one(ev, golden, {}, False, args.port)
    print("\n".join(describe("baseline", results["baseline"])))
    if need_full:
        print("\nbaseline FULL run (needed to compare stage F) ... slow")
        results["baseline_full"] = run_one(ev, golden, {}, True, args.port)
        print("\n".join(describe("baseline (full)", results["baseline_full"])))
    for stage in stages:
        cfg = STAGES[stage]
        print(f"\nstage {stage} ({cfg['name']}): {cfg['env']} ...")
        results[stage] = run_one(ev, golden, cfg["env"], cfg["full"], args.port)
        print("\n".join(describe(f"stage {stage} {cfg['env']}", results[stage])))
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}. Copy the symptoms into docs/FAILURE-ANALYSIS.md, then write the "
          "diagnosis and remedy yourself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
