#!/usr/bin/env python3
"""The release gate (Assignment 3, C2): candidate vs baseline, thresholds from a config file.

    python eval/gate.py --candidate eval/results/suite-<stamp>-candidate.json
    python eval/gate.py --candidate ... --baseline eval/results/baseline.json --config eval/gate.yaml

Prints every rule with its measured value and PASS/FAIL, then a side-by-side comparison of
all metrics (module, component and end-to-end levels). Exit code 0 = release allowed,
1 = blocked, 2 = could not run (missing file). Thresholds live in the config file only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
LOWER_IS_BETTER = ("latency", "embed_ms", "over_abstention", "determinism", "injection")


def evaluate(rules: list[dict[str, Any]], cand: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in rules:
        metric, check, limit = r["metric"], r["check"], r.get("value")
        c, b = cand.get(metric), base.get(metric)
        row = {"id": r["id"], "metric": metric, "check": check, "limit": limit,
               "candidate": c, "baseline": b, "why": r.get("why", "")}
        if c is None or (check in ("max_drop_vs_baseline", "max_rise_ratio", "not_below_baseline") and b is None):
            row["status"] = "SKIP" if r.get("optional") else "FAIL"
            row["detail"] = "not measured" + (" (optional rule)" if r.get("optional") else "")
        elif check == "min":
            ok = c >= limit
            row["status"], row["detail"] = ("PASS" if ok else "FAIL"), f"{c:.4f} must be >= {limit}"
        elif check == "max":
            ok = c <= limit
            row["status"], row["detail"] = ("PASS" if ok else "FAIL"), f"{c:.4f} must be <= {limit}"
        elif check == "max_drop_vs_baseline":
            drop = b - c
            ok = drop <= limit + 1e-9
            row["status"], row["detail"] = ("PASS" if ok else "FAIL"), f"dropped {drop:+.4f} (allowed {limit}) from {b:.4f}"
        elif check == "max_rise_ratio":
            ratio = c / b if b else float("inf")
            ok = ratio <= limit
            row["status"], row["detail"] = ("PASS" if ok else "FAIL"), f"{ratio:.2f}x the baseline (allowed {limit}x)"
        elif check == "not_below_baseline":
            ok = c >= b - 1e-9
            row["status"], row["detail"] = ("PASS" if ok else "FAIL"), f"{c:.4f} vs baseline {b:.4f}"
        else:
            raise ValueError(f"rule {r['id']}: unknown check {check!r}")
        out.append(row)
    return out


def comparison(cand: dict[str, Any], base: dict[str, Any]) -> list[str]:
    lines = [f"{'metric':<34}{'baseline':>12}{'candidate':>12}{'change':>12}  better?", "-" * 80]
    for k in sorted(set(cand) | set(base)):
        c, b = cand.get(k), base.get(k)
        if not isinstance(c, (int, float)) or not isinstance(b, (int, float)):
            continue
        delta = c - b
        lower = any(w in k for w in LOWER_IS_BETTER)
        mark = "" if abs(delta) < 1e-9 else ("better" if (delta < 0) == lower else "worse")
        lines.append(f"{k:<34}{b:>12.4f}{c:>12.4f}{delta:>+12.4f}  {mark}")
    return lines


def newest_candidate(directory: Path, baseline: Path) -> Path | None:
    files = sorted(p for p in directory.glob("suite-*.json") if p.resolve() != baseline.resolve())
    return files[-1] if files else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=Path, help="suite result of the change (default: newest in eval/results)")
    ap.add_argument("--baseline", type=Path, default=Path(os.environ.get("GATE_BASELINE", HERE / "results" / "baseline.json")))
    ap.add_argument("--config", type=Path, default=Path(os.environ.get("GATE_CONFIG", HERE / "gate.yaml")))
    ap.add_argument("--out", type=Path, help="write the gate result here (default: eval/results/gate-<stamp>.json)")
    args = ap.parse_args()
    cand_path = args.candidate or newest_candidate(args.baseline.parent, args.baseline)
    for label, path in (("candidate", cand_path), ("baseline", args.baseline), ("config", args.config)):
        if path is None or not Path(path).exists():
            print(f"gate: cannot find the {label} file {path}", file=sys.stderr)
            return 2
    cand = json.loads(Path(cand_path).read_text())
    base = json.loads(args.baseline.read_text())
    rules = yaml.safe_load(args.config.read_text())["rules"]
    rows = evaluate(rules, cand["flat"], base["flat"])

    print(f"candidate: {cand_path}\nbaseline:  {args.baseline}\nconfig:    {args.config}\n")
    for r in rows:
        print(f"  {r['status']:<5} {r['id']:<22} {r['metric']:<24} {r['detail']}")
    print("\n" + "\n".join(comparison(cand["flat"], base["flat"])))
    failed = [r for r in rows if r["status"] == "FAIL"]
    verdict = "FAIL" if failed else "PASS"
    print(f"\nRELEASE GATE: {verdict}" + (f"  ({len(failed)} rule(s) failed: {', '.join(r['id'] for r in failed)})" if failed else ""))
    out = args.out or args.baseline.parent / f"gate-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}.json"
    try:
        out.write_text(json.dumps({"verdict": verdict, "candidate": str(cand_path), "baseline": str(args.baseline),
                                   "config": str(args.config), "rules": rows}, indent=2))
    except OSError as exc:   # a read-only volume must not change the exit code
        print(f"(gate result not written: {exc})", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
