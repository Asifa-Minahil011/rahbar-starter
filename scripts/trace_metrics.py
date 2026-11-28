#!/usr/bin/env python3
"""D3. Summarise the trace log over a time window.

    python scripts/trace_metrics.py                          # whole file
    python scripts/trace_metrics.py --since 1h               # last hour (m, h, d) or an ISO time
    python scripts/trace_metrics.py --since 2026-11-27T09:00:00Z --until 2026-11-27T12:00:00Z
    python scripts/trace_metrics.py --json --out eval/results/trace-metrics.json

Reports: top-1 retrieval similarity distribution, abstention rate, cache hit rate, p50/p95
latency per stage, number of injection attempts flagged, and the outcome mix.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BUCKETS = (0.3, 0.4, 0.5, 0.6, 0.68, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)


def parse_time(text: str) -> datetime:
    m = re.fullmatch(r"(\d+)([mhd])", text.strip())
    if m:
        unit = {"m": "minutes", "h": "hours", "d": "days"}[m.group(2)]
        return datetime.now(timezone.utc) - timedelta(**{unit: int(m.group(1))})
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def pct(values: list[float], q: float) -> float | None:
    """Percentile by linear interpolation; None for an empty list."""
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (pos - lo), 4)


def load(path: str, since: datetime | None, until: datetime | None) -> list[dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec.get("trace"), dict):
            rec = rec["trace"]
        if "trace_id" not in rec:
            continue
        ts = datetime.fromisoformat(rec["timestamp"].replace("Z", "+00:00"))
        if (since and ts < since) or (until and ts > until):
            continue
        out.append(rec)
    return out


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    outcomes: dict[str, int] = {}
    for r in records:
        outcomes[r.get("outcome") or "?"] = outcomes.get(r.get("outcome") or "?", 0) + 1
    abstained = sum(c for o, c in outcomes.items() if o.startswith("abstain"))
    fallbacks = sum(c for o, c in outcomes.items() if o.startswith("fallback"))
    hits = sum(1 for r in records if r.get("cache") == "hit")
    misses = sum(1 for r in records if r.get("cache") == "miss")
    top1 = []
    for r in records:
        sims = [c["similarity"] for c in (r.get("retrieval") or {}).get("retrieved", [])
                if c.get("similarity") is not None]
        if sims and r.get("cache") != "hit":
            top1.append(max(sims))
    hist = {f"<= {b}": 0 for b in BUCKETS}
    for s in top1:
        for b in BUCKETS:
            if s <= b:
                hist[f"<= {b}"] += 1
                break
    stages: dict[str, list[float]] = {}
    for r in records:
        for k, v in (r.get("latency_ms") or {}).items():
            stages.setdefault(k, []).append(float(v))
    flags: dict[str, int] = {}
    for r in records:
        for a in r.get("guardrail_actions") or []:
            if a.get("action") in ("flagged", "dropped", "redacted"):
                key = f"{a.get('rail')}:{a.get('action')}@{'question' if a.get('where') == 'question' else 'passage'}"
                flags[key] = flags.get(key, 0) + 1
    return {
        "queries": n,
        "window": {"first": records[0]["timestamp"] if n else None,
                   "last": records[-1]["timestamp"] if n else None},
        "outcomes": outcomes,
        "abstention_rate": round(abstained / n, 4) if n else None,
        "fallback_rate": round(fallbacks / n, 4) if n else None,
        "cache": {"hits": hits, "misses": misses,
                  "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None},
        "top1_similarity": {"n": len(top1), "min": pct(top1, 0), "p50": pct(top1, 0.5),
                            "mean": round(sum(top1) / len(top1), 4) if top1 else None,
                            "p95": pct(top1, 0.95), "max": pct(top1, 1), "histogram": hist},
        "latency_ms": {k: {"n": len(v), "p50": pct(v, 0.5), "p95": pct(v, 0.95)}
                       for k, v in sorted(stages.items())},
        "injection_flags": {"total": sum(flags.values()), "by_kind": flags},
    }


def render(s: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        return "n/a" if x is None else (f"{x:.1%}" if isinstance(x, float) and x <= 1 else str(x))
    t1 = s["top1_similarity"]
    lines = [f"queries: {s['queries']}   window: {s['window']['first']} .. {s['window']['last']}",
             f"outcomes: {s['outcomes']}",
             f"abstention rate: {f(s['abstention_rate'])}   fallback rate: {f(s['fallback_rate'])}",
             f"cache: {s['cache']['hits']} hits / {s['cache']['misses']} misses  "
             f"hit rate: {f(s['cache']['hit_rate'])}",
             f"top-1 similarity (n={t1['n']}): min={t1['min']} p50={t1['p50']} mean={t1['mean']} "
             f"p95={t1['p95']} max={t1['max']}"]
    lines += [f"   {k:<8} {'#' * v} {v}" for k, v in t1["histogram"].items() if v]
    lines.append("latency per stage (ms):")
    lines += [f"   {k:<18} n={v['n']:<4} p50={v['p50']}  p95={v['p95']}" for k, v in s["latency_ms"].items()]
    lines.append(f"injection attempts flagged: {s['injection_flags']['total']}  {s['injection_flags']['by_kind']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="traces/traces.jsonl")
    ap.add_argument("--since", help="30m, 2h, 1d or an ISO time")
    ap.add_argument("--until", help="an ISO time")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--out", help="also write the JSON summary to this file")
    args = ap.parse_args()
    try:
        recs = load(args.file, parse_time(args.since) if args.since else None,
                    parse_time(args.until) if args.until else None)
    except OSError as exc:
        print(f"cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    summary = summarise(recs)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
