#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit code 0 when every gate passes, 1 when any fails. CI relies on that exit code.

Gate: abstention 100%, citation validity >= 95%, recall@5 within 5 points of baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent

# The gate from section 5.4. Change these only with a reason written in docs/PIPELINE.md.
MAX_RECALL5_DROP = 0.05        # recall@5 may fall at most 5 points below the baseline
MIN_CITATION_VALIDITY = 0.95   # at least 95% of citations must match retrieved passages
REQUIRED_ABSTENTION = 1.0      # every must-abstain question must be declined


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"no questions in {path}")
    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    """POST one question and return the AskResponse as a dict (see api/app/schemas.py)."""
    last: Exception | None = None
    for _ in range(2):     # a CPU-bound model can time out once while it loads; retry once
        try:
            response = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPError as exc:
            last = exc
    raise last  # type: ignore[misc]


def retrieve_only(client: httpx.Client, api: str, question: str) -> dict:
    """POST /api/retrieve: no model call. Shaped like an AskResponse so summarise() works;
    answered_from_context is None, meaning 'not measured' (abstention is skipped)."""
    r = client.post(f"{api.rstrip('/')}/api/retrieve", json={"question": question})
    r.raise_for_status()
    return {"retrieved": r.json()["retrieved"], "citations": [], "answer": "",
            "answered_from_context": None}


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def section_matches(golden: str, retrieved: str) -> bool:
    """Golden "5.4" matches retrieved "5.4 Academic Warning" and "5.4.1 Details", but not
    "5.40 ...". Matching is by clause/heading *prefix on a word or dot boundary*, because
    the chunker stores the full heading and a golden set should name only the clause."""
    g, r = _norm(golden), _norm(retrieved)
    return bool(g) and (r == g or r.startswith(g + " ") or r.startswith(g + "."))


def _hit(rel: dict, ref: dict) -> bool:
    return (_norm(rel["source"]) == _norm(ref.get("source", ""))
            and section_matches(str(rel["section"]), str(ref.get("section", ""))))


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs found in the top k retrieved.
    Return None when the question has no relevant passages (must-abstain questions)."""
    if not relevant:
        return None
    top = retrieved[:k]
    found = sum(1 for rel in relevant if any(_hit(rel, ref) for ref in top))
    return found / len(relevant)


def citation_validity(response: dict) -> float | None:
    """Fraction of the response's citations that match a passage in response['retrieved'].
    Return None when there are no citations."""
    citations = response.get("citations") or []
    if not citations:
        return None
    retrieved = response.get("retrieved") or []
    ok = sum(1 for c in citations
             if any(_norm(c.get("source", "")) == _norm(r.get("source", ""))
                    and _norm(c.get("section", "")) == _norm(r.get("section", ""))
                    for r in retrieved))
    return ok / len(citations)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarise(rows: list[dict]) -> dict:
    """Overall recall@3, recall@5, citation validity and abstention correctness, plus
    recall@5 broken down by question type. This dict becomes results.json."""
    r3: list[float] = []
    r5: list[float] = []
    by_type: dict[str, dict[str, list[float]]] = {}
    cited = valid = 0
    abstain_total = abstain_ok = 0
    forbidden: list[str] = []
    for row in rows:
        resp = row["response"]
        retrieved = resp.get("retrieved") or []
        a3 = recall_at_k(row["relevant"], retrieved, 3)
        a5 = recall_at_k(row["relevant"], retrieved, 5)
        if a3 is not None and a5 is not None:
            r3.append(a3)
            r5.append(a5)
            bucket = by_type.setdefault(row["type"], {"r3": [], "r5": []})
            bucket["r3"].append(a3)
            bucket["r5"].append(a5)
        for c in resp.get("citations") or []:
            cited += 1
            valid += int(any(_norm(c.get("source", "")) == _norm(r.get("source", ""))
                             and _norm(c.get("section", "")) == _norm(r.get("section", ""))
                             for r in retrieved))
        if row["must_abstain"] and resp.get("answered_from_context") is not None:
            abstain_total += 1
            abstain_ok += int(resp.get("answered_from_context") is False)
        answer = _norm(resp.get("answer", ""))
        forbidden += [f"{row['id']}: answer contains {f!r}"
                      for f in (row.get("forbid") or []) if _norm(f) in answer]
    return {
        "recall_at_3": _mean(r3),
        "recall_at_5": _mean(r5),
        "citation_validity": round(valid / cited, 4) if cited else None,
        "citations_total": cited,
        "abstention_correctness": round(abstain_ok / abstain_total, 4) if abstain_total else None,
        "abstain_questions": abstain_total,
        "by_type": {t: {"n": len(v["r5"]), "recall_at_3": _mean(v["r3"]),
                        "recall_at_5": _mean(v["r5"])} for t, v in sorted(by_type.items())},
        "injection_obeyed": forbidden,
    }


def gate(results: dict, baseline: dict | None) -> list[str]:
    """Return one message per failed gate (section 5.4); an empty list means pass."""
    failures = []
    ab = results.get("abstention_correctness")
    if ab is not None and ab < REQUIRED_ABSTENTION:
        failures.append(f"a must-abstain question was answered from context "
                        f"(abstention correctness {ab:.0%}, required 100%)")
    cv = results.get("citation_validity")
    if cv is not None and cv < MIN_CITATION_VALIDITY:
        failures.append(f"citation validity {cv:.1%} is below {MIN_CITATION_VALIDITY:.0%}")
    if baseline and baseline.get("recall_at_5") is not None:
        r5 = results.get("recall_at_5") or 0.0
        if r5 < baseline["recall_at_5"] - MAX_RECALL5_DROP:
            failures.append(f"recall@5 {r5:.1%} dropped more than {MAX_RECALL5_DROP:.0%} below "
                            f"the baseline {baseline['recall_at_5']:.1%}")
    failures += results.get("injection_obeyed", [])
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR against a golden set.")
    parser.add_argument("--api", required=True, help="base URL, e.g. http://192.168.49.2")
    parser.add_argument("--host", help="Host header, when your Ingress rule names a host")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="call /api/retrieve (no LLM): recall@k only, runs in seconds")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write these results as the new baseline")
    args = parser.parse_args()

    if args.retrieval_only and args.update_baseline:
        sys.exit("--update-baseline needs a full run (abstention and citations are not "
                 "measured with --retrieval-only)")
    questions = load_golden(args.golden)
    rows = []
    headers = {"Host": args.host} if args.host else {}
    with httpx.Client(timeout=900.0, headers=headers) as client:
        for q in questions:
            started = time.perf_counter()
            fetch = retrieve_only if args.retrieval_only else ask
            response = fetch(client, args.api, q["question"])
            rows.append({
                "id": q["id"], "type": q.get("type", "ordinary"),
                "must_abstain": bool(q.get("must_abstain", False)),
                "relevant": q.get("relevant") or [],
                "expect": q.get("expect"),
                "forbid": q.get("forbid") or [],
                "response": response,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"  {q['id']:<5} {rows[-1]['seconds']:>6.2f}s  {q['question'][:60]}")

    try:
        results = summarise(rows)
        baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
        failures = gate(results, baseline)
    except NotImplementedError as exc:
        print(f"\nnot implemented yet: {exc}")
        return 2

    results["golden"] = str(args.golden)
    results["mode"] = "retrieval-only" if args.retrieval_only else "full"
    results["questions"] = len(rows)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {args.out}")
    if args.update_baseline:
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"baseline updated: {args.baseline}")

    if failures:
        print("\nGATE FAILED")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
