#!/usr/bin/env python3
"""D2. Print the exact context and prompt that produced one answer.

    python scripts/trace_show.py <trace_id>                       # reads traces/traces.jsonl
    python scripts/trace_show.py <trace_id> --file /data/traces.jsonl
    kubectl logs deploy/api -n rahbar | python scripts/trace_show.py <trace_id> --file -
    python scripts/trace_show.py --list 10                        # the last 10 trace ids

A trace id is the request_id the API returns with every answer (the X-Request-ID header).
A unique prefix is enough. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from typing import Any


def read_traces(path: str) -> Iterator[dict[str, Any]]:
    fh = sys.stdin if path == "-" else open(path, encoding="utf-8")
    with fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec.get("trace"), dict):      # a `kubectl logs` line: trace is nested
                rec = rec["trace"]
            if "trace_id" in rec:
                yield rec


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(3, 66 - len(title)))


def show(rec: dict[str, Any]) -> None:
    print(f"trace_id   {rec['trace_id']}\ntimestamp  {rec['timestamp']}\noutcome    {rec.get('outcome')}")
    print(f"query      {rec['query']}")
    if rec.get("rewritten_query"):
        print(f"rewritten  {rec['rewritten_query']}")
    emb, col, model = rec.get("embedding", {}), rec.get("collection", {}), rec.get("model", {})
    rule("pipeline versions")
    print(f"embedding model   {emb.get('model')}  (index built with: {emb.get('index_model')}, "
          f"dimension {emb.get('dimension')})")
    print(f"collection        {col.get('name')}  version/index_id: {col.get('version')}")
    print(f"llm               {model.get('provider')} / {model.get('name')}  temperature={model.get('temperature')}")
    print(f"retrieval params  top_k={model.get('top_k')} floor={model.get('relevance_floor')} "
          f"hybrid_boost={model.get('hybrid_boost')}   cache={rec.get('cache')}")
    ret = rec.get("retrieval", {})
    rule("retrieval")
    print(f"filters applied   {json.dumps(ret.get('filters'))}")
    print(f"exact terms       {ret.get('exact_terms')}")
    print(f"order by similarity  {ret.get('order_by_similarity')}")
    print(f"reranked order       {ret.get('reranked_order')}")
    for i, c in enumerate(ret.get("retrieved", []), 1):
        print(f"  {i}. {c['chunk_id']}  {c['document_id']} / {c['section']}  p.{c.get('page')}  "
              f"v={c.get('doc_version')}  superseded={c.get('superseded')}  "
              f"sim={c.get('similarity')} final={c.get('final_score')}")
    rule("guardrail actions")
    actions = rec.get("guardrail_actions") or []
    for a in actions:
        print(f"  {a.get('rail')}: {a.get('action')} @ {a.get('where')}  {a.get('detail', '')}")
    if not actions:
        print("  none")
    rule(f"prompt sent to the model  ({rec.get('prompt_hash')})")
    if rec.get("prompt"):
        print(f"chunks in the prompt: {rec.get('prompt_chunk_ids')}")
        for m in rec["prompt"]:
            print(f"\n--- {m['role']} " + "-" * 50)
            print(m["content"])
    else:
        print("  no prompt: the model was not called "
              f"({rec.get('outcome')}).")
    rule("response")
    r = rec.get("response") or {}
    print(f"answered_from_context={r.get('answered_from_context')} confidence={r.get('confidence')}")
    print(r.get("answer"))
    for c in rec.get("citations", []):
        print(f"  cites {c['source']} / {c['section']} p.{c.get('page')}")
    rule("latency (ms)")
    print("  " + "  ".join(f"{k}={v}" for k, v in (rec.get("latency_ms") or {}).items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_id", nargs="?", help="trace id (or a unique prefix)")
    ap.add_argument("--file", default="traces/traces.jsonl", help="trace file, or - for stdin")
    ap.add_argument("--list", type=int, metavar="N", help="list the last N trace ids and exit")
    args = ap.parse_args()
    try:
        records = list(read_traces(args.file))
    except OSError as exc:
        print(f"cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    if args.list:
        for rec in records[-args.list:]:
            print(f"{rec['trace_id']}  {rec['timestamp']}  {rec.get('outcome'):<22} {rec['query'][:60]}")
        return 0
    if not args.trace_id:
        ap.error("give a trace id, or use --list N")
    found = [r for r in records if r["trace_id"].startswith(args.trace_id)]
    if not found:
        print(f"no trace starting with {args.trace_id!r} in {args.file} ({len(records)} traces)",
              file=sys.stderr)
        return 1
    if len(found) > 1 and len({r["trace_id"] for r in found}) > 1:
        print("more than one trace matches; use a longer prefix:", file=sys.stderr)
        for r in found:
            print("  ", r["trace_id"], file=sys.stderr)
        return 1
    show(found[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
