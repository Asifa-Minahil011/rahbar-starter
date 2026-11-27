#!/usr/bin/env python3
"""Part B: the three-level offline test suite, in one command (Assignment 3).

    python eval/ragops_suite.py --api http://localhost:8010                     # everything
    python eval/ragops_suite.py --api ... --retrieval-only                      # no LLM: minutes, not hours
    python eval/ragops_suite.py --api ... --judge --repeat 2 --tag baseline     # the full run for the report
    python eval/ragops_suite.py --api ... --limit 6                             # a quick smoke run

Levels (paper Table 1):
  module     the embedding model: paraphrase vs unrelated similarity, time per chunk, determinism
  component  retrieval: recall@3, recall@5, MRR, nDCG@5, then the same on typo-noised queries
  e2e        answers: faithfulness, abstention, citation validity, latency p50/p95
Stochasticity: --repeat N runs the suite N times with the same seeds and reports the difference
of every metric. The semantic cache is bypassed (X-Bypass-Cache) so a cached answer cannot
hide variation.

Writes eval/results/suite-<UTC timestamp>[-tag].json and .md, and prints the summary table.
It never fails the build: eval/gate.py applies the release gate to the result files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ragops_metrics as M  # noqa: E402

DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


# ------------------------------------------------------------------ io helpers

def load_golden(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        sys.exit(f"no questions in {path}")
    return rows


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def post_json(client: httpx.Client, url: str, body: dict[str, Any], retries: int = 2,
              headers: dict[str, str] | None = None) -> dict[str, Any]:
    last: Exception | None = None
    for _ in range(retries):   # a CPU model can time out once while it loads
        try:
            r = client.post(url, json=body, headers=headers or {})
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]
        except httpx.HTTPError as exc:
            last = exc
    raise last  # type: ignore[misc]


def read_traces(path: Path) -> dict[str, dict[str, Any]]:
    return parse_traces(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_traces(text: str) -> dict[str, dict[str, Any]]:
    """Trace records from JSON lines: a trace file, or `kubectl logs` of the API pods
    (a trace sits inside the log line under "trace")."""
    out: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        rec = rec.get("trace", rec) if isinstance(rec.get("trace"), dict) else rec
        if "trace_id" in rec:
            out[rec["trace_id"]] = rec
    return out


def traces_from_command(cmd: str) -> dict[str, dict[str, Any]]:
    """Run a command that prints the API's trace lines (e.g. kubectl logs) and parse them."""
    done = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=600, check=False)
    if done.returncode != 0:
        print(f"  (trace command failed: {done.stderr.strip()[:200]})")
    return parse_traces(done.stdout)


CONTEXT_BLOCK = re.compile(r"<context>\n\[\d+\][^\n]*\n(.*?)\n</context>", re.S)


def context_from_trace(rec: dict[str, Any]) -> str | None:
    prompt = rec.get("prompt")
    if not prompt:
        return None
    return "\n".join(CONTEXT_BLOCK.findall(prompt[1]["content"])) or None


# ------------------------------------------------------------------ module level

def embed(client: httpx.Client, ollama: str, model: str, texts: list[str], prefix: str) -> list[list[float]]:
    r = post_json(client, f"{ollama.rstrip('/')}/api/embed", {"model": model, "input": [prefix + t for t in texts]})
    return [[float(x) for x in v] for v in r["embeddings"]]


def sample_chunks(args: argparse.Namespace, n: int = 30) -> tuple[list[str], str]:
    try:
        import chromadb
        from chromadb.config import Settings as CS
        col = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port,
                                  settings=CS(anonymized_telemetry=False)).get_collection(args.collection)
        docs = col.get(limit=n, include=["documents"])["documents"]
        if docs:
            return list(docs), f"{len(docs)} chunks from the Chroma collection '{args.collection}'"
    except Exception as exc:   # chromadb-client missing, or Chroma not reachable from here
        print(f"  (chunk sample: Chroma not reachable: {exc!r:.80}; using fixture sentences)")
    return [], "fixture sentences (Chroma not reachable)"


def module_level(args: argparse.Namespace, client: httpx.Client) -> dict[str, Any]:
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    prefix = QUERY_PREFIX if args.embed_model.startswith("nomic-embed") else ""
    sims: dict[str, list[float]] = {}
    for kind in ("paraphrase", "unrelated"):
        sims[kind] = []
        for a, b in pairs[kind]:
            va, vb = embed(client, args.ollama, args.embed_model, [a, b], prefix)
            sims[kind].append(M.cosine(va, vb))
    mp, mu = sum(sims["paraphrase"]) / len(sims["paraphrase"]), sum(sims["unrelated"]) / len(sims["unrelated"])

    chunks, source = sample_chunks(args)
    if not chunks:
        chunks = [a for a, _ in pairs["paraphrase"]]
    times = []
    for text in chunks:
        t = time.perf_counter()
        embed(client, args.ollama, args.embed_model, [text], DOC_PREFIX if prefix else "")
        times.append((time.perf_counter() - t) * 1000)

    texts = [a for a, _ in pairs["paraphrase"][:5]]
    first = embed(client, args.ollama, args.embed_model, texts, prefix)
    second = embed(client, args.ollama, args.embed_model, texts, prefix)
    return {
        "model": args.embed_model, "dimension": len(first[0]),
        "paraphrase": {"pairs": len(sims["paraphrase"]), "mean_cosine": round(mp, 4),
                       "min": round(min(sims["paraphrase"]), 4)},
        "unrelated": {"pairs": len(sims["unrelated"]), "mean_cosine": round(mu, 4),
                      "max": round(max(sims["unrelated"]), 4)},
        "gap": round(mp - mu, 4),
        "separable": min(sims["paraphrase"]) > max(sims["unrelated"]),
        "time_per_chunk_ms": {"mean": round(sum(times) / len(times), 1), "p50": M.percentile(times, 0.5),
                              "p95": M.percentile(times, 0.95), "n": len(times), "source": source},
        "determinism": {"texts": len(texts), "max_abs_diff": max(M.max_abs_diff(a, b) for a, b in zip(first, second)),
                        "max_cosine_gap": max(1 - M.cosine(a, b) for a, b in zip(first, second))},
    }


# ------------------------------------------------------------------ component level

def retrieve(client: httpx.Client, args: argparse.Namespace, question: str) -> dict[str, Any]:
    return post_json(client, f"{args.api.rstrip('/')}/api/retrieve", {"question": question})


def component_level(args: argparse.Namespace, client: httpx.Client, golden: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [q for q in golden if q["category"] in ("answerable", "hard")]
    rng = random.Random(args.seed)
    clean_rows, noisy_rows, changed = [], [], []
    top_k = None
    for q in ans:
        got = retrieve(client, args, q["question"])
        top_k = got.get("top_k", top_k)
        clean_rows.append({"q": q, "relevant": q["relevant"], "retrieved": got["retrieved"]})
        noisy = M.add_typos(q["question"], rng, args.noise_rate)
        got_n = retrieve(client, args, noisy) if noisy != q["question"] else got
        noisy_rows.append({"q": q, "relevant": q["relevant"], "retrieved": got_n["retrieved"]})
        r_clean = M.recall_at_k(q["relevant"], got["retrieved"], 5)
        r_noisy = M.recall_at_k(q["relevant"], got_n["retrieved"], 5)
        if noisy != q["question"] and r_clean != r_noisy:
            changed.append({"id": q["id"], "clean": q["question"], "noisy": noisy,
                            "recall_at_5": [r_clean, r_noisy]})

    def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        keys = sorted({r["q"][key] for r in rows})
        return {k: M.retrieval_scores([r for r in rows if r["q"][key] == k]) for k in keys}

    clean, noisy_s = M.retrieval_scores(clean_rows), M.retrieval_scores(noisy_rows)
    delta = {k: (round(noisy_s[k] - clean[k], 4) if noisy_s[k] is not None and clean[k] is not None else None)
             for k in ("recall_at_3", "recall_at_5", "mrr", "ndcg_at_5")}
    # best similarity of the unanswerable questions: shows how far they sit below the floor
    unans = []
    for q in golden:
        if q["category"] == "unanswerable":
            got = retrieve(client, args, q["question"])["retrieved"]
            unans.append({"id": q["id"], "top1_similarity": round(got[0]["score"], 4) if got else None})
    per_question = []
    for r_clean, r_noisy in zip(clean_rows, noisy_rows):
        ret = r_clean["retrieved"]
        flags = M.relevance_vector(r_clean["relevant"], ret)
        per_question.append({
            "id": r_clean["q"]["id"], "recall_at_5": M.recall_at_5_value(r_clean), "first_hit_rank":
            (flags.index(1) + 1) if 1 in flags else None,
            "top1": ({"source": ret[0]["source"], "section": ret[0]["section"], "score": round(ret[0]["score"], 4)}
                     if ret else None),
            "sources": sorted({x["source"] for x in ret})})
    return {"top_k": top_k, "per_question": per_question, "overall": clean, "by_category": grouped(clean_rows, "category"),
            "by_kind": grouped(clean_rows, "kind"),
            "noise": {"kind": "typos", "rate": args.noise_rate, "seed": args.seed, "overall": noisy_s,
                      "delta": delta, "queries_changed": changed},
            "unanswerable_top1": unans}


# ------------------------------------------------------------------ end-to-end level

JUDGE_SCHEMA = {"type": "object", "properties": {"supported": {"type": "boolean"},
                                                 "unsupported_claims": {"type": "array", "items": {"type": "string"}}},
                "required": ["supported", "unsupported_claims"]}
JUDGE_SYSTEM = ("You check whether an ANSWER is fully supported by the CONTEXT. Supported means every factual "
                "claim in the answer can be found in the context. Ignore citation brackets. Reply with JSON only.")


def judge(client: httpx.Client, args: argparse.Namespace, context: str, answer: str) -> bool | None:
    body = {"model": args.judge_model, "stream": False, "format": JUDGE_SCHEMA,
            "options": {"temperature": 0, "seed": args.seed},
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"}]}
    try:
        r = post_json(client, f"{args.ollama.rstrip('/')}/api/chat", body)
        return bool(json.loads(r["message"]["content"])["supported"])
    except Exception as exc:
        print(f"  (judge failed: {exc!r:.80})")
        return None


def e2e_level(args: argparse.Namespace, client: httpx.Client, golden: list[dict[str, Any]],
              answers_out: Path | None) -> dict[str, Any]:
    results = []
    for q in golden:
        t = time.perf_counter()
        resp = post_json(client, f"{args.api.rstrip('/')}/api/ask", {"question": q["question"]},
                         headers={"X-Bypass-Cache": "1"})
        results.append({"q": q, "resp": resp, "seconds": time.perf_counter() - t})
        print(f"  {q['id']:<5} {results[-1]['seconds']:>7.1f}s  {q['question'][:58]}")
    traces = read_traces(Path(args.trace_file))
    if args.trace_cmd:
        traces.update(traces_from_command(args.trace_cmd))

    answerable = [r for r in results if r["q"]["category"] in ("answerable", "hard")]
    unans = [r for r in results if r["q"]["category"] == "unanswerable" or r["q"].get("must_abstain")]
    answered = [r for r in answerable if r["resp"].get("answered_from_context")]
    valid = total = 0
    for r in results:
        v, n = M.citation_validity(r["resp"])
        valid, total = valid + v, total + n
    forbidden = [f"{r['q']['id']}: answer contains {f!r}" for r in results for f in (r["q"].get("forbid") or [])
                 if M._norm(f) in M._norm(r["resp"].get("answer", ""))]
    with_expect = [r for r in answered if r["q"].get("expect")]
    hits = [M._norm(r["q"]["expect"]) in M._norm(r["resp"]["answer"]) for r in with_expect]

    lex, jud, export = [], [], []
    hand = {}
    if args.hand_labels and Path(args.hand_labels).exists():
        for line in Path(args.hand_labels).read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                hand[x["id"]] = x
    pairs, skipped = [], 0
    for r in answered:
        rec = traces.get(r["resp"].get("request_id", ""))
        ctx = context_from_trace(rec) if rec else None
        answer = r["resp"]["answer"]
        sha = hashlib.sha256(answer.encode()).hexdigest()[:12]
        entry = {"id": r["q"]["id"], "question": r["q"]["question"], "answer": answer, "answer_sha": sha,
                 "request_id": r["resp"].get("request_id"), "context_excerpt": (ctx or "")[:700],
                 "lexical": None, "judge": None}
        if ctx:
            entry["lexical"] = M.lexical_faithfulness(answer, ctx)
            if entry["lexical"] is not None:
                lex.append(entry["lexical"])
            if args.judge:
                v = judge(client, args, ctx, answer)
                entry["judge"] = v
                if v is not None:
                    jud.append(1.0 if v else 0.0)
                    h = hand.get(r["q"]["id"])
                    if h is not None:
                        if h.get("answer_sha") in (None, sha):
                            pairs.append((v, bool(h["hand_supported"])))
                        else:
                            skipped += 1
        export.append(entry)
    if answers_out is not None:
        answers_out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in export), encoding="utf-8")

    lat = [r["seconds"] * 1000 for r in results]
    stage: dict[str, list[float]] = {}
    for r in results:
        rec = traces.get(r["resp"].get("request_id", ""))
        for k, v in ((rec or {}).get("latency_ms") or {}).items():
            stage.setdefault(k, []).append(float(v))
    pipeline = {}
    first = next((traces[r["resp"]["request_id"]] for r in results if r["resp"].get("request_id") in traces), None)
    if first:
        pipeline = {"model": first.get("model"), "embedding": first.get("embedding"), "collection": first.get("collection")}
    per_question = [{"id": r["q"]["id"], "category": r["q"]["category"],
                     "answered_from_context": r["resp"].get("answered_from_context"),
                     "seconds": round(r["seconds"], 1), "request_id": r["resp"].get("request_id")} for r in results]
    return {
        "per_question": per_question,
        "questions": len(results), "answerable_answered_rate": M.mean([1.0 if r in answered else 0.0 for r in answerable]),
        "over_abstention_rate": M.mean([0.0 if r in answered else 1.0 for r in answerable]),
        "answer_match_rate": M.mean([1.0 if h else 0.0 for h in hits]), "answer_match_n": len(hits),
        "abstention_correctness": M.mean([0.0 if r["resp"].get("answered_from_context") else 1.0 for r in unans]),
        "unanswerable_n": len(unans),
        "citation_validity": round(valid / total, 4) if total else None, "citations_total": total,
        "faithfulness_lexical": M.mean(lex), "faithfulness_lexical_n": len(lex),
        "faithfulness_judge": M.mean(jud), "faithfulness_judge_n": len(jud),
        "judge_vs_hand": {**M.agreement(pairs), "skipped_answer_changed": skipped},
        "latency_ms": {"p50": M.percentile(lat, 0.5), "p95": M.percentile(lat, 0.95), "n": len(lat),
                       "by_stage": {k: {"p50": M.percentile(v, 0.5), "p95": M.percentile(v, 0.95), "n": len(v)}
                                    for k, v in sorted(stage.items())}},
        "injection_obeyed": forbidden, "pipeline": pipeline,
        "traces_found": sum(1 for r in results if r["resp"].get("request_id") in traces),
    }


# ------------------------------------------------------------------ assembling

def flatten(res: dict[str, Any]) -> dict[str, Any]:
    """The flat metric dict that eval/gate.py compares between baseline and candidate."""
    f: dict[str, Any] = {}
    if "module" in res:
        m = res["module"]
        f.update(embed_gap=m["gap"], embed_mean_paraphrase=m["paraphrase"]["mean_cosine"],
                 embed_mean_unrelated=m["unrelated"]["mean_cosine"],
                 embed_ms_per_chunk=m["time_per_chunk_ms"]["mean"],
                 embed_determinism_max_abs_diff=m["determinism"]["max_abs_diff"])
    if "component" in res:
        c = res["component"]
        f.update({k: c["overall"][k] for k in ("recall_at_3", "recall_at_5", "mrr", "ndcg_at_5")})
        f.update(noise_recall_at_5=c["noise"]["overall"]["recall_at_5"], noise_recall_at_5_delta=c["noise"]["delta"]["recall_at_5"])
    if "e2e" in res:
        e = res["e2e"]
        f.update(abstention_correctness=e["abstention_correctness"], citation_validity=e["citation_validity"],
                 over_abstention_rate=e["over_abstention_rate"], answer_match_rate=e["answer_match_rate"],
                 faithfulness_lexical=e["faithfulness_lexical"], faithfulness_judge=e["faithfulness_judge"],
                 faithfulness=e["faithfulness_judge"] if e["faithfulness_judge"] is not None else e["faithfulness_lexical"],
                 latency_p50_ms=e["latency_ms"]["p50"], latency_p95_ms=e["latency_ms"]["p95"],
                 judge_agreement=e["judge_vs_hand"].get("agreement"),
                 injection_obeyed_count=len(e["injection_obeyed"]))
    return f


def fmt(x: Any) -> str:
    return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def summary_table(res: dict[str, Any]) -> str:
    rows = ["| Level | Metric | Value |", "| --- | --- | --- |"]
    if "module" in res:
        m = res["module"]
        rows += [f"| module | mean cosine, {m['paraphrase']['pairs']} paraphrase pairs | {fmt(m['paraphrase']['mean_cosine'])} |",
                 f"| module | mean cosine, {m['unrelated']['pairs']} unrelated pairs | {fmt(m['unrelated']['mean_cosine'])} |",
                 f"| module | gap (paraphrase - unrelated) | {fmt(m['gap'])} |",
                 f"| module | embedding time per chunk (mean / p95 ms) | {m['time_per_chunk_ms']['mean']} / {m['time_per_chunk_ms']['p95']} |",
                 f"| module | determinism, largest difference | {m['determinism']['max_abs_diff']:.2e} |"]
    if "component" in res:
        c = res["component"]
        o = c["overall"]
        rows += [f"| component | recall@3 / recall@5 (n={o['n']}) | {fmt(o['recall_at_3'])} / {fmt(o['recall_at_5'])} |",
                 f"| component | MRR / nDCG@5 | {fmt(o['mrr'])} / {fmt(o['ndcg_at_5'])} |"]
        for k, v in c["by_kind"].items():
            rows.append(f"| component | recall@5 for '{k}' (n={v['n']}) | {fmt(v['recall_at_5'])} |")
        n = c["noise"]
        rows += [f"| noise ({n['kind']}) | recall@5 / MRR with noise | {fmt(n['overall']['recall_at_5'])} / {fmt(n['overall']['mrr'])} |",
                 f"| noise | change in recall@5 / MRR | {fmt(n['delta']['recall_at_5'])} / {fmt(n['delta']['mrr'])} |"]
    if "e2e" in res:
        e = res["e2e"]
        rows += [f"| e2e | faithfulness: lexical (n={e['faithfulness_lexical_n']}) | {fmt(e['faithfulness_lexical'])} |",
                 f"| e2e | faithfulness: LLM judge (n={e['faithfulness_judge_n']}) | {fmt(e['faithfulness_judge'])} |",
                 f"| e2e | judge vs hand labels (n / agreement / kappa) | {e['judge_vs_hand'].get('n')} / {fmt(e['judge_vs_hand'].get('agreement'))} / {fmt(e['judge_vs_hand'].get('kappa'))} |",
                 f"| e2e | abstention correctness (n={e['unanswerable_n']}) | {fmt(e['abstention_correctness'])} |",
                 f"| e2e | citation validity ({e['citations_total']} citations) | {fmt(e['citation_validity'])} |",
                 f"| e2e | over-abstention on answerable questions | {fmt(e['over_abstention_rate'])} |",
                 f"| e2e | latency p50 / p95 (ms) | {fmt(e['latency_ms']['p50'])} / {fmt(e['latency_ms']['p95'])} |"]
    if "stochasticity" in res:
        sd = res["stochasticity"]
        quality = {k: v for k, v in sd["differences"].items() if not k.startswith(("latency", "embed_ms"))}
        worst = max(quality.items(), key=lambda kv: kv[1], default=("-", 0))
        rows.append(f"| stochasticity | {sd['runs']} runs, same seed: largest quality-metric difference | {worst[0]} = {worst[1]} |")
        lat = sd["differences"].get("latency_p50_ms")
        if lat is not None:
            rows.append(f"| stochasticity | latency p50 difference between runs (ms) | {lat} |")
    return "\n".join(rows)


def run_once(args: argparse.Namespace, client: httpx.Client, golden: list[dict[str, Any]], answers_out: Path | None) -> dict[str, Any]:
    res: dict[str, Any] = {}
    levels = set(args.levels.split(","))
    if "module" in levels:
        print("module level (embedding)")
        res["module"] = module_level(args, client)
    if "component" in levels:
        print("component level (retrieval)")
        res["component"] = component_level(args, client, golden)
    if "e2e" in levels:
        print("end-to-end level (answers)")
        res["e2e"] = e2e_level(args, client, golden, answers_out)
    res["flat"] = flatten(res)
    return res


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=HERE, timeout=5).stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    ap.add_argument("--api", required=True)
    ap.add_argument("--host", help="Host header when the Ingress rule names a host")
    ap.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    ap.add_argument("--pairs", type=Path, default=HERE / "embedding_pairs.json")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--tag", default="", help="label in the file name, e.g. baseline or candidate-topk8")
    ap.add_argument("--levels", default="module,component,e2e")
    ap.add_argument("--retrieval-only", action="store_true", help="same as --levels module,component")
    ap.add_argument("--limit", type=int, help="use only the first N golden questions (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1, help="run the suite N times (stochasticity test)")
    ap.add_argument("--noise-rate", type=float, default=0.15)
    ap.add_argument("--judge", action="store_true", help="LLM-judge faithfulness (slow on CPU)")
    ap.add_argument("--judge-model", default=env("JUDGE_MODEL", env("LLM_MODEL", "llama3.2:3b")))
    ap.add_argument("--hand-labels", type=Path, default=HERE / "judge_labels.jsonl")
    ap.add_argument("--trace-file", default=env("TRACE_FILE", "traces/traces.jsonl"))
    ap.add_argument("--trace-cmd", default=env("TRACE_CMD", ""),
                    help="command that prints the API's trace lines, e.g. "
                         "'kubectl logs -n rahbar -l app=api --tail=-1' (run once, after all answers)")
    ap.add_argument("--ollama", default=env("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--embed-model", default=env("EMBED_MODEL", "nomic-embed-text"))
    ap.add_argument("--chroma-host", default=env("CHROMA_HOST", "localhost"))
    ap.add_argument("--chroma-port", type=int, default=int(env("CHROMA_PORT", "8000")))
    ap.add_argument("--collection", default=env("CHROMA_COLLECTION", "rahbar"))
    args = ap.parse_args()
    if args.retrieval_only:
        args.levels = "module,component"
    random.seed(args.seed)

    golden = load_golden(args.golden)
    if args.limit:
        golden = golden[: args.limit]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"suite-{stamp}" + (f"-{args.tag}" if args.tag else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Host": args.host} if args.host else {}
    runs = []
    with httpx.Client(timeout=900.0, headers=headers) as client:
        for i in range(args.repeat):
            print(f"== run {i + 1} of {args.repeat} (seed {args.seed})")
            answers = args.out_dir / f"{name}-answers.jsonl" if "e2e" in args.levels and i == 0 else None
            runs.append(run_once(args, client, golden, answers))

    result = runs[0]
    result["meta"] = {"timestamp": stamp, "tag": args.tag, "api": args.api, "git_sha": git_sha(), "seed": args.seed,
                      "golden": str(args.golden), "golden_sha256": sha256_of(args.golden),
                      "questions": len(golden), "levels": args.levels, "repeat": args.repeat,
                      "judge_model": args.judge_model if args.judge else None, "cache_bypassed": True}
    if args.repeat > 1:
        diffs = {}
        for later in runs[1:]:
            for k, v in M.metric_differences(runs[0]["flat"], later["flat"]).items():
                diffs[k] = max(diffs.get(k, 0.0), v)
        result["stochasticity"] = {"runs": args.repeat, "seed": args.seed, "differences": diffs,
                                   "all_runs": [r["flat"] for r in runs]}
    table = summary_table(result)
    (args.out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / f"{name}.md").write_text(f"# Suite {name}\n\n{table}\n", encoding="utf-8")
    print("\n" + table + f"\n\nwritten: {args.out_dir / (name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
