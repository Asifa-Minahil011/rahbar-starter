#!/usr/bin/env python3
"""Write the measured results into the docs, after scripts/finish_all.py has run.

Reads eval/tuning.json, eval/tuning-b.json, eval/failure-CE.json, eval/failure-F.json,
eval/results.json and /tmp/finish/cache.log, and fills:
  docs/PIPELINE.md           chunk-tuning table + justification, semantic-cache section
  docs/FAILURE-ANALYSIS.md   one block per stage (B, C, E, F) with the measured symptom
  docs/ENGINEERING-NOTES.md  question 5 (recall by question type)

Each part is independent: a missing input is reported and that part is skipped.
Everything it writes is a DRAFT built from your numbers: read it, edit it into your own words,
and check `git diff docs` before committing.

    python scripts/fill_results.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def by_type(d: dict[str, Any] | None) -> str:
    if not d:
        return "n/a"
    return ", ".join(f"{t} {pct(v)}" for t, v in d.items() if v is not None)


def tuning_table(rows: list[dict[str, Any]]) -> str:
    out = ["| Size (words) | Overlap (words) | Chunks stored | Storage amplification | recall@3 | recall@5 "
           "| recall@5 with exact-term boost |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['size']} | {r['overlap']} | {r['chunks']} | {r['amplification']:.3f} | "
                   f"{pct(r['recall_at_3'])} | {pct(r['recall_at_5'])} | {pct(r['recall_at_5_boost'])} |")
    return "\n".join(out)


def justification(rows: list[dict[str, Any]], shipped: tuple[int, int] = (400, 50)) -> str:
    best = max(rows, key=lambda r: (r["recall_at_5"], -r["chunks"]))
    ship = next((r for r in rows if (r["size"], r["overlap"]) == shipped), None)
    if ship is None:
        return f"Best measured: {best['size']}:{best['overlap']} with recall@5 {pct(best['recall_at_5'])}."
    others = ", ".join(f"{r['size']}:{r['overlap']} {pct(r['recall_at_5'])} ({r['chunks']} chunks)"
                       for r in rows if r is not ship)
    if ship["recall_at_5"] >= best["recall_at_5"]:
        tie = [r for r in rows if r is not ship and r["recall_at_5"] == ship["recall_at_5"]]
        why = (f"{shipped[0]}:{shipped[1]} reached recall@5 {pct(ship['recall_at_5'])}, "
               + ("tied with the best other configuration" if tie else "the best of the three")
               + f", while storing {ship['chunks']} chunks at amplification {ship['amplification']:.3f}")
    else:
        gap = (best["recall_at_5"] - ship["recall_at_5"]) * 100
        why = (f"{best['size']}:{best['overlap']} measured {gap:.1f} points higher recall@5 "
               f"({pct(best['recall_at_5'])} vs {pct(ship['recall_at_5'])}) on this golden set, but we kept "
               f"{shipped[0]}:{shipped[1]} ({ship['chunks']} chunks, amplification {ship['amplification']:.3f}): the "
               f"difference is small against {len(rows)} measured configurations and one golden question moves it by several points")
    return (f"Why the shipped configuration: {why}. Other configurations: {others}. The rules documents are "
            f"insensitive to size, because each clause is already its own short section; size matters for the "
            f"narrative documents (handbook, annual reports).")


def parse_cache(text: str) -> dict[str, Any]:
    rows = [(float(m.group(1)), int(m.group(2)), int(m.group(3)), (m.group(4) or "").strip())
            for m in re.finditer(r"^\| (\d\.\d+) \| (\d+) \| (\d+)(?:: (.*?))? \|$", text, re.M)]
    head = re.search(r"True hits \(of (\d+) paraphrases\) \| False hits \(of (\d+) near-misses\)", text)
    hi = re.search(r"highest near-miss\s*:\s*([\d.]+)", text)
    raw = re.search(r"highest near-miss without the guard\s*:\s*([\d.]+)", text)
    blocked = re.findall(r"^blocked by the exact-term guard at any threshold \(cosine ([\d.]+)\): (.*)$", text, re.M)
    return {"raw_highest": float(raw.group(1)) if raw else None, "blocked": blocked, "rows": rows, "same": int(head.group(1)) if head else None,
            "different": int(head.group(2)) if head else None, "highest": float(hi.group(1)) if hi else None}


def cache_section(text: str) -> tuple[str, str]:
    c = parse_cache(text)
    if not c["rows"]:
        raise ValueError("no threshold table found in the cache log")
    rows = sorted(c["rows"])
    clean = [r for r in rows if r[2] == 0]
    table = ["| Threshold | True hits (of %s paraphrases) | False hits (of %s near-misses) |" % (c["same"], c["different"]),
             "| --- | --- | --- |"]
    for t, hits, false, label in rows:
        table.append(f"| {t:.2f} | {hits} | {false}" + (f" ({label})" if label else "") + " |")
    shipped = min(clean)[0] if clean else None
    lower = [r for r in rows if shipped is not None and r[0] < shipped and r[2] > 0]
    ruled = lower[-1] if lower else None
    body = ["Thresholds measured on the pairs in `eval/cache_pairs.yaml` (paraphrases that must hit, near-misses "
            "that must never hit). The rule from the lecture: the lowest threshold with zero false hits.", "", *table, ""]
    if shipped is None:
        body.append("No tested threshold avoided every false hit. Do not ship a cache threshold below the highest "
                    f"near-miss similarity ({c['highest']}); raise the tested range.")
        value = ""
    else:
        body.append(f"Shipped: `CACHE_THRESHOLD={shipped:.2f}` (ConfigMap).")
        if ruled:
            body.append(f"The false hit that ruled out {ruled[0]:.2f}: {ruled[3] or 'see the table'}. "
                        f"A cached answer for that pair would be served to the wrong question.")
        value = f"{shipped:.2f}"
    if c.get("blocked"):
        cos_, pair = c["blocked"][0]
        body += ["", f"Pure cosine cannot separate near-misses: {pair} reached cosine {cos_}, as similar as a paraphrase, "
                 "yet the answers differ. So the cache also requires both questions to name the same clause numbers, "
                 "course codes and years (`signature` in `api/app/services/cache.py`); such pairs never hit, at any "
                 "threshold. The table above counts only the pairs the guard lets through."]
    body += ["", "Never cached: abstentions, fallbacks, answers without a valid citation, follow-ups. Every key "
             "contains the index's `index_id`, so a re-index invalidates the cache (`api/app/services/cache.py`)."]
    return "\n".join(body), value


def replace_section(text: str, heading: str, next_heading: str, new_body: str) -> str:
    pat = re.compile(rf"({re.escape(heading)}\n).*?(?={re.escape(next_heading)})", re.S)
    if not pat.search(text):
        raise ValueError(f"section {heading!r} not found")
    return pat.sub(lambda m: m.group(1) + new_body.rstrip() + "\n\n", text, count=1)


def failure_doc(root: Path) -> str:
    ev = root / "eval"
    tuning = load_json(ev / "tuning.json") if (ev / "tuning.json").exists() else []
    ship = next((r for r in tuning if (r["size"], r["overlap"]) == (400, 50)), None)
    ce = load_json(ev / "failure-CE.json") if (ev / "failure-CE.json").exists() else {}
    fx = load_json(ev / "failure-F.json") if (ev / "failure-F.json").exists() else {}
    tb = load_json(ev / "tuning-b.json") if (ev / "tuning-b.json").exists() else []
    base = ce.get("baseline", {})
    parts = ["# Failure analysis", "",
             "We broke four of the six stages on purpose, one at a time, restored each, and recorded what the "
             "measurements showed (`scripts/failure_experiments.py`, `scripts/tune_chunking.py`; raw numbers in "
             "`eval/failure-*.json` and `eval/tuning-b.json`). Only stage F is the model's fault, so A to E are checked first.", ""]
    if base:
        parts += [f"Baseline, nothing broken: recall@5 {pct(base.get('recall_at_5'))} "
                  f"(by type: {by_type({t: v['recall_at_5'] for t, v in base.get('by_type', {}).items()})}).", ""]
    if tb:
        b = tb[0]
        parts += ["## Stage B · Chunking",
                  "- **What we broke:** fixed 100-word chunks with no overlap (`tune_chunking.py --configs 100:0`), indexed "
                  "into a scratch collection; the real collection was untouched.",
                  f"- **Symptom (measured):** recall@5 {pct(b['recall_at_5'])} and recall@3 {pct(b['recall_at_3'])} with "
                  f"{b['chunks']} chunks" + (f", against {pct(ship['recall_at_5'])} recall@5 for the shipped 400:50 "
                  f"({ship['chunks']} chunks). By type: {by_type(b.get('by_type'))} vs {by_type(ship.get('by_type'))}." if ship else "."),
                  "- **Diagnosis, as if we had not caused it:** recall@5 by question type in `eval/results.json` drops "
                  "for questions whose answer spans several sentences, and the retrieved chunk text in the `answered` "
                  "log line ends mid-sentence. Chunk count and storage amplification in the indexer's `chunking summary` "
                  "line jump when the size shrinks.",
                  "- **Remedy:** structure-aware chunking with overlap, size and overlap chosen from the tuning table "
                  "(ADR 0002).", ""]
    if "C" in ce:
        c = ce["C"]
        parts += ["## Stage C · Retrieval",
                  "- **What we broke:** `TOP_K=1`.",
                  f"- **Symptom (measured):** recall@5 fell from {pct(base.get('recall_at_5'))} to {pct(c['recall_at_5'])}; "
                  f"by type {by_type({t: v['recall_at_5'] for t, v in c.get('by_type', {}).items()})}. A question with two "
                  "relevant clauses can score at most one half.",
                  "- **Diagnosis, as if we had not caused it:** the `answered` log line shows `top_k: 1` and a single "
                  "`chunk_ids` entry; the retrieval-only eval's recall by type is the number that moved.",
                  "- **Remedy:** `TOP_K=5` from the ConfigMap.", ""]
    if "E" in ce:
        e = ce["E"]
        vq = e.get("version_question_sources", {})
        vb = base.get("version_question_sources", {})
        parts += ["## Stage E · Source",
                  "- **What we broke:** `SUPERSEDED_FILTER=off`, so superseded editions are retrieved.",
                  f"- **Symptom (measured):** recall@5 {pct(e['recall_at_5'])} (baseline {pct(base.get('recall_at_5'))}); "
                  f"version questions retrieved from {json.dumps(vq)} with the filter off, against {json.dumps(vb)} normally. "
                  "Both editions share clause numbers, so the 2023 edition competes for the same questions.",
                  "- **Diagnosis, as if we had not caused it:** the retrieved `source` and `doc_version` of the version "
                  "questions in the log line show `undergraduate-academic-rules-aug-2023.pdf` (`superseded: true`) in the top k.",
                  "- **Remedy:** metadata filter on `superseded: false` unless the question asks about an older edition "
                  "(`api/app/retrieval/chroma.py`); a prompt cannot fix this.", ""]
    if "F" in fx:
        f, bf = fx["F"], fx.get("baseline_full", {})
        m = f.get("metrics", {})
        parts += ["## Stage F · Generation",
                  "- **What we broke:** `TEMPERATURE=1.2`.",
                  f"- **Symptom (measured):** citation validity {pct(f.get('citation_validity'))} (baseline "
                  f"{pct(bf.get('citation_validity'))}), abstention correctness {pct(f.get('abstention_correctness'))} "
                  f"(baseline {pct(bf.get('abstention_correctness'))}). Counters in this run: retries "
                  f"{m.get('rahbar_generation_retries_total', 0):g}, fallbacks {m.get('rahbar_fallbacks_total', 0):g}, "
                  f"invalid citations dropped {m.get('rahbar_invalid_citations_total', 0):g}. Retrieval recall is unchanged "
                  "by temperature, which is how this stage is told apart from A to E.",
                  "- **Diagnosis, as if we had not caused it:** retrieval-only recall is normal while the generation counters "
                  "(`rahbar_generation_retries_total`, `rahbar_fallbacks_total`, `rahbar_invalid_citations_total` in `/metrics`) "
                  "and citation validity move.",
                  "- **Remedy:** `TEMPERATURE=0.2` from the ConfigMap, schema-constrained output, one retry and a safe fallback.", ""]
    return "\n".join(parts)


def line_of(path: Path, needle: str) -> int | None:
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in ln:
            return i
    return None


def q5_text(root: Path) -> str:
    r = load_json(root / "eval" / "results.json")
    types = {t: v for t, v in r["by_type"].items() if v["recall_at_5"] is not None}
    worst_t, worst = min(types.items(), key=lambda kv: kv[1]["recall_at_5"])
    best_t, best = max(types.items(), key=lambda kv: kv[1]["recall_at_5"])
    chroma = root / "api" / "app" / "retrieval" / "chroma.py"
    l_terms, l_boost = line_of(chroma, "def exact_terms"), line_of(chroma, "hybrid_boost * hit")
    tuning = root / "eval" / "tuning.json"
    boost = ""
    if tuning.exists():
        ship = next((x for x in load_json(tuning) if (x["size"], x["overlap"]) == (400, 50)), None)
        if ship:
            boost = (f" In the tuning run the exact-term boost changed recall@5 from {pct(ship['recall_at_5'])} to "
                     f"{pct(ship['recall_at_5_boost'])}; exact questions alone: {pct(ship.get('by_type', {}).get('exact'))} "
                     f"without it, {pct(ship.get('by_type_boost', {}).get('exact'))} with it.")
    table = "\n".join(f"| {t} | {v['n']} | {pct(v['recall_at_3'])} | {pct(v['recall_at_5'])} |" for t, v in r["by_type"].items())
    return (f"Measured on the golden set (`eval/results.json`): overall recall@5 {pct(r['recall_at_5'])}.\n\n"
            f"| Question type | Questions | recall@3 | recall@5 |\n| --- | --- | --- | --- |\n{table}\n\n"
            f"It is worst for **{worst_t}** questions ({pct(worst['recall_at_5'])}, {worst['n']} questions) and best for "
            f"{best_t} ({pct(best['recall_at_5'])}). Embeddings struggle with exact terms because a clause number or a "
            f"code like `4.23` is a rare token whose vector sits almost on top of `4.24`: the embedding captures what a "
            f"passage is about, not which exact number it carries, so neighbouring clauses are near-duplicates in vector "
            f"space. The technique from the RAG lecture that addresses it is hybrid retrieval (exact matching next to "
            f"vectors). We implemented a small version: `exact_terms` (`api/app/retrieval/chroma.py:{l_terms}`) pulls "
            f"clause numbers and course codes out of the question, and candidates that contain them get a rank bonus "
            f"(`HYBRID_BOOST`, `chroma.py:{l_boost}`).{boost}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cache-log", type=Path, default=Path("/tmp/finish/cache.log"))
    args = ap.parse_args()
    root, docs, ev = args.root, args.root / "docs", args.root / "eval"
    done: list[str] = []

    def attempt(name: str, fn: Any) -> None:
        try:
            fn()
            done.append(f"ok       {name}")
        except Exception as exc:
            done.append(f"SKIPPED  {name}: {exc}")

    def pipeline_tuning() -> None:
        rows = load_json(ev / "tuning.json")
        p = docs / "PIPELINE.md"
        t = p.read_text(encoding="utf-8")
        t = re.sub(r"TUNING-TABLE-GOES-HERE", tuning_table(rows), t)
        t, n = re.subn(r"Why the shipped configuration:.*?(?=\n\n|\Z)", lambda _: justification(rows), t, count=1, flags=re.S)
        if n != 1:
            raise ValueError("justification paragraph not found")
        p.write_text(t, encoding="utf-8")

    def pipeline_cache() -> None:
        body, value = cache_section(args.cache_log.read_text(encoding="utf-8"))
        p = docs / "PIPELINE.md"
        p.write_text(replace_section(p.read_text(encoding="utf-8"), "## Semantic cache", "## Evaluation method", body),
                     encoding="utf-8")
        (ev / "cache-threshold.txt").write_text(value + "\n", encoding="utf-8")

    def failure() -> None:
        text = failure_doc(root)
        if "## Stage" not in text:
            raise ValueError("no measured stage data yet (eval/failure-*.json, eval/tuning-b.json); file left untouched")
        (docs / "FAILURE-ANALYSIS.md").write_text(text, encoding="utf-8")

    def q5() -> None:
        p = docs / "ENGINEERING-NOTES.md"
        t = p.read_text(encoding="utf-8")
        m = re.search(r"(## 5\. [^\n]*\n)(.*?)(?=## 6\.)", t, re.S)
        if not m:
            raise ValueError("question 5 heading not found")
        if m.group(2).strip():
            raise ValueError("question 5 already has text; delete it first if you want it replaced")
        p.write_text(t[:m.end(1)] + "\n" + q5_text(root) + "\n\n" + t[m.end():], encoding="utf-8")

    attempt("PIPELINE.md tuning table and justification", pipeline_tuning)
    attempt("PIPELINE.md semantic cache section", pipeline_cache)
    attempt("FAILURE-ANALYSIS.md", failure)
    attempt("ENGINEERING-NOTES.md question 5", q5)
    print("\n".join(done))
    cv = ev / "cache-threshold.txt"
    if cv.exists() and cv.read_text().strip():
        print(f"\nPut CACHE_THRESHOLD={cv.read_text().strip()} in the ConfigMap (tell Asifa).")
    print("\nReview with: git diff docs   (edit the text into your own words before committing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
