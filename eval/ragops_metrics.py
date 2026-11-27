"""Pure metric functions for the Assignment 3 suite (Part B). No network, no model.

Kept apart from ragops_suite.py so that every number in the report comes from code
that has unit tests (api/tests/test_ragops_metrics.py).
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

# ---------------------------------------------------------------- matching (same rule as run_eval.py)


def _norm(text: Any) -> str:
    return " ".join(str(text).lower().split())


def section_matches(golden: str, retrieved: str) -> bool:
    """Golden "5.4" matches "5.4 Academic Warning" and "5.4.1 ..." but not "5.40 ..."."""
    g, r = _norm(golden), _norm(retrieved)
    return bool(g) and (r == g or r.startswith(g + " ") or r.startswith(g + "."))


def is_relevant(rel: dict[str, Any], ref: dict[str, Any]) -> bool:
    return (_norm(rel["source"]) == _norm(ref.get("source", ""))
            and section_matches(str(rel["section"]), str(ref.get("section", ""))))


def relevance_vector(relevant: list[dict[str, Any]], retrieved: list[dict[str, Any]]) -> list[int]:
    """1 where the retrieved chunk at that rank matches any relevant (source, section)."""
    return [int(any(is_relevant(rel, ref) for rel in relevant)) for ref in retrieved]


# ---------------------------------------------------------------- retrieval metrics


def recall_at_k(relevant: list[dict[str, Any]], retrieved: list[dict[str, Any]], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs that appear in the top k."""
    if not relevant:
        return None
    top = retrieved[:k]
    return sum(1 for rel in relevant if any(is_relevant(rel, ref) for ref in top)) / len(relevant)


def reciprocal_rank(relevant: list[dict[str, Any]], retrieved: list[dict[str, Any]]) -> float | None:
    """1 / rank of the first relevant chunk; 0 when none was retrieved."""
    if not relevant:
        return None
    for rank, flag in enumerate(relevance_vector(relevant, retrieved), start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: list[dict[str, Any]], retrieved: list[dict[str, Any]], k: int) -> float | None:
    """Binary-gain nDCG. The ideal ranking puts min(#relevant sections, k) hits first."""
    if not relevant:
        return None
    # A relevant section can earn its gain once: two chunks of the same clause (it can span
    # pages) must not count twice, or nDCG would pass 1.
    open_rel = list(relevant)
    gains = []
    for ref in retrieved[:k]:
        hit = next((r for r in open_rel if is_relevant(r, ref)), None)
        if hit is not None:
            open_rel.remove(hit)
        gains.append(int(hit is not None))
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_5_value(row: dict[str, Any]) -> float | None:
    return recall_at_k(row["relevant"], row["retrieved"], 5)


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def retrieval_scores(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """rows: {"relevant": [...], "retrieved": [...]} for answerable questions only."""
    return {
        "recall_at_3": mean([recall_at_k(r["relevant"], r["retrieved"], 3) for r in rows]),
        "recall_at_5": mean([recall_at_k(r["relevant"], r["retrieved"], 5) for r in rows]),
        "mrr": mean([reciprocal_rank(r["relevant"], r["retrieved"]) for r in rows]),
        "ndcg_at_5": mean([ndcg_at_k(r["relevant"], r["retrieved"], 5) for r in rows]),
        "n": len(rows),  # type: ignore[dict-item]
    }


# ---------------------------------------------------------------- noise: typos


def add_typos(text: str, rng: random.Random, rate: float = 0.15) -> str:
    """Corrupt about `rate` of the words of 4+ letters with one edit (swap, drop or double a
    letter). Seeded, so the same seed gives the same noisy query. Numbers (clause numbers)
    are left alone: the test is about spelling mistakes, not about changing the question."""
    out = []
    for word in text.split(" "):
        letters = re.sub(r"[^A-Za-z]", "", word)
        if len(letters) >= 4 and word.isalpha() and rng.random() < rate:
            i = rng.randrange(1, len(word) - 1)
            kind = rng.choice(("swap", "drop", "double"))
            if kind == "swap":
                word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            elif kind == "drop":
                word = word[:i] + word[i + 1:]
            else:
                word = word[:i] + word[i] + word[i:]
        out.append(word)
    return " ".join(out)


# ---------------------------------------------------------------- embeddings


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def max_abs_diff(a: list[float], b: list[float]) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


# ---------------------------------------------------------------- latency


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile; None for no data."""
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (pos - lo), 3)


# ---------------------------------------------------------------- faithfulness (lexical)

STOP = set("""a an the and or of to in on for with by at from as is are was were be been it its this
that these those which who whom whose what when where why how not no nor than then so such can may
must shall should would could will also any all each both per into over under about above below
after before between during their there they them his her him she he you your our we us has have
had do does did if but""".split())
CITATION = re.compile(r"\[[^\]]*\]|\([^)]*\.pdf[^)]*\)")


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower())
            if w not in STOP and (len(w) > 2 or w.isdigit())}


def split_sentences(text: str) -> list[str]:
    text = CITATION.sub(" ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 3]


def lexical_faithfulness(answer: str, context: str, threshold: float = 0.6) -> float | None:
    """Share of answer sentences whose content words are mostly present in the context.
    A deterministic proxy for "is the answer supported by the retrieved context"; it cannot
    see paraphrase or negation, which is why the suite also offers an LLM judge."""
    ctx = content_words(context)
    sentences = split_sentences(answer)
    scored = []
    for s in sentences:
        words = content_words(s)
        if words:
            scored.append(len(words & ctx) / len(words) >= threshold)
    return sum(scored) / len(scored) if scored else None


# ---------------------------------------------------------------- end to end


def citation_validity(response: dict[str, Any]) -> tuple[int, int]:
    """(valid, total) citations that match a chunk retrieved for the same request."""
    cites = response.get("citations") or []
    retrieved = response.get("retrieved") or []
    ok = sum(1 for c in cites if any(
        _norm(c.get("source", "")) == _norm(r.get("source", ""))
        and _norm(c.get("section", "")) == _norm(r.get("section", "")) for r in retrieved))
    return ok, len(cites)


def agreement(pairs: list[tuple[bool, bool]]) -> dict[str, float | int | None]:
    """Judge vs hand labels: raw agreement and Cohen's kappa."""
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    agree = sum(1 for j, h in pairs if j == h) / n
    pj = sum(1 for j, _ in pairs if j) / n
    ph = sum(1 for _, h in pairs if h) / n
    expected = pj * ph + (1 - pj) * (1 - ph)
    kappa = (agree - expected) / (1 - expected) if expected < 1 else 1.0
    return {"n": n, "agreement": round(agree, 4), "kappa": round(kappa, 4)}


def metric_differences(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, float]:
    """Absolute difference of every numeric metric present in both flat dicts."""
    return {k: round(abs(run_a[k] - run_b[k]), 6) for k in sorted(run_a)
            if isinstance(run_a[k], (int, float)) and isinstance(run_b.get(k), (int, float))}
