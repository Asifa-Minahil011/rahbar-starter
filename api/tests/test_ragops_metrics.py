"""The Part B metric functions are code too, and the report's numbers depend on them."""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ragops_metrics", Path(__file__).resolve().parents[2] / "eval" / "ragops_metrics.py")
m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(m)                 # type: ignore[union-attr]

REL = [{"source": "r.pdf", "section": "5.4"}]


def ref(section: str, source: str = "r.pdf") -> dict:
    return {"source": source, "section": section}


def test_mrr_is_one_over_the_rank_of_the_first_hit() -> None:
    got = [ref("9 x"), ref("8 y"), ref("5.4 Warning")]
    assert math.isclose(m.reciprocal_rank(REL, got), 1 / 3)
    assert m.reciprocal_rank(REL, [ref("9 x")]) == 0.0
    assert m.reciprocal_rank([], got) is None


def test_ndcg_is_one_when_the_hit_is_first_and_lower_when_it_is_not() -> None:
    assert m.ndcg_at_k(REL, [ref("5.4 W"), ref("9")], 5) == 1.0
    assert math.isclose(m.ndcg_at_k(REL, [ref("9"), ref("5.4 W")], 5), 1 / math.log2(3))
    assert m.ndcg_at_k(REL, [ref("9")], 5) == 0.0


def test_ndcg_with_two_relevant_sections_rewards_finding_both() -> None:
    rel = [{"source": "r.pdf", "section": "1.1"}, {"source": "r.pdf", "section": "2.2"}]
    both = m.ndcg_at_k(rel, [ref("1.1 a"), ref("2.2 b")], 5)
    one = m.ndcg_at_k(rel, [ref("1.1 a"), ref("9 z")], 5)
    assert both == 1.0 and 0 < one < 1


def test_section_matching_does_not_confuse_5_4_with_5_40() -> None:
    assert not m.section_matches("5.4", "5.40 Other")
    assert m.section_matches("5.4", "5.4.1 Detail")


def test_typos_are_seeded_and_leave_clause_numbers_alone() -> None:
    q = "What does clause 4.24 say about withdrawing from a course quickly"
    a = m.add_typos(q, random.Random(1), rate=1.0)
    b = m.add_typos(q, random.Random(1), rate=1.0)
    assert a == b and a != q and "4.24" in a
    assert m.add_typos(q, random.Random(2), rate=1.0) != a


def test_cosine_and_max_abs_diff() -> None:
    assert math.isclose(m.cosine([1, 0], [1, 0]), 1.0) and math.isclose(m.cosine([1, 0], [0, 1]), 0.0)
    assert m.max_abs_diff([1.0, 2.0], [1.0, 2.5]) == 0.5


def test_percentile_interpolates() -> None:
    assert m.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert m.percentile([10], 0.95) == 10 and m.percentile([], 0.5) is None
    assert m.percentile(list(range(1, 101)), 0.95) == 95.05


def test_lexical_faithfulness_separates_supported_from_invented() -> None:
    ctx = "A student may repeat at most two courses. The grade of the most recent attempt counts."
    assert m.lexical_faithfulness("A student may repeat at most two courses [rules.pdf / 6.6].", ctx) == 1.0
    assert m.lexical_faithfulness("Students pay a library membership fee every semester.", ctx) == 0.0
    half = m.lexical_faithfulness("At most two courses may be repeated. The hostel closes at midnight.", ctx)
    assert half == 0.5


def test_citation_validity_counts_only_retrieved_citations() -> None:
    resp = {"retrieved": [ref("5.4 W")], "citations": [ref("5.4 W"), ref("7 invented")]}
    assert m.citation_validity(resp) == (1, 2)


def test_agreement_and_kappa() -> None:
    perfect = m.agreement([(True, True), (False, False), (True, True), (False, False)])
    assert perfect["agreement"] == 1.0 and perfect["kappa"] == 1.0
    assert m.agreement([])["agreement"] is None
    assert m.agreement([(True, False), (True, False)])["agreement"] == 0.0


def test_metric_differences_only_compares_numbers_present_in_both() -> None:
    assert m.metric_differences({"a": 1.0, "b": 0.5, "c": "x"}, {"a": 0.75, "b": 0.5}) == {"a": 0.25, "b": 0.0}


def test_ndcg_never_exceeds_one_when_a_clause_spans_two_chunks() -> None:
    """Regression: two retrieved chunks of the same relevant section gave nDCG 1.17."""
    got = m.ndcg_at_k(REL, [ref("5.4 W"), ref("5.4 W more text"), ref("9")], 5)
    assert got == 1.0


def test_parse_traces_reads_kubectl_log_lines_and_trace_files() -> None:
    import json as _json
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "eval"))
    import ragops_suite as suite

    log_line = _json.dumps({"msg": "trace", "trace": {"trace_id": "abc", "prompt": []}})
    file_line = _json.dumps({"trace_id": "def"})
    got = suite.parse_traces("\n".join(["not json", log_line, '{"msg": "x"}', file_line, "{broken"]))
    assert set(got) == {"abc", "def"}
