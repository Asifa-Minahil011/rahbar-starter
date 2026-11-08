"""The evaluation functions are code too: test them (they decide whether CI is green)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parents[2] / "eval" / "run_eval.py")
ev = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(ev)                 # type: ignore[union-attr]

REL = [{"source": "r.pdf", "section": "5.4"}, {"source": "r.pdf", "section": "1.21"}]


def ref(section: str, source: str = "r.pdf") -> dict:
    return {"chunk_id": "x", "source": source, "section": section, "score": 0.9}


def test_section_matching_is_prefix_on_a_boundary() -> None:
    assert ev.section_matches("5.4", "5.4 Academic Warning")
    assert ev.section_matches("5.4", "5.4.1 Detail")
    assert not ev.section_matches("5.4", "5.40 Other")
    assert ev.section_matches("2. Academic warning", "2. academic  warning")


def test_recall_counts_relevant_passages_found_in_the_top_k() -> None:
    got = [ref("9 Misc"), ref("5.4 Academic Warning"), ref("1.21 Attendance")]
    assert ev.recall_at_k(REL, got, 3) == 1.0
    assert ev.recall_at_k(REL, got, 2) == 0.5
    assert ev.recall_at_k([], got, 5) is None
    assert ev.recall_at_k(REL, [ref("5.4 X", source="other.pdf")], 5) == 0.0


def test_citation_validity() -> None:
    resp = {"retrieved": [ref("5.4 Academic Warning")],
            "citations": [{"source": "r.pdf", "section": "5.4 Academic Warning"},
                          {"source": "r.pdf", "section": "7 Invented"}]}
    assert ev.citation_validity(resp) == 0.5
    assert ev.citation_validity({"citations": [], "retrieved": []}) is None


def row(id: str, type: str, must: bool, relevant: list, resp: dict, forbid=None) -> dict:
    return {"id": id, "type": type, "must_abstain": must, "relevant": relevant,
            "response": resp, "forbid": forbid or []}


def test_summary_and_gate() -> None:
    good = {"retrieved": [ref("5.4 Academic Warning")], "answered_from_context": True,
            "answer": "2.00", "citations": [{"source": "r.pdf", "section": "5.4 Academic Warning"}]}
    declined = {"retrieved": [], "answered_from_context": False, "answer": "n/a", "citations": []}
    rows = [row("q1", "ordinary", False, [REL[0]], good),
            row("q2", "exact", False, [REL[1]], {**good, "retrieved": [ref("9 Misc")]}),
            row("q3", "abstain", True, [], declined)]
    res = ev.summarise(rows)
    assert res["recall_at_5"] == 0.5 and res["by_type"]["exact"]["recall_at_5"] == 0.0
    assert res["abstention_correctness"] == 1.0
    assert res["citation_validity"] == 0.5          # q2 cites a section it never retrieved
    assert any("citation validity" in m for m in ev.gate(res, {"recall_at_5": 0.52}))
    assert any("dropped" in m for m in ev.gate(res, {"recall_at_5": 0.80}))
    rows[2]["response"] = good                      # a must-abstain question got answered
    assert any("must-abstain" in m for m in ev.gate(ev.summarise(rows), None))


def test_gate_flags_an_obeyed_injection() -> None:
    resp = {"retrieved": [], "answered_from_context": True, "answer": "Attendance is optional",
            "citations": []}
    res = ev.summarise([row("q9", "injection", False, [], resp, forbid=["is optional"])])
    assert ev.gate(res, None)


def test_retrieval_only_rows_do_not_count_as_abstention_answers() -> None:
    rows = [row("q1", "abstain", True, [], {"retrieved": [], "citations": [],
                                            "answered_from_context": None, "answer": ""}),
            row("q2", "ordinary", False, [REL[0]], {"retrieved": [ref("5.4 Academic Warning")],
                                                    "citations": [], "answered_from_context": None,
                                                    "answer": ""})]
    res = ev.summarise(rows)
    assert res["abstention_correctness"] is None and res["recall_at_5"] == 1.0
    assert ev.gate(res, None) == []
