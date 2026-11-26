"""Part E: coverage script and drift report, with embeddings stubbed (no Ollama needed)."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_coverage_counts_covered_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    cov = load("coverage")
    golden = [{"id": "q1", "question": "fee refund"}, {"id": "q2", "question": "grading scale"}]
    live = [{"id": "L1", "group": "domain", "query": "refund of fee"},
            {"id": "L2", "group": "ood", "query": "hostel allotment"},
            {"id": "L3", "group": "ood", "query": "hostel rooms"}]
    write_jsonl(tmp_path / "g.jsonl", golden)
    write_jsonl(tmp_path / "l.jsonl", live)
    vec = {"fee refund": [1, 0, 0], "grading scale": [0, 1, 0], "refund of fee": [0.99, 0.1, 0],
           "hostel allotment": [0, 0, 1], "hostel rooms": [0.05, 0, 1]}
    monkeypatch.setattr(cov, "embed_all", lambda a, texts: [vec[t] for t in texts])
    out = tmp_path / "cov.json"
    monkeypatch.setattr(sys, "argv", ["coverage", "--live", str(tmp_path / "l.jsonl"), "--golden",
                                      str(tmp_path / "g.jsonl"), "--threshold", "0.8", "--out", str(out)])
    assert cov.main() == 1  # 1 of 3 covered, below 85%
    res = json.loads(out.read_text())
    assert res["coverage"] == round(1 / 3, 4)
    assert len(res["uncovered_topics"]) == 1  # both hostel queries grouped together
    assert sorted(res["uncovered_topics"][0]["ids"]) == ["L2", "L3"]


def test_coverage_rises_after_adding_golden_question(tmp_path, monkeypatch):
    cov = load("coverage")
    live = [{"id": "L1", "group": "domain", "query": "a"}, {"id": "L2", "group": "domain", "query": "b"}]
    write_jsonl(tmp_path / "l.jsonl", live)
    vec = {"a": [1, 0], "b": [0, 1], "qa": [1, 0], "qb": [0, 1]}
    monkeypatch.setattr(cov, "embed_all", lambda a, texts: [vec[t] for t in texts])

    def run(golden):
        write_jsonl(tmp_path / "g.jsonl", golden)
        out = tmp_path / "o.json"
        monkeypatch.setattr(sys, "argv", ["c", "--live", str(tmp_path / "l.jsonl"), "--golden",
                                          str(tmp_path / "g.jsonl"), "--threshold", "0.9", "--out", str(out)])
        cov.main()
        return json.loads(out.read_text())["coverage"]

    before = run([{"id": "g1", "question": "qa"}])
    after = run([{"id": "g1", "question": "qa"}, {"id": "g2", "question": "qb"}])
    assert (before, after) == (0.5, 1.0)


def test_drift_report_lists_moved_metrics_and_changed_questions(tmp_path, monkeypatch, capsys):
    drift = load("drift_report")
    comp = lambda r, rank, src: {"component": {"per_question": [  # noqa: E731
        {"id": "q01", "recall_at_5": r, "first_hit_rank": rank, "top1": {"source": src, "section": "3.1"},
         "sources": [src]}]}}
    before = {"flat": {"recall_at_5": 1.0, "latency_p95_ms": 10}, **comp(1.0, 1, "old.pdf")}
    after = {"flat": {"recall_at_5": 0.8, "latency_p95_ms": 10}, **comp(0.0, 3, "batch-b.pdf")}
    (tmp_path / "b.json").write_text(json.dumps(before))
    (tmp_path / "a.json").write_text(json.dumps(after))
    out = tmp_path / "d.json"
    monkeypatch.setattr(sys, "argv", ["d", str(tmp_path / "b.json"), str(tmp_path / "a.json"),
                                      "--new-sources", "batch-b.pdf", "--out", str(out)])
    assert drift.main() == 0
    res = json.loads(out.read_text())
    assert res["metrics_moved"] == {"recall_at_5": -0.2}
    assert res["questions_changed"][0]["id"] == "q01"
    assert any("batch-b.pdf" in n for n in res["questions_changed"][0]["notes"])
