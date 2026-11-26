"""The release gate decides what ships, so it is tested like any other code."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("gate", Path(__file__).resolve().parents[2] / "eval" / "gate.py")
g = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(g)                 # type: ignore[union-attr]

BASE = {"recall_at_5": 0.90, "faithfulness": 0.80, "latency_p95_ms": 100000, "abstention_correctness": 1.0,
        "citation_validity": 1.0, "injection_obeyed_count": 0}
RULES = [
    {"id": "r5", "metric": "recall_at_5", "check": "max_drop_vs_baseline", "value": 0.02},
    {"id": "f", "metric": "faithfulness", "check": "min", "value": 0.6},
    {"id": "lat", "metric": "latency_p95_ms", "check": "max", "value": 150000},
    {"id": "ab", "metric": "abstention_correctness", "check": "not_below_baseline"},
]


def status(cand: dict) -> dict[str, str]:
    return {r["id"]: r["status"] for r in g.evaluate(RULES, cand, BASE)}


def test_an_identical_candidate_passes_every_rule() -> None:
    assert set(status(dict(BASE)).values()) == {"PASS"}


def test_recall_may_drop_two_points_but_not_three() -> None:
    assert status({**BASE, "recall_at_5": 0.88})["r5"] == "PASS"
    assert status({**BASE, "recall_at_5": 0.87})["r5"] == "FAIL"


def test_each_rule_can_fail_on_its_own() -> None:
    assert status({**BASE, "faithfulness": 0.5})["f"] == "FAIL"
    assert status({**BASE, "latency_p95_ms": 200000})["lat"] == "FAIL"
    assert status({**BASE, "abstention_correctness": 0.8333})["ab"] == "FAIL"


def test_a_metric_that_was_not_measured_fails_unless_the_rule_is_optional() -> None:
    cand = {k: v for k, v in BASE.items() if k != "faithfulness"}
    assert status(cand)["f"] == "FAIL"
    optional = [{"id": "f", "metric": "faithfulness", "check": "min", "value": 0.6, "optional": True}]
    assert g.evaluate(optional, cand, BASE)[0]["status"] == "SKIP"


def test_changing_the_threshold_changes_the_outcome_without_touching_the_code() -> None:
    cand = {**BASE, "latency_p95_ms": 120000}
    strict = [{"id": "lat", "metric": "latency_p95_ms", "check": "max", "value": 110000}]
    loose = [{"id": "lat", "metric": "latency_p95_ms", "check": "max", "value": 130000}]
    assert g.evaluate(strict, cand, BASE)[0]["status"] == "FAIL"
    assert g.evaluate(loose, cand, BASE)[0]["status"] == "PASS"


def test_exit_codes_through_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml
    cfg = tmp_path / "gate.yaml"
    cfg.write_text(yaml.safe_dump({"rules": RULES}))
    (tmp_path / "baseline.json").write_text(json.dumps({"flat": BASE}))
    good, bad = tmp_path / "suite-good.json", tmp_path / "suite-bad.json"
    good.write_text(json.dumps({"flat": BASE}))
    bad.write_text(json.dumps({"flat": {**BASE, "recall_at_5": 0.5}}))

    def run(cand: Path) -> int:
        monkeypatch.setattr("sys.argv", ["gate", "--candidate", str(cand), "--baseline",
                                         str(tmp_path / "baseline.json"), "--config", str(cfg)])
        return g.main()

    assert run(good) == 0 and run(bad) == 1
    assert run(tmp_path / "missing.json") == 2
