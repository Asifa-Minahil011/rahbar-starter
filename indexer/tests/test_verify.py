"""RAGOps A1: the verify stage. No model, no network: embeddings are a stub."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from pipeline import run as run_mod
from pipeline.config import IndexerSettings
from pipeline.models import Document, Page
from pipeline.verify import (
    Thresholds,
    manifest_problems,
    quarantine_files,
    verify_documents,
    version_year,
    write_report,
)

TODAY = dt.date(2026, 11, 27)
GOOD = "The Head of Department may condone absence for documented medical reasons. " * 12


def doc(name: str = "a.pdf", text: str = GOOD, version: str = "2026-06", superseded: bool = False,
        pages: int = 3, ocr: bool = False) -> Document:
    return Document(name, version, superseded, [Page(name, i + 1, f"{text} page {i}", ocr) for i in range(pages)])


def by_name(report) -> dict[str, str]:
    return {c.name: c.status for c in report.checks}


def verify(docs, embed_fn=None):
    return verify_documents(docs, Thresholds(), embed_fn, TODAY)


def test_a_good_document_passes_all_four_checks() -> None:
    (r,) = verify([doc()])
    assert r.decision == "accept"
    assert set(by_name(r).values()) <= {"pass"}


def test_quality_fails_on_empty_pages_and_on_garbled_text() -> None:
    empty = Document("e.pdf", "2026", False, [Page("e.pdf", i, "", False) for i in range(1, 5)])
    garbled = doc("g.pdf", text="ab��� " * 80)
    r_empty, r_garbled = verify([empty, garbled])
    assert by_name(r_empty)["quality"] == "fail" and r_empty.decision == "quarantine"
    assert by_name(r_garbled)["quality"] == "fail"


def test_a_clean_scan_is_kept_but_a_thin_scan_is_quarantined() -> None:
    (clean_scan,) = verify([doc(ocr=True)])           # every page OCR'd, plenty of clean text
    assert by_name(clean_scan)["quality"] == "warn" and clean_scan.decision == "accept"
    thin = Document("t.pdf", "2026", False, [Page("t.pdf", i, "word " * 8, True) for i in range(1, 5)])
    (r_thin,) = verify([thin])
    assert by_name(r_thin)["quality"] == "fail"       # OCR'd, and the OCR found almost nothing
    mixed = Document("m.pdf", "2026", False,
                     [Page("m.pdf", 1, GOOD, True), Page("m.pdf", 2, GOOD, False), Page("m.pdf", 3, GOOD, False)])
    (m,) = verify([mixed])
    assert by_name(m)["quality"] == "warn" and m.decision == "accept"


def test_completeness_needs_a_version_and_enough_text() -> None:
    no_version, tiny = verify([doc("n.pdf", version="unknown"), doc("t.pdf", text="hi", pages=1)])
    assert by_name(no_version)["completeness"] == "fail"
    assert by_name(tiny)["completeness"] == "fail"


def test_recency_superseded_is_kept_old_current_warns_and_very_old_fails() -> None:
    sup, stale, ancient, fiscal = verify([doc("s.pdf", version="2023-08", superseded=True),
                                          doc("o.pdf", text=GOOD + "x1", version="2021"),
                                          doc("x.pdf", text=GOOD + "x2", version="2015"),
                                          doc("f.pdf", text=GOOD + "x3", version="2019-20")])
    assert by_name(sup)["recency"] == "warn" and sup.decision == "accept"
    assert by_name(stale)["recency"] == "warn" and stale.decision == "accept"
    assert by_name(ancient)["recency"] == "fail" and ancient.decision == "quarantine"
    assert by_name(fiscal)["recency"] == "warn"   # 2019-20 counts as 2020, 6 years old


def test_version_year_parsing() -> None:
    assert [version_year(v) for v in ("2026-06", "2019-20", "2026", "2026-27", "x")] == [2026, 2020, 2026, 2027, None]


def test_exact_duplicate_is_quarantined_and_the_first_copy_wins() -> None:
    first = doc("first.pdf")
    copy = doc("copy.pdf")
    copy.pages = [Page("copy.pdf", p.page, p.text.upper().replace("  ", " "), False) for p in first.pages]
    r1, r2 = verify([first, copy])
    assert r1.decision == "accept"
    assert r2.decision == "quarantine" and "exact duplicate of first.pdf" in r2.reasons[0]


def test_near_duplicate_by_embedding_fails_and_unrelated_passes() -> None:
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.999, 0.01, 0.0], "c": [0.0, 1.0, 0.0]}
    docs = [doc("a.pdf", text=GOOD + " A"), doc("b.pdf", text=GOOD + " B"), doc("c.pdf", text=GOOD + " C")]
    keys = ["a", "b", "c"]
    r = verify(docs, lambda texts: [vectors[k] for k in keys[:len(texts)]])
    assert [x.decision for x in r] == ["accept", "quarantine", "accept"]
    assert "near-duplicate of a.pdf" in r[1].reasons[0]


def test_an_embedding_failure_skips_the_near_duplicate_check_without_stopping() -> None:
    def broken(texts):
        raise RuntimeError("ollama down")

    (r,) = verify([doc()], broken)
    assert r.decision == "accept" and "embedding failed" in r.checks[-1].detail


def test_report_and_quarantine_folder(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    src.mkdir()
    (src / "bad.pdf").write_bytes(b"%PDF-1.4 bad")
    docs = [doc("ok.pdf"), doc("bad.pdf", version="unknown")]
    reports = verify(docs)
    jp, mp = write_report(reports, str(tmp_path / "out"), Thresholds(), ["manifest lists x.pdf but the file is missing"],
                          now=dt.datetime(2026, 11, 27, 9, 0, tzinfo=dt.timezone.utc))
    data = json.loads(jp.read_text())
    assert (data["documents"], data["accepted"], data["quarantined"]) == (2, 1, 1)
    assert "bad.pdf" in mp.read_text() and "quarantine" in mp.read_text()
    quarantine_files(reports, {"bad.pdf": src / "bad.pdf"}, str(tmp_path / "out"))
    assert (tmp_path / "out" / "quarantine" / "bad.pdf").exists()
    assert "completeness" in json.loads((tmp_path / "out" / "quarantine" / "bad.pdf.reason.json").read_text())["reasons"][0]


def test_manifest_problems_lists_missing_files(tmp_path: Path) -> None:
    (tmp_path / "_index.json").write_text(json.dumps([{"file": "here.pdf"}, {"file": "gone.pdf"}]))
    (tmp_path / "here.pdf").write_bytes(b"x")
    assert manifest_problems(str(tmp_path)) == ["manifest lists gone.pdf but the file is missing"]


def test_pipeline_does_not_chunk_or_embed_a_quarantined_document(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.md").write_text(f"---\ndoc_version: 2026\nsuperseded: false\n---\n# Rules\n\n{GOOD}\n")
    (corpus / "nover.md").write_text(f"# Other\n\n{GOOD} different words entirely here\n")   # no front matter: version unknown
    seen: dict[str, list[str]] = {}

    def fake_embed(texts, settings):
        seen.setdefault("calls", []).append(texts[0][:20])
        return [[1.0, float(i)] for i, _ in enumerate(texts)]

    def fake_store(chunks, vectors, settings):
        seen["sources"] = sorted({c.source for c in chunks})
        return len(chunks)

    monkeypatch.setattr("pipeline.embed.embed", fake_embed)
    monkeypatch.setattr("pipeline.store.store", fake_store)
    monkeypatch.setenv("VERIFY_DIR", str(tmp_path / "verification"))
    settings = IndexerSettings("m", "h", 1, "c", str(corpus), 400, 50, "http://x", str(tmp_path / "verification"))
    assert run_mod.run_pipeline(settings) > 0
    assert seen["sources"] == ["good.md"]
    assert (tmp_path / "verification" / "quarantine" / "nover.md").exists()
    assert list((tmp_path / "verification").glob("verification-*.md"))
