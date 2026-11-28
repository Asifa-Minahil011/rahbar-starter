"""RAGOps A1: the verify stage. No model, no network: embeddings are a stub."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import dataclasses
import uuid

from pipeline import run as run_mod
from pipeline.config import get_settings
from pipeline.models import Document, Page
from pipeline.verify import (
    Thresholds,
    manifest_problems,
    quarantine_files,
    verify_documents,
    version_year,
    write_report,
)
from tests.test_incremental import FakeClient

TODAY = dt.date(2026, 11, 27)
GOOD = "The Head of Department may condone absence for documented medical reasons. " * 12


def doc(name: str = "a.pdf", text: str = GOOD, version: str = "2026-06", superseded: bool = False,
        pages: int = 3, ocr: bool = False) -> Document:
    return Document(name, version, superseded, [Page(name, i + 1, f"{text} page {i}.", ocr) for i in range(pages)])


def by_name(report) -> dict[str, str]:
    return {c.name: c.status for c in report.checks}


def verify(docs, embed_fn=None):
    return verify_documents(docs, Thresholds(), embed_fn, TODAY)


def test_a_good_document_passes_all_four_checks() -> None:
    (r,) = verify([doc()])
    assert r.action == "indexed"
    assert set(by_name(r).values()) <= {"pass"}


def test_quality_fails_on_empty_pages_and_on_garbled_text() -> None:
    empty = Document("e.pdf", "2026", False, [Page("e.pdf", i, "", False) for i in range(1, 5)])
    garbled = doc("g.pdf", text="ab\ufffd\ufffd\ufffd " * 80)
    r_empty, r_garbled = verify([empty, garbled])
    assert by_name(r_empty)["quality"] == "fail" and r_empty.action == "quarantined"
    assert by_name(r_garbled)["quality"] == "fail"


def test_a_clean_scan_is_kept_but_a_thin_scan_is_quarantined() -> None:
    (clean_scan,) = verify([doc(ocr=True)])
    assert by_name(clean_scan)["quality"] == "warn" and clean_scan.action == "indexed"
    thin = Document("t.pdf", "2026", False, [Page("t.pdf", i, "word " * 8, True) for i in range(1, 5)])
    (r_thin,) = verify([thin])
    assert by_name(r_thin)["quality"] == "fail"


def test_completeness_needs_a_version_a_title_and_enough_text() -> None:
    no_version, tiny = verify([doc("n.pdf", version="unknown"), doc("t.pdf", text="hi", pages=1)])
    assert by_name(no_version)["completeness"] == "fail"
    assert by_name(tiny)["completeness"] == "fail"


def test_a_truncated_markdown_file_fails_completeness() -> None:
    cut = Document("c.md", "2026", False, [Page("c.md", 1, "# Rules\n\n" + GOOD + " and the student must then subm", False)])
    (r,) = verify_documents([cut], Thresholds(), None, TODAY, paths={"c.md": Path("c.md")})
    assert by_name(r)["completeness"] == "fail" and "mid-sentence" in r.reasons[0]


def test_recency_a_newer_version_flags_the_old_one_without_quarantining_it() -> None:
    old = doc("rules-2024.pdf", text=GOOD + " old", version="2024-08")
    new = doc("rules-2026.pdf", text=GOOD + " new", version="2026-06")
    r_old, r_new = verify_documents([old, new], Thresholds(), None, TODAY,
                                    doc_ids={"rules-2024.pdf": "rules", "rules-2026.pdf": "rules"})
    assert by_name(r_old)["recency"] == "fail" and r_old.action == "indexed"   # archived later, never deleted
    assert by_name(r_new)["recency"] == "pass"


def test_recency_superseded_and_stale_documents_warn() -> None:
    sup, stale = verify([doc("s.pdf", version="2023-08", superseded=True), doc("o.pdf", text=GOOD + "x1", version="2021")])
    assert by_name(sup)["recency"] == "warn" and sup.action == "indexed"
    assert by_name(stale)["recency"] == "warn"


def test_version_year_parsing() -> None:
    assert [version_year(v) for v in ("2026-06", "2019-20", "2026", "2026-27", "x")] == [2026, 2020, 2026, 2027, None]


def test_exact_duplicate_is_quarantined_and_the_first_copy_wins() -> None:
    first = doc("first.pdf")
    copy = doc("copy.pdf")
    copy.pages = [Page("copy.pdf", p.page, p.text.upper().replace("  ", " "), False) for p in first.pages]
    r1, r2 = verify([first, copy])
    assert r1.action == "indexed"
    assert r2.action == "quarantined" and "exact duplicate of first.pdf" in r2.reasons[0]


def test_near_duplicate_by_embedding_fails_and_unrelated_passes() -> None:
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.999, 0.01, 0.0], "c": [0.0, 1.0, 0.0]}
    docs = [doc("a.pdf", text=GOOD + " A"), doc("b.pdf", text=GOOD + " B"), doc("c.pdf", text=GOOD + " C")]
    keys = ["a", "b", "c"]
    r = verify(docs, lambda texts: [vectors[k] for k in keys[:len(texts)]])
    assert [x.action for x in r] == ["indexed", "quarantined", "indexed"]
    assert "near-duplicate of a.pdf" in r[1].reasons[0]


def test_an_embedding_failure_skips_the_near_duplicate_check_without_stopping() -> None:
    def broken(texts):
        raise RuntimeError("ollama down")

    (r,) = verify([doc()], broken)
    assert r.action == "indexed" and "embedding failed" in r.checks[-1].detail


def test_report_and_quarantine_folder(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    src.mkdir()
    (src / "bad.pdf").write_bytes(b"%PDF-1.4 bad")
    reports = verify([doc("ok.pdf"), doc("bad.pdf", version="unknown")])
    jp, mp = write_report(reports, tmp_path / "out", Thresholds(), ["manifest lists x.pdf but the file is missing"],
                          now=dt.datetime(2026, 11, 27, 9, 0, tzinfo=dt.timezone.utc))
    data = json.loads(jp.read_text())
    assert jp.parent == tmp_path / "out" / "reports"
    assert (data["documents"], data["indexed"], data["quarantined"]) == (2, 1, 1)
    bad = next(r for r in data["reports"] if r["source"] == "bad.pdf")
    assert bad["checks"]["completeness"]["result"] == "fail" and bad["action"] == "quarantined"
    assert "bad.pdf" in mp.read_text()
    quarantine_files(reports, {"bad.pdf": src / "bad.pdf"}, tmp_path / "out")
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
    (corpus / "nover.md").write_text(f"# Other\n\n{GOOD} different words entirely here\n")   # no version
    (corpus / "broken.pdf").write_bytes(b"not a pdf at all")                                   # extraction fails
    embedded: list[str] = []

    def fake_embed(texts, settings):
        embedded.extend(texts)
        return [[1.0, float(i % 7)] for i, _ in enumerate(texts)]

    stored: list[str] = []

    def fake_store(chunks, vectors, settings, client=None):
        stored.extend(c.source for c in chunks)
        return len(chunks)

    monkeypatch.setattr("pipeline.embed.embed", fake_embed)
    monkeypatch.setattr("pipeline.store.store", fake_store)
    monkeypatch.setattr("pipeline.store.get_client", lambda settings: FakeClient())
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    settings = dataclasses.replace(get_settings(), corpus_dir=str(corpus), collection=f"t{uuid.uuid4().hex[:8]}")
    assert run_mod.run_pipeline(settings) > 0
    assert set(stored) == {"good.md"}
    q = tmp_path / "data" / "quarantine"
    assert (q / "nover.md").exists() and (q / "broken.pdf").exists()
    (report,) = (tmp_path / "data" / "reports").glob("verification-*.json")
    by_src = {r["source"]: r for r in json.loads(report.read_text())["reports"]}
    assert by_src["broken.pdf"]["checks"]["quality"]["result"] == "fail"
    assert by_src["good.md"]["action"] == "indexed"
