"""Pipeline tests: no Chroma server, no model, no network. Tesseract is only needed for
the OCR test, which skips itself when the binary is missing."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pipeline.chunk import chunk
from pipeline.clean import clean, repeated_lines
from pipeline.config import get_settings
from pipeline.extract import _extract_html, extract, render_table
from pipeline.models import Document, Page
from pipeline.store import store

FIXTURES = Path(__file__).resolve().parents[2] / "eval" / "fixtures"


def _doc(*texts: str) -> Document:
    return Document("d.pdf", "2026", False,
                    [Page("d.pdf", i + 1, t) for i, t in enumerate(texts)])


def test_markdown_front_matter_is_not_indexed() -> None:
    doc = extract(str(FIXTURES / "academic-rules-2026.md"), "2026", False)
    assert "doc_version" not in doc.pages[0].text


def test_chunks_follow_sections_and_never_cross_them() -> None:
    doc = clean(extract(str(FIXTURES / "academic-rules-2026.md"), "2026", False))
    chunks = chunk(doc, 400, 50)
    sections = {c.section for c in chunks}
    assert {"1. Attendance", "2. Academic warning", "3. Repeating a course"} <= sections
    warn = next(c for c in chunks if c.section == "2. Academic warning")
    assert "2.00" in warn.text and "Attendance" not in warn.text
    assert warn.doc_version == "2026" and warn.superseded is False


def test_overlap_repeats_the_tail_and_size_is_respected() -> None:
    sentences = " ".join(f"Sentence number {i} says something about rule {i}." for i in range(60))
    doc = _doc("# 1. Rules\n\n" + sentences)
    chunks = chunk(doc, 60, 15)
    assert len(chunks) > 3
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text.split()[-15:]
        assert tail == b.text.split()[1 + 1:1 + 1 + 15] or " ".join(tail) in b.text
    # heading line ("1. Rules" = 2 words) is on top of the 60-word body budget
    assert max(len(c.text.split()) for c in chunks) <= 60 + 2


def test_ids_are_stable_and_unique() -> None:
    doc = clean(extract(str(FIXTURES / "fees-2026.md"), "2026", False))
    a, b = chunk(doc, 50, 10), chunk(doc, 50, 10)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk(_doc("# A\n\nsome text here"), 10, 10)


def test_scan_without_headings_gets_page_sections() -> None:
    doc = _doc("plain words on page one with no heading at all", "page two also has no heading")
    sections = [c.section for c in chunk(doc, 100, 10)]
    assert sections == ["Page 1", "Page 2"]


def test_clean_strips_running_header_footer_and_keeps_clauses() -> None:
    pages = [f"FAST-NUCES Academic Rules\n5.{n} Rule heading {n}\n"
             f"5.{n}.1 The body of clause {n} says a thing.\nPage {n} of 6" for n in range(1, 7)]
    doc = clean(_doc(*pages))
    text = "\n".join(p.text for p in doc.pages)
    assert "FAST-NUCES Academic Rules" not in text
    assert "of 6" not in text
    assert "5.3.1 The body of clause 3 says a thing." in text
    assert "5.3 Rule heading 3" in text


def test_clean_joins_wrapped_lines_and_dehyphenates() -> None:
    doc = clean(_doc("A student must complete the exam-\nination within the\nstated period."))
    assert doc.pages[0].text == "A student must complete the examination within the stated period."


def test_repeat_detection_needs_four_pages() -> None:
    assert repeated_lines([["Header", "x"]] * 3) == set()


def test_table_rows_keep_their_headers() -> None:
    rows = [["Campus", "Faculty"], ["Lahore", "120"], ["Karachi", "95"]]
    assert render_table(rows) == ["| Campus: Lahore | Faculty: 120 |",
                                  "| Campus: Karachi | Faculty: 95 |"]


def test_html_table_and_headings() -> None:
    html = ("<nav>menu</nav><h2>Tuition</h2><p>Fees per credit hour.</p>"
            "<table><tr><th>Program</th><th>Fee</th></tr>"
            "<tr><td>BS</td><td>10,000</td></tr></table>")
    text = _extract_html(html)
    assert "## Tuition" in text and "menu" not in text
    assert "| Program: BS | Fee: 10,000 |" in text


def test_extract_pdf_text_table_and_ocr(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    if not shutil.which("tesseract"):
        pytest.skip("tesseract not installed")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from PIL import Image, ImageDraw, ImageFont

    pdf = tmp_path / "t.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(72, 750, "5.4 Academic Warning")
    c.drawString(72, 730, "A student below 2.00 CGPA is placed on warning.")
    c.showPage()
    img = Image.new("RGB", (1240, 400), "white")           # a page that is only a picture
    ImageDraw.Draw(img).text((40, 40), "Scanned policy clause 7.3 refund", fill="black",
                             font=ImageFont.load_default(size=40))
    c.drawImage(ImageReader(img), 72, 400, width=450, height=145)
    c.showPage()
    c.save()
    doc = extract(str(pdf), "2026", False)
    assert [p.ocr for p in doc.pages] == [False, True]
    assert "2.00" in doc.pages[0].text
    assert "refund" in doc.pages[1].text.lower()


class FakeCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name, self.metadata, self.rows = name, metadata, {}

    def add(self, ids, documents, embeddings, metadatas) -> None:  # type: ignore[no-untyped-def]
        for i, d, e, m in zip(ids, documents, embeddings, metadatas):
            self.rows[i] = (d, e, m)

    def count(self) -> int:
        return len(self.rows)

    def modify(self, name: str) -> None:
        self.name = name
        self.client.collections[name] = self      # type: ignore[attr-defined]
        self.client.collections.pop(self.old, None)  # type: ignore[attr-defined]


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def delete_collection(self, name: str) -> None:
        if name not in self.collections:
            raise ValueError("no such collection")
        del self.collections[name]

    def create_collection(self, name: str, metadata: dict) -> FakeCollection:
        col = FakeCollection(name, metadata)
        col.client, col.old = self, name          # type: ignore[attr-defined]
        self.collections[name] = col
        return col


def test_store_swaps_in_a_new_index_and_records_the_pin() -> None:
    settings = get_settings()
    doc = clean(extract(str(FIXTURES / "academic-rules-2026.md"), "2026", False))
    chunks = chunk(doc, 400, 50)
    client = FakeClient()
    n1 = store(chunks, [[0.1, 0.2, 0.3]] * len(chunks), settings, client)
    first = client.collections["rahbar"].metadata
    n2 = store(chunks, [[0.1, 0.2, 0.3]] * len(chunks), settings, client)
    second = client.collections["rahbar"].metadata
    assert n1 == n2 == len(chunks)
    assert set(client.collections) == {"rahbar"}            # scratch collection is gone
    assert first["embed_model"] == settings.embed_model and first["dimension"] == 3
    assert first["index_id"] != second["index_id"]          # the cache key changes per build
    meta = next(iter(client.collections["rahbar"].rows.values()))[2]
    assert set(meta) == {"source", "section", "page", "doc_version", "superseded"}


def test_store_refuses_mismatched_vectors() -> None:
    with pytest.raises(ValueError):
        store([], [], get_settings(), FakeClient())


def test_clause_line_is_labelled_by_number_and_its_text_is_kept() -> None:
    clause = "1.21 Students are required to maintain 100% attendance in lectures"
    doc = clean(_doc("1. General\n\n" + clause + "\n\n1.22 Failure to meet attendance requirements "
                     "in a course will result in an F grade"))
    chunks = chunk(doc, 400, 50)
    c121 = next(c for c in chunks if c.section.startswith("1.21"))
    assert c121.section == "1.21 Students are required to maintain 100% attendance"
    assert "maintain 100% attendance in lectures" in c121.text      # nothing dropped
    assert any("F grade" in c.text for c in chunks)


def test_table_of_contents_dot_leaders_are_removed() -> None:
    doc = clean(_doc("Table of Contents\nAttendance...........12\nI N T R O D U C T I O N ccsse....cccc\n"
                     "Real sentence stays here."))
    assert "......" not in doc.pages[0].text and "Real sentence stays here." in doc.pages[0].text


def test_footer_rendered_as_a_table_row_is_stripped() -> None:
    pages = [f"{n}.1 Real clause {n} text here.\n| {n + 20} | Undergraduate Academic Rules and Regulations 2026 |"
             for n in range(1, 7)]
    text = "\n".join(p.text for p in clean(_doc(*pages)).pages)
    assert "Undergraduate Academic Rules" not in text and "3.1 Real clause 3 text here." in text
