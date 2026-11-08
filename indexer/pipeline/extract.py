"""Stage 1: Extract. PDF (or HTML, or Markdown) to text, page by page.

Design notes (reported in docs/PIPELINE.md):
- PDFs: pdfplumber for the text layer. A page whose text layer is (almost) empty is
  rasterised and OCR'd with Tesseract; those pages carry ocr=True.
- Tables are cut out of the running text and rendered one row per line with the column
  header repeated in every cell ("| Campus: Lahore | Faculty: 120 |"). A raw text dump
  of a table puts the headers far from the numbers, so an embedding cannot connect them.
- HTML (fee and grading pages) goes through the standard-library parser: headings become
  Markdown headings, table rows are rendered like PDF tables.
"""

from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .models import Document, Page

MIN_TEXT_CHARS = 25  # a page with fewer characters than this has no usable text layer
OCR_DPI = 300


def _log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer.extract", "msg": msg, **fields}), flush=True)


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def render_table(rows: list[list[Any]]) -> list[str]:
    """One line per data row, header repeated in every cell. A table with no usable
    header (or one row) falls back to plain pipe-separated cells."""
    grid = [[_clean_cell(c) for c in r] for r in rows if any(_clean_cell(c) for c in r)]
    if not grid:
        return []
    header = grid[0]
    usable_header = len(grid) > 1 and all(header) and len(set(header)) == len(header)
    lines = []
    for row in grid[1:] if usable_header else grid:
        if usable_header:
            cells = [f"{h}: {v}" for h, v in zip(header, row) if v]
        else:
            cells = [v for v in row if v]
        if cells:
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def _inside(obj: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _text_layer(page: Any) -> str:
    """Text of one pdfplumber page, with tables pulled out and re-rendered."""
    try:
        tables = page.find_tables()
    except Exception:  # pdfplumber can choke on odd drawings; fall back to plain text
        tables = []
    if not tables:
        return str(page.extract_text() or "")
    boxes = [t.bbox for t in tables]
    outside = page.filter(lambda o: o.get("object_type") != "char" or
                          not any(_inside(o, b) for b in boxes))
    text = str(outside.extract_text() or "")
    table_lines: list[str] = []
    for t in tables:
        table_lines.extend(render_table(t.extract()))
    return (text + "\n\n" + "\n".join(table_lines)).strip()


def _ocr_page(path: str, number: int) -> str:
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(path, dpi=OCR_DPI, first_page=number, last_page=number)
    return "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images)


def _extract_pdf(path: str, source: str) -> list[Page]:
    import pdfplumber

    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = _text_layer(page)
            ocr = False
            if len(text.strip()) < MIN_TEXT_CHARS:
                try:
                    text, ocr = _ocr_page(path, number), True
                except Exception as exc:  # missing binary, corrupt image: lose the page, say so
                    _log("ocr failed; page left empty", source=source, page=number,
                         error=repr(exc))
                    text = ""
            pages.append(Page(source=source, page=number, text=text, ocr=ocr))
    return pages


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "form", "svg"}
    BLOCK = {"p", "div", "li", "br", "section", "article", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.heading: int | None = None
        self.rows: list[list[str]] | None = None
        self.cell: list[str] | None = None
        self.row: list[str] | None = None

    def _emit(self, s: str) -> None:
        if self.cell is not None:
            self.cell.append(s)
        else:
            self.out.append(s)

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "table":
            self.rows = []
        elif tag == "tr" and self.rows is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif re.fullmatch(r"h[1-6]", tag):
            self.heading = int(tag[1])
            self.out.append("\n\n" + "#" * self.heading + " ")
        elif tag in self.BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.rows is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.rows is not None:
            self.out.append("\n\n" + "\n".join(render_table(self.rows)) + "\n\n")
            self.rows = None
        elif re.fullmatch(r"h[1-6]", tag):
            self.heading = None
            self.out.append("\n\n")
        elif tag in self.BLOCK:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self._emit(data if self.heading else " ".join(data.split()) + " ")


def _extract_html(text: str) -> str:
    parser = _HTMLText()
    parser.feed(text)
    return re.sub(r"\n{3,}", "\n\n", "".join(parser.out)).strip()


def _strip_front_matter(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1:])
    return text


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    """Return the document's text page by page.

    PDF pages with no text layer are OCR'd and marked ocr=True. Markdown and HTML have no
    pages, so they come back as page 1.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf(str(p), p.name)
    elif suffix in (".html", ".htm"):
        pages = [Page(p.name, 1, _extract_html(p.read_text(encoding="utf-8", errors="replace")))]
    elif suffix in (".md", ".txt"):
        pages = [Page(p.name, 1, _strip_front_matter(p.read_text(encoding="utf-8")))]
    else:
        raise ValueError(f"unsupported file type: {p.name}")
    ocr_pages = sum(1 for pg in pages if pg.ocr)
    _log("extracted", source=p.name, pages=len(pages), ocr_pages=ocr_pages,
         chars=sum(len(pg.text) for pg in pages))
    return Document(source=p.name, doc_version=doc_version, superseded=superseded, pages=pages)
