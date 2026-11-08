#!/usr/bin/env python3
"""Fill the Extraction section of docs/PIPELINE.md and the table row of docs/CORPUS.md from facts
measured on the real corpus and index. The table 'before' is pdfplumber's plain text of the page,
the 'after' is the chunk the pipeline stored; both are read live, not typed by hand.

    python scripts/fill_docs.py            # edits the two docs in place; run `git diff docs` to review
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "corpus" / "raw" / "academic-rules-handbook-jun-2026.pdf"


def raw_page_lines(pdf_path: Path, needles: tuple[str, ...] = ("A+", "4.00", "C+", "2.33")) -> tuple[int, list[str]]:
    """First page whose plain text holds the grade-point table, and that table's lines."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if all(n in text for n in needles):
                lines = text.splitlines()
                start = next((i for i, ln in enumerate(lines) if "A+" in ln and "4.00" in ln), 0)
                return number, lines[max(start - 1, 0): start + 5]
    raise LookupError("grade-point table not found in " + str(pdf_path))


def stored_table_chunk() -> tuple[str, str, str]:
    """(section, page, text) of the stored chunk that holds the grade-point table."""
    import chromadb

    col = chromadb.HttpClient(host="localhost", port=8001).get_collection("rahbar")
    got = col.get(where={"source": "academic-rules-handbook-jun-2026.pdf"}, include=["metadatas", "documents"])
    for meta, doc in zip(got["metadatas"], got["documents"]):
        if "| Grade | Points" in doc and "4.00" in doc:
            return str(meta["section"]), str(meta["page"]), " ".join(doc.split())
    raise LookupError("no stored chunk holds the grade-point table")


def extraction_section(page: int, before: list[str], section: str, stored_page: str, after: str) -> str:
    before_txt = "\n".join("    " + ln for ln in before)
    return f"""## Extraction
`indexer/pipeline/extract.py`. PDFs go through pdfplumber page by page. A page with fewer than 25 characters of
text is rasterised at 300 dpi and OCR'd with Tesseract; those pages carry `ocr=True` and are counted in the
`extracted` log line.

OCR'd pages (indexing run log): start-ups policy 8 of 8 pages (a pure scan, `pdffonts` lists no fonts), 2023 rules
24 of 26, rules handbook 2026 5 of 58, student handbook 4 of 64.

Tables are cut out of the running text and re-rendered one line per row, with the column header repeated in every
cell, because a plain dump puts the headings far from the numbers and an embedding cannot connect them. A real
stored chunk from `fee-structure-page.html` shows the result: `| Program: BBA / BS | Fee: Rs. 12,000 |`, and the
refund table is stored as `| Timeline: Upto 10th day of commencement of classes | Percentage of Fee: 100% ...`.

Before and after for a table that did not work well: the grade-point table of the 2026 rules handbook (PDF page
{page}). Plain `pdfplumber` text of the page:

{before_txt}

What we stored (chunk of section "{section}", page {stored_page}):

    {after[:600]}

Two problems are visible. The table has two column pairs with the same header names (Grade, Points, Grade, Points),
so the header cannot be repeated per cell and the renderer falls back to plain pipe-separated cells. And tables are
appended at the end of their page's text, so this table is stored under the last section on the page ("{section}")
and not under clause 2.3 where it belongs.

What could not be recovered, found by inspecting the real index:
- 43 of 769 chunks still carry a page-footer fragment (for example `| 22 | Undergraduate Academic Rules and
  Regulations 2026 |`), because the footer is rendered as a table row and its page number changes.
- The first chapter of the 2026 handbook has a duplicated text layer: heading lines repeat up to four times in
  one chunk (clauses 1.1, 1.17, 1.18).
- The handbook's two-column contents page leaks into clauses: clause 1.6's chunk contains `Use in Research 45
  ... Detecting Generative AI Use in Assessments 45`.
- The 2023 rules are 24 of 26 pages of OCR text. Its table of contents came out as garbage (`IMtrOdUCtION....`), so
  dot-leader lines are removed; OCR also misreads some digits, so any 2023 value used in a golden question was
  checked in the PDF.
- Text inside images and charts is not extracted.

"""


def main() -> int:
    docs = ROOT / "docs"
    try:
        page, before = raw_page_lines(PDF)
        section, stored_page, after = stored_table_chunk()
    except Exception as exc:
        print(f"could not read the real example ({exc!r}); docs not changed.", file=sys.stderr)
        print("Needs corpus/raw/ and Chroma on localhost:8001 with the final index.", file=sys.stderr)
        return 1
    p = docs / "PIPELINE.md"
    text = p.read_text(encoding="utf-8")
    new_section = extraction_section(page, before, section, stored_page, after)
    text, n = re.subn(r"## Extraction\n.*?(?=## Cleaning)", lambda _: new_section, text, count=1, flags=re.S)
    if n != 1:
        print("could not find the Extraction section in PIPELINE.md; nothing written", file=sys.stderr)
        return 1
    p.write_text(text, encoding="utf-8")

    c = docs / "CORPUS.md"
    corpus = c.read_text(encoding="utf-8")
    corpus = re.sub(r" VERIFY: confirm with `find_clause.py \"Campus\"` that rows survive\.", "", corpus)
    corpus = corpus.replace(
        "see PIPELINE.md for a before and after.",
        "see PIPELINE.md for a before and after. Stored rows, for example: "
        "`| Program: BBA / BS | Fee: Rs. 12,000 |` (fee page) and `| Timeline: Upto 10th day of commencement of "
        "classes | Percentage of Fee: 100% ...` (refund table).")
    c.write_text(corpus, encoding="utf-8")
    print("PIPELINE.md Extraction section and CORPUS.md table row updated. Review with: git diff docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
