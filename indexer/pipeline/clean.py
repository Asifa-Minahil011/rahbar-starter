"""Stage 2: Clean. Remove artefacts without destroying structure.

Rules (reported in docs/PIPELINE.md):
1. Drop lines that repeat in the first or last three lines of at least 40% of the pages
   (running headers, footers, "Page 3 of 53"). Page counters and bare leading or trailing numbers are ignored when comparing, so
   page numbers do not hide a repeat (clause numbers like 5.4 are kept). Only runs on documents of 4+ pages.
2. Drop lone page-number lines at the top or bottom of a page.
3. Re-join lines that the PDF wrapped mid-sentence, and undo "exam-\\nination" hyphenation.
4. Drop OCR junk lines (no letters or digits at all, or a single stray character).
5. Collapse runs of spaces; end each page with paragraphs separated by one blank line.

Deliberately NOT stripped: clause numbers ("1.21"), heading lines and table-row lines
("| Header: value |"). Exact-term questions and citations depend on them.
"""

from __future__ import annotations

import re
from collections import Counter

from .models import Document
from .structure import heading_text, starts_new_block

PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?$", re.I)
EDGE = 3
REPEAT_SHARE = 0.4


def _norm(line: str) -> str:
    """Key for repeat detection: page counters and a leading/trailing bare number are
    ignored, but digits inside clause numbers ("5.4") are kept, so two different
    headings never look like one repeated header."""
    t = re.sub(r"\s+", " ", line.strip().lower())
    t = re.sub(r"\bpage\s*\d+(\s*(of|/)\s*\d+)?\b", "", t)
    t = re.sub(r"^\d+\s+|\s*[|\-\u2013]?\s*\d+$", "", t)
    return t.strip()


def _edges(lines: list[str]) -> list[tuple[int, str]]:
    idx = [i for i, ln in enumerate(lines) if ln.strip()]
    return [(i, lines[i]) for i in idx[:EDGE] + idx[-EDGE:]]


def repeated_lines(page_lines: list[list[str]]) -> set[str]:
    n = len(page_lines)
    if n < 4:
        return set()
    counts: Counter[str] = Counter()
    for lines in page_lines:
        counts.update({_norm(text) for _, text in _edges(lines)})
    return {k for k, c in counts.items() if c >= max(3, REPEAT_SHARE * n) and len(k) > 2}


def rebuild_paragraphs(lines: list[str]) -> str:
    paragraphs: list[str] = []
    cur = ""
    for raw in lines:
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            if cur:
                paragraphs.append(cur)
                cur = ""
            continue
        if not cur:
            cur = line
        elif starts_new_block(line) or cur.startswith("|") or heading_text(cur):
            paragraphs.append(cur)
            cur = line
        elif cur.endswith("-") and line[:1].islower():
            cur = cur[:-1] + line          # exam-/ination
        else:
            cur = cur + " " + line
    if cur:
        paragraphs.append(cur)
    return "\n\n".join(paragraphs)


def clean(doc: Document) -> Document:
    """Strip repeating headers and footers; keep headings and section boundaries."""
    page_lines = [p.text.splitlines() for p in doc.pages]
    repeats = repeated_lines(page_lines)
    for page, lines in zip(doc.pages, page_lines):
        edge_idx = {i for i, _ in _edges(lines)}
        kept = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i in edge_idx and (_norm(line) in repeats or PAGE_NUMBER.match(stripped)):
                continue
            if stripped and not re.search(r"[A-Za-z0-9]", stripped):
                continue
            if page.ocr and len(stripped) == 1:
                continue
            kept.append(line)
        page.text = rebuild_paragraphs(kept)
    return doc
