"""Stage 3: Chunk. Structure-aware, with overlap.

Unit: **words** (whitespace-separated), not model tokens. It is deterministic, needs no
tokenizer in the image, and is within ~1.3x of tokens for English. Say so in PIPELINE.md.

Algorithm:
1. Walk the pages; a heading line starts a new section and closes the current chunk, so a
   chunk never straddles two sections.
2. Inside a section, pack whole paragraphs until the next one would exceed `size` words.
3. A paragraph longer than `size` is split on sentence boundaries; a sentence longer than
   `size` is split on words (last resort).
4. Each new chunk starts with the last `overlap` words of the previous chunk of the same
   section, so a sentence cut at the boundary appears whole in one of them.
5. Every chunk text starts with its section heading, so the embedding "knows" where it is.
"""

from __future__ import annotations

import re

from .models import Chunk, Document
from .structure import heading_text

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
MIN_WORDS = 3


def _slug(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:60] or "doc"


def _split_long(text: str, size: int) -> list[str]:
    """Break one over-long paragraph into pieces of at most `size` words."""
    pieces: list[str] = []
    cur: list[str] = []
    count = 0
    for sentence in SENTENCE_END.split(text):
        words = sentence.split()
        if len(words) > size:                     # last resort: hard split on words
            if cur:
                pieces.append(" ".join(cur))
                cur, count = [], 0
            pieces.extend(" ".join(words[i:i + size]) for i in range(0, len(words), size))
            continue
        if count + len(words) > size and cur:
            pieces.append(" ".join(cur))
            cur, count = [], 0
        cur.append(sentence)
        count += len(words)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _pack(units: list[tuple[str, int]], size: int, overlap: int) -> list[tuple[str, int]]:
    """Greedy packing of (text, page) units into (text, first_page) chunks. A chunk is at
    most `size` words including the overlap carried in from the previous chunk."""
    chunks: list[tuple[str, int]] = []
    body: list[str] = []
    words = 0
    fresh = False          # does `body` hold anything besides carried-over overlap?
    page = units[0][1] if units else 1
    for text, unit_page in units:
        n = len(text.split())
        if fresh and words + n > size:
            joined = "\n\n".join(body)
            chunks.append((joined, page))
            tail = " ".join(joined.split()[-overlap:]) if overlap > 0 else ""
            body = [tail] if tail else []
            words = len(tail.split())
            fresh = False
        if not fresh:
            page = unit_page
        body.append(text)
        words += n
        fresh = True
    if fresh:
        chunks.append(("\n\n".join(body), page))
    return chunks


def chunk(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split on sections, then paragraphs, then sentences, keeping each chunk under
    `size` words, with `overlap` words shared between neighbours. Every chunk keeps the
    section heading it sits under, its page, doc_version and superseded."""
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    sections: list[tuple[str, list[tuple[str, int]]]] = []   # (heading, [(paragraph, page)])
    current: list[tuple[str, int]] = []
    heading = ""
    seen_heading = False
    for page in doc.pages:
        for para in (p.strip() for p in page.text.split("\n\n")):
            if not para:
                continue
            h = heading_text(para) if "\n" not in para else None
            if h:
                if current:
                    sections.append((heading, current))
                heading, current, seen_heading = h, [], True
                continue
            if not seen_heading and heading != f"Page {page.page}":
                # no heading yet (scans, flat text): one section per page, so citations
                # still point somewhere
                if current:
                    sections.append((heading, current))
                heading, current = f"Page {page.page}", []
            for piece in (_split_long(para, size - overlap)
                          if len(para.split()) > size - overlap else [para]):
                current.append((piece, page.page))
    if current:
        sections.append((heading, current))

    stem = _slug(doc.source)
    out: list[Chunk] = []
    for heading, units in sections:
        for text, first_page in _pack(units, size, overlap):
            if len(text.split()) < MIN_WORDS:
                continue
            out.append(Chunk(
                id=f"{stem}-{len(out):04d}",
                text=f"{heading}\n{text}" if heading else text,
                source=doc.source, section=heading or "(no heading)", page=first_page,
                doc_version=doc.doc_version, superseded=doc.superseded,
            ))
    return out
