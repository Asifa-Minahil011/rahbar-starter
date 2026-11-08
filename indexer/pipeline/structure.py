"""Shared rules for recognising headings, so clean and chunk agree on what a section is."""

from __future__ import annotations

import re

# "5.4 Academic Warning", "1.21 Attendance", "2. Academic warning". No trailing full stop:
# a numbered *sentence* ("1.1 A student must attend at least 80 percent ...") is a clause,
# not a heading.
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3}\.?)\s+([A-Z][^.!?;]{2,90})$")
KEYWORD_HEADING = re.compile(r"^(chapter|section|article|part|appendix)\s+[\dIVXA-Z]+\b.{0,80}$", re.I)
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
CLAUSE_START = re.compile(r"^\d+(?:\.\d+)+\.?\s")
BULLET = re.compile(r"^([-*\u2022\u25cf\u25aa]|\(?[a-z0-9ivx]{1,4}[).])\s+\S")


def heading_text(line: str) -> str | None:
    """Return the heading's text if `line` is a heading, else None."""
    s = line.strip()
    if not s or len(s) > 100:
        return None
    m = MARKDOWN_HEADING.match(s)
    if m:
        return m.group(1).strip()
    if NUMBERED_HEADING.match(s) or KEYWORD_HEADING.match(s):
        return s
    # ALL-CAPS line, at least two letters words long, no sentence punctuation
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 5 and s == s.upper() and not s.endswith((".", ",", ":")) and len(s.split()) <= 10:
        return s.title()
    return None


def starts_new_block(line: str) -> bool:
    """A line that must not be glued onto the previous line when rebuilding paragraphs."""
    s = line.strip()
    return bool(
        heading_text(s) or CLAUSE_START.match(s) or BULLET.match(s) or s.startswith("|")
    )


def short_label(heading: str, words: int = 8) -> str:
    """Section label stored in metadata and quoted in citations: clause number plus the
    first few words. Short enough for a model to copy exactly, long enough to be readable."""
    parts = heading.split()
    return " ".join(parts[:words])
