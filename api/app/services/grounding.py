"""Grounding checks that run after the model answers (section 3.4)."""

from __future__ import annotations

import re

from ..schemas import Citation, Passage

ABSTAIN_TEXT = ("I could not find this in the documents I have access to, so I can't answer "
                "it reliably. Try rephrasing, or check with the relevant university office.")
FALLBACK_TEXT = ("Sorry, I could not produce a reliable answer right now. "
                 "Please try again in a moment.")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def matches_retrieved(citation: Citation, passages: list[Passage]) -> bool:
    """A citation is real only if its source and section equal a passage retrieved for
    THIS request. Anything else is fabricated."""
    return any(_norm(citation.source) == _norm(p.source)
               and _norm(citation.section) == _norm(p.section) for p in passages)


def split_citations(citations: list[Citation], passages: list[Passage]
                    ) -> tuple[list[Citation], list[Citation]]:
    valid: list[Citation] = []
    invalid: list[Citation] = []
    for c in citations:
        (valid if matches_retrieved(c, passages) else invalid).append(c)
    return valid, invalid
