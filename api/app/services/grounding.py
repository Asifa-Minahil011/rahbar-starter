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


CLAUSE_NUMBER = re.compile(r"^\d+(\.\d+)*\.?$")


def _same_section(cited: str, actual: str) -> bool:
    """Equal after whitespace/case folding, or `cited` is only the clause number of `actual`
    ("5.25" for "5.25 Rechecking fee will be refunded if there"). Section headers in this corpus
    are the first words of a clause, and a 3B model often cites just the number. This is not
    leniency toward invented sources: the clause must still belong to a passage retrieved for
    THIS request, from the same document."""
    c, a = _norm(cited), _norm(actual)
    if c == a:
        return True
    return bool(CLAUSE_NUMBER.match(c)) and a.split(" ", 1)[0].rstrip(".") == c.rstrip(".")


def matches_retrieved(citation: Citation, passages: list[Passage]) -> bool:
    """A citation is real only if its source and section equal a passage retrieved for
    THIS request (a bare clause number of that passage's section counts). Anything else is
    fabricated."""
    return any(_norm(citation.source) == _norm(p.source)
               and _same_section(citation.section, p.section) for p in passages)


def split_citations(citations: list[Citation], passages: list[Passage]
                    ) -> tuple[list[Citation], list[Citation]]:
    valid: list[Citation] = []
    invalid: list[Citation] = []
    for c in citations:
        (valid if matches_retrieved(c, passages) else invalid).append(c)
    return valid, invalid
