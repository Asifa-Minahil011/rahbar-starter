"""D5 fix: a citation that is only the clause number of a retrieved passage is accepted."""

from __future__ import annotations

from app.schemas import Citation, Passage
from app.services.grounding import split_citations

SRC = "academic-rules-handbook-jun-2026.pdf"


def passage(section: str, source: str = SRC) -> Passage:
    return Passage(chunk_id="c1", text="t", source=source, section=section, page=27, doc_version="2026-06",
                   superseded=False, score=0.8)


P = [passage("5.25 Rechecking fee will be refunded if there")]


def test_bare_clause_number_of_a_retrieved_passage_is_valid() -> None:
    valid, invalid = split_citations([Citation(source=SRC, section="5.25", page=27)], P)
    assert len(valid) == 1 and not invalid


def test_exact_header_still_valid_and_trailing_dot_ok() -> None:
    cites = [Citation(source=SRC, section="5.25 Rechecking fee will be refunded if there"),
             Citation(source=SRC, section="5.25.")]
    assert len(split_citations(cites, P)[0]) == 2


def test_other_clause_other_source_and_prefix_of_a_number_are_still_fabricated() -> None:
    cites = [Citation(source=SRC, section="5.26"), Citation(source=SRC, section="5.2"),
             Citation(source="other.pdf", section="5.25"), Citation(source=SRC, section="Rechecking")]
    valid, invalid = split_citations(cites, P)
    assert not valid and len(invalid) == 4
