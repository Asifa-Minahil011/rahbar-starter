"""Retrieval rail against prompt injection (Assignment 3, D4; Assignment 2, section 3.4).

Two layers already existed: the prompt tells the model that passages are data, and
`prompt._neutralise` stops a passage from closing its own <context> tag. This module adds
a third, visible one: it scans every retrieved passage (and the question) for instruction
phrases and returns the actions it took, so they can be written into the trace.

INJECTION_RAIL=drop  flagged passages are kept out of the prompt (they still count as
                     "retrieved" for evaluation)
INJECTION_RAIL=flag  flagged passages stay in the prompt; the flag is only logged
INJECTION_RAIL=off   the rail does nothing
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import Passage

# Specific phrases only: a handbook says "must" and "shall" all the time, so generic
# imperative words would flag normal rules.
INJECTION = re.compile(
    r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+|your\s+|my\s+)*"
    r"(?:previous|prior|above|earlier|preceding|system)\s+"
    r"(?:instructions?|rules?|prompts?|context|messages?)"
    r"|(?:new|updated|real)\s+instructions?\s*:"
    r"|(?:reveal|print|show|repeat|leak)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)"
    r"|do\s+not\s+follow\s+(?:the|your|any)\s+(?:previous|above|system)",
    re.I,
)
DELIMITER = re.compile(r"</?\s*context\b", re.I)


@dataclass
class RailResult:
    usable: list[Passage]
    actions: list[dict[str, object]] = field(default_factory=list)

    @property
    def flagged(self) -> int:
        return sum(1 for a in self.actions if a["action"] in ("flagged", "dropped"))


def scan_question(question: str) -> list[dict[str, object]]:
    if INJECTION.search(question):
        return [{"rail": "injection_phrase", "where": "question", "action": "flagged",
                 "detail": "instruction phrase in the user's question; answered normally, "
                           "the prompt rules and citation check still apply"}]
    return []


def apply_rail(passages: list[Passage], mode: str) -> RailResult:
    """Return the passages the prompt may use, plus what the rail did, for the trace."""
    result = RailResult(usable=list(passages))
    # Delimiter neutralisation always runs in prompt.format_context; record that it will fire.
    for p in passages:
        escapes = len(DELIMITER.findall(p.text))
        if escapes:
            result.actions.append({"rail": "delimiter_escape", "where": p.chunk_id,
                                   "action": "neutralised",
                                   "detail": f"{escapes} <context> tag(s) rewritten in the passage text"})
    if mode == "off":
        return result
    kept: list[Passage] = []
    for p in passages:
        hit = INJECTION.search(p.text)
        if hit is None:
            kept.append(p)
            continue
        dropped = mode == "drop"
        result.actions.append({"rail": "injection_phrase", "where": p.chunk_id,
                               "source": p.source, "section": p.section,
                               "action": "dropped" if dropped else "flagged",
                               "detail": f"matched: {hit.group(0)[:80]!r}"})
        if not dropped:
            kept.append(p)
    result.usable = kept
    return result
