from __future__ import annotations

import re

from pydantic import ValidationError

from ...schemas import Answer
from .errors import GenerationError

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_answer(raw: str) -> Answer:
    """Structure can be guaranteed; truth cannot. Parse and validate; never trust."""
    text = FENCE.sub("", (raw or "").strip())
    try:
        return Answer.model_validate_json(text)
    except ValidationError as exc:
        raise GenerationError(f"model output failed the Answer schema: {exc.error_count()} "
                              f"error(s); first: {exc.errors()[0]['msg']}") from exc
