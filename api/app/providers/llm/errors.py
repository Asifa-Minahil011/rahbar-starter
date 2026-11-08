from __future__ import annotations


class GenerationError(Exception):
    """The model call failed or its output did not satisfy the Answer schema."""
