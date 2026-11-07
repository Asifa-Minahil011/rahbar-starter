"""Ollama provider: ollama-host or ollama-incluster (section 3.5).

Enforcement: the Answer JSON schema goes in `format`, so decoding is constrained to the
schema. We still validate with Pydantic afterwards (parse_answer).
"""

from __future__ import annotations

import logging
import time

import httpx

from ...config import Settings
from ...schemas import Answer, Passage
from ...services import telemetry
from .errors import GenerationError
from .parse import parse_answer
from .prompt import build_messages

log = logging.getLogger("rahbar.provider.ollama")


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings   # ollama_url, llm_model, temperature

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        payload = {
            "model": self.settings.llm_model,
            "messages": build_messages(question, passages),
            "stream": False,
            "format": Answer.model_json_schema(),
            "options": {"temperature": self.settings.temperature},
        }
        started = time.perf_counter()
        try:
            r = httpx.post(f"{self.settings.ollama_url.rstrip('/')}/api/chat",
                           json=payload, timeout=180.0)
            r.raise_for_status()
            body = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"ollama call failed: {exc!r}") from exc
        finally:
            telemetry.GENERATION_SECONDS.observe(time.perf_counter() - started)
        tokens_in, tokens_out = body.get("prompt_eval_count", 0), body.get("eval_count", 0)
        telemetry.TOKENS_IN.inc(tokens_in)
        telemetry.TOKENS_OUT.inc(tokens_out)
        log.info("generated", extra={"model": self.settings.llm_model,
                                     "tokens_in": tokens_in, "tokens_out": tokens_out})
        return parse_answer(body.get("message", {}).get("content", ""))
