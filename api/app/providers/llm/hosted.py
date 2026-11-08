"""Hosted free-tier provider (Groq, Google AI Studio, OpenRouter), OpenAI-compatible.

The key arrives as LLM_API_KEY from a Kubernetes Secret created on the command line.
It never appears in a file you commit, and never in a log line.
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

log = logging.getLogger("rahbar.provider.hosted")


class HostedProvider:
    name = "hosted"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings   # hosted_base_url, llm_model, llm_api_key, temperature

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        if not self.settings.hosted_base_url or not self.settings.llm_api_key:
            raise GenerationError("HOSTED_BASE_URL or LLM_API_KEY is not set")
        payload = {
            "model": self.settings.llm_model,
            "messages": build_messages(question, passages),
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},   # schema is in the system prompt
        }
        started = time.perf_counter()
        try:
            r = httpx.post(
                f"{self.settings.hosted_base_url.rstrip('/')}/chat/completions", json=payload,
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"}, timeout=60.0)
            r.raise_for_status()
            body = r.json()
            content = body["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            # repr of an HTTPError carries the URL, never the Authorization header
            raise GenerationError(f"hosted call failed: {exc!r}") from exc
        finally:
            telemetry.GENERATION_SECONDS.observe(time.perf_counter() - started)
        usage = body.get("usage", {})
        telemetry.TOKENS_IN.inc(usage.get("prompt_tokens", 0))
        telemetry.TOKENS_OUT.inc(usage.get("completion_tokens", 0))
        log.info("generated", extra={"model": self.settings.llm_model,
                                     "tokens_in": usage.get("prompt_tokens", 0),
                                     "tokens_out": usage.get("completion_tokens", 0)})
        return parse_answer(content)
