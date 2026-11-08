"""Semantic cache tests with an in-memory Redis and hand-made vectors."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from app.schemas import AskResponse, Citation
from app.services.cache import RedisSemanticCache

VECTORS = {
    "what is the attendance rule": [1.0, 0.0, 0.0],
    "tell me the attendance rule": [0.99, 0.14, 0.0],      # paraphrase: cosine about 0.99
    "what does clause 4.23 say": [0.0, 1.0, 0.0],
    "what does clause 4.24 say": [0.0, 0.96, 0.28],        # near miss: cosine about 0.96
}


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value

    def scan_iter(self, match: str, count: int = 0) -> Any:
        return iter([k for k in self.data if fnmatch(k, match)])

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self.data.get(k) for k in keys]


def reply(grounded: bool = True, cited: bool = True) -> AskResponse:
    return AskResponse(
        request_id="r", answer="100%", answered_from_context=grounded, confidence="high",
        citations=[Citation(source="s.pdf", section="1.21")] if cited else [],
        retrieved=[], provider="t")


def make(index_id: list[str], threshold: float = 0.97) -> tuple[RedisSemanticCache, FakeRedis]:
    r = FakeRedis()
    return RedisSemanticCache(r, lambda q: VECTORS[q.lower()], lambda: index_id[0],
                              threshold=threshold), r


def test_paraphrase_hits_and_near_miss_does_not() -> None:
    cache, _ = make(["i1"])
    cache.store("What is the attendance rule", reply())
    cache.store("What does clause 4.23 say", reply())
    assert cache.lookup("tell me the attendance rule") is not None          # true hit
    assert cache.lookup("What does clause 4.24 say") is None                # false hit avoided


def test_a_lower_threshold_without_the_guard_returns_the_wrong_clause() -> None:
    cache, _ = make(["i1"], threshold=0.9)
    cache.exact_guard = False                    # what pure cosine does: the mistake the threshold report is about
    cache.store("What does clause 4.23 say", reply())
    assert cache.lookup("What does clause 4.24 say") is not None


def test_the_exact_term_guard_blocks_a_near_miss_even_at_a_low_threshold() -> None:
    cache, _ = make(["i1"], threshold=0.9)
    cache.store("What does clause 4.23 say", reply())
    assert cache.lookup("What does clause 4.24 say") is None
    assert cache.lookup("What does clause 4.23 say") is not None


def test_signature_separates_clauses_years_and_older_editions() -> None:
    from app.services.cache import signature
    assert signature("what does clause 4.23 say") != signature("what does clause 4.24 say")
    assert signature("attendance rule in 2023") != signature("attendance rule")
    assert signature("the previous attendance rule") != signature("the attendance rule")
    assert signature("How many courses can I repeat?") == signature("How many courses may a student repeat")


def test_reindex_invalidates_everything() -> None:
    index = ["index-A"]
    cache, _ = make(index)
    cache.store("What is the attendance rule", reply())
    assert cache.lookup("What is the attendance rule") is not None
    index[0] = "index-B"                         # the indexer wrote a new index_id
    assert cache.lookup("What is the attendance rule") is None


def test_abstentions_and_uncited_answers_are_never_stored() -> None:
    cache, r = make(["i1"])
    cache.store("What is the attendance rule", reply(grounded=False))
    cache.store("What is the attendance rule", reply(cited=False))
    assert r.data == {}


def test_a_broken_redis_is_a_miss_not_an_error() -> None:
    class Broken(FakeRedis):
        def scan_iter(self, match: str, count: int = 0) -> Any:
            raise ConnectionError("redis down")

        def set(self, key: str, value: str, ex: int | None = None) -> None:
            raise ConnectionError("redis down")

    cache = RedisSemanticCache(Broken(), lambda q: VECTORS[q.lower()], lambda: "i1", threshold=0.97)
    assert cache.lookup("What is the attendance rule") is None
    cache.store("What is the attendance rule", reply())       # must not raise
