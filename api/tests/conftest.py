"""Shared fixtures. Every test runs against fake Chroma and Redis: no network, no cluster.

Flip the flags on the `deps` fixture to simulate a failure, then call /readyz.
"""

from __future__ import annotations

import pytest

from app.routes import health


class Deps:
    chroma_down = False
    collection_exists = True
    count = 3
    model = "nomic-embed-text"      # the Settings default for EMBED_MODEL
    redis_down = False


class FakeCollection:
    def __init__(self, deps: Deps) -> None:
        self._deps = deps
        self.metadata = {"embed_model": deps.model}

    def count(self) -> int:
        return self._deps.count


class FakeChroma:
    def __init__(self, deps: Deps) -> None:
        self._deps = deps

    def heartbeat(self) -> int:
        if self._deps.chroma_down:
            raise ConnectionError("chroma is unreachable")
        return 1

    def get_collection(self, name: str) -> FakeCollection:
        if self._deps.chroma_down:
            raise ConnectionError("chroma is unreachable")
        if not self._deps.collection_exists:
            raise ValueError(f"Collection {name} does not exist")
        return FakeCollection(self._deps)


class FakeRedis:
    def __init__(self, deps: Deps) -> None:
        self._deps = deps

    def ping(self) -> bool:
        if self._deps.redis_down:
            raise ConnectionError("redis is unreachable")
        return True


@pytest.fixture(autouse=True)
def deps(monkeypatch: pytest.MonkeyPatch) -> Deps:
    d = Deps()
    monkeypatch.setattr(health, "_chroma_client", lambda settings: FakeChroma(d))
    monkeypatch.setattr(health, "_redis_client", lambda settings: FakeRedis(d))
    health._last_failure.clear()
    return d

