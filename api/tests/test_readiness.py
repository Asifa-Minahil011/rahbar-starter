"""Readiness (sections 4.3 and 4.4): /readyz tells the truth, /healthz never depends on anything."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.routes import health
from tests.conftest import Deps


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_ready_when_every_dependency_is_healthy(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"chroma", "collection", "documents", "embed_model", "redis"}


def test_not_ready_when_chroma_is_down(client: TestClient, deps: Deps) -> None:
    deps.chroma_down = True
    r = client.get("/readyz")
    assert r.status_code == 503
    assert "chroma" in r.json()["failed"]


def test_not_ready_when_the_collection_is_missing(client: TestClient, deps: Deps) -> None:
    deps.collection_exists = False
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["failed"] == ["collection"]


def test_not_ready_when_the_collection_is_empty(client: TestClient, deps: Deps) -> None:
    deps.count = 0
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["failed"] == ["documents"]


def test_not_ready_when_redis_is_down(client: TestClient, deps: Deps) -> None:
    deps.redis_down = True
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["failed"] == ["redis"]


def test_embedding_pin_mismatch_blocks_readiness_and_names_both_models(
    client: TestClient, deps: Deps, caplog: pytest.LogCaptureFixture
) -> None:
    deps.model = "dummy-768"            # the index says one model, the API says another
    with caplog.at_level(logging.ERROR, logger="rahbar.api"):
        r = client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["failed"] == ["embed_model"]
    detail = body["checks"]["embed_model"]["detail"]
    assert "dummy-768" in detail and "nomic-embed-text" in detail
    assert any("not ready" in record.getMessage() for record in caplog.records)


def test_liveness_ignores_every_dependency(client: TestClient, deps: Deps) -> None:
    deps.chroma_down = True
    deps.redis_down = True
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


def test_startup_pin_check_reports_ok_mismatch_and_unavailable(deps: Deps) -> None:
    settings = Settings(embed_model="nomic-embed-text")
    assert health.check_embedding_pin(settings)[0] == "ok"

    deps.model = "dummy-768"
    status, detail = health.check_embedding_pin(settings)
    assert status == "mismatch"
    assert "dummy-768" in detail and "nomic-embed-text" in detail

    deps.chroma_down = True
    assert health.check_embedding_pin(settings)[0] == "unavailable"
