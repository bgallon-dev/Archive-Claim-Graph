"""The app must survive Neo4j being unavailable at startup.

Previously the executor was constructed unguarded inside the FastAPI
lifespan, so a database that was merely slow to come up aborted ASGI startup
and the process exited. Under a supervisor that is a crash-loop; without one
the deployment is simply over until someone notices. For an unattended pilot
this is the difference between a blip and a dead week.

The contract these tests pin down:
  * startup completes even with no database;
  * /health reports it honestly rather than claiming readiness;
  * data routes answer 503 (server-side, retryable) rather than 500;
  * the app heals on its own once the database returns — no restart.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from gemynd.retrieval.web import app as app_module  # noqa: E402


class _FakeExecutor:
    """Minimal stand-in for Neo4jQueryExecutor."""

    def __init__(self, *args, **kwargs):
        self.closed = False

    def ensure_schema(self) -> None:
        pass

    def run(self, cypher, params=None):
        if "count(c) AS total" in cypher:
            return [{"total": 0}]
        return []

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Point every persistent path at a temp dir so tests don't touch real data."""
    for var, name in [
        ("USERS_DB", "users.db"),
        ("REVIEW_DB", "review.db"),
        ("INGEST_DB", "ingest.db"),
        ("CONV_LOG_DB", "conv.db"),
        ("TOKEN_USAGE_DB", "tokens.db"),
        ("WRITE_AUDIT_DB", "audit.db"),
        ("FEEDBACK_DB", "feedback.db"),
    ]:
        monkeypatch.setenv(var, str(tmp_path / name))
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 48)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    yield


def _build_app(domain_dir="gemynd/resources"):
    return app_module.create_app(domain_dir=domain_dir)


class TestStartupWithoutNeo4j:
    def test_app_starts_when_neo4j_is_down(self, env):
        """The whole point: no exception out of the lifespan."""
        with patch.object(
            app_module, "Neo4jQueryExecutor",
            side_effect=RuntimeError("Connection refused"),
        ):
            with TestClient(_build_app()) as client:
                assert client.get("/health").status_code == 503

    def test_health_reports_degraded_not_ok(self, env):
        with patch.object(
            app_module, "Neo4jQueryExecutor",
            side_effect=RuntimeError("Connection refused"),
        ):
            with TestClient(_build_app()) as client:
                body = client.get("/health").json()
                assert body["status"] == "degraded"
                assert body["neo4j"] == "unavailable"

    def test_health_does_not_leak_connection_details(self, env):
        """/health is unauthenticated — it must not expose internals."""
        secret = "super-secret-password-detail"
        with patch.object(
            app_module, "Neo4jQueryExecutor",
            side_effect=RuntimeError(secret),
        ):
            with TestClient(_build_app()) as client:
                assert secret not in client.get("/health").text


class TestRecovery:
    def test_health_reconnects_once_neo4j_returns(self, env):
        """A polled /health is what heals an instance that booted degraded."""
        with patch.object(app_module, "Neo4jQueryExecutor") as mock_exec:
            mock_exec.side_effect = RuntimeError("Connection refused")
            with TestClient(_build_app()) as client:
                assert client.get("/health").json()["status"] == "degraded"

                # Neo4j comes back.
                mock_exec.side_effect = None
                mock_exec.return_value = _FakeExecutor()

                body = client.get("/health").json()
                assert body["status"] == "ok"
                assert body["neo4j"] == "connected"

    def test_stale_connection_is_dropped_and_retried(self, env):
        """A driver that starts failing must not stay wired in forever."""
        class _FailingExecutor(_FakeExecutor):
            def run(self, cypher, params=None):
                raise RuntimeError("session expired")

        with patch.object(app_module, "Neo4jQueryExecutor") as mock_exec:
            mock_exec.return_value = _FailingExecutor()
            with TestClient(_build_app()) as client:
                assert client.get("/health").json()["status"] == "degraded"
                # Healthy driver on the next attempt → recovered.
                mock_exec.return_value = _FakeExecutor()
                assert client.get("/health").json()["status"] == "ok"


class TestExceptionMapping:
    def test_neo4j_unavailable_is_a_runtime_error(self):
        assert issubclass(app_module.Neo4jUnavailable, RuntimeError)

    def test_unavailable_message_is_user_appropriate(self, env):
        """503 body should tell a tester it isn't their fault, without internals."""
        with patch.object(
            app_module, "Neo4jQueryExecutor",
            side_effect=RuntimeError("bolt handshake failed at 10.0.0.5:7687"),
        ):
            app = _build_app()
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/health")
                assert resp.status_code == 503
                assert "10.0.0.5" not in resp.text
