"""Tests for the operator status report."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gemynd.shared.database_manager import DatabaseManager
from gemynd.shared.settings import Settings
from gemynd.shared.status import (
    StatusReport,
    _age_text,
    _human_size,
    collect_status,
    format_status_json,
    format_status_text,
)


@pytest.fixture()
def tmp_settings(tmp_path: Path, monkeypatch) -> Settings:
    """Settings with every DB path under tmp_path and Neo4j unset."""
    monkeypatch.delenv("CLAIM_CACHE_PATH", raising=False)
    monkeypatch.setenv("CLAIM_CACHE_PATH", str(tmp_path / "claim_cache.db"))
    return Settings(
        users_db=str(tmp_path / "users.db"),
        review_db=str(tmp_path / "review.db"),
        ingest_db=str(tmp_path / "ingest.db"),
        conv_log_db=str(tmp_path / "conversation_log.db"),
        token_usage_db=str(tmp_path / "token_usage.db"),
        write_audit_db=str(tmp_path / "write_audit.db"),
        annotation_db=str(tmp_path / "annotation.db"),
    )


@pytest.fixture()
def migrated(tmp_settings: Settings) -> DatabaseManager:
    mgr = DatabaseManager(tmp_settings)
    mgr.run_migrations()
    yield mgr
    mgr.close_all()


def _seed_conversation(path: Path, created_at: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO conversation "
            "(conversation_id, query_text, bucket, classifier_confidence, "
            " year_min, year_max, retrieval_path, created_at, turn_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("conv_test", "hello?", "test", 1.0, None, None, "test", created_at, 1),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_token_usage(path: Path, created_at: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO token_usage "
            "(usage_id, caller, model, input_tokens, output_tokens, total_tokens, "
            " input_cost_usd, output_cost_usd, total_cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("u1", "test", "claude-test", 10, 10, 20, 0.0, 0.0, 0.0, created_at),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_claim_cache(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS claim_cache ("
            " cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, "
            " response TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        conn.executemany(
            "INSERT INTO claim_cache (cache_key, model, response) VALUES (?, ?, ?)",
            [(f"k{i}", "test-model", "[]") for i in range(rows)],
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# collect_status — happy path & exit code
# ---------------------------------------------------------------------------

class TestCollectStatus:
    def test_runs_without_raising_on_empty_dbs(self, tmp_settings, migrated):
        report = collect_status(tmp_settings)
        assert isinstance(report, StatusReport)
        names = {c.name for c in report.checks}
        assert {
            "neo4j",
            "users",
            "review",
            "ingest",
            "conv_log",
            "token_usage",
            "write_audit",
            "annotation",
            "claim_cache",
            "conv_log_freshness",
            "token_usage_freshness",
        } <= names

    def test_neo4j_unset_yields_warn(self, tmp_settings, migrated):
        report = collect_status(tmp_settings)
        neo4j = next(c for c in report.checks if c.name == "neo4j")
        assert neo4j.status == "warn"
        assert "NEO4J_URI" in neo4j.detail

    def test_sqlite_dbs_ok_after_migration(self, tmp_settings, migrated):
        report = collect_status(tmp_settings)
        for name in ("users", "review", "ingest", "conv_log", "token_usage", "write_audit"):
            check = next(c for c in report.checks if c.name == name)
            assert check.status == "ok", f"{name}: {check.detail}"
            assert check.metrics["exists"] is True
            assert check.metrics["size_bytes"] > 0

    def test_missing_sqlite_db_fails(self, tmp_settings, migrated, tmp_path):
        # Close the manager first so Windows releases its handle, then delete
        # the review DB to simulate corruption / loss.
        migrated.close_all()
        Path(tmp_settings.review_db).unlink()
        for sidecar in ("review.db-wal", "review.db-shm"):
            sc = tmp_path / sidecar
            if sc.exists():
                sc.unlink()
        report = collect_status(tmp_settings)
        review = next(c for c in report.checks if c.name == "review")
        assert review.status == "fail"
        assert "missing" in review.detail
        assert report.overall == "fail"
        assert report.exit_code == 2

    def test_overall_warn_exit_code(self, tmp_settings, migrated):
        # No Neo4j (warn) and no fails -> overall warn -> exit 1
        report = collect_status(tmp_settings)
        assert report.overall == "warn"
        assert report.exit_code == 1


# ---------------------------------------------------------------------------
# Claim cache check
# ---------------------------------------------------------------------------

class TestClaimCacheCheck:
    def test_absent_yields_warn(self, tmp_settings, migrated):
        report = collect_status(tmp_settings)
        cache = next(c for c in report.checks if c.name == "claim_cache")
        assert cache.status == "warn"
        assert "absent" in cache.detail

    def test_row_count_below_threshold_ok(self, tmp_settings, migrated, tmp_path):
        _seed_claim_cache(tmp_path / "claim_cache.db", rows=3)
        report = collect_status(tmp_settings, cache_warn_rows=100)
        cache = next(c for c in report.checks if c.name == "claim_cache")
        assert cache.status == "ok"
        assert cache.metrics["row_count"] == 3

    def test_row_count_over_threshold_warns(self, tmp_settings, migrated, tmp_path):
        _seed_claim_cache(tmp_path / "claim_cache.db", rows=5)
        report = collect_status(tmp_settings, cache_warn_rows=2)
        cache = next(c for c in report.checks if c.name == "claim_cache")
        assert cache.status == "warn"
        assert "exceeds" in cache.detail
        assert cache.metrics["row_count"] == 5
        assert cache.metrics["warn_threshold_rows"] == 2


# ---------------------------------------------------------------------------
# Liveness checks
# ---------------------------------------------------------------------------

class TestFreshnessChecks:
    def test_conv_log_empty_warns(self, tmp_settings, migrated):
        report = collect_status(tmp_settings)
        conv = next(c for c in report.checks if c.name == "conv_log_freshness")
        assert conv.status == "warn"
        assert "no conversations" in conv.detail

    def test_token_usage_stale_vs_conv_log_warns(self, tmp_settings, migrated):
        # Conversation written *after* token usage -> writer thread suspect.
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        new = datetime.now(timezone.utc).isoformat()
        _seed_token_usage(Path(tmp_settings.token_usage_db), old)
        _seed_conversation(Path(tmp_settings.conv_log_db), new)

        report = collect_status(tmp_settings)
        tok = next(c for c in report.checks if c.name == "token_usage_freshness")
        assert tok.status == "warn"
        assert "stale" in tok.detail

    def test_token_usage_fresh_vs_conv_log_ok(self, tmp_settings, migrated):
        # Token write newer than conversation write -> ok.
        old = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        new = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        _seed_conversation(Path(tmp_settings.conv_log_db), old)
        _seed_token_usage(Path(tmp_settings.token_usage_db), new)

        report = collect_status(tmp_settings)
        tok = next(c for c in report.checks if c.name == "token_usage_freshness")
        assert tok.status == "ok"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class TestFormatters:
    def test_text_contains_section_headers(self, tmp_settings, migrated):
        report = collect_status(tmp_settings)
        text = format_status_text(report)
        assert "gemynd status" in text
        # Header must remain ASCII so Windows cp1252 stdout can print it.
        text.encode("ascii")
        assert "Databases" in text
        assert "Liveness" in text
        assert "Overall:" in text

    def test_json_round_trip(self, tmp_settings, migrated):
        report = collect_status(tmp_settings)
        payload = json.loads(format_status_json(report))
        assert payload["overall"] in {"ok", "warn", "fail"}
        assert payload["exit_code"] == report.exit_code
        assert isinstance(payload["checks"], list)
        assert any(c["name"] == "neo4j" for c in payload["checks"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_human_size(self):
        assert _human_size(0) == "0 B"
        assert _human_size(512) == "512 B"
        assert _human_size(2048).endswith("KB")
        assert _human_size(5 * 1024 * 1024).endswith("MB")

    def test_age_text(self):
        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)
        assert _age_text((now - timedelta(seconds=30)).isoformat(), now=now) == "30s ago"
        assert _age_text((now - timedelta(minutes=5)).isoformat(), now=now) == "5m ago"
        assert _age_text((now - timedelta(hours=3)).isoformat(), now=now) == "3h ago"
        assert _age_text((now - timedelta(days=2)).isoformat(), now=now) == "2d ago"
        assert _age_text("not-a-date", now=now) == "unknown age"
