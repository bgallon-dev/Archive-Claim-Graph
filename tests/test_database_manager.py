"""Tests for the centralised DatabaseManager."""
from __future__ import annotations

import sqlite3

import pytest

from graphrag_pipeline.shared.database_manager import DatabaseManager
from graphrag_pipeline.shared.settings import Settings


@pytest.fixture()
def tmp_settings(tmp_path):
    """Return a Settings instance with all DB paths under tmp_path."""
    return Settings(
        users_db=str(tmp_path / "users.db"),
        review_db=str(tmp_path / "review.db"),
        ingest_db=str(tmp_path / "ingest.db"),
        conv_log_db=str(tmp_path / "conv_log.db"),
        token_usage_db=str(tmp_path / "token_usage.db"),
        write_audit_db=str(tmp_path / "write_audit.db"),
        annotation_db=str(tmp_path / "annotation.db"),
    )


@pytest.fixture()
def manager(tmp_settings):
    mgr = DatabaseManager(tmp_settings)
    yield mgr
    mgr.close_all()


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

class TestGetConnection:
    def test_creates_db_file(self, manager, tmp_settings):
        conn = manager.get_connection("users")
        assert conn is not None
        from pathlib import Path
        assert Path(tmp_settings.users_db).exists()

    def test_returns_same_connection(self, manager):
        c1 = manager.get_connection("review")
        c2 = manager.get_connection("review")
        assert c1 is c2

    def test_wal_mode(self, manager):
        conn = manager.get_connection("users")
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_on(self, manager):
        conn = manager.get_connection("users")
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_row_factory_set(self, manager):
        conn = manager.get_connection("users")
        assert conn.row_factory is sqlite3.Row

    def test_unknown_name_raises(self, manager):
        with pytest.raises(KeyError):
            manager.get_connection("nonexistent")


class TestGetPath:
    def test_returns_path(self, manager, tmp_settings):
        p = manager.get_path("review")
        assert str(p) == tmp_settings.review_db

    def test_unknown_name_raises(self, manager):
        with pytest.raises(KeyError):
            manager.get_path("nonexistent")


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

class TestMigrations:
    def test_creates_all_tables(self, manager):
        manager.run_migrations()

        # Spot-check expected tables per database.
        def _tables(name):
            conn = manager.get_connection(name)
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            return {r[0] for r in rows}

        assert "user" in _tables("users")
        assert "proposal" in _tables("review")
        assert "anti_pattern_class" in _tables("review")
        assert "ingest_job" in _tables("ingest")
        assert "ingest_document" in _tables("ingest")
        assert "document_annotation" in _tables("annotation")
        assert "conversation" in _tables("conv_log")
        assert "saved_search" in _tables("conv_log")
        assert "token_usage" in _tables("token_usage")
        assert "token_usage_daily" in _tables("token_usage")
        assert "write_event" in _tables("write_audit")

    def test_idempotent(self, manager):
        manager.run_migrations()
        manager.run_migrations()  # should not raise


class TestAnnotationOptional:
    def test_no_annotation_db(self, tmp_path):
        settings = Settings(
            users_db=str(tmp_path / "users.db"),
            review_db=str(tmp_path / "review.db"),
            ingest_db=str(tmp_path / "ingest.db"),
            conv_log_db=str(tmp_path / "conv_log.db"),
            token_usage_db=str(tmp_path / "token_usage.db"),
            write_audit_db=str(tmp_path / "write_audit.db"),
            annotation_db="",
        )
        mgr = DatabaseManager(settings)
        mgr.run_migrations()
        with pytest.raises(KeyError):
            mgr.get_connection("annotation")
        mgr.close_all()


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------

class TestCloseAll:
    def test_connections_closed(self, manager):
        manager.run_migrations()
        conn = manager.get_connection("users")
        manager.close_all()
        # After close_all, the connection should be unusable.
        with pytest.raises(Exception):
            conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# Stores accept managed connections / skip_init
# ---------------------------------------------------------------------------

class TestStoresWithManagedConnections:
    def test_review_store_with_conn(self, manager):
        manager.run_migrations()
        from graphrag_pipeline.review.store import ReviewStore
        conn = manager.get_connection("review")
        store = ReviewStore(manager.get_path("review"), conn=conn)
        # Should be able to query without error.
        counts = store.proposal_counts_by_status()
        assert isinstance(counts, dict)

    def test_annotation_store_with_conn(self, manager):
        manager.run_migrations()
        from graphrag_pipeline.ingest.annotation.store import AnnotationStore
        conn = manager.get_connection("annotation")
        store = AnnotationStore(manager.get_path("annotation"), conn=conn)
        result = store.get_current_note("doc-1")
        assert result is None

    def test_query_history_store_with_conn(self, manager):
        manager.run_migrations()
        from graphrag_pipeline.retrieval.conversation_log import QueryHistoryStore
        conn = manager.get_connection("conv_log")
        store = QueryHistoryStore(manager.get_path("conv_log"), conn=conn)
        queries = store.list_queries(limit=5)
        assert isinstance(queries, list)

    def test_token_usage_store_with_conn(self, manager):
        manager.run_migrations()
        from graphrag_pipeline.shared.token_tracker import TokenUsageStore
        conn = manager.get_connection("token_usage")
        store = TokenUsageStore(manager.get_path("token_usage"), conn=conn)
        summary = store.today_summary()
        assert isinstance(summary, dict)

    def test_user_store_skip_init(self, manager):
        manager.run_migrations()
        from graphrag_pipeline.auth.store import UserStore
        store = UserStore(manager.get_path("users"), skip_init=True)
        users = store.list_users()
        assert isinstance(users, list)

    def test_ingest_store_skip_init(self, manager):
        manager.run_migrations()
        from graphrag_pipeline.ingest.store import IngestStore
        store = IngestStore(str(manager.get_path("ingest")), skip_init=True)
        result = store.get_job("nonexistent")
        assert result is None

    def test_write_audit_skip_init(self, manager):
        manager.run_migrations()
        from graphrag_pipeline.retrieval.web.write_audit_log import WriteAuditLogger
        wal = WriteAuditLogger(manager.get_path("write_audit"), skip_init=True)
        # Should be able to log without error.
        wal.log("ingestion", "doc-1", "Test", "test-inst", "cli")
