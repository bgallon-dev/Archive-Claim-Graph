"""Centralised SQLite connection and migration manager.

Owns all SQLite connections for a deployment, runs schema migrations in a
single ordered sequence at startup, and provides pre-configured connections
to the individual stores.  Stores keep their business logic unchanged — only
their ``__init__`` connection setup is replaced.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from .settings import Settings

_log = logging.getLogger(__name__)


class DatabaseManager:
    """Centralised owner of all SQLite connections for a web deployment.

    Parameters
    ----------
    settings:
        Application :class:`Settings` instance with resolved DB paths.
    """

    def __init__(self, settings: Settings) -> None:
        self._conns: dict[str, sqlite3.Connection] = {}
        self._paths: dict[str, Path] = {}
        self._register(settings)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register(self, s: Settings) -> None:
        for name, path_str in [
            ("users", s.users_db),
            ("review", s.review_db),
            ("ingest", s.ingest_db),
            ("conv_log", s.conv_log_db),
            ("token_usage", s.token_usage_db),
            ("write_audit", s.write_audit_db),
        ]:
            self._paths[name] = Path(path_str)
        if s.annotation_db:
            self._paths["annotation"] = Path(s.annotation_db)

    # ------------------------------------------------------------------
    # Connection access
    # ------------------------------------------------------------------

    def get_connection(self, name: str) -> sqlite3.Connection:
        """Return a managed connection for the named database.

        Creates the connection (and parent directories) on first access.
        All connections use WAL mode, ``check_same_thread=False``, and
        ``foreign_keys=ON``.
        """
        if name not in self._paths:
            raise KeyError(f"Unknown database name: {name!r}")
        if name not in self._conns:
            path = self._paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._conns[name] = conn
        return self._conns[name]

    def get_path(self, name: str) -> Path:
        """Return the resolved filesystem path for the named database."""
        if name not in self._paths:
            raise KeyError(f"Unknown database name: {name!r}")
        return self._paths[name]

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def run_migrations(self) -> None:
        """Run all schema migrations in dependency order.

        Safe to call multiple times — every migration uses
        ``CREATE TABLE IF NOT EXISTS`` and guards ``ALTER TABLE`` with
        try/except for already-existing columns.
        """
        for name, migrate_fn in _MIGRATIONS:
            if name in self._paths:
                conn = self.get_connection(name)
                try:
                    migrate_fn(conn)
                    conn.commit()
                except Exception:
                    _log.warning("Migration failed for %s", name, exc_info=True)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close_all(self) -> None:
        """Close every managed connection."""
        for name, conn in self._conns.items():
            try:
                conn.close()
            except Exception:
                _log.debug("Failed to close %s", name, exc_info=True)
        self._conns.clear()


# ======================================================================
# Migration functions — one per database
#
# Each function uses deferred imports to pull _SCHEMA_SQL constants from
# the store modules they belong to.  No SQL is duplicated.
# ======================================================================


def _migrate_users(conn: sqlite3.Connection) -> None:
    from gemynd.auth.store import _SCHEMA, _MIGRATE_TOKEN_VERSION

    conn.executescript(_SCHEMA)
    try:
        conn.execute(_MIGRATE_TOKEN_VERSION)
    except sqlite3.OperationalError:
        pass  # column already exists


def _migrate_review(conn: sqlite3.Connection) -> None:
    from gemynd.review.store import _DEFAULT_ANTI_PATTERNS, _SCHEMA_SQL

    conn.executescript(_SCHEMA_SQL)
    for ap in _DEFAULT_ANTI_PATTERNS:
        conn.execute(
            "INSERT OR IGNORE INTO anti_pattern_class"
            " (anti_pattern_id, name, description, queue_name)"
            " VALUES (?, ?, ?, ?)",
            (ap["anti_pattern_id"], ap["name"], ap["description"], ap["queue_name"]),
        )
    conn.commit()

    # Column migrations matching ReviewStore._migrate()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(proposal_target)")}
    if "reviewer_override" not in cols:
        conn.execute(
            "ALTER TABLE proposal_target ADD COLUMN reviewer_override TEXT DEFAULT NULL"
        )
        conn.commit()

    proposal_cols = {r[1] for r in conn.execute("PRAGMA table_info(proposal)")}
    if "review_tier" not in proposal_cols:
        conn.execute(
            "ALTER TABLE proposal ADD COLUMN review_tier TEXT NOT NULL DEFAULT 'needs_review'"
        )
        conn.commit()

    ce_cols = {r[1] for r in conn.execute("PRAGMA table_info(correction_event)")}
    if "error_root_cause" not in ce_cols:
        conn.execute(
            "ALTER TABLE correction_event ADD COLUMN error_root_cause TEXT NOT NULL DEFAULT ''"
        )
        conn.execute(
            "ALTER TABLE correction_event ADD COLUMN error_type TEXT NOT NULL DEFAULT ''"
        )
        conn.commit()


def _migrate_ingest(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ingest_job (
            job_id         TEXT PRIMARY KEY,
            status         TEXT NOT NULL DEFAULT 'running',
            out_dir        TEXT NOT NULL,
            created_at     TEXT NOT NULL,
            completed_at   TEXT,
            total_docs     INTEGER NOT NULL DEFAULT 0,
            completed_docs INTEGER NOT NULL DEFAULT 0,
            failed_docs    INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS ingest_document (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id        TEXT NOT NULL REFERENCES ingest_job(job_id),
            filename      TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'queued',
            error_message TEXT,
            output_dir    TEXT,
            claims_count  INTEGER,
            mention_count INTEGER,
            created_at    TEXT NOT NULL,
            completed_at  TEXT
        );
    """)


def _migrate_annotation(conn: sqlite3.Connection) -> None:
    from gemynd.ingest.annotation.store import _SCHEMA_SQL

    conn.executescript(_SCHEMA_SQL)


def _migrate_conv_log(conn: sqlite3.Connection) -> None:
    from gemynd.retrieval.conversation_log import (
        _SAVED_SEARCH_SCHEMA_SQL,
        _SCHEMA_SQL,
    )

    conn.executescript(_SCHEMA_SQL)
    # Column migrations for databases predating newer columns.
    for ddl in (
        "ALTER TABLE conversation ADD COLUMN session_id TEXT",
        "ALTER TABLE conversation ADD COLUMN turn_number INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE conversation ADD COLUMN request_id TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.executescript(_SAVED_SEARCH_SCHEMA_SQL)


def _migrate_token_usage(conn: sqlite3.Connection) -> None:
    from gemynd.shared.token_tracker import _SCHEMA_SQL

    conn.executescript(_SCHEMA_SQL)


def _migrate_write_audit(conn: sqlite3.Connection) -> None:
    from gemynd.retrieval.web.write_audit_log import (
        _CREATE_INDEX,
        _CREATE_TABLE,
    )

    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_INDEX)


# Ordered migration list.  conv_log before QueryHistoryStore reads,
# token_usage before TokenUsageStore reads.
_MIGRATIONS: list[tuple[str, callable]] = [
    ("users", _migrate_users),
    ("review", _migrate_review),
    ("ingest", _migrate_ingest),
    ("annotation", _migrate_annotation),
    ("conv_log", _migrate_conv_log),
    ("token_usage", _migrate_token_usage),
    ("write_audit", _migrate_write_audit),
]
