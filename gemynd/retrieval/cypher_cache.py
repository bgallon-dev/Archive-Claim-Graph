"""SQLite-backed cache for validated LLM-generated Cypher (plan B4).

Lives in the existing ``conversation_log.db`` so we don't introduce a new
managed database. Keyed by ``(domain_version, normalized_query)`` so a schema
change automatically invalidates every entry without an explicit migration.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time

_log = logging.getLogger(__name__)


# Migration applied via :func:`gemynd.shared.database_manager._migrate_conv_log`.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS query_cypher_cache (
    domain_version    TEXT NOT NULL,
    normalized_query  TEXT NOT NULL,
    cypher            TEXT NOT NULL,
    created_at        REAL NOT NULL,
    hit_count         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (domain_version, normalized_query)
);
"""


class SqliteCypherCache:
    """Cache implementing the protocol expected by :class:`LLMCypherGenerator`.

    Parameters
    ----------
    conn:
        A live SQLite connection. Typically the ``conv_log`` connection from
        :class:`gemynd.shared.database_manager.DatabaseManager`.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        # Ensure the table exists for callers who instantiate the cache against
        # a connection that pre-dates the database_manager migration step.
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    def get(self, domain_version: str, normalized_query: str) -> str | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT cypher FROM query_cypher_cache "
                "WHERE domain_version = ? AND normalized_query = ?",
                (domain_version, normalized_query),
            )
            row = cur.fetchone()
            if row is None:
                return None
            # Best-effort hit counter. Failure to bump is non-fatal.
            try:
                self._conn.execute(
                    "UPDATE query_cypher_cache SET hit_count = hit_count + 1 "
                    "WHERE domain_version = ? AND normalized_query = ?",
                    (domain_version, normalized_query),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                _log.debug("hit_count bump failed: %s", exc)
            return row[0] if not isinstance(row, sqlite3.Row) else row["cypher"]

    def put(
        self,
        domain_version: str,
        normalized_query: str,
        cypher: str,
    ) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO query_cypher_cache "
                    "(domain_version, normalized_query, cypher, created_at, hit_count) "
                    "VALUES (?, ?, ?, ?, 0)",
                    (domain_version, normalized_query, cypher, time.time()),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                _log.debug("cypher cache write failed: %s", exc)
