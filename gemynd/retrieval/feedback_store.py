"""Tester feedback capture.

A pilot's entire value is learning what breaks, and until now the only
channel was out-of-band email. This store backs a "Report a problem" control
that any logged-in user can reach from any page, so friction gets recorded
where it happens instead of being reconstructed afterwards.

Deliberately minimal: free text, plus whatever context we can capture without
asking the tester for it (the page they were on, an optional query/answer the
report is about, their role). Submissions are never blocking and never
user-visible to other testers.
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Categories the UI offers. Free-form `message` carries the detail; the
# category exists so the developer can triage a pile of reports quickly.
FEEDBACK_CATEGORIES = (
    "wrong_answer",
    "missing_source",
    "confusing",
    "slow",
    "error",
    "other",
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id  TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    user_id      TEXT,
    user_email   TEXT,
    user_role    TEXT,
    category     TEXT NOT NULL DEFAULT 'other',
    message      TEXT NOT NULL,
    page_path    TEXT,
    query_text   TEXT,
    request_id   TEXT,
    resolved     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_resolved ON feedback(resolved, created_at DESC);
"""


class FeedbackStore:
    """SQLite-backed store for tester-submitted problem reports."""

    def __init__(
        self,
        db_path: str | Path = "data/feedback.db",
        conn: sqlite3.Connection | None = None,
        skip_init: bool = False,
    ) -> None:
        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._owns_conn = True
        if not skip_init:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add(
        self,
        message: str,
        *,
        category: str = "other",
        user_id: str | None = None,
        user_email: str | None = None,
        user_role: str | None = None,
        page_path: str | None = None,
        query_text: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """Record one report and return its id.

        ``message`` is trimmed and capped; an empty message is rejected by the
        caller, not here. Unknown categories fall back to ``other`` rather than
        raising — a tester's report is never worth losing to validation.
        """
        if category not in FEEDBACK_CATEGORIES:
            category = "other"
        feedback_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO feedback (feedback_id, created_at, user_id, user_email,"
            " user_role, category, message, page_path, query_text, request_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                feedback_id,
                datetime.now(timezone.utc).isoformat(),
                user_id,
                user_email,
                user_role,
                category,
                message.strip()[:4000],
                page_path,
                (query_text or "")[:2000] or None,
                request_id,
            ),
        )
        self._conn.commit()
        return feedback_id

    def mark_resolved(self, feedback_id: str, resolved: bool = True) -> None:
        self._conn.execute(
            "UPDATE feedback SET resolved = ? WHERE feedback_id = ?",
            (1 if resolved else 0, feedback_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reads (admin-only surface)
    # ------------------------------------------------------------------

    def list_recent(
        self, limit: int = 100, include_resolved: bool = True
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM feedback"
        if not include_resolved:
            sql += " WHERE resolved = 0"
        sql += " ORDER BY created_at DESC LIMIT ?"
        return [dict(r) for r in self._conn.execute(sql, (limit,))]

    def count_unresolved(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM feedback WHERE resolved = 0"
        ).fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        if self._owns_conn:
            try:
                self._conn.close()
            except Exception:
                _log.debug("Failed to close feedback store", exc_info=True)
