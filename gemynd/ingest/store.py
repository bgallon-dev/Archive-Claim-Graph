"""SQLite-backed job store for the document ingestion UI."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestStore:
    """Tracks ingest jobs and per-document processing status in SQLite."""

    def __init__(self, db_path: str, *, skip_init: bool = False) -> None:
        self._db_path = db_path
        if skip_init:
            return
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ingest_job (
                    job_id         TEXT PRIMARY KEY,
                    status         TEXT NOT NULL DEFAULT 'running',
                    out_dir        TEXT NOT NULL,
                    institution_id TEXT NOT NULL DEFAULT '',
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
            # Migration: add institution_id column to existing databases.
            try:
                conn.execute(
                    "ALTER TABLE ingest_job ADD COLUMN institution_id TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass  # column already exists

    def create_job(
        self, out_dir: str, filenames: list[str], *, institution_id: str = ""
    ) -> str:
        """Create a new ingest job and return its job_id."""
        job_id = uuid.uuid4().hex
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ingest_job (job_id, status, out_dir, institution_id, created_at, total_docs)"
                " VALUES (?, 'running', ?, ?, ?, ?)",
                (job_id, out_dir, institution_id, now, len(filenames)),
            )
            for fn in filenames:
                conn.execute(
                    "INSERT INTO ingest_document (job_id, filename, status, created_at)"
                    " VALUES (?, ?, 'queued', ?)",
                    (job_id, fn, now),
                )
        return job_id

    def update_document_status(
        self,
        job_id: str,
        filename: str,
        status: str,
        *,
        error_message: str | None = None,
        output_dir: str | None = None,
        claims_count: int | None = None,
        mention_count: int | None = None,
    ) -> None:
        now = _utcnow()
        completed_at = now if status in ("completed", "failed") else None
        with self._connect() as conn:
            conn.execute(
                """UPDATE ingest_document
                   SET status        = ?,
                       error_message = ?,
                       output_dir    = COALESCE(?, output_dir),
                       claims_count  = COALESCE(?, claims_count),
                       mention_count = COALESCE(?, mention_count),
                       completed_at  = COALESCE(?, completed_at)
                   WHERE job_id = ? AND filename = ?""",
                (
                    status, error_message, output_dir,
                    claims_count, mention_count,
                    completed_at, job_id, filename,
                ),
            )
            if status in ("completed", "failed"):
                col = "completed_docs" if status == "completed" else "failed_docs"
                conn.execute(
                    f"UPDATE ingest_job SET {col} = {col} + 1 WHERE job_id = ?",
                    (job_id,),
                )
                # Seal the job once every document has a terminal status.
                conn.execute(
                    """UPDATE ingest_job
                       SET status = 'completed', completed_at = ?
                       WHERE job_id = ?
                         AND total_docs = completed_docs + failed_docs
                         AND status = 'running'""",
                    (now, job_id),
                )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingest_job WHERE job_id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ingest_document WHERE job_id = ? ORDER BY id",
                (job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        pass  # Connections are opened and closed per operation.
