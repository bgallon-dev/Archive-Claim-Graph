"""Write operation audit log.

Maintains a SQLite record of all write operations performed against the graph:
document ingestion (via CLI), soft-deletion, and restoration. This log is the
authoritative trail for answering "who changed what and when."

The database is stored at the path configured by the WRITE_AUDIT_DB environment
variable (default: data/write_audit.db) and is separate from the conversation
log so that security-relevant events are not co-mingled with query telemetry.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_VALID_EVENT_TYPES = frozenset({"ingestion", "soft_delete", "restore"})

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS write_event (
    event_id      TEXT PRIMARY KEY,
    event_type    TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    doc_title     TEXT,
    institution_id TEXT NOT NULL,
    performed_by  TEXT NOT NULL,
    performed_at  TEXT NOT NULL,
    details       TEXT
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS write_event_doc_id ON write_event (doc_id)
"""


class WriteAuditLogger:
    """Append-only SQLite audit log for write operations.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Parent directories are created
        automatically. Defaults to ``data/write_audit.db``.
    """

    def __init__(self, db_path: str | Path = "data/write_audit.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def log(
        self,
        event_type: str,
        doc_id: str,
        doc_title: str | None,
        institution_id: str,
        performed_by: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a write event.

        Parameters
        ----------
        event_type:
            One of ``"ingestion"``, ``"soft_delete"``, ``"restore"``.
        doc_id:
            The document's unique identifier.
        doc_title:
            Human-readable title (may be None if not yet known).
        institution_id:
            Institution that owns the document.
        performed_by:
            Identity string — ``"cli"`` for command-line ingestion,
            ``"role/institution_id"`` for API callers.
        details:
            Optional JSON-serialisable dict with extra context
            (e.g. access_level, source_file path).
        """
        if event_type not in _VALID_EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {event_type!r}")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO write_event
                    (event_id, event_type, doc_id, doc_title, institution_id,
                     performed_by, performed_at, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    event_type,
                    doc_id,
                    doc_title,
                    institution_id,
                    performed_by,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(details) if details else None,
                ),
            )
