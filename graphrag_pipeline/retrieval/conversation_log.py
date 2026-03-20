"""Fire-and-forget conversation logger for the retrieval pipeline.

Captures the full causal chain per query — conversation → retrieval event →
claim interactions — without blocking the response path.  Records are assembled
from data already in flight and handed to a bounded background writer that
drains to SQLite.

Schema lives in a separate ``conversation_log.db`` file (path resolved from the
``CONV_LOG_DB`` env var, defaulting to ``conversation_log.db`` in the working
directory) so the append-only query log stays isolated from ``review.db``.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def make_conversation_id(query_text: str, created_at: str) -> str:
    from ..ids import stable_hash
    return f"conv_{stable_hash(query_text, created_at)}"


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class ClaimInteraction:
    """One row in the claim_interaction table."""

    claim_id: str
    claim_type: str
    traversal_rel_types: list[str]   # empty list for fulltext-path claims
    was_cited: bool
    extraction_confidence: float


@dataclass
class LogRecord:
    """Everything needed to write all three log tables for one query."""

    conversation_id: str
    query_text: str
    bucket: str
    classifier_confidence: float
    year_min: int | None
    year_max: int | None
    retrieval_path: str              # "entity_anchored" or "fulltext"
    created_at: str                  # ISO-8601 UTC
    # retrieval_event fields
    entity_ids_resolved: list[str]
    entity_types_resolved: list[str]
    candidates_retrieved: int
    ocr_dropped: int
    claims_in_context: int
    # per-claim rows
    claim_interactions: list[ClaimInteraction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversation (
    conversation_id       TEXT PRIMARY KEY,
    query_text            TEXT NOT NULL,
    bucket                TEXT NOT NULL,
    classifier_confidence REAL NOT NULL,
    year_min              INTEGER,
    year_max              INTEGER,
    retrieval_path        TEXT NOT NULL,
    created_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_event (
    conversation_id       TEXT PRIMARY KEY REFERENCES conversation(conversation_id),
    entity_ids_resolved   TEXT NOT NULL,
    entity_types_resolved TEXT NOT NULL,
    candidates_retrieved  INTEGER NOT NULL,
    ocr_dropped           INTEGER NOT NULL,
    claims_in_context     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_interaction (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id       TEXT NOT NULL REFERENCES conversation(conversation_id),
    claim_id              TEXT NOT NULL,
    claim_type            TEXT NOT NULL,
    traversal_rel_types   TEXT NOT NULL,
    was_cited             INTEGER NOT NULL DEFAULT 0,
    extraction_confidence REAL NOT NULL
);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_record(conn: sqlite3.Connection, record: LogRecord) -> None:
    conn.execute("BEGIN")
    conn.execute(
        """INSERT OR IGNORE INTO conversation
           (conversation_id, query_text, bucket, classifier_confidence,
            year_min, year_max, retrieval_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record.conversation_id,
            record.query_text,
            record.bucket,
            record.classifier_confidence,
            record.year_min,
            record.year_max,
            record.retrieval_path,
            record.created_at,
        ),
    )
    conn.execute(
        """INSERT OR IGNORE INTO retrieval_event
           (conversation_id, entity_ids_resolved, entity_types_resolved,
            candidates_retrieved, ocr_dropped, claims_in_context)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            record.conversation_id,
            json.dumps(record.entity_ids_resolved),
            json.dumps(record.entity_types_resolved),
            record.candidates_retrieved,
            record.ocr_dropped,
            record.claims_in_context,
        ),
    )
    for ci in record.claim_interactions:
        conn.execute(
            """INSERT INTO claim_interaction
               (conversation_id, claim_id, claim_type, traversal_rel_types,
                was_cited, extraction_confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record.conversation_id,
                ci.claim_id,
                ci.claim_type,
                json.dumps(ci.traversal_rel_types),
                1 if ci.was_cited else 0,
                ci.extraction_confidence,
            ),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# ConversationLogger
# ---------------------------------------------------------------------------

class ConversationLogger:
    """Bounded, fire-and-forget SQLite writer for query log records.

    Records are enqueued from the request path (non-blocking) and drained by a
    daemon thread.  When the queue is full the oldest record is dropped to make
    room — query logs are statistical signal, not financial records.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created on first use.
    maxsize:
        Maximum number of records buffered in memory.  When full, the oldest
        record is discarded rather than blocking the caller.
    """

    def __init__(self, db_path: Path, maxsize: int = 200) -> None:
        self._db_path = db_path
        self._queue: queue.Queue[LogRecord] = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="conv-log-writer")
        self._thread.start()

    def enqueue(self, record: LogRecord) -> None:
        """Add *record* to the write queue (non-blocking).

        Drops the oldest buffered record if the queue is full rather than
        blocking or raising.
        """
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                pass  # accept the loss

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _init_db(conn)
        while True:
            try:
                record = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                _write_record(conn, record)
            except Exception:  # noqa: BLE001 — never crash the writer thread
                pass
            finally:
                self._queue.task_done()
