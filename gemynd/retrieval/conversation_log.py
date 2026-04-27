"""Fire-and-forget conversation logger and query history store for the retrieval pipeline.

Captures the full causal chain per query — conversation → retrieval event →
claim interactions — without blocking the response path.  Records are assembled
from data already in flight and handed to a bounded background writer that
drains to SQLite.

Schema lives in a separate ``conversation_log.db`` file (path resolved from the
``CONV_LOG_DB`` env var, defaulting to ``data/conversation_log.db`` relative to
the working directory) so the append-only query log stays isolated from
``data/review.db``.

``QueryHistoryStore`` opens its own read/write connection to the same database
for history browsing and saved-search management.
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
    from gemynd.core.ids import stable_hash
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
    # session tracking
    session_id: str | None = None    # client-generated; groups multi-turn exchanges
    turn_number: int = 1             # 1-based position within the session
    # request tracing
    request_id: str | None = None    # UUID generated per HTTP request by the middleware
    # conversation cache: authenticated owner + full post-PII-redaction response JSON,
    # used to replay past conversations in the chat UI without re-calling the LLM.
    user_id: str | None = None
    response_json: str | None = None
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
    created_at            TEXT NOT NULL,
    session_id            TEXT,
    turn_number           INTEGER NOT NULL DEFAULT 1,
    request_id            TEXT,
    user_id               TEXT,
    response_json         TEXT
);

CREATE INDEX IF NOT EXISTS idx_conversation_user_session
    ON conversation (user_id, session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_conversation_created_at
    ON conversation (created_at);

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
    # Migrate existing databases that predate newer columns.
    for col, ddl in (
        ("session_id", "ALTER TABLE conversation ADD COLUMN session_id TEXT"),
        ("turn_number", "ALTER TABLE conversation ADD COLUMN turn_number INTEGER NOT NULL DEFAULT 1"),
        ("request_id", "ALTER TABLE conversation ADD COLUMN request_id TEXT"),
        ("user_id", "ALTER TABLE conversation ADD COLUMN user_id TEXT"),
        ("response_json", "ALTER TABLE conversation ADD COLUMN response_json TEXT"),
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_record(conn: sqlite3.Connection, record: LogRecord) -> None:
    conn.execute("BEGIN")
    conn.execute(
        """INSERT OR IGNORE INTO conversation
           (conversation_id, query_text, bucket, classifier_confidence,
            year_min, year_max, retrieval_path, created_at,
            session_id, turn_number, request_id, user_id, response_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record.conversation_id,
            record.query_text,
            record.bucket,
            record.classifier_confidence,
            record.year_min,
            record.year_max,
            record.retrieval_path,
            record.created_at,
            record.session_id,
            record.turn_number,
            record.request_id,
            record.user_id,
            record.response_json,
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

    def __init__(self, db_path: Path, maxsize: int = 200, *, skip_init: bool = False) -> None:
        self._db_path = db_path
        self._skip_init = skip_init
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
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        if not self._skip_init:
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


# ---------------------------------------------------------------------------
# Saved-search table (added to the same conversation_log.db)
# ---------------------------------------------------------------------------

_SAVED_SEARCH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS saved_search (
    saved_id        TEXT    NOT NULL PRIMARY KEY,
    conversation_id TEXT,
    query_text      TEXT    NOT NULL,
    label           TEXT    NOT NULL DEFAULT '',
    bucket          TEXT    NOT NULL DEFAULT '',
    year_min        INTEGER,
    year_max        INTEGER,
    created_by      TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_saved_search_user
    ON saved_search (created_by, created_at DESC);
"""


class QueryHistoryStore:
    """Read-optimised access to the conversation log plus saved-search management.

    Opens its own SQLite connection to the same ``conversation_log.db`` used by
    :class:`ConversationLogger`.  WAL mode allows concurrent reads alongside the
    logger's background write thread without blocking.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created on first use.
    """

    def __init__(self, db_path: "str | Path", *, conn: sqlite3.Connection | None = None) -> None:
        if conn is not None:
            self._conn = conn
            self._lock = threading.Lock()
            return
        _p = Path(db_path)
        _p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(_p), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Ensure base tables exist (harmless if ConversationLogger already created them).
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.executescript(_SAVED_SEARCH_SCHEMA_SQL)
        self._conn.commit()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # History reads
    # ------------------------------------------------------------------

    def list_queries(
        self,
        limit: int = 50,
        offset: int = 0,
        q: str = "",
        bucket: str = "",
    ) -> "list[dict]":
        """Return recent queries, newest first, with optional filtering."""
        filters: list[str] = []
        params: list = []
        if q:
            filters.append("query_text LIKE ?")
            params.append(f"%{q}%")
        if bucket:
            filters.append("bucket = ?")
            params.append(bucket)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        params += [max(1, min(200, limit)), max(0, offset)]
        rows = self._conn.execute(
            f"SELECT conversation_id, query_text, bucket, classifier_confidence, "
            f"year_min, year_max, retrieval_path, created_at "
            f"FROM conversation {where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def count_queries(self, q: str = "", bucket: str = "") -> int:
        """Return the total number of rows matching the filters."""
        filters: list[str] = []
        params: list = []
        if q:
            filters.append("query_text LIKE ?")
            params.append(f"%{q}%")
        if bucket:
            filters.append("bucket = ?")
            params.append(bucket)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM conversation {where}", params
        ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Saved searches
    # ------------------------------------------------------------------

    def get_saved_searches(self, created_by: str = "") -> "list[dict]":
        """Return saved searches for *created_by*, or all if empty."""
        if created_by:
            rows = self._conn.execute(
                "SELECT * FROM saved_search WHERE created_by = ? ORDER BY created_at DESC",
                (created_by,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM saved_search ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def save_search(
        self,
        query_text: str,
        label: str,
        bucket: str,
        year_min: "int | None",
        year_max: "int | None",
        created_by: str,
        conversation_id: "str | None" = None,
    ) -> str:
        """Persist a saved search and return its ``saved_id``."""
        import uuid as _uuid
        from datetime import datetime as _dt, timezone as _tz

        saved_id = str(_uuid.uuid4())
        created_at = _dt.now(_tz.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO saved_search "
                "(saved_id, conversation_id, query_text, label, bucket, "
                "year_min, year_max, created_by, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    saved_id,
                    conversation_id,
                    query_text.strip(),
                    label.strip()[:200],
                    bucket,
                    year_min,
                    year_max,
                    created_by,
                    created_at,
                ),
            )
            self._conn.commit()
        return saved_id

    def delete_saved_search(self, saved_id: str, created_by: str) -> bool:
        """Delete a saved search owned by *created_by*.  Returns True if deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM saved_search WHERE saved_id = ? AND created_by = ?",
                (saved_id, created_by),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Conversation cache (per-user multi-turn history for chat replay)
    # ------------------------------------------------------------------

    def list_user_sessions(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> "list[dict]":
        """Return one row per session owned by *user_id*, most-recent first.

        Each dict: ``{session_id, title, turn_count, created_at, updated_at}``.
        ``title`` is the first turn's ``query_text`` truncated to 60 characters.
        Sessions with a NULL ``session_id`` (pre-feature rows) are excluded.
        """
        if not user_id:
            return []
        limit = max(1, min(200, limit))
        offset = max(0, offset)
        rows = self._conn.execute(
            """
            SELECT
                session_id,
                MIN(created_at) AS created_at,
                MAX(created_at) AS updated_at,
                COUNT(*)        AS turn_count
            FROM conversation
            WHERE user_id = ? AND session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        ).fetchall()
        result: list[dict] = []
        for r in rows:
            sid = r["session_id"]
            title_row = self._conn.execute(
                "SELECT query_text FROM conversation "
                "WHERE user_id = ? AND session_id = ? "
                "ORDER BY turn_number ASC, created_at ASC LIMIT 1",
                (user_id, sid),
            ).fetchone()
            title = (title_row["query_text"] if title_row else "") or ""
            if len(title) > 60:
                title = title[:57] + "..."
            result.append({
                "session_id": sid,
                "title": title,
                "turn_count": int(r["turn_count"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return result

    def get_session_turns(self, user_id: str, session_id: str) -> "list[dict]":
        """Return all turns for (user_id, session_id) ordered by turn_number.

        Each dict: ``{turn_number, query_text, response_json, created_at}``.
        Returns ``[]`` if the session is not owned by *user_id* — this is the
        only cross-user access control; callers must not look up by
        ``session_id`` alone.
        """
        if not user_id or not session_id:
            return []
        rows = self._conn.execute(
            """
            SELECT turn_number, query_text, response_json, created_at
            FROM conversation
            WHERE user_id = ? AND session_id = ?
            ORDER BY turn_number ASC, created_at ASC
            """,
            (user_id, session_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, user_id: str, session_id: str) -> int:
        """Delete every turn for (user_id, session_id) and cascade child rows.

        Returns the number of ``conversation`` rows deleted. Returns 0 (no-op)
        if the session is not owned by *user_id*, so callers can safely pass
        a client-supplied ``session_id`` without an explicit ownership check.
        """
        if not user_id or not session_id:
            return 0
        with self._lock:
            conv_ids = [
                row["conversation_id"]
                for row in self._conn.execute(
                    "SELECT conversation_id FROM conversation "
                    "WHERE user_id = ? AND session_id = ?",
                    (user_id, session_id),
                ).fetchall()
            ]
            if not conv_ids:
                return 0
            placeholders = ",".join("?" * len(conv_ids))
            self._conn.execute(
                f"DELETE FROM claim_interaction WHERE conversation_id IN ({placeholders})",
                conv_ids,
            )
            self._conn.execute(
                f"DELETE FROM retrieval_event WHERE conversation_id IN ({placeholders})",
                conv_ids,
            )
            cur = self._conn.execute(
                "DELETE FROM conversation WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            self._conn.commit()
            return int(cur.rowcount)

    def purge_older_than(self, cutoff_iso: str) -> int:
        """Delete every conversation (and cascaded children) older than *cutoff_iso*.

        Returns the number of ``conversation`` rows removed. Used by the
        retention daemon. Scans all users — not user-scoped — because
        retention is a system-wide policy.
        """
        with self._lock:
            conv_ids = [
                row["conversation_id"]
                for row in self._conn.execute(
                    "SELECT conversation_id FROM conversation WHERE created_at < ?",
                    (cutoff_iso,),
                ).fetchall()
            ]
            if not conv_ids:
                return 0
            # Chunk the IN-clause to avoid SQLite's default 999 host-param limit.
            chunk = 500
            for i in range(0, len(conv_ids), chunk):
                batch = conv_ids[i : i + chunk]
                placeholders = ",".join("?" * len(batch))
                self._conn.execute(
                    f"DELETE FROM claim_interaction WHERE conversation_id IN ({placeholders})",
                    batch,
                )
                self._conn.execute(
                    f"DELETE FROM retrieval_event WHERE conversation_id IN ({placeholders})",
                    batch,
                )
            cur = self._conn.execute(
                "DELETE FROM conversation WHERE created_at < ?",
                (cutoff_iso,),
            )
            self._conn.commit()
            return int(cur.rowcount)

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Retention purge daemon
# ---------------------------------------------------------------------------

def start_retention_purge(
    store: "QueryHistoryStore",
    *,
    days: int = 90,
    interval_hours: float = 24.0,
) -> threading.Thread:
    """Spawn a daemon thread that purges conversations older than *days*.

    Runs once immediately, then every *interval_hours*. Errors are swallowed
    so the loop never crashes — this is an opportunistic retention sweep, not
    a correctness mechanism.
    """
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    def _loop() -> None:
        while True:
            try:
                cutoff = (_dt.now(_tz.utc) - _td(days=days)).isoformat()
                store.purge_older_than(cutoff)
            except Exception:  # noqa: BLE001
                pass
            # threading.Event().wait allows a clean interpreter shutdown.
            threading.Event().wait(timeout=interval_hours * 3600.0)

    t = threading.Thread(target=_loop, daemon=True, name="conv-log-retention")
    t.start()
    return t
