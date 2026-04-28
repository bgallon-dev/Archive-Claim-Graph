"""Tests for per-user cached conversation history.

Covers the new ``QueryHistoryStore`` methods (``list_user_sessions``,
``get_session_turns``, ``delete_session``, ``purge_older_than``) plus the
idempotent schema migration. These let the chat UI replay past conversations
without re-calling the LLM.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from gemynd.retrieval.conversation_log import (
    LogRecord,
    QueryHistoryStore,
    _init_db,
    _write_record,
)


def _make_record(
    *,
    conversation_id: str,
    query_text: str,
    created_at: str,
    session_id: str,
    turn_number: int,
    user_id: str | None,
    response_json: str | None,
) -> LogRecord:
    return LogRecord(
        conversation_id=conversation_id,
        query_text=query_text,
        bucket="conversational",
        classifier_confidence=0.9,
        year_min=None,
        year_max=None,
        retrieval_path="fulltext",
        created_at=created_at,
        entity_ids_resolved=[],
        entity_types_resolved=[],
        candidates_retrieved=1,
        ocr_dropped=0,
        claims_in_context=1,
        session_id=session_id,
        turn_number=turn_number,
        request_id=None,
        user_id=user_id,
        response_json=response_json,
        claim_interactions=[],
    )


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "conv_log.db"
    # Pre-init schema via a writer-style connection, then hand a separate
    # QueryHistoryStore connection the same file — this mirrors how the
    # FastAPI app wires the daemon writer alongside the reader store.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_db(conn)
    s = QueryHistoryStore(db_path)
    # Seed directly via _write_record through the same file.
    def seed(rec: LogRecord) -> None:
        _write_record(conn, rec)
    s._seed = seed  # type: ignore[attr-defined]
    yield s
    conn.close()
    s.close()


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Schema migration is idempotent
# ---------------------------------------------------------------------------

def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "conv_log.db"
    conn = sqlite3.connect(str(db_path))
    _init_db(conn)
    _init_db(conn)  # running twice should not raise
    # Confirm the new columns exist.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(conversation)").fetchall()}
    assert "user_id" in cols
    assert "response_json" in cols
    conn.close()


# ---------------------------------------------------------------------------
# list_user_sessions
# ---------------------------------------------------------------------------

def test_list_user_sessions_groups_by_session_and_sorts_newest_first(store):
    base = datetime.now(timezone.utc)
    # Session A, user "alice", two turns.
    store._seed(_make_record(
        conversation_id="conv_a1",
        query_text="what species are at Turnbull?",
        created_at=_iso(base - timedelta(minutes=10)),
        session_id="sess_A",
        turn_number=1,
        user_id="alice",
        response_json='{"answer": "Many."}',
    ))
    store._seed(_make_record(
        conversation_id="conv_a2",
        query_text="any waterfowl?",
        created_at=_iso(base - timedelta(minutes=9)),
        session_id="sess_A",
        turn_number=2,
        user_id="alice",
        response_json='{"answer": "Mallards, pintails."}',
    ))
    # Session B, user "alice", newer — should appear first.
    store._seed(_make_record(
        conversation_id="conv_b1",
        query_text="which refuges had fires in the 1990s?",
        created_at=_iso(base - timedelta(minutes=1)),
        session_id="sess_B",
        turn_number=1,
        user_id="alice",
        response_json='{"answer": "Several."}',
    ))
    # Session C, different user — must not appear in alice's list.
    store._seed(_make_record(
        conversation_id="conv_c1",
        query_text="bob's private query",
        created_at=_iso(base),
        session_id="sess_C",
        turn_number=1,
        user_id="bob",
        response_json='{"answer": "secret"}',
    ))

    sessions = store.list_user_sessions("alice")
    assert [s["session_id"] for s in sessions] == ["sess_B", "sess_A"]
    assert sessions[0]["turn_count"] == 1
    assert sessions[1]["turn_count"] == 2
    # Title = first-turn query_text (truncated to 60 chars).
    assert sessions[1]["title"].startswith("what species are at Turnbull?")


def test_list_user_sessions_truncates_long_titles(store):
    long_q = "a" * 100
    store._seed(_make_record(
        conversation_id="conv_long",
        query_text=long_q,
        created_at=_iso(datetime.now(timezone.utc)),
        session_id="sess_long",
        turn_number=1,
        user_id="alice",
        response_json=None,
    ))
    sessions = store.list_user_sessions("alice")
    assert len(sessions[0]["title"]) <= 60
    assert sessions[0]["title"].endswith("...")


def test_list_user_sessions_empty_for_unknown_user(store):
    assert store.list_user_sessions("ghost") == []
    assert store.list_user_sessions("") == []


# ---------------------------------------------------------------------------
# get_session_turns
# ---------------------------------------------------------------------------

def test_get_session_turns_returns_ordered_turns_with_response_json(store):
    base = datetime.now(timezone.utc)
    for i in range(1, 4):
        store._seed(_make_record(
            conversation_id=f"conv_s_{i}",
            query_text=f"turn {i}",
            created_at=_iso(base + timedelta(seconds=i)),
            session_id="sess_S",
            turn_number=i,
            user_id="alice",
            response_json=f'{{"answer": "reply {i}"}}',
        ))

    turns = store.get_session_turns("alice", "sess_S")
    assert [t["turn_number"] for t in turns] == [1, 2, 3]
    assert [t["query_text"] for t in turns] == ["turn 1", "turn 2", "turn 3"]
    assert turns[1]["response_json"] == '{"answer": "reply 2"}'


def test_get_session_turns_cross_user_isolation(store):
    """Cross-user access must return [] even with a correct session_id."""
    now = _iso(datetime.now(timezone.utc))
    store._seed(_make_record(
        conversation_id="conv_x1",
        query_text="alice's private turn",
        created_at=now,
        session_id="sess_shared_name",
        turn_number=1,
        user_id="alice",
        response_json='{"answer": "private"}',
    ))
    # Bob asks for alice's session_id by name — must see nothing.
    assert store.get_session_turns("bob", "sess_shared_name") == []
    # Alice herself still sees it.
    assert len(store.get_session_turns("alice", "sess_shared_name")) == 1


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------

def test_delete_session_removes_only_targeted_rows(store):
    now = _iso(datetime.now(timezone.utc))
    # Alice has two sessions.
    for sid, qid in (("sess_keep", "conv_k"), ("sess_drop", "conv_d")):
        store._seed(_make_record(
            conversation_id=qid,
            query_text=f"query for {sid}",
            created_at=now,
            session_id=sid,
            turn_number=1,
            user_id="alice",
            response_json='{"answer": "x"}',
        ))
    deleted = store.delete_session("alice", "sess_drop")
    assert deleted == 1
    # The other session survives.
    remaining = [s["session_id"] for s in store.list_user_sessions("alice")]
    assert remaining == ["sess_keep"]


def test_delete_session_no_op_for_foreign_owner(store):
    now = _iso(datetime.now(timezone.utc))
    store._seed(_make_record(
        conversation_id="conv_bob",
        query_text="bob's query",
        created_at=now,
        session_id="sess_B",
        turn_number=1,
        user_id="bob",
        response_json='{"answer": "bob"}',
    ))
    # Alice tries to delete bob's session — must be a silent no-op.
    assert store.delete_session("alice", "sess_B") == 0
    assert len(store.get_session_turns("bob", "sess_B")) == 1


# ---------------------------------------------------------------------------
# purge_older_than
# ---------------------------------------------------------------------------

def test_purge_older_than_drops_old_and_keeps_recent(store):
    now = datetime.now(timezone.utc)
    store._seed(_make_record(
        conversation_id="conv_old",
        query_text="ancient",
        created_at=_iso(now - timedelta(days=200)),
        session_id="sess_old",
        turn_number=1,
        user_id="alice",
        response_json='{"answer": "old"}',
    ))
    store._seed(_make_record(
        conversation_id="conv_new",
        query_text="fresh",
        created_at=_iso(now - timedelta(days=5)),
        session_id="sess_new",
        turn_number=1,
        user_id="alice",
        response_json='{"answer": "new"}',
    ))
    cutoff = _iso(now - timedelta(days=90))
    deleted = store.purge_older_than(cutoff)
    assert deleted == 1
    remaining = [s["session_id"] for s in store.list_user_sessions("alice")]
    assert remaining == ["sess_new"]


# ---------------------------------------------------------------------------
# Per-stage timing instrumentation (timing_json column on retrieval_event)
# ---------------------------------------------------------------------------

import json as _json


def test_timing_json_persisted_when_present(tmp_path):
    """LogRecord.timing_ms is serialised into retrieval_event.timing_json."""
    db_path = tmp_path / "conv_log.db"
    conn = sqlite3.connect(str(db_path))
    _init_db(conn)
    rec = _make_record(
        conversation_id="conv_timing_a",
        query_text="q",
        created_at="2026-04-28T00:00:00",
        session_id="sess_a",
        turn_number=1,
        user_id="alice",
        response_json=None,
    )
    rec.timing_ms = {"assemble_ms": 412.7, "synthesise_ms": 35210.5, "synthesise_ok": 1.0}
    _write_record(conn, rec)
    row = conn.execute(
        "SELECT timing_json FROM retrieval_event WHERE conversation_id = ?",
        ("conv_timing_a",),
    ).fetchone()
    conn.close()
    assert row[0] is not None
    parsed = _json.loads(row[0])
    assert parsed["assemble_ms"] == 412.7
    assert parsed["synthesise_ms"] == 35210.5
    assert parsed["synthesise_ok"] == 1.0


def test_timing_json_null_when_absent(tmp_path):
    """Empty timing_ms dict persists as NULL — preserves space + grep simplicity."""
    db_path = tmp_path / "conv_log.db"
    conn = sqlite3.connect(str(db_path))
    _init_db(conn)
    rec = _make_record(
        conversation_id="conv_timing_b",
        query_text="q",
        created_at="2026-04-28T00:00:00",
        session_id="sess_b",
        turn_number=1,
        user_id="alice",
        response_json=None,
    )
    # rec.timing_ms is the default empty dict; we don't override it
    _write_record(conn, rec)
    row = conn.execute(
        "SELECT timing_json FROM retrieval_event WHERE conversation_id = ?",
        ("conv_timing_b",),
    ).fetchone()
    conn.close()
    assert row[0] is None


def test_init_db_migrates_pre_existing_retrieval_event_table(tmp_path):
    """_init_db must add timing_json to retrieval_event tables that predate the column.

    Simulates the realistic upgrade path: the conversation table is already
    fully migrated (post all the user_id/session_id/etc. ALTERs), and only
    retrieval_event is missing the new timing_json column.
    """
    db_path = tmp_path / "conv_log.db"
    conn = sqlite3.connect(str(db_path))
    # Initialise the current schema first (this runs all conversation column
    # migrations), then drop and recreate retrieval_event WITHOUT timing_json
    # to simulate the gap.
    _init_db(conn)
    conn.execute("DROP TABLE retrieval_event")
    conn.execute(
        """CREATE TABLE retrieval_event (
            conversation_id TEXT PRIMARY KEY REFERENCES conversation(conversation_id),
            entity_ids_resolved TEXT NOT NULL,
            entity_types_resolved TEXT NOT NULL,
            candidates_retrieved INTEGER NOT NULL,
            ocr_dropped INTEGER NOT NULL,
            claims_in_context INTEGER NOT NULL
        )"""
    )
    conn.commit()
    cols_before = {r[1] for r in conn.execute("PRAGMA table_info(retrieval_event)").fetchall()}
    assert "timing_json" not in cols_before
    # Re-running _init_db should add the missing column via the ALTER migration.
    _init_db(conn)
    cols_after = {r[1] for r in conn.execute("PRAGMA table_info(retrieval_event)").fetchall()}
    assert "timing_json" in cols_after
    # Migration must be idempotent — running again must not error.
    _init_db(conn)
    conn.close()
