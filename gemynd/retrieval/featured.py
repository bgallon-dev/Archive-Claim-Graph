"""Featured questions + curated answer cache.

Two surfaces, one store:

* **Suggested questions** — a curator-maintained set of starter questions shown
  on the empty chat landing, so a walk-up visitor is not staring at a blank box.
* **Curated answer cache** — when a question the curator has *captured* is asked
  (tapped as a suggestion, or typed verbatim), the vetted answer is returned
  instantly instead of running the ~100s retrieval+synthesis pipeline.

Matching is intentionally **exact on a normalised form** (lowercase, collapsed
whitespace, trailing punctuation stripped). Tapping a suggestion sends the exact
text, so it always hits; a typed paraphrase misses and falls through to the live
pipeline. This can never serve a *wrong* cached answer — the worst case is a
miss, which is just the current behaviour.

An empty store = exactly today's behaviour: no suggestions, no cache hits. The
store is populated out of band by the ``featured`` CLI (see cli.py), which
captures a vetted answer from the conversation log — nothing in the request path
writes to it.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS featured (
    question_norm    TEXT PRIMARY KEY,
    question_display TEXT NOT NULL,
    response_json    TEXT,               -- NULL until an answer is captured
    sort_order       INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT NOT NULL
);
"""

_WS_RE = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Fold a question to its match key: lowercased, whitespace-collapsed, with
    trailing punctuation removed. Deterministic and dependency-free so the CLI
    and the request path agree."""
    t = _WS_RE.sub(" ", (text or "").strip().lower())
    return t.rstrip(" ?!.")


class FeaturedStore:
    """SQLite-backed featured-question store. WAL + check_same_thread=False, per
    the repo convention (see shared/CLAUDE.md)."""

    def __init__(self, db_path: str | Path = "data/featured.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # NB: `with sqlite3.connect(...)` commits but does NOT close the
        # connection — so every call must be wrapped in closing() to avoid
        # leaking connections (and, on Windows, locking the file open).
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # -- request path (read-only) --------------------------------------------
    def list_questions(self) -> list[dict[str, Any]]:
        """Suggestions in curator order. `cached` marks the ones that answer
        instantly (an answer has been captured)."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT question_display, response_json FROM featured "
                "ORDER BY sort_order, question_display"
            ).fetchall()
        return [
            {"display": r["question_display"], "cached": r["response_json"] is not None}
            for r in rows
        ]

    def get_cached(self, text: str) -> dict | None:
        """Return the captured response dict for *text*, or None. None means
        'run the query live' — never an error, so a bad row cannot break a
        query (a JSON-decode failure is logged and treated as a miss)."""
        norm = normalize_question(text)
        if not norm:
            return None
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT response_json FROM featured "
                "WHERE question_norm = ? AND response_json IS NOT NULL",
                (norm,),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["response_json"])
        except (json.JSONDecodeError, TypeError):
            _log.warning("featured: cached response for %r is not valid JSON; miss", norm)
            return None

    # -- curator path (write; called by the CLI, not the request path) -------
    def add(self, display: str, sort_order: int = 0) -> None:
        """Add/keep a suggestion. Preserves an existing captured answer."""
        norm = normalize_question(display)
        if not norm:
            raise ValueError("question is empty after normalisation")
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO featured (question_norm, question_display, sort_order, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(question_norm) DO UPDATE SET "
                "  question_display = excluded.question_display, "
                "  sort_order = excluded.sort_order, "
                "  updated_at = excluded.updated_at",
                (norm, display.strip(), sort_order, now),
            )
            conn.commit()

    def capture(self, display: str, response_json: str) -> None:
        """Store the vetted answer for *display* so it answers instantly."""
        norm = normalize_question(display)
        if not norm:
            raise ValueError("question is empty after normalisation")
        # Validate it parses now, so a malformed capture fails loudly at the CLI
        # rather than silently missing later in the request path.
        json.loads(response_json)
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO featured (question_norm, question_display, response_json, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(question_norm) DO UPDATE SET "
                "  question_display = excluded.question_display, "
                "  response_json = excluded.response_json, "
                "  updated_at = excluded.updated_at",
                (norm, display.strip(), response_json, now),
            )
            conn.commit()

    def remove(self, display: str) -> bool:
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "DELETE FROM featured WHERE question_norm = ?",
                (normalize_question(display),),
            )
            conn.commit()
            return cur.rowcount > 0
