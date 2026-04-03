"""Token usage tracking for Anthropic API calls.

Instruments API calls via a transparent ``MeteredAnthropicClient`` wrapper and
persists per-request records + daily aggregates to a dedicated SQLite database.
Follows the same daemon-thread / fire-and-forget pattern as
:class:`~graphrag_pipeline.retrieval.conversation_log.ConversationLogger`.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing loader
# ---------------------------------------------------------------------------

_DEFAULT_RESOURCES_DIR: Path = Path(__file__).parent.parent / "resources"

_HARDCODED_DEFAULT = {"input_per_million": 3.00, "output_per_million": 15.00}


def _model_key(model: str) -> str:
    """Convert a model name to an env-var key fragment.

    ``claude-sonnet-4-6`` → ``CLAUDE_SONNET_4_6``
    """
    return re.sub(r"[^A-Za-z0-9]", "_", model).upper()


def load_pricing(resources_dir: Path | None = None) -> dict[str, dict[str, float]]:
    """Load per-model token pricing.

    Resolution order (per model):
    1. ``TOKEN_PRICE_<MODEL_KEY>_INPUT`` / ``_OUTPUT`` env vars
    2. ``token_pricing.yaml`` entry under ``models.<model>``
    3. ``default`` block in the YAML
    4. Hardcoded fallback if the YAML is missing entirely
    """
    rdir = resources_dir or _DEFAULT_RESOURCES_DIR
    yaml_path = rdir / "token_pricing.yaml"

    yaml_models: dict[str, dict[str, float]] = {}
    yaml_default: dict[str, float] = dict(_HARDCODED_DEFAULT)

    try:
        with yaml_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        yaml_models = data.get("models", {})
        yaml_default = data.get("default", yaml_default)
    except FileNotFoundError:
        _log.warning("token_pricing.yaml not found at %s — using hardcoded defaults", yaml_path)
    except Exception:
        _log.warning("Failed to load token_pricing.yaml — using hardcoded defaults", exc_info=True)

    pricing: dict[str, dict[str, float]] = {}

    # Pre-populate all models declared in the YAML
    for model_name, rates in yaml_models.items():
        mk = _model_key(model_name)
        pricing[model_name] = {
            "input_per_million": float(
                os.environ.get(f"TOKEN_PRICE_{mk}_INPUT", rates.get("input_per_million", yaml_default["input_per_million"]))
            ),
            "output_per_million": float(
                os.environ.get(f"TOKEN_PRICE_{mk}_OUTPUT", rates.get("output_per_million", yaml_default["output_per_million"]))
            ),
        }

    # Store the default for unknown models
    pricing["__default__"] = {
        "input_per_million": float(os.environ.get("TOKEN_PRICE_DEFAULT_INPUT", yaml_default["input_per_million"])),
        "output_per_million": float(os.environ.get("TOKEN_PRICE_DEFAULT_OUTPUT", yaml_default["output_per_million"])),
    }
    return pricing


def _get_rates(pricing: dict[str, dict[str, float]], model: str) -> dict[str, float]:
    """Look up pricing for *model*, falling back to the default block."""
    if model in pricing:
        return pricing[model]
    # Try env-var override for an unknown model
    mk = _model_key(model)
    env_in = os.environ.get(f"TOKEN_PRICE_{mk}_INPUT")
    env_out = os.environ.get(f"TOKEN_PRICE_{mk}_OUTPUT")
    if env_in is not None and env_out is not None:
        return {"input_per_million": float(env_in), "output_per_million": float(env_out)}
    return pricing.get("__default__", _HARDCODED_DEFAULT)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TokenUsageRecord:
    """Lightweight record emitted by :class:`MeteredAnthropicClient`."""

    caller: str  # "synthesis" | "claim_extraction" | "claim_cache"
    model: str
    input_tokens: int
    output_tokens: int
    cache_hit: bool = False
    stop_reason: str = ""
    request_id: str | None = None
    institution_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS token_usage (
    usage_id        TEXT PRIMARY KEY,
    caller          TEXT NOT NULL,
    model           TEXT NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    total_tokens    INTEGER NOT NULL,
    input_cost_usd  REAL NOT NULL,
    output_cost_usd REAL NOT NULL,
    total_cost_usd  REAL NOT NULL,
    cache_hit       INTEGER NOT NULL DEFAULT 0,
    stop_reason     TEXT NOT NULL DEFAULT '',
    request_id      TEXT,
    institution_id  TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_token_usage_caller ON token_usage(caller);
CREATE INDEX IF NOT EXISTS idx_token_usage_model ON token_usage(model);
CREATE INDEX IF NOT EXISTS idx_token_usage_created ON token_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_token_usage_institution ON token_usage(institution_id);

CREATE TABLE IF NOT EXISTS token_usage_daily (
    date_key        TEXT NOT NULL,
    caller          TEXT NOT NULL,
    model           TEXT NOT NULL,
    institution_id  TEXT NOT NULL DEFAULT '',
    total_requests  INTEGER NOT NULL DEFAULT 0,
    total_input     INTEGER NOT NULL DEFAULT 0,
    total_output    INTEGER NOT NULL DEFAULT 0,
    total_cost_usd  REAL NOT NULL DEFAULT 0.0,
    cache_hits      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date_key, caller, model, institution_id)
);

CREATE TABLE IF NOT EXISTS token_budget (
    budget_id       TEXT PRIMARY KEY,
    period          TEXT NOT NULL DEFAULT 'daily',
    institution_id  TEXT NOT NULL DEFAULT '',
    max_cost_usd    REAL NOT NULL,
    alert_threshold REAL NOT NULL DEFAULT 0.80,
    enforcement     TEXT NOT NULL DEFAULT 'warn',
    created_at      TEXT NOT NULL
);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Writer helpers
# ---------------------------------------------------------------------------

def _write_record(
    conn: sqlite3.Connection,
    record: TokenUsageRecord,
    pricing: dict[str, dict[str, float]],
) -> None:
    rates = _get_rates(pricing, record.model)
    input_cost = record.input_tokens * rates["input_per_million"] / 1_000_000
    output_cost = record.output_tokens * rates["output_per_million"] / 1_000_000
    total_tokens = record.input_tokens + record.output_tokens
    total_cost = input_cost + output_cost

    usage_id = str(uuid.uuid4())
    date_key = record.created_at[:10]  # "2026-04-02"

    conn.execute(
        """INSERT INTO token_usage
           (usage_id, caller, model, input_tokens, output_tokens, total_tokens,
            input_cost_usd, output_cost_usd, total_cost_usd, cache_hit,
            stop_reason, request_id, institution_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            usage_id, record.caller, record.model,
            record.input_tokens, record.output_tokens, total_tokens,
            input_cost, output_cost, total_cost,
            1 if record.cache_hit else 0,
            record.stop_reason, record.request_id,
            record.institution_id, record.created_at,
        ),
    )

    conn.execute(
        """INSERT INTO token_usage_daily
           (date_key, caller, model, institution_id,
            total_requests, total_input, total_output, total_cost_usd, cache_hits)
           VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
           ON CONFLICT(date_key, caller, model, institution_id) DO UPDATE SET
              total_requests = total_requests + 1,
              total_input    = total_input + excluded.total_input,
              total_output   = total_output + excluded.total_output,
              total_cost_usd = total_cost_usd + excluded.total_cost_usd,
              cache_hits     = cache_hits + excluded.cache_hits""",
        (
            date_key, record.caller, record.model, record.institution_id,
            record.input_tokens, record.output_tokens, total_cost,
            1 if record.cache_hit else 0,
        ),
    )
    conn.commit()


def _check_budget(
    conn: sqlite3.Connection,
    record: TokenUsageRecord,
    budget_callback: Callable[[str, str, float, float], None] | None,
) -> None:
    """Check daily/monthly budgets after a write and fire callback if exceeded."""
    if budget_callback is None:
        return
    date_key = record.created_at[:10]
    month_key = record.created_at[:7]  # "2026-04"
    inst = record.institution_id

    rows = conn.execute(
        "SELECT budget_id, period, max_cost_usd, alert_threshold, enforcement "
        "FROM token_budget WHERE institution_id = ?",
        (inst,),
    ).fetchall()

    for budget_id, period, max_cost, threshold, enforcement in rows:
        if period == "daily":
            row = conn.execute(
                "SELECT COALESCE(SUM(total_cost_usd), 0) FROM token_usage_daily "
                "WHERE date_key = ? AND institution_id = ?",
                (date_key, inst),
            ).fetchone()
        elif period == "monthly":
            row = conn.execute(
                "SELECT COALESCE(SUM(total_cost_usd), 0) FROM token_usage_daily "
                "WHERE date_key LIKE ? AND institution_id = ?",
                (month_key + "%", inst),
            ).fetchone()
        else:
            continue

        current_cost = row[0] if row else 0.0
        if current_cost >= max_cost * threshold:
            budget_callback(budget_id, period, current_cost, max_cost)


# ---------------------------------------------------------------------------
# TokenUsageLogger (daemon-thread writer)
# ---------------------------------------------------------------------------

class TokenUsageLogger:
    """Asynchronous, fire-and-forget logger for token usage records.

    Follows the same bounded-queue / daemon-thread pattern as
    :class:`~graphrag_pipeline.retrieval.conversation_log.ConversationLogger`.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created on first use.
    pricing:
        Model pricing dict from :func:`load_pricing`.
    maxsize:
        Maximum number of records buffered in memory.  When full, the oldest
        record is discarded rather than blocking the caller.
    """

    def __init__(
        self,
        db_path: Path,
        pricing: dict[str, dict[str, float]],
        maxsize: int = 200,
        *,
        skip_init: bool = False,
    ) -> None:
        self._db_path = db_path
        self._skip_init = skip_init
        if not skip_init:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pricing = pricing
        self._queue: queue.Queue[TokenUsageRecord] = queue.Queue(maxsize=maxsize)
        self._budget_callback: Callable[[str, str, float, float], None] | None = _default_budget_callback
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="token-usage-writer")
        self._thread.start()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def set_budget_callback(self, cb: Callable[[str, str, float, float], None] | None) -> None:
        """Set a callback ``(budget_id, period, current_cost, max_cost)`` for budget alerts."""
        self._budget_callback = cb

    def enqueue(self, record: TokenUsageRecord) -> None:
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
        if not self._skip_init:
            _init_db(conn)
        while True:
            try:
                record = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                _write_record(conn, record, self._pricing)
                _check_budget(conn, record, self._budget_callback)
            except Exception:  # noqa: BLE001 — never crash the writer thread
                _log.debug("token_tracker: failed to write record", exc_info=True)
            finally:
                self._queue.task_done()


def _default_budget_callback(budget_id: str, period: str, current: float, limit: float) -> None:
    pct = (current / limit * 100) if limit else 0
    _log.warning(
        "Token budget alert [%s/%s]: $%.4f / $%.2f (%.1f%%)",
        budget_id, period, current, limit, pct,
    )


# ---------------------------------------------------------------------------
# TokenUsageStore (read-only access for API endpoints)
# ---------------------------------------------------------------------------

class TokenUsageStore:
    """Read-optimised access to the token usage database.

    Opens its own WAL-mode connection so reads never block the writer thread.
    """

    def __init__(self, db_path: Path | str, *, conn: sqlite3.Connection | None = None) -> None:
        self._db_path = Path(db_path)
        if conn is not None:
            self._conn = conn
            return
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def today_summary(self, institution_id: str = "") -> dict[str, Any]:
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._period_summary(date_key, date_key, institution_id)

    def month_summary(self, institution_id: str = "") -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        month_start = now.strftime("%Y-%m-01")
        month_end = now.strftime("%Y-%m-%d")
        return self._period_summary(month_start, month_end, institution_id)

    def _period_summary(self, start: str, end: str, institution_id: str) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT caller, "
            "  SUM(total_requests) AS total_requests, "
            "  SUM(total_input) AS total_input, "
            "  SUM(total_output) AS total_output, "
            "  SUM(total_cost_usd) AS total_cost_usd, "
            "  SUM(cache_hits) AS cache_hits "
            "FROM token_usage_daily "
            "WHERE date_key >= ? AND date_key <= ? AND institution_id = ? "
            "GROUP BY caller",
            (start, end, institution_id),
        ).fetchall()

        by_caller: dict[str, Any] = {}
        total_cost = 0.0
        total_requests = 0
        total_input = 0
        total_output = 0
        total_cache_hits = 0

        for row in rows:
            caller = row["caller"]
            cost = row["total_cost_usd"] or 0.0
            reqs = row["total_requests"] or 0
            inp = row["total_input"] or 0
            out = row["total_output"] or 0
            hits = row["cache_hits"] or 0
            by_caller[caller] = {"cost": round(cost, 4), "requests": reqs}
            total_cost += cost
            total_requests += reqs
            total_input += inp
            total_output += out
            total_cache_hits += hits

        return {
            "total_cost_usd": round(total_cost, 4),
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "cache_hits": total_cache_hits,
            "by_caller": by_caller,
        }

    def budget_status(self, institution_id: str = "") -> dict[str, Any]:
        today = self.today_summary(institution_id)
        month = self.month_summary(institution_id)
        budgets = self._conn.execute(
            "SELECT period, max_cost_usd, alert_threshold, enforcement "
            "FROM token_budget WHERE institution_id = ?",
            (institution_id,),
        ).fetchall()

        status: dict[str, Any] = {}
        for row in budgets:
            period = row["period"]
            limit_usd = row["max_cost_usd"]
            used = today["total_cost_usd"] if period == "daily" else month["total_cost_usd"]
            pct = (used / limit_usd * 100) if limit_usd else 0
            status[period] = {
                "limit": limit_usd,
                "used": round(used, 4),
                "pct": round(pct, 1),
                "alert": pct >= row["alert_threshold"] * 100,
                "enforcement": row["enforcement"],
            }
        return status

    def history(
        self,
        start: str,
        end: str,
        caller: str = "",
        institution_id: str = "",
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT date_key, caller, model, total_requests, "
            "  total_input, total_output, total_cost_usd, cache_hits "
            "FROM token_usage_daily "
            "WHERE date_key >= ? AND date_key <= ? AND institution_id = ?"
        )
        params: list[Any] = [start, end, institution_id]
        if caller:
            sql += " AND caller = ?"
            params.append(caller)
        sql += " ORDER BY date_key DESC, caller, model"
        return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# MeteredAnthropicClient (transparent wrapper)
# ---------------------------------------------------------------------------

class _MeteredMessages:
    """Proxy for ``client.messages`` that meters ``create()`` calls."""

    def __init__(
        self,
        real_messages: Any,
        logger: TokenUsageLogger,
        caller: str,
        institution_id: str,
        request_id_func: Callable[[], str | None] | None,
    ) -> None:
        self._real = real_messages
        self._logger = logger
        self._caller = caller
        self._institution_id = institution_id
        self._request_id_func = request_id_func

    def create(self, **kwargs: Any) -> Any:
        response = self._real.create(**kwargs)
        try:
            usage = response.usage
            self._logger.enqueue(TokenUsageRecord(
                caller=self._caller,
                model=response.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_hit=False,
                stop_reason=response.stop_reason or "",
                request_id=self._request_id_func() if self._request_id_func else None,
                institution_id=self._institution_id,
            ))
        except Exception:  # noqa: BLE001 — metering must never break the call
            _log.debug("token_tracker: failed to record usage", exc_info=True)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class MeteredAnthropicClient:
    """Drop-in wrapper around ``anthropic.Anthropic`` that meters token usage.

    Transparently proxies all attribute access to the underlying client.  Only
    ``messages.create()`` is intercepted to read the ``usage`` field from the
    response and enqueue a :class:`TokenUsageRecord`.

    Parameters
    ----------
    client:
        The real ``anthropic.Anthropic`` instance.
    logger:
        :class:`TokenUsageLogger` to receive usage records.
    caller:
        Tag for the call site (``"synthesis"``, ``"claim_extraction"``).
    institution_id:
        Default institution ID for tenant isolation.
    request_id_func:
        Optional callable returning the current request ID (for correlation
        with conversation logs).
    """

    def __init__(
        self,
        client: Any,
        logger: TokenUsageLogger,
        caller: str,
        institution_id: str = "",
        request_id_func: Callable[[], str | None] | None = None,
    ) -> None:
        self._client = client
        self._logger = logger
        self._caller = caller
        self._institution_id = institution_id
        self._request_id_func = request_id_func
        self._messages = _MeteredMessages(
            client.messages, logger, caller, institution_id, request_id_func,
        )

    @property
    def messages(self) -> _MeteredMessages:
        return self._messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Cache-hit helper
# ---------------------------------------------------------------------------

def log_cache_hit(
    logger: TokenUsageLogger,
    model: str,
    caller: str = "claim_cache",
    institution_id: str = "",
    request_id: str | None = None,
) -> None:
    """Emit a zero-cost record for a claim-cache hit."""
    logger.enqueue(TokenUsageRecord(
        caller=caller,
        model=model,
        input_tokens=0,
        output_tokens=0,
        cache_hit=True,
        stop_reason="cache_hit",
        request_id=request_id,
        institution_id=institution_id,
    ))
