"""Tests for the token usage tracking module."""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graphrag_pipeline.shared.token_tracker import (
    MeteredAnthropicClient,
    TokenUsageLogger,
    TokenUsageRecord,
    TokenUsageStore,
    _get_rates,
    _model_key,
    load_pricing,
    log_cache_hit,
)


# ---------------------------------------------------------------------------
# Pricing tests
# ---------------------------------------------------------------------------


def test_model_key_normalization() -> None:
    assert _model_key("claude-sonnet-4-6") == "CLAUDE_SONNET_4_6"
    assert _model_key("claude-haiku-4-5") == "CLAUDE_HAIKU_4_5"
    assert _model_key("claude.opus.4.6") == "CLAUDE_OPUS_4_6"


def test_load_pricing_from_yaml() -> None:
    pricing = load_pricing()
    assert "claude-sonnet-4-6" in pricing
    assert pricing["claude-sonnet-4-6"]["input_per_million"] == 3.00
    assert pricing["claude-sonnet-4-6"]["output_per_million"] == 15.00
    assert "claude-haiku-4-5" in pricing
    assert pricing["claude-haiku-4-5"]["input_per_million"] == 0.80
    assert "__default__" in pricing


def test_load_pricing_env_override() -> None:
    with patch.dict(os.environ, {
        "TOKEN_PRICE_CLAUDE_SONNET_4_6_INPUT": "5.00",
        "TOKEN_PRICE_CLAUDE_SONNET_4_6_OUTPUT": "25.00",
    }):
        pricing = load_pricing()
        assert pricing["claude-sonnet-4-6"]["input_per_million"] == 5.00
        assert pricing["claude-sonnet-4-6"]["output_per_million"] == 25.00


def test_load_pricing_missing_yaml(tmp_path: Path) -> None:
    pricing = load_pricing(resources_dir=tmp_path)
    assert "__default__" in pricing
    assert pricing["__default__"]["input_per_million"] == 3.00


def test_get_rates_known_model() -> None:
    pricing = {"claude-sonnet-4-6": {"input_per_million": 3.0, "output_per_million": 15.0}, "__default__": {"input_per_million": 1.0, "output_per_million": 5.0}}
    rates = _get_rates(pricing, "claude-sonnet-4-6")
    assert rates["input_per_million"] == 3.0


def test_get_rates_unknown_model_falls_back() -> None:
    pricing = {"__default__": {"input_per_million": 1.0, "output_per_million": 5.0}}
    rates = _get_rates(pricing, "some-new-model")
    assert rates["input_per_million"] == 1.0


# ---------------------------------------------------------------------------
# TokenUsageRecord tests
# ---------------------------------------------------------------------------


def test_record_auto_created_at() -> None:
    record = TokenUsageRecord(caller="synthesis", model="claude-sonnet-4-6", input_tokens=100, output_tokens=50)
    assert record.created_at  # auto-populated


def test_record_explicit_created_at() -> None:
    record = TokenUsageRecord(caller="synthesis", model="m", input_tokens=0, output_tokens=0, created_at="2026-04-02T00:00:00+00:00")
    assert record.created_at == "2026-04-02T00:00:00+00:00"


# ---------------------------------------------------------------------------
# TokenUsageLogger tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def token_db(tmp_path: Path) -> Path:
    return tmp_path / "token_usage.db"


@pytest.fixture()
def pricing() -> dict:
    return {
        "claude-sonnet-4-6": {"input_per_million": 3.0, "output_per_million": 15.0},
        "claude-haiku-4-5": {"input_per_million": 0.8, "output_per_million": 4.0},
        "__default__": {"input_per_million": 3.0, "output_per_million": 15.0},
    }


def test_logger_writes_record(token_db: Path, pricing: dict) -> None:
    logger = TokenUsageLogger(token_db, pricing)
    record = TokenUsageRecord(
        caller="synthesis",
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=200,
        stop_reason="end_turn",
    )
    logger.enqueue(record)
    # Wait for daemon thread to process
    logger._queue.join()
    time.sleep(0.1)

    conn = sqlite3.connect(str(token_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM token_usage").fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["caller"] == "synthesis"
    assert row["model"] == "claude-sonnet-4-6"
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 200
    assert row["total_tokens"] == 1200
    assert row["cache_hit"] == 0
    # Cost: 1000 * 3.0 / 1_000_000 = 0.003, 200 * 15.0 / 1_000_000 = 0.003
    assert abs(row["input_cost_usd"] - 0.003) < 1e-6
    assert abs(row["output_cost_usd"] - 0.003) < 1e-6
    assert abs(row["total_cost_usd"] - 0.006) < 1e-6
    conn.close()


def test_logger_writes_daily_aggregate(token_db: Path, pricing: dict) -> None:
    logger = TokenUsageLogger(token_db, pricing)
    for i in range(3):
        logger.enqueue(TokenUsageRecord(
            caller="claim_extraction",
            model="claude-haiku-4-5",
            input_tokens=500,
            output_tokens=100,
        ))
    logger._queue.join()
    time.sleep(0.1)

    conn = sqlite3.connect(str(token_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM token_usage_daily").fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["total_requests"] == 3
    assert row["total_input"] == 1500
    assert row["total_output"] == 300
    conn.close()


def test_logger_cache_hit_record(token_db: Path, pricing: dict) -> None:
    logger = TokenUsageLogger(token_db, pricing)
    log_cache_hit(logger, model="claude-haiku-4-5")
    logger._queue.join()
    time.sleep(0.1)

    conn = sqlite3.connect(str(token_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM token_usage").fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["cache_hit"] == 1
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 0
    assert row["total_cost_usd"] == 0.0
    assert row["stop_reason"] == "cache_hit"

    daily = conn.execute("SELECT * FROM token_usage_daily").fetchall()
    assert len(daily) == 1
    assert dict(daily[0])["cache_hits"] == 1
    conn.close()


def test_budget_callback_fires(token_db: Path, pricing: dict) -> None:
    logger = TokenUsageLogger(token_db, pricing)
    # Wait for DB init
    time.sleep(0.2)

    # Insert a budget row directly
    conn = sqlite3.connect(str(token_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "INSERT INTO token_budget (budget_id, period, institution_id, max_cost_usd, alert_threshold, enforcement, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("b1", "daily", "", 0.001, 0.80, "warn", "2026-04-02T00:00:00"),
    )
    conn.commit()
    conn.close()

    alerts: list[tuple] = []
    logger.set_budget_callback(lambda bid, period, cur, lim: alerts.append((bid, period, cur, lim)))

    # Send a record that will exceed the budget
    logger.enqueue(TokenUsageRecord(
        caller="synthesis",
        model="claude-sonnet-4-6",
        input_tokens=10000,
        output_tokens=2000,
    ))
    logger._queue.join()
    time.sleep(0.1)

    assert len(alerts) >= 1
    assert alerts[0][0] == "b1"
    assert alerts[0][1] == "daily"


# ---------------------------------------------------------------------------
# MeteredAnthropicClient tests
# ---------------------------------------------------------------------------


def test_metered_client_intercepts_create(token_db: Path, pricing: dict) -> None:
    logger = TokenUsageLogger(token_db, pricing)

    # Mock the Anthropic client
    mock_response = MagicMock()
    mock_response.usage.input_tokens = 500
    mock_response.usage.output_tokens = 100
    mock_response.model = "claude-sonnet-4-6"
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(text="Hello")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    metered = MeteredAnthropicClient(mock_client, logger, caller="synthesis")

    result = metered.messages.create(model="claude-sonnet-4-6", max_tokens=100, messages=[])
    assert result is mock_response  # transparent pass-through
    mock_client.messages.create.assert_called_once()

    logger._queue.join()
    time.sleep(0.1)

    conn = sqlite3.connect(str(token_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM token_usage").fetchall()
    assert len(rows) == 1
    assert dict(rows[0])["caller"] == "synthesis"
    assert dict(rows[0])["input_tokens"] == 500
    conn.close()


def test_metered_client_proxies_attributes() -> None:
    mock_client = MagicMock()
    mock_client.api_key = "sk-test"
    mock_logger = MagicMock()
    metered = MeteredAnthropicClient(mock_client, mock_logger, caller="test")
    assert metered.api_key == "sk-test"


def test_metered_client_doesnt_break_on_metering_error(token_db: Path, pricing: dict) -> None:
    logger = TokenUsageLogger(token_db, pricing)

    # Response without usage attr
    mock_response = MagicMock(spec=[])  # no attributes
    mock_response.content = [MagicMock(text="Hi")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    metered = MeteredAnthropicClient(mock_client, logger, caller="synthesis")
    result = metered.messages.create(model="m", max_tokens=1, messages=[])
    assert result is mock_response  # still returns even if metering fails


# ---------------------------------------------------------------------------
# TokenUsageStore tests
# ---------------------------------------------------------------------------


def test_store_today_summary(token_db: Path, pricing: dict) -> None:
    logger = TokenUsageLogger(token_db, pricing)
    logger.enqueue(TokenUsageRecord(
        caller="synthesis", model="claude-sonnet-4-6",
        input_tokens=1000, output_tokens=200,
    ))
    logger.enqueue(TokenUsageRecord(
        caller="claim_extraction", model="claude-haiku-4-5",
        input_tokens=500, output_tokens=100,
    ))
    logger._queue.join()
    time.sleep(0.1)

    store = TokenUsageStore(token_db)
    summary = store.today_summary()
    assert summary["total_requests"] == 2
    assert "synthesis" in summary["by_caller"]
    assert "claim_extraction" in summary["by_caller"]
    store.close()


def test_store_history(token_db: Path, pricing: dict) -> None:
    logger = TokenUsageLogger(token_db, pricing)
    logger.enqueue(TokenUsageRecord(
        caller="synthesis", model="claude-sonnet-4-6",
        input_tokens=1000, output_tokens=200,
    ))
    logger._queue.join()
    time.sleep(0.1)

    store = TokenUsageStore(token_db)
    history = store.history("2020-01-01", "2030-12-31")
    assert len(history) >= 1
    assert history[0]["caller"] == "synthesis"
    store.close()


# ---------------------------------------------------------------------------
# CachedClaimAdapter integration
# ---------------------------------------------------------------------------


def test_cached_adapter_logs_cache_hit(tmp_path: Path, pricing: dict) -> None:
    """Cache hit emits zero-cost record; cache miss triggers inner adapter (no double-count)."""
    token_db = tmp_path / "token_usage.db"
    logger = TokenUsageLogger(token_db, pricing)

    # Build a CachedClaimAdapter with a mock inner
    from graphrag_pipeline.ingest.extractors.claim_cache import CachedClaimAdapter

    inner = MagicMock()
    inner.extract_claims.return_value = [{"claim": "test"}]

    adapter = CachedClaimAdapter(
        inner=inner,
        cache_path=tmp_path / "cache.db",
        model="claude-haiku-4-5",
        system_prompt_hash="abc123",
        token_logger=logger,
    )

    # First call: cache miss — inner adapter is called
    result1 = adapter.extract_claims("Some paragraph text for extraction.")
    assert result1 == [{"claim": "test"}]
    inner.extract_claims.assert_called_once()

    # Second call: cache hit — inner adapter NOT called again
    result2 = adapter.extract_claims("Some paragraph text for extraction.")
    assert result2 == [{"claim": "test"}]
    assert inner.extract_claims.call_count == 1  # no double-count

    logger._queue.join()
    time.sleep(0.1)

    conn = sqlite3.connect(str(token_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM token_usage ORDER BY created_at").fetchall()
    # Should have exactly 1 row: the cache-hit record (the miss doesn't emit
    # because the inner mock doesn't go through MeteredAnthropicClient)
    assert len(rows) == 1
    assert dict(rows[0])["cache_hit"] == 1
    assert dict(rows[0])["input_tokens"] == 0

    assert adapter.cache_stats == {"hits": 1, "misses": 1}
    conn.close()
    adapter.close()
