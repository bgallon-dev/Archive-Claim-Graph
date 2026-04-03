"""Tests for the SQLite-backed paragraph-level claim extraction cache."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gemynd.ingest.extractors.claim_cache import (
    CachedClaimAdapter,
    _cache_key,
)


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "test_cache.db"


def _make_adapter(
    tmp_cache: Path,
    inner: MagicMock | None = None,
    model: str = "test-model",
    prompt_hash: str = "abc123",
) -> CachedClaimAdapter:
    if inner is None:
        inner = MagicMock()
        inner.extract_claims.return_value = [
            {"source_sentence": "50 mallards observed.", "claim_type": "population_estimate"}
        ]
    return CachedClaimAdapter(
        inner=inner,
        cache_path=tmp_cache,
        model=model,
        system_prompt_hash=prompt_hash,
    )


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------

def test_cache_key_deterministic() -> None:
    k1 = _cache_key("model", "prompt", "some text")
    k2 = _cache_key("model", "prompt", "some text")
    assert k1 == k2


def test_cache_key_varies_by_model() -> None:
    k1 = _cache_key("model-a", "prompt", "same text")
    k2 = _cache_key("model-b", "prompt", "same text")
    assert k1 != k2


def test_cache_key_varies_by_prompt() -> None:
    k1 = _cache_key("model", "prompt-a", "same text")
    k2 = _cache_key("model", "prompt-b", "same text")
    assert k1 != k2


def test_cache_key_strips_whitespace() -> None:
    k1 = _cache_key("m", "p", "  hello  ")
    k2 = _cache_key("m", "p", "hello")
    assert k1 == k2


# ---------------------------------------------------------------------------
# CachedClaimAdapter tests
# ---------------------------------------------------------------------------

def test_cache_miss_delegates_to_inner(tmp_cache: Path) -> None:
    inner = MagicMock()
    inner.extract_claims.return_value = [{"source_sentence": "test"}]
    adapter = _make_adapter(tmp_cache, inner=inner)

    result = adapter.extract_claims("A paragraph of text.")
    assert result == [{"source_sentence": "test"}]
    inner.extract_claims.assert_called_once_with("A paragraph of text.")
    assert adapter.cache_stats == {"hits": 0, "misses": 1}


def test_cache_hit_returns_stored(tmp_cache: Path) -> None:
    inner = MagicMock()
    inner.extract_claims.return_value = [{"source_sentence": "cached"}]
    adapter = _make_adapter(tmp_cache, inner=inner)

    adapter.extract_claims("Same paragraph.")
    result = adapter.extract_claims("Same paragraph.")

    assert result == [{"source_sentence": "cached"}]
    assert inner.extract_claims.call_count == 1  # only called once
    assert adapter.cache_stats == {"hits": 1, "misses": 1}


def test_empty_paragraph_not_cached(tmp_cache: Path) -> None:
    inner = MagicMock()
    adapter = _make_adapter(tmp_cache, inner=inner)

    result = adapter.extract_claims("")
    assert result == []
    inner.extract_claims.assert_not_called()
    assert adapter.cache_stats == {"hits": 0, "misses": 0}


def test_whitespace_only_paragraph_not_cached(tmp_cache: Path) -> None:
    inner = MagicMock()
    adapter = _make_adapter(tmp_cache, inner=inner)

    result = adapter.extract_claims("   \n  ")
    assert result == []
    inner.extract_claims.assert_not_called()


def test_cache_persists_across_instances(tmp_cache: Path) -> None:
    inner = MagicMock()
    inner.extract_claims.return_value = [{"source_sentence": "persisted"}]

    adapter1 = _make_adapter(tmp_cache, inner=inner)
    adapter1.extract_claims("Persistent paragraph.")
    adapter1.close()

    # New adapter instance, same cache DB — should get a hit
    inner2 = MagicMock()
    adapter2 = _make_adapter(tmp_cache, inner=inner2)
    result = adapter2.extract_claims("Persistent paragraph.")

    assert result == [{"source_sentence": "persisted"}]
    inner2.extract_claims.assert_not_called()
    assert adapter2.cache_stats == {"hits": 1, "misses": 0}
    adapter2.close()


def test_different_model_misses_cache(tmp_cache: Path) -> None:
    inner = MagicMock()
    inner.extract_claims.return_value = [{"source_sentence": "v1"}]

    adapter1 = _make_adapter(tmp_cache, inner=inner, model="model-a")
    adapter1.extract_claims("Same text.")
    adapter1.close()

    inner2 = MagicMock()
    inner2.extract_claims.return_value = [{"source_sentence": "v2"}]
    adapter2 = _make_adapter(tmp_cache, inner=inner2, model="model-b")
    result = adapter2.extract_claims("Same text.")

    assert result == [{"source_sentence": "v2"}]
    inner2.extract_claims.assert_called_once()
    adapter2.close()


def test_cache_stats_tracks_correctly(tmp_cache: Path) -> None:
    inner = MagicMock()
    inner.extract_claims.return_value = []
    adapter = _make_adapter(tmp_cache, inner=inner)

    adapter.extract_claims("Para 1.")
    adapter.extract_claims("Para 2.")
    adapter.extract_claims("Para 1.")  # hit
    adapter.extract_claims("Para 3.")
    adapter.extract_claims("Para 2.")  # hit

    assert adapter.cache_stats == {"hits": 2, "misses": 3}
