"""SQLite-backed paragraph-level cache for LLM claim extraction responses."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/claim_cache.db")

_INIT_SQL = """\
CREATE TABLE IF NOT EXISTS claim_cache (
    cache_key   TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _cache_key(model: str, system_prompt_hash: str, paragraph_text: str) -> str:
    """Deterministic key from model + prompt fingerprint + paragraph content."""
    normalized = paragraph_text.strip()
    payload = f"{model}||{system_prompt_hash}||{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedClaimAdapter:
    """Transparent caching wrapper around any ClaimLLMAdapter."""

    def __init__(
        self,
        inner: Any,
        cache_path: Path | str | None = None,
        model: str = "",
        system_prompt_hash: str = "",
    ) -> None:
        self._inner = inner
        self._model = model
        self._prompt_hash = system_prompt_hash
        resolved = Path(
            cache_path
            or os.environ.get("CLAIM_CACHE_PATH", "")
            or DEFAULT_CACHE_PATH
        )
        self._db_path = resolved
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_INIT_SQL)
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
        text = paragraph_text.strip()
        if not text:
            return []
        key = _cache_key(self._model, self._prompt_hash, text)

        row = self._conn.execute(
            "SELECT response FROM claim_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is not None:
            self._hits += 1
            _log.debug("Cache hit for paragraph (key=%s…)", key[:12])
            return json.loads(row[0])

        self._misses += 1
        result = self._inner.extract_claims(paragraph_text)
        self._conn.execute(
            "INSERT OR REPLACE INTO claim_cache (cache_key, model, response) VALUES (?, ?, ?)",
            (key, self._model, json.dumps(result)),
        )
        self._conn.commit()
        return result

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def close(self) -> None:
        self._conn.close()
