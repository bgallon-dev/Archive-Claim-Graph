"""LLM claim VALIDATOR for the rules-propose / LLM-validate pipeline.

Instead of extracting claims from scratch (expensive: one call per paragraph,
full claim objects), this *validates* the candidate sentences the rule extractor
proposed — its low-confidence ``unclassified_assertion`` drafts (see
``RuleBasedClaimExtractor`` fallback). For each candidate the model decides
whether it is a genuine factual claim and, if so, assigns the best claim_type.

Validation is much cheaper than extraction: candidates are **batched** into one
call and each verdict is tiny (is_claim + type + confidence), not a full claim
object with links and offsets.

Targets any OpenAI-compatible ``/v1/chat/completions`` server (LM Studio, vLLM,
…) and reuses the same endpoint / extra-body handling as ``openai_claim_adapter``.

Configuration (environment variables, shared with the extractor seam):
  - ``CLAIM_LLM_BASE_URL`` / ``CLAIM_LLM_MODEL`` / ``CLAIM_LLM_API_KEY``
  - ``CLAIM_LLM_EXTRA_BODY`` merged last (e.g. ``{"reasoning_effort":"none"}``)
  - ``CLAIM_VALIDATOR_BATCH`` candidates per call (default 8). Keep it small for
                             small local models: e4b reliably validates batches of
                             ~10 but returns an empty array (gives up) around 16.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, TYPE_CHECKING

from gemynd.ingest.extractors.openai_claim_adapter import _parse_extra_body
from gemynd.shared.openai_compat import _normalize_endpoint

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig

_log = logging.getLogger(__name__)


def _build_validator_system_prompt(claim_types: list[str]) -> str:
    types = ", ".join(claim_types)
    return (
        "You validate whether each numbered sentence, taken from historical "
        "newspaper OCR, is a factual CLAIM: an assertive statement of a fact, "
        "event, action, or condition that could be true or false.\n\n"
        "NOT claims (mark is_claim=false): advertisements, prices, product lists, "
        "headlines, mastheads, tables, name/office lists, non-assertive fragments, "
        "and garbled OCR that does not form a coherent statement.\n\n"
        "For each input sentence return an object with:\n"
        '  - "index" (int): the sentence number, exactly as given\n'
        '  - "is_claim" (bool)\n'
        '  - "claim_type" (string or null): if is_claim, the single best type from '
        f"the allowed list; otherwise null. Allowed types: {types}\n"
        '  - "confidence" (float 0-1): your confidence that it is a claim of that type\n\n'
        "Return ONLY a JSON array with exactly one object per input sentence, in "
        "the same order. No prose, no markdown."
    )


def _build_verdict_schema(claim_types: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "claim_validation",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "is_claim"],
                    "properties": {
                        "index": {"type": "integer"},
                        "is_claim": {"type": "boolean"},
                        "claim_type": {"type": ["string", "null"], "enum": [*claim_types, None]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
        },
    }


class ClaimValidatorAdapter:
    """Validate rule-proposed candidate sentences via a batched LLM call."""

    def __init__(
        self,
        claim_types: list[str],
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
        max_tokens: int = 2048,
        timeout: float = 180.0,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        resolved_base = base_url or os.environ.get("CLAIM_LLM_BASE_URL")
        if not resolved_base:
            raise ValueError("No base URL. Set CLAIM_LLM_BASE_URL (e.g. http://localhost:1234/v1).")
        self._endpoint = _normalize_endpoint(resolved_base)
        self._model = model or os.environ.get("CLAIM_LLM_MODEL") or "local-model"
        self._api_key = api_key or os.environ.get("CLAIM_LLM_API_KEY") or "not-needed"
        self._claim_types = list(claim_types)
        self._batch_size = batch_size or int(os.environ.get("CLAIM_VALIDATOR_BATCH", "8"))
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._temperature = temperature
        self._system_prompt = _build_validator_system_prompt(self._claim_types)
        self._schema = _build_verdict_schema(self._claim_types)
        self._extra_body = {**(extra_body or {}), **_parse_extra_body()}

    def validate(self, sentences: list[str]) -> list[dict[str, Any]]:
        """Return a verdict dict per input sentence, aligned to input order.

        Each verdict: ``{"is_claim": bool, "claim_type": str|None, "confidence": float}``.
        Fails **open**: on any call/parse failure the affected candidates come back
        ``is_claim=True, claim_type=None`` so a broken validator never silently
        deletes rule proposals (they stay as unclassified rather than vanish).
        """
        verdicts: list[dict[str, Any]] = []
        for i in range(0, len(sentences), self._batch_size):
            batch = sentences[i : i + self._batch_size]
            verdicts.extend(self._validate_batch(batch))
        return verdicts

    def _fail_open(self, n: int) -> list[dict[str, Any]]:
        return [{"is_claim": True, "claim_type": None, "confidence": 0.3} for _ in range(n)]

    def _validate_batch(self, batch: list[str]) -> list[dict[str, Any]]:
        numbered = "\n".join(f"{idx}. {s}" for idx, s in enumerate(batch))
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": numbered},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "response_format": self._schema,
        }
        body.update(self._extra_body)
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw = payload["choices"][0]["message"]["content"].strip()
            parsed = json.loads(raw)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            _log.warning("Claim validation call failed (%s) — failing open for %d candidates", exc, len(batch))
            return self._fail_open(len(batch))
        if not isinstance(parsed, list):
            _log.warning("Validator returned non-list — failing open for %d candidates", len(batch))
            return self._fail_open(len(batch))

        # Align verdicts to input order by their reported index; fail-open on gaps.
        by_index: dict[int, dict[str, Any]] = {}
        for item in parsed:
            if isinstance(item, dict) and isinstance(item.get("index"), int):
                by_index[item["index"]] = item
        out: list[dict[str, Any]] = []
        for idx in range(len(batch)):
            v = by_index.get(idx)
            if v is None:
                out.append({"is_claim": True, "claim_type": None, "confidence": 0.3})
                continue
            ctype = v.get("claim_type")
            if ctype is not None and ctype not in self._claim_types:
                ctype = None
            out.append({
                "is_claim": bool(v.get("is_claim", True)),
                "claim_type": ctype,
                "confidence": float(v.get("confidence", 0.5)) if isinstance(v.get("confidence"), (int, float)) else 0.5,
            })
        return out


def try_create_validator(config: "DomainConfig | None" = None, **kwargs: Any) -> "ClaimValidatorAdapter | None":
    """Build a ClaimValidatorAdapter from config; None if no server configured."""
    if not os.environ.get("CLAIM_LLM_BASE_URL") and not kwargs.get("base_url"):
        return None
    from gemynd.core.claim_contract import ALLOWED_CLAIM_TYPES

    claim_types = sorted(config.allowed_claim_types) if config is not None else sorted(ALLOWED_CLAIM_TYPES)
    try:
        return ClaimValidatorAdapter(claim_types=claim_types, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to create ClaimValidatorAdapter: %s", exc)
        return None
