"""OpenAI-compatible ClaimLLMAdapter for self-hosted / open-source models.

This adapter targets any server that speaks the OpenAI ``/v1/chat/completions``
API — vLLM, Ollama, Text-Generation-Inference, LM Studio, llama.cpp's server,
and OpenAI itself. It is the seam for running claim extraction on an
open-weights model (e.g. Qwen2.5-32B-Instruct, Llama-3.1-70B) served from a
GPU box such as a Colab A100, instead of the metered Anthropic API.

It reuses the exact same domain-driven system prompt and JSON response parser
as the Anthropic adapter, so the rows it returns satisfy the identical
``ClaimLLMAdapter`` contract (``extract_claims(paragraph) -> list[dict]``) and
flow through the same ``CachedClaimAdapter`` SQLite cache.

Configuration (environment variables):
  - ``CLAIM_LLM_BASE_URL``   server base, e.g. ``http://localhost:8000/v1`` or a
                             Colab tunnel like ``https://xxxx.trycloudflare.com/v1``
  - ``CLAIM_LLM_MODEL``      served model id, e.g. ``Qwen/Qwen2.5-32B-Instruct-AWQ``
  - ``CLAIM_LLM_API_KEY``    bearer token (optional; vLLM accepts any value)
  - ``CLAIM_LLM_EXTRA_BODY`` JSON merged into the request body last (wins over
                             everything). Mirrors ``SYNTHESIS_LLM_EXTRA_BODY``:
                             use it to curb a reasoning model's thinking
                             (``{"reasoning_effort": "none"}`` for LM Studio) or
                             to override ``response_format``.
  - ``CLAIM_LLM_JSON_SCHEMA`` set to ``0``/``false`` to disable the auto-built
                             ``response_format`` json-schema constraint (on by
                             default).
  - ``CLAIM_LLM_SCHEMA_PROFILE`` ``full`` (default) | ``trim`` | ``min_epi`` |
                             ``min`` — how many fields the model must emit. Lean
                             profiles cut output tokens (≈ latency on a
                             decode-bound local model); ``min_epi`` is the tuned
                             bulk-ingest default (~40% faster than ``full`` on
                             e4b, same claim yield). The constraint forces the exact
                             array-of-claims shape ``_parse_response`` expects,
                             which structurally eliminates the "model emitted
                             two bare objects instead of an array" parse
                             failures small local models produce.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, TYPE_CHECKING

from gemynd.ingest.extractors.anthropic_claim_adapter import (
    _build_system_prompt,
    _parse_response,
)
# Single source of truth for the chat-completions endpoint rule (shared with
# the query-time OpenAI-compatible seam in shared/openai_compat.py).
from gemynd.shared.openai_compat import _normalize_endpoint

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig

_log = logging.getLogger(__name__)

# Contract-level epistemic values (see the claim contract in the system prompt).
# These are domain-independent, unlike claim/entity types, so they can be a
# fixed enum in the response schema.
_EPISTEMIC_STATUSES = ("observed", "estimated", "uncertain", "reported")

# Schema profiles trade output tokens (≈ latency, decode-bound) against field
# richness. Benchmarked on e4b over real newspaper OCR: "min_epi" is ~40% faster
# than "full" with identical claim/link yield, because it drops only fields that
# are recomputed or defaulted downstream (claim_extractor.py consumes LLM rows
# with .get(...) fallbacks): normalized_sentence -> normalize_for_match(),
# extraction_confidence -> 0.6, char offsets -> None. epistemic_status is the one
# non-recomputable signal, so "min_epi" keeps it (vs bare "min").
_VALID_PROFILES = ("full", "trim", "min_epi", "min")


def _resolve_schema_profile(profile: str | None) -> str:
    resolved = (profile or os.environ.get("CLAIM_LLM_SCHEMA_PROFILE") or "full").strip().lower()
    if resolved not in _VALID_PROFILES:
        _log.warning("Unknown CLAIM_LLM_SCHEMA_PROFILE=%r — using 'full'", resolved)
        return "full"
    return resolved


def _build_claim_response_format(
    config: "DomainConfig | None", profile: str | None = None
) -> dict[str, Any]:
    """Build an OpenAI ``response_format`` json-schema for the claim array.

    The root is an **array** of claim objects — exactly what ``_parse_response``
    accepts. Only ``source_sentence`` is required (matching the prompt); all
    other fields are optional. ``claim_type`` / ``entity_type_hint`` enums are
    injected from the domain config when present (same values the system prompt
    advertises), so the model can't invent out-of-vocabulary types; when config
    is ``None`` they stay free strings.

    *profile* (or ``CLAIM_LLM_SCHEMA_PROFILE`` env) selects how many fields the
    model must emit — see ``_VALID_PROFILES``. Fewer fields = fewer output tokens
    = faster on a decode-bound local model.
    """
    profile = _resolve_schema_profile(profile)
    claim_type_field: dict[str, Any] = {"type": "string"}
    entity_hint_field: dict[str, Any] = {"type": ["string", "null"]}
    if config is not None:
        from gemynd.core.claim_contract import UNCLASSIFIED_TYPE

        claim_types = sorted(
            {ct for ct, _, _ in config.claim_type_patterns} | {UNCLASSIFIED_TYPE}
        )
        if claim_types:
            claim_type_field = {"type": "string", "enum": claim_types}
        entity_types = sorted({e.entity_type for e in config.seed_entities})
        if entity_types:
            entity_hint_field = {"type": ["string", "null"], "enum": [*entity_types, None]}

    # Link fields: lean profiles keep only the identity + type hint (offsets and
    # normalized_form are reconstructed / non-load-bearing downstream).
    link_props: dict[str, Any] = {"surface_form": {"type": "string"}}
    if profile in ("full", "trim"):
        link_props["normalized_form"] = {"type": "string"}
        link_props["relation"] = {"type": ["string", "null"]}
    link_props["entity_type_hint"] = entity_hint_field
    if profile == "full":
        link_props["start_offset"] = {"type": ["integer", "null"]}
        link_props["end_offset"] = {"type": ["integer", "null"]}
    link_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["surface_form"],
        "properties": link_props,
    }

    claim_props: dict[str, Any] = {"source_sentence": {"type": "string"}}
    if profile in ("full", "trim"):
        claim_props["normalized_sentence"] = {"type": "string"}
    claim_props["claim_type"] = claim_type_field
    if profile != "min":
        claim_props["epistemic_status"] = {"type": "string", "enum": list(_EPISTEMIC_STATUSES)}
    if profile in ("full", "trim"):
        claim_props["extraction_confidence"] = {"type": "number", "minimum": 0.0, "maximum": 1.0}
        claim_props["claim_date"] = {"type": ["string", "null"]}
    if profile == "full":
        claim_props["notes"] = {"type": "string"}
        claim_props["evidence_start"] = {"type": ["integer", "null"]}
        claim_props["evidence_end"] = {"type": ["integer", "null"]}
    claim_props["claim_links"] = {"type": "array", "items": link_schema}
    claim_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_sentence"],
        "properties": claim_props,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "claim_extraction",
            "strict": True,
            "schema": {"type": "array", "items": claim_schema},
        },
    }


def _parse_extra_body() -> dict[str, Any]:
    """Parse ``CLAIM_LLM_EXTRA_BODY`` (JSON) into a dict; empty on error/unset."""
    raw = os.environ.get("CLAIM_LLM_EXTRA_BODY")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        _log.warning("CLAIM_LLM_EXTRA_BODY is not valid JSON — ignored")
        return {}
    if not isinstance(parsed, dict):
        _log.warning("CLAIM_LLM_EXTRA_BODY must be a JSON object — ignored")
        return {}
    return parsed


class OpenAICompatibleClaimAdapter:
    """ClaimLLMAdapter backed by an OpenAI-compatible chat endpoint."""

    _MIN_LENGTH = int(os.environ.get("CLAIM_MIN_PARAGRAPH_LENGTH", "40"))

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        temperature: float = 0.0,
        config: "DomainConfig | None" = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        resolved_base = base_url or os.environ.get("CLAIM_LLM_BASE_URL")
        if not resolved_base:
            raise ValueError(
                "No OpenAI-compatible base URL found. Set CLAIM_LLM_BASE_URL "
                "(e.g. http://localhost:8000/v1)."
            )
        self._endpoint = _normalize_endpoint(resolved_base)
        self._model = (
            model
            or os.environ.get("CLAIM_LLM_MODEL")
            or os.environ.get("CLAIM_EXTRACTION_MODEL")
            or "local-model"
        )
        # Most local servers ignore the key; send a placeholder so auth-enabled
        # gateways still accept the request.
        self._api_key = api_key or os.environ.get("CLAIM_LLM_API_KEY") or "not-needed"
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._temperature = temperature
        self._system_prompt = _build_system_prompt(config)
        # Auto-build the array-of-claims json-schema (on by default) so local
        # models are constrained to the exact shape _parse_response accepts.
        # Disable with CLAIM_LLM_JSON_SCHEMA=0/false when a server doesn't
        # support response_format.
        if response_format is not None:
            self._response_format = response_format
        elif (os.environ.get("CLAIM_LLM_JSON_SCHEMA", "1").strip().lower()
              in {"0", "false", "no", "off"}):
            self._response_format = None
        else:
            self._response_format = _build_claim_response_format(config)
        # Server-specific knobs merged into the body last (e.g. reasoning_effort
        # for LM Studio); env-provided values win over the constructor arg.
        self._extra_body = {**(extra_body or {}), **_parse_extra_body()}

    def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
        text = paragraph_text.strip()
        if not text:
            return []
        if len(text) < self._MIN_LENGTH:
            _log.debug("Skipping short paragraph (%d chars): %.30s…", len(text), text)
            return []

        payload_body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        want_schema = self._response_format is not None and "response_format" not in self._extra_body
        if self._response_format is not None:
            payload_body["response_format"] = self._response_format
        # extra_body wins last — lets callers override response_format or add
        # reasoning_effort without touching the base body.
        payload_body.update(self._extra_body)

        payload = self._post(payload_body)
        # A server that doesn't understand response_format typically 400s; drop
        # the constraint and retry once so an unsupported backend degrades to
        # free-form JSON instead of returning nothing.
        if payload is None and want_schema:
            _log.warning("Retrying claim extraction without response_format (server rejected it)")
            payload_body.pop("response_format", None)
            payload = self._post(payload_body)
        if payload is None:
            return []

        try:
            raw = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            _log.warning("Unexpected chat-completions response shape: %s", exc)
            return []
        return _parse_response(raw)

    def _post(self, payload_body: dict[str, Any]) -> dict[str, Any] | None:
        """POST the chat-completions body; return parsed JSON or None on failure."""
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            _log.warning("OpenAI-compatible claim extraction call failed: %s", exc)
            return None


def try_create_openai_adapter(
    config: "DomainConfig | None" = None,
    *,
    cache_path: str | None = None,
    disable_cache: bool = False,
    token_logger: Any | None = None,
) -> Any:
    """Create an OpenAICompatibleClaimAdapter; fall back to NullLLMAdapter.

    Mirrors ``try_create_anthropic_adapter``: when no base URL is configured the
    LLM path degrades to a no-op rather than raising, and (unless disabled) the
    adapter is wrapped in the same SQLite response cache so re-runs are free.
    """
    from gemynd.ingest.extractors.claim_extractor import NullLLMAdapter

    if not os.environ.get("CLAIM_LLM_BASE_URL"):
        _log.debug("No CLAIM_LLM_BASE_URL set — using NullLLMAdapter for claim LLM path")
        return NullLLMAdapter()
    try:
        adapter = OpenAICompatibleClaimAdapter(config=config)
    except Exception as exc:
        _log.warning(
            "Failed to create OpenAICompatibleClaimAdapter: %s — falling back to NullLLMAdapter",
            exc,
        )
        return NullLLMAdapter()

    if not disable_cache:
        from gemynd.ingest.extractors.claim_cache import (
            CachedClaimAdapter,
            DEFAULT_CACHE_PATH,
            system_prompt_hash,
        )

        prompt_hash = system_prompt_hash(adapter._system_prompt)
        resolved_path = cache_path or os.environ.get("CLAIM_CACHE_PATH") or str(DEFAULT_CACHE_PATH)
        adapter = CachedClaimAdapter(
            inner=adapter,
            cache_path=resolved_path,
            model=adapter._model,
            system_prompt_hash=prompt_hash,
            token_logger=token_logger,
        )
        _log.info("LLM claim extraction cache enabled at %s", resolved_path)

    return adapter
