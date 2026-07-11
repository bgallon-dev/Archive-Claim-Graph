"""OpenAI-compatible ClaimLLMAdapter for local models (e.g. LM Studio).

LM Studio (and Ollama, vLLM, OpenAI cloud, etc.) expose the OpenAI
``/v1/chat/completions`` API rather than the Anthropic Messages API, so this
adapter talks to them via the ``openai`` SDK instead of ``MeteredAnthropicClient``.
It reuses the provider-neutral system prompt and response parser from
``anthropic_claim_adapter`` so the cache key (which hashes the system prompt)
stays identical across providers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, TYPE_CHECKING

from gemynd.ingest.extractors.anthropic_claim_adapter import (
    _build_system_prompt,
    _FENCE_RE,
)

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig

_log = logging.getLogger(__name__)

try:
    import openai as _openai_pkg
except ImportError:  # pragma: no cover
    _openai_pkg = None  # type: ignore[assignment]

_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_LOCAL_PROVIDERS = {"openai", "lmstudio", "lm-studio", "local"}

# Appended only to the local/OpenAI-compatible system prompt (never the Anthropic
# one). Small local models tend to paraphrase the source sentence, which breaks the
# hybrid extractor's dedup-merge with rule-based claims and produces near-duplicate,
# inconsistently-typed claims. These rules push the model toward verbatim, single-fact,
# single-type output.
_LOCAL_PROMPT_ADDENDUM = """

ADDITIONAL RULES FOR THIS TASK — follow them exactly:
- "source_sentence" MUST be copied VERBATIM from the input paragraph: character for \
character, preserving the original spelling, capitalization, punctuation, and any OCR \
errors. Never rephrase, correct, clean, summarize, or translate it.
- Do NOT include a "normalized_sentence" field. Omit it entirely.
- Output each distinct fact only ONCE. Never emit two claims describing the same fact.
- Give each claim exactly ONE "claim_type" — the single best-fitting type."""


class OpenAICompatClaimAdapter:
    """ClaimLLMAdapter backed by any OpenAI-compatible chat-completions server."""

    _MIN_LENGTH = int(os.environ.get("CLAIM_MIN_PARAGRAPH_LENGTH", "40"))

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        temperature: float = 0.0,
        config: "DomainConfig | None" = None,
        token_logger: Any | None = None,  # accepted for parity; unused (local is free)
    ) -> None:
        if _openai_pkg is None:
            raise RuntimeError(
                "openai package is not installed. Install with: pip install openai"
            )
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL
        # Local servers ignore the key, but the SDK requires a non-empty string.
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "lm-studio"
        self._client = _openai_pkg.OpenAI(base_url=resolved_base, api_key=resolved_key)
        self._model = model or os.environ.get("LLM_MODEL") or "local-model"
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._temperature = temperature
        self._system_prompt = _build_system_prompt(config) + _LOCAL_PROMPT_ADDENDUM
        # JSON-schema constrained decoding: forces valid JSON, claim_type/relation_type
        # enums, and the flat claim_links shape at the decoder level. Set
        # OPENAI_STRUCTURED_OUTPUT=0 to disable (e.g. a server that rejects json_schema).
        if os.environ.get("OPENAI_STRUCTURED_OUTPUT", "1").strip().lower() in {"0", "false", "no", "off"}:
            self._response_format: dict[str, Any] | None = None
        else:
            self._response_format = _build_response_format(config)
        self._schema_disabled = False  # set True after a server rejects json_schema

    def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
        """Call the local model to extract structured claims from a paragraph."""
        text = paragraph_text.strip()
        if not text:
            return []
        if len(text) < self._MIN_LENGTH:
            _log.debug(
                "Skipping short paragraph (%d chars < %d): %.30s…",
                len(text), self._MIN_LENGTH, text,
            )
            return []

        use_schema = bool(self._response_format) and not self._schema_disabled
        # Try once with the schema; if the server rejects it, disable and retry plain.
        for attempt_with_schema in ([True, False] if use_schema else [False]):
            kwargs: dict[str, Any] = dict(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                timeout=self._timeout,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            if attempt_with_schema:
                kwargs["response_format"] = self._response_format
            try:
                response = self._client.chat.completions.create(**kwargs)
                raw = (response.choices[0].message.content or "").strip()
                return _normalize_claims(_parse_claims(raw))
            except Exception as exc:
                if attempt_with_schema:
                    _log.warning(
                        "Structured-output (json_schema) call failed: %s — disabling it and "
                        "retrying without a schema.", exc,
                    )
                    self._schema_disabled = True
                    continue
                _log.warning("OpenAI-compatible claim extraction API call failed: %s", exc)
                return []
        return []


def _normalize_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize raw local-model claims into the shape the pipeline expects.

    Two fixes, both targeting weak local-model output:

    * Flatten the nested ``claim_links`` shape small models often emit
      (``{"SPECIES_FOCUS": [{"surface_form": ...}]}``); the downstream parser only
      accepts flat rows with a top-level ``relation_type`` and silently drops the
      rest, so entity edges would never reach the graph.
    * Drop any model-supplied ``normalized_sentence``.  Small models paraphrase it,
      which becomes the hybrid extractor's merge key and prevents the LLM claim from
      deduping against the matching rule-based claim.  Removing it forces the
      deterministic ``normalize_for_match`` of the verbatim ``source_sentence``,
      which aligns the two so they merge instead of duplicating.

    Already-flat rows pass through untouched.
    """
    for claim in claims:
        claim.pop("normalized_sentence", None)
        links = claim.get("claim_links")
        if not isinstance(links, list):
            continue
        flat: list[dict[str, Any]] = []
        for entry in links:
            if not isinstance(entry, dict):
                continue
            if "relation_type" in entry or "surface_form" in entry or "text" in entry:
                flat.append(entry)
                continue
            # Nested {RELATION: {...}} or {RELATION: [{...}, ...]}
            for relation, value in entry.items():
                items = value if isinstance(value, list) else [value]
                for item in items:
                    if isinstance(item, dict):
                        flat.append({**item, "relation_type": relation})
        claim["claim_links"] = flat
    return claims


def _parse_claims(raw: str) -> list[dict[str, Any]]:
    """Parse the model response into a list of claim dicts.

    Handles both a bare top-level JSON array and the object-wrapped form
    (``{"claims": [...]}``) that ``json_schema`` constrained decoding returns, plus
    markdown code fences. Returns ``[]`` on any parse failure.
    """
    match = _FENCE_RE.match(raw)
    if match:
        raw = match.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _log.warning("Failed to parse claim extraction JSON: %s", exc)
        return []
    if isinstance(data, dict):
        for key in ("claims", "items", "results", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            _log.warning("Claim extraction returned an object with no claims array")
            return []
    if not isinstance(data, list):
        _log.warning("Claim extraction returned non-list JSON: %s", type(data).__name__)
        return []
    return [item for item in data if isinstance(item, dict)]


def _build_response_format(config: "DomainConfig | None") -> dict[str, Any]:
    """Build an OpenAI ``json_schema`` response_format for constrained decoding.

    Enums are derived from the SAME domain config the system prompt uses, so the
    grammar restricts ``claim_type`` / ``relation_type`` to valid values and forces
    ``claim_links`` into the flat ``{relation_type, surface_form}`` shape the parser
    needs. Wrapped in a top-level object (``{"claims": [...]}``) for OpenAI-spec
    compatibility; :func:`_parse_claims` unwraps it.
    """
    from gemynd.core.claim_contract import (
        ALLOWED_CLAIM_TYPES,
        EXTRACTOR_CLAIM_LINK_RELATIONS,
        UNCLASSIFIED_TYPE,
    )

    if config is not None:
        claim_types = sorted({ct for ct, _, _ in config.claim_type_patterns} | {UNCLASSIFIED_TYPE})
        entity_types = sorted({e.entity_type for e in config.seed_entities})
    else:
        claim_types = sorted(ALLOWED_CLAIM_TYPES)
        entity_types = []
    relation_types = sorted(EXTRACTOR_CLAIM_LINK_RELATIONS)

    entity_hint_schema: dict[str, Any] = {"type": ["string", "null"]}
    if entity_types:
        entity_hint_schema = {"type": ["string", "null"], "enum": entity_types + [None]}

    link_schema = {
        "type": "object",
        "properties": {
            "relation_type": {"type": "string", "enum": relation_types},
            "surface_form": {"type": "string"},
            "normalized_form": {"type": "string"},
            "entity_type_hint": entity_hint_schema,
        },
        "required": ["relation_type", "surface_form"],
    }
    claim_schema = {
        "type": "object",
        "properties": {
            "source_sentence": {"type": "string"},
            "claim_type": {"type": "string", "enum": claim_types},
            "epistemic_status": {
                "type": "string",
                "enum": ["observed", "estimated", "uncertain", "reported"],
            },
            "extraction_confidence": {"type": "number"},
            "claim_date": {"type": ["string", "null"]},
            "notes": {"type": "string"},
            "claim_links": {"type": "array", "items": link_schema},
        },
        "required": ["source_sentence", "claim_type"],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "claim_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"claims": {"type": "array", "items": claim_schema}},
                "required": ["claims"],
            },
        },
    }


def create_claim_llm_adapter(
    config: "DomainConfig | None" = None,
    *,
    token_logger: Any | None = None,
    cache_path: str | None = None,
    disable_cache: bool = False,
    provider: str | None = None,
) -> Any:
    """Provider-aware factory for the claim-extraction LLM adapter.

    Reads ``LLM_PROVIDER`` (default ``anthropic``) unless *provider* is given.
    The ``anthropic`` path delegates to the existing factory unchanged.  Local
    OpenAI-compatible providers build an :class:`OpenAICompatClaimAdapter` and
    wrap it in the shared SQLite cache.  Local cache hits are NOT metered
    (``token_logger=None``) because local inference is free.
    """
    resolved = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).strip().lower()

    if resolved not in _LOCAL_PROVIDERS:
        from gemynd.ingest.extractors.anthropic_claim_adapter import (
            try_create_anthropic_adapter,
        )
        return try_create_anthropic_adapter(
            config=config,
            cache_path=cache_path,
            disable_cache=disable_cache,
            token_logger=token_logger,
        )

    from gemynd.ingest.extractors.claim_extractor import NullLLMAdapter

    if _openai_pkg is None:
        _log.warning(
            "LLM_PROVIDER=%s but openai package is not installed — using NullLLMAdapter. "
            "Install with: pip install 'gemynd[local]'",
            resolved,
        )
        return NullLLMAdapter()
    try:
        adapter: Any = OpenAICompatClaimAdapter(config=config)
    except Exception as exc:
        _log.warning(
            "Failed to create OpenAICompatClaimAdapter: %s — falling back to NullLLMAdapter",
            exc,
        )
        return NullLLMAdapter()

    if disable_cache:
        return adapter

    from gemynd.ingest.extractors.claim_cache import CachedClaimAdapter, DEFAULT_CACHE_PATH

    prompt_hash = hashlib.sha256(
        adapter._system_prompt.encode("utf-8")
    ).hexdigest()[:16]
    resolved_path = cache_path or os.environ.get("CLAIM_CACHE_PATH") or str(DEFAULT_CACHE_PATH)
    wrapped = CachedClaimAdapter(
        inner=adapter,
        cache_path=resolved_path,
        model=adapter._model,
        system_prompt_hash=prompt_hash,
        token_logger=None,  # local models are free — never log fabricated cost
    )
    _log.info(
        "Local (%s) claim extraction enabled (model=%s); cache at %s",
        resolved, adapter._model, resolved_path,
    )
    return wrapped
