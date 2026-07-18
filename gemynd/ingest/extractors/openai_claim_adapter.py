"""OpenAI-compatible ClaimLLMAdapter for self-hosted / open-source models.

This module targets any server that speaks the OpenAI ``/v1/chat/completions``
API — vLLM, Ollama, Text-Generation-Inference, LM Studio, llama.cpp's server,
and OpenAI itself. It is the seam for running claim extraction on an
open-weights model (e.g. Qwen2.5-32B-Instruct, Llama-3.1-70B) served from a
GPU box such as a Colab A100, instead of the metered Anthropic API.

Both adapters reuse the exact same domain-driven system prompt and JSON
response parser as the Anthropic adapter, so the rows they return satisfy the
identical ``ClaimLLMAdapter`` contract (``extract_claims(paragraph) ->
list[dict]``) and flow through the same ``CachedClaimAdapter`` SQLite cache.

Two adapters live here, reached by two different entry points:

* :class:`OpenAICompatibleClaimAdapter` / :func:`try_create_openai_adapter` —
  dependency-free (``urllib``), driven by the ``CLAIM_LLM_*`` env vars, and
  selected by ``ingest/extractors/llm_adapter_factory.py``. Use this for
  airgapped deployments: it adds no third-party package to the install.
* :class:`OpenAICompatClaimAdapter` / :func:`create_claim_llm_adapter` — built
  on the ``openai`` SDK, driven by ``LLM_PROVIDER`` / ``OPENAI_*`` env vars,
  and selected by ``claim_extractor.py``. Adds a prompt addendum that pushes
  weak local models toward verbatim, single-fact, single-type output.

Configuration for the urllib adapter (environment variables):
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
    _FENCE_RE,
    _parse_response,
)
# Single source of truth for the chat-completions endpoint rule (shared with
# the query-time OpenAI-compatible seam in shared/openai_compat.py).
from gemynd.shared.openai_compat import _normalize_endpoint

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig

_log = logging.getLogger(__name__)

try:
    import openai as _openai_pkg
except ImportError:  # pragma: no cover
    _openai_pkg = None  # type: ignore[assignment]

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
    """ClaimLLMAdapter backed by an OpenAI-compatible chat endpoint.

    Dependency-free (stdlib ``urllib``) — preferred for airgapped installs.
    Reached via :func:`try_create_openai_adapter`.
    """

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


# ---------------------------------------------------------------------------
# SDK-backed adapter (``openai`` package). Reached via create_claim_llm_adapter.
# ---------------------------------------------------------------------------


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

    from gemynd.ingest.extractors.claim_cache import (
        CachedClaimAdapter,
        DEFAULT_CACHE_PATH,
        system_prompt_hash,
    )

    # Canonical fingerprint helper — must not be hand-rolled here, or pre-warmed
    # cache entries land under keys the runtime never looks up.
    prompt_hash = system_prompt_hash(adapter._system_prompt)
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
