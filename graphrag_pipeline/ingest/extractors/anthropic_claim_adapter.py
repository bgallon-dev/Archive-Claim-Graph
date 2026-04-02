"""Anthropic-backed ClaimLLMAdapter for the HybridClaimExtractor LLM path."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from graphrag_pipeline.core.domain_config import DomainConfig

_log = logging.getLogger(__name__)

try:
    import anthropic as _anthropic_pkg
except ImportError:  # pragma: no cover
    _anthropic_pkg = None  # type: ignore[assignment]

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)


def _build_system_prompt(config: "DomainConfig | None" = None) -> str:
    """Build the claim extraction system prompt from domain config.

    When *config* is provided, claim types, entity types, and domain context
    are derived dynamically so the adapter works for any domain — not just
    Turnbull.  When *config* is ``None``, the prompt uses the contract-level
    defaults from ``claim_contract.py``.
    """
    from graphrag_pipeline.core.claim_contract import (
        ALLOWED_CLAIM_TYPES,
        EXTRACTOR_CLAIM_LINK_RELATIONS,
        RELATION_TO_ENTITY_TYPE_HINTS,
        UNCLASSIFIED_TYPE,
    )

    # --- claim types ---
    if config is not None:
        claim_types = sorted({ct for ct, _, _ in config.claim_type_patterns} | {UNCLASSIFIED_TYPE})
    else:
        claim_types = sorted(ALLOWED_CLAIM_TYPES)

    # --- entity types ---
    if config is not None:
        entity_types = sorted({e.entity_type for e in config.seed_entities})
    else:
        entity_type_set: set[str] = set()
        for types in RELATION_TO_ENTITY_TYPE_HINTS.values():
            entity_type_set |= types
        entity_types = sorted(entity_type_set)

    # --- relation types + hints ---
    relation_types = sorted(EXTRACTOR_CLAIM_LINK_RELATIONS)
    hint_lines: list[str] = []
    for rel in relation_types:
        hints = RELATION_TO_ENTITY_TYPE_HINTS.get(rel)
        if hints:
            hint_lines.append(f"    - {rel} → entity types: {', '.join(sorted(hints))}")
        else:
            hint_lines.append(f"    - {rel}")
    relation_hint_block = "\n".join(hint_lines)

    # --- domain context ---
    if config is not None and config.synthesis_context:
        domain_context = config.synthesis_context.strip()
    else:
        domain_context = "domain-specific documents"

    return f"""\
You are a structured information extraction engine for {domain_context}.

Extract factual claims from the given paragraph. Each claim is a single \
assertive sentence that states a fact, observation, action, or condition.

Return a JSON array of claim objects. If no claims can be extracted, return [].

Each claim object must have these fields:

Required:
- "source_sentence" (string): the exact sentence from the input text

Optional:
- "normalized_sentence" (string): cleaned/normalized version of the sentence
- "claim_type" (string): one of the following types:
    {', '.join(claim_types)}
- "epistemic_status" (string): one of "observed", "estimated", "uncertain", "reported"
- "extraction_confidence" (float 0.0–1.0): your confidence in the extraction
- "claim_date" (string or null): date referenced in the claim (ISO format preferred)
- "notes" (string): any relevant extraction notes
- "evidence_start" (int or null): character offset where the sentence begins in the input
- "evidence_end" (int or null): character offset where the sentence ends in the input
- "claim_links" (array): entities linked to this claim, each object with:
{relation_hint_block}
    - "surface_form" (string): the entity text as it appears in the sentence
    - "normalized_form" (string): standardized name
    - "entity_type_hint" (string or null): one of {', '.join(entity_types)}
    - "start_offset" (int or null): character offset of entity within the sentence
    - "end_offset" (int or null): character offset of entity end within the sentence

Return ONLY valid JSON — no markdown fences, no explanatory text."""


class AnthropicClaimAdapter:
    """ClaimLLMAdapter implementation backed by the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        config: "DomainConfig | None" = None,
        token_logger: Any | None = None,
    ) -> None:
        if _anthropic_pkg is None:
            raise RuntimeError(
                "anthropic package is not installed. "
                "Install with: pip install anthropic"
            )
        resolved_key = (
            api_key
            or os.environ.get("Anthropic_API_Key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key found. Set Anthropic_API_Key or "
                "ANTHROPIC_API_KEY in your environment."
            )
        _raw_client = _anthropic_pkg.Anthropic(api_key=resolved_key)
        if token_logger is not None:
            from graphrag_pipeline.shared.token_tracker import MeteredAnthropicClient
            self._client = MeteredAnthropicClient(_raw_client, token_logger, caller="claim_extraction")
        else:
            self._client = _raw_client
        self._model = (
            model
            or os.environ.get("CLAIM_EXTRACTION_MODEL")
            or os.environ.get("SYNTHESIS_MODEL")
            or "claude-haiku-4-5"
        )
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._system_prompt = _build_system_prompt(config)

    _MIN_LENGTH = int(os.environ.get("CLAIM_MIN_PARAGRAPH_LENGTH", "40"))

    def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
        """Call Claude to extract structured claims from a paragraph."""
        text = paragraph_text.strip()
        if not text:
            return []
        if len(text) < self._MIN_LENGTH:
            _log.debug(
                "Skipping short paragraph (%d chars < %d): %.30s…",
                len(text), self._MIN_LENGTH, text,
            )
            return []
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=[{"role": "user", "content": text}],
                timeout=self._timeout,
            )
            raw = response.content[0].text.strip()
        except Exception as exc:
            _log.warning("Anthropic claim extraction API call failed: %s", exc)
            return []

        return _parse_response(raw)


def _parse_response(raw: str) -> list[dict[str, Any]]:
    """Parse Claude's JSON response, stripping markdown fences if present."""
    fence_match = _FENCE_RE.match(raw)
    if fence_match:
        raw = fence_match.group(1)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        _log.warning("Failed to parse claim extraction JSON: %s", exc)
        return []
    if not isinstance(parsed, list):
        _log.warning("Claim extraction returned non-list JSON: %s", type(parsed).__name__)
        return []
    return [item for item in parsed if isinstance(item, dict)]


def try_create_anthropic_adapter(
    config: "DomainConfig | None" = None,
    *,
    cache_path: str | None = None,
    disable_cache: bool = False,
    token_logger: Any | None = None,
) -> Any:
    """Attempt to create an AnthropicClaimAdapter; fall back to NullLLMAdapter.

    Returns an object satisfying the ClaimLLMAdapter protocol.  When caching is
    enabled (the default), the adapter is wrapped in a ``CachedClaimAdapter``
    that stores responses in a SQLite database so re-runs skip the API call for
    already-processed paragraphs.
    """
    from graphrag_pipeline.ingest.extractors.claim_extractor import NullLLMAdapter

    if _anthropic_pkg is None:
        _log.debug("anthropic package not installed — using NullLLMAdapter for claim LLM path")
        return NullLLMAdapter()
    api_key = os.environ.get("Anthropic_API_Key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _log.debug("No Anthropic API key found — using NullLLMAdapter for claim LLM path")
        return NullLLMAdapter()
    try:
        adapter = AnthropicClaimAdapter(api_key=api_key, config=config, token_logger=token_logger)
    except Exception as exc:
        _log.warning("Failed to create AnthropicClaimAdapter: %s — falling back to NullLLMAdapter", exc)
        return NullLLMAdapter()

    if not disable_cache:
        import hashlib as _hl
        from graphrag_pipeline.ingest.extractors.claim_cache import (
            CachedClaimAdapter,
            DEFAULT_CACHE_PATH,
        )

        prompt_hash = _hl.sha256(
            adapter._system_prompt.encode("utf-8")
        ).hexdigest()[:16]
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
