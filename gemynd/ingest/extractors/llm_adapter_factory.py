"""Provider dispatch for the claim-extraction LLM adapter.

A single seam that selects which ``ClaimLLMAdapter`` backs the
``HybridClaimExtractor`` LLM path, so callers don't hardcode Anthropic.

Selection:
  - ``CLAIM_LLM_PROVIDER`` (explicit) wins: ``anthropic`` | ``openai`` /
    ``openai_compatible`` / ``vllm`` / ``ollama`` / ``tgi`` / ``local``.
  - Otherwise, if ``CLAIM_LLM_BASE_URL`` is set, the OpenAI-compatible
    (self-hosted / open-source) path is chosen automatically.
  - Otherwise the Anthropic path is used (existing default behavior).

Either factory degrades to ``NullLLMAdapter`` when its backend is unconfigured,
so an unset environment never raises — the LLM layer simply goes quiet.
"""
from __future__ import annotations

import logging
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig

_log = logging.getLogger(__name__)

_OPENAI_PROVIDERS = {
    "openai",
    "openai_compatible",
    "openai-compatible",
    "vllm",
    "ollama",
    "tgi",
    "local",
    "oss",
}


def _resolve_provider() -> str:
    explicit = (os.environ.get("CLAIM_LLM_PROVIDER") or "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("CLAIM_LLM_BASE_URL"):
        return "openai_compatible"
    return "anthropic"


def try_create_llm_adapter(
    config: "DomainConfig | None" = None,
    *,
    token_logger: Any | None = None,
) -> Any:
    """Create the configured claim-extraction LLM adapter (provider-dispatched)."""
    provider = _resolve_provider()
    if provider in _OPENAI_PROVIDERS:
        from gemynd.ingest.extractors.openai_claim_adapter import try_create_openai_adapter

        _log.info("Claim LLM provider: %s (OpenAI-compatible)", provider)
        return try_create_openai_adapter(config=config, token_logger=token_logger)

    if provider != "anthropic":
        _log.warning("Unknown CLAIM_LLM_PROVIDER=%r — defaulting to anthropic", provider)
    from gemynd.ingest.extractors.anthropic_claim_adapter import try_create_anthropic_adapter

    return try_create_anthropic_adapter(config=config, token_logger=token_logger)
