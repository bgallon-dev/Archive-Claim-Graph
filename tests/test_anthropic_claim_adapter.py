"""Tests for the Anthropic-backed ClaimLLMAdapter."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gemynd.ingest.extractors.anthropic_claim_adapter import (
    AnthropicClaimAdapter,
    _build_system_prompt,
    _parse_response,
    try_create_anthropic_adapter,
)
from gemynd.ingest.extractors.claim_extractor import (
    LLMClaimExtractor,
    NullLLMAdapter,
)


# ---------------------------------------------------------------------------
# _parse_response unit tests
# ---------------------------------------------------------------------------

def test_parse_valid_json_array() -> None:
    raw = json.dumps([{"source_sentence": "50 mallards observed.", "claim_type": "population_estimate"}])
    result = _parse_response(raw)
    assert len(result) == 1
    assert result[0]["source_sentence"] == "50 mallards observed."


def test_parse_empty_array() -> None:
    assert _parse_response("[]") == []


def test_parse_strips_markdown_fences() -> None:
    inner = json.dumps([{"source_sentence": "Fire broke out.", "claim_type": "fire_incident"}])
    raw = f"```json\n{inner}\n```"
    result = _parse_response(raw)
    assert len(result) == 1
    assert result[0]["claim_type"] == "fire_incident"


def test_parse_malformed_json_returns_empty() -> None:
    assert _parse_response("not json at all") == []


def test_parse_non_list_json_returns_empty() -> None:
    assert _parse_response('{"source_sentence": "hello"}') == []


def test_parse_filters_non_dict_items() -> None:
    raw = json.dumps([{"source_sentence": "A."}, "stray string", 42, {"source_sentence": "B."}])
    result = _parse_response(raw)
    assert len(result) == 2
    assert result[0]["source_sentence"] == "A."
    assert result[1]["source_sentence"] == "B."


# ---------------------------------------------------------------------------
# _build_system_prompt tests
# ---------------------------------------------------------------------------

def test_build_prompt_without_config_uses_defaults() -> None:
    prompt = _build_system_prompt(None)
    assert "domain-specific documents" in prompt
    # No-config fallback carries only the sentinel claim type and the
    # domain-neutral relation vocabulary.
    assert "unclassified_assertion" in prompt
    assert "SPECIES_FOCUS" in prompt
    assert "source_sentence" in prompt


def test_build_prompt_with_config_uses_domain_context() -> None:
    """When a DomainConfig is provided, the prompt uses its synthesis_context,
    claim types from patterns, and entity types from seed entities."""
    import re
    from gemynd.core.domain_config import DomainConfig

    config = DomainConfig(
        seed_entities=[],
        claim_type_patterns=[
            ("accident_finding", re.compile(r"accident"), 1.0),
            ("probable_cause", re.compile(r"cause"), 1.0),
        ],
        claim_role_policy={},
        measurement_units={},
        measurement_species={},
        ocr_corrections=frozenset(),
        ocr_correction_map={},
        negative_lexicon=frozenset(),
        preferred_entity_types={},
        compatibility_matrix={},
        derivation_registry={},
        domain_anchor=None,
        year_validation=None,
        synthesis_context="NTSB aviation accident docket reports",
        claim_entity_relation_precedence=("AIRCRAFT_FOCUS", "CAUSE_FACTOR"),
        claim_entity_relations=frozenset({"AIRCRAFT_FOCUS", "CAUSE_FACTOR"}),
        relation_to_entity_type_hints={},
        claim_location_relation="OCCURRED_AT",
        entity_labels=frozenset(),
        legacy_renames={},
        allowed_claim_types=frozenset({"accident_finding", "probable_cause", "unclassified_assertion"}),
        observation_eligible_types=frozenset(),
        event_eligible_types=frozenset(),
        concept_rules=[],
        query_intent_to_claim_types={},
    )
    prompt = _build_system_prompt(config)
    assert "NTSB aviation accident docket reports" in prompt
    assert "accident_finding" in prompt
    assert "probable_cause" in prompt
    assert "unclassified_assertion" in prompt  # always included


# ---------------------------------------------------------------------------
# AnthropicClaimAdapter tests
# ---------------------------------------------------------------------------

def _make_adapter_with_mock_client(
    config: "DomainConfig | None" = None,
) -> tuple[AnthropicClaimAdapter, MagicMock]:
    """Create an adapter with a mocked Anthropic client, bypassing __init__."""
    adapter = AnthropicClaimAdapter.__new__(AnthropicClaimAdapter)
    mock_client = MagicMock()
    adapter._client = mock_client
    adapter._model = "claude-sonnet-4-6"
    adapter._max_tokens = 2048
    adapter._timeout = 60.0
    adapter._system_prompt = _build_system_prompt(config)
    return adapter, mock_client


def _mock_response(text: str) -> MagicMock:
    """Build a mock Anthropic response with given text content."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


def test_extract_claims_happy_path() -> None:
    adapter, client = _make_adapter_with_mock_client()
    claims_json = json.dumps([
        {
            "source_sentence": "Approximately 50 mallards were observed.",
            "claim_type": "population_estimate",
            "epistemic_status": "observed",
            "extraction_confidence": 0.85,
            "claim_links": [
                {
                    "relation_type": "SPECIES_FOCUS",
                    "surface_form": "mallards",
                    "normalized_form": "mallard",
                    "entity_type_hint": "Species",
                }
            ],
        }
    ])
    client.messages.create.return_value = _mock_response(claims_json)

    result = adapter.extract_claims("Approximately 50 mallards were observed on the refuge.")
    assert len(result) == 1
    assert result[0]["claim_type"] == "population_estimate"
    assert result[0]["claim_links"][0]["relation_type"] == "SPECIES_FOCUS"


def test_extract_claims_empty_text_returns_empty() -> None:
    adapter, _ = _make_adapter_with_mock_client()
    assert adapter.extract_claims("") == []
    assert adapter.extract_claims("   ") == []


def test_extract_claims_api_error_returns_empty() -> None:
    adapter, client = _make_adapter_with_mock_client()
    client.messages.create.side_effect = Exception("API rate limit exceeded")
    result = adapter.extract_claims("Some paragraph text.")
    assert result == []


def test_extract_claims_malformed_response_returns_empty() -> None:
    adapter, client = _make_adapter_with_mock_client()
    client.messages.create.return_value = _mock_response("I can't parse this document.")
    result = adapter.extract_claims("Some paragraph text.")
    assert result == []


def test_extract_claims_uses_system_prompt() -> None:
    """Verify the adapter passes its system prompt to the API call."""
    adapter, client = _make_adapter_with_mock_client()
    client.messages.create.return_value = _mock_response("[]")
    adapter.extract_claims("Some paragraph text that is long enough to pass the minimum length filter for claims.")
    call_kwargs = client.messages.create.call_args
    assert call_kwargs.kwargs["system"] == adapter._system_prompt


# ---------------------------------------------------------------------------
# Integration: adapter -> LLMClaimExtractor -> ClaimDraft
# ---------------------------------------------------------------------------

def test_adapter_to_llm_extractor_produces_claim_drafts() -> None:
    adapter, client = _make_adapter_with_mock_client()
    claims_json = json.dumps([
        {
            "source_sentence": "A fire broke out on August 15.",
            "claim_type": "fire_incident",
            "epistemic_status": "reported",
            "extraction_confidence": 0.9,
            "claim_date": "1945-08-15",
        },
    ])
    client.messages.create.return_value = _mock_response(claims_json)

    extractor = LLMClaimExtractor(adapter)
    drafts = extractor.extract("A fire broke out on August 15 in the south unit.")

    assert len(drafts) == 1
    assert drafts[0].claim_type == "fire_incident"
    assert drafts[0].extraction_source == "llm"
    assert drafts[0].extraction_confidence == 0.9
    assert drafts[0].claim_date == "1945-08-15"


# ---------------------------------------------------------------------------
# try_create_anthropic_adapter tests
# ---------------------------------------------------------------------------

@patch.dict("os.environ", {}, clear=True)
def test_try_create_no_api_key_returns_null_adapter() -> None:
    result = try_create_anthropic_adapter()
    assert isinstance(result, NullLLMAdapter)


@patch.dict("os.environ", {"Anthropic_API_Key": "sk-test-key"})
def test_try_create_with_key_returns_cached_adapter() -> None:
    import gemynd.ingest.extractors.anthropic_claim_adapter as mod
    if mod._anthropic_pkg is None:
        pytest.skip("anthropic package not installed")
    from gemynd.ingest.extractors.claim_cache import CachedClaimAdapter
    result = try_create_anthropic_adapter()
    assert isinstance(result, CachedClaimAdapter)
    assert isinstance(result._inner, AnthropicClaimAdapter)


@patch.dict("os.environ", {"Anthropic_API_Key": "sk-test-key"})
def test_try_create_disable_cache_returns_raw_adapter() -> None:
    import gemynd.ingest.extractors.anthropic_claim_adapter as mod
    if mod._anthropic_pkg is None:
        pytest.skip("anthropic package not installed")
    result = try_create_anthropic_adapter(disable_cache=True)
    assert isinstance(result, AnthropicClaimAdapter)


@patch("gemynd.ingest.extractors.anthropic_claim_adapter._anthropic_pkg", None)
def test_try_create_no_anthropic_pkg_returns_null_adapter() -> None:
    result = try_create_anthropic_adapter()
    assert isinstance(result, NullLLMAdapter)
