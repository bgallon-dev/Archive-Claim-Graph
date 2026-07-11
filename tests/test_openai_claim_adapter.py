"""Tests for the OpenAI-compatible (LM Studio) claim extraction adapter."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from gemynd.ingest.extractors.anthropic_claim_adapter import _build_system_prompt
from gemynd.ingest.extractors.claim_extractor import LLMClaimExtractor, NullLLMAdapter
from gemynd.ingest.extractors.openai_claim_adapter import (
    OpenAICompatClaimAdapter,
    create_claim_llm_adapter,
)

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter_with_mock_client(
    config: "DomainConfig | None" = None,
    response_format: dict | None = None,
) -> tuple[OpenAICompatClaimAdapter, MagicMock]:
    """Create an adapter with a mocked OpenAI client, bypassing __init__."""
    adapter = OpenAICompatClaimAdapter.__new__(OpenAICompatClaimAdapter)
    mock_client = MagicMock()
    adapter._client = mock_client
    adapter._model = "local-model"
    adapter._max_tokens = 2048
    adapter._timeout = 120.0
    adapter._temperature = 0.0
    adapter._system_prompt = _build_system_prompt(config)
    adapter._response_format = response_format
    adapter._schema_disabled = False
    return adapter, mock_client


def _mock_response(text: str) -> MagicMock:
    """Build a mock OpenAI chat-completions response: choices[0].message.content."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# OpenAICompatClaimAdapter tests
# ---------------------------------------------------------------------------

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
    client.chat.completions.create.return_value = _mock_response(claims_json)

    result = adapter.extract_claims("Approximately 50 mallards were observed on the refuge.")
    assert len(result) == 1
    assert result[0]["claim_type"] == "population_estimate"
    assert result[0]["claim_links"][0]["relation_type"] == "SPECIES_FOCUS"


def test_extract_claims_empty_text_returns_empty() -> None:
    adapter, client = _make_adapter_with_mock_client()
    assert adapter.extract_claims("") == []
    assert adapter.extract_claims("   ") == []
    client.chat.completions.create.assert_not_called()


def test_extract_claims_short_paragraph_skips_call() -> None:
    adapter, client = _make_adapter_with_mock_client()
    assert adapter.extract_claims("too short") == []
    client.chat.completions.create.assert_not_called()


def test_extract_claims_api_error_returns_empty() -> None:
    adapter, client = _make_adapter_with_mock_client()
    client.chat.completions.create.side_effect = Exception("connection refused")
    result = adapter.extract_claims("Some paragraph text that is plenty long enough here.")
    assert result == []


def test_extract_claims_malformed_response_returns_empty() -> None:
    adapter, client = _make_adapter_with_mock_client()
    client.chat.completions.create.return_value = _mock_response("I cannot parse this.")
    result = adapter.extract_claims("Some paragraph text that is plenty long enough here.")
    assert result == []


def test_extract_claims_none_content_returns_empty() -> None:
    adapter, client = _make_adapter_with_mock_client()
    client.chat.completions.create.return_value = _mock_response(None)  # type: ignore[arg-type]
    result = adapter.extract_claims("Some paragraph text that is plenty long enough here.")
    assert result == []


def test_extract_claims_uses_system_and_user_messages() -> None:
    adapter, client = _make_adapter_with_mock_client()
    client.chat.completions.create.return_value = _mock_response("[]")
    adapter.extract_claims(
        "Some paragraph text that is long enough to pass the minimum length filter."
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": adapter._system_prompt}
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["temperature"] == 0.0


def test_extract_claims_flattens_nested_claim_links() -> None:
    """Small models often group links by relation type; they must be flattened."""
    adapter, client = _make_adapter_with_mock_client()
    nested_json = json.dumps([
        {
            "source_sentence": "Approximately 50 mallards were observed on the refuge.",
            "normalized_sentence": "a paraphrased version that would break the merge",
            "claim_type": "species_presence",
            "claim_links": [
                {"SPECIES_FOCUS": [{"surface_form": "mallards", "normalized_form": "Mallard"}]},
                {"HABITAT_FOCUS": {"surface_form": "refuge", "normalized_form": "Refuge"}},
                {"relation_type": "TOPIC_OF_CLAIM", "surface_form": "observed"},
            ],
        }
    ])
    client.chat.completions.create.return_value = _mock_response(nested_json)

    result = adapter.extract_claims("Approximately 50 mallards were observed on the refuge today.")
    # paraphrased normalized_sentence is dropped so the deterministic merge key wins
    assert "normalized_sentence" not in result[0]
    links = result[0]["claim_links"]
    assert len(links) == 3
    by_rel = {l["relation_type"]: l["surface_form"] for l in links}
    assert by_rel == {
        "SPECIES_FOCUS": "mallards",
        "HABITAT_FOCUS": "refuge",   # single-dict (non-list) nesting also handled
        "TOPIC_OF_CLAIM": "observed",  # already-flat row passes through
    }


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
    client.chat.completions.create.return_value = _mock_response(claims_json)

    drafts = LLMClaimExtractor(adapter).extract("A fire broke out on August 15 in the south unit.")
    assert len(drafts) == 1
    assert drafts[0].claim_type == "fire_incident"
    assert drafts[0].extraction_source == "llm"


# ---------------------------------------------------------------------------
# create_claim_llm_adapter (provider switch) tests
# ---------------------------------------------------------------------------

@patch("gemynd.ingest.extractors.openai_claim_adapter._openai_pkg", None)
def test_factory_local_without_openai_pkg_returns_null() -> None:
    result = create_claim_llm_adapter(provider="lmstudio")
    assert isinstance(result, NullLLMAdapter)


@patch("gemynd.ingest.extractors.openai_claim_adapter._openai_pkg")
def test_factory_local_returns_cached_adapter_unmetered(mock_openai, tmp_path) -> None:
    from gemynd.ingest.extractors.claim_cache import CachedClaimAdapter

    sentinel_logger = object()
    result = create_claim_llm_adapter(
        provider="lmstudio",
        token_logger=sentinel_logger,
        cache_path=str(tmp_path / "claim_cache.db"),
    )
    assert isinstance(result, CachedClaimAdapter)
    assert isinstance(result._inner, OpenAICompatClaimAdapter)
    # Local models are free: the cache must NOT carry a token logger.
    assert result._token_logger is None


@patch("gemynd.ingest.extractors.openai_claim_adapter._openai_pkg")
def test_factory_local_disable_cache_returns_raw_adapter(mock_openai) -> None:
    result = create_claim_llm_adapter(provider="lmstudio", disable_cache=True)
    assert isinstance(result, OpenAICompatClaimAdapter)


@patch.dict("os.environ", {}, clear=True)
def test_factory_anthropic_without_key_returns_null() -> None:
    result = create_claim_llm_adapter(provider="anthropic")
    assert isinstance(result, NullLLMAdapter)


# ---------------------------------------------------------------------------
# JSON-schema constrained decoding (structured output)
# ---------------------------------------------------------------------------

def test_schema_passed_and_object_wrapped_response_parsed() -> None:
    rf = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {}}}
    adapter, client = _make_adapter_with_mock_client(response_format=rf)
    # json_schema mode returns the array wrapped in a top-level object
    wrapped = json.dumps({"claims": [{"source_sentence": "A fire broke out.", "claim_type": "fire_incident"}]})
    client.chat.completions.create.return_value = _mock_response(wrapped)

    result = adapter.extract_claims("A fire broke out in the south unit of the refuge today.")
    assert len(result) == 1 and result[0]["claim_type"] == "fire_incident"
    # the schema must have been sent on the request
    assert client.chat.completions.create.call_args.kwargs["response_format"] is rf


def test_schema_rejected_falls_back_to_plain_call() -> None:
    rf = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {}}}
    adapter, client = _make_adapter_with_mock_client(response_format=rf)
    plain = _mock_response(json.dumps([{"source_sentence": "A fire broke out.", "claim_type": "fire_incident"}]))
    # first call (with schema) raises, second call (no schema) succeeds
    client.chat.completions.create.side_effect = [Exception("json_schema unsupported"), plain]

    result = adapter.extract_claims("A fire broke out in the south unit of the refuge today.")
    assert len(result) == 1
    assert client.chat.completions.create.call_count == 2
    assert "response_format" not in client.chat.completions.create.call_args.kwargs
    assert adapter._schema_disabled is True


def test_build_response_format_enum_from_contract() -> None:
    from gemynd.ingest.extractors.openai_claim_adapter import _build_response_format
    from gemynd.core.claim_contract import EXTRACTOR_CLAIM_LINK_RELATIONS
    rf = _build_response_format(None)
    schema = rf["json_schema"]["schema"]
    claim = schema["properties"]["claims"]["items"]
    assert claim["required"] == ["source_sentence", "claim_type"]
    assert "claim_type" in claim["properties"] and claim["properties"]["claim_type"]["enum"]
    link = claim["properties"]["claim_links"]["items"]
    assert set(link["properties"]["relation_type"]["enum"]) == set(EXTRACTOR_CLAIM_LINK_RELATIONS)
