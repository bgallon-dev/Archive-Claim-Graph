"""Tests for the OpenAI-compatible (open-source / self-hosted) claim adapter."""
from __future__ import annotations

import io
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from gemynd.ingest.extractors.anthropic_claim_adapter import _build_system_prompt
from gemynd.ingest.extractors.claim_extractor import LLMClaimExtractor, NullLLMAdapter
from gemynd.ingest.extractors.openai_claim_adapter import (
    OpenAICompatClaimAdapter,
    OpenAICompatibleClaimAdapter,
    _build_claim_response_format,
    _normalize_endpoint,
    create_claim_llm_adapter,
    try_create_openai_adapter,
)

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig


def _capture_request():
    """A fake urlopen that records the request body and returns one claim."""
    captured: dict = {}

    @contextmanager
    def _fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["body"] = json.loads(request.data.decode("utf-8"))
        out = json.dumps(
            {"choices": [{"message": {"content": json.dumps([{"source_sentence": "A fact stated plainly here."}])}}]}
        ).encode("utf-8")
        yield io.BytesIO(out)

    return _fake_urlopen, captured


@pytest.mark.parametrize(
    "base, expected",
    [
        ("http://h:8000/v1", "http://h:8000/v1/chat/completions"),
        ("http://h:8000/v1/", "http://h:8000/v1/chat/completions"),
        ("http://h:8000/v1/chat/completions", "http://h:8000/v1/chat/completions"),
        ("https://x.trycloudflare.com/v1  ", "https://x.trycloudflare.com/v1/chat/completions"),
    ],
)
def test_normalize_endpoint(base: str, expected: str) -> None:
    assert _normalize_endpoint(base) == expected


def test_missing_base_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIM_LLM_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatibleClaimAdapter(base_url=None, config=None)


def _mock_chat_response(content: str):
    @contextmanager
    def _fake_urlopen(request, timeout=None):  # noqa: ANN001
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        yield io.BytesIO(body)

    return _fake_urlopen


def test_extract_claims_parses_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAICompatibleClaimAdapter(
        base_url="http://localhost:8000/v1", model="qwen", config=None
    )
    claims_json = json.dumps(
        [{"source_sentence": "The strike began on Monday.", "claim_type": "labor_action"}]
    )
    monkeypatch.setattr(
        "gemynd.ingest.extractors.openai_claim_adapter.urllib.request.urlopen",
        _mock_chat_response(claims_json),
    )
    rows = adapter.extract_claims("The strike began on Monday in the rail yards downtown.")
    assert len(rows) == 1
    assert rows[0]["claim_type"] == "labor_action"


def test_extract_claims_handles_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAICompatibleClaimAdapter(base_url="http://localhost:8000/v1", config=None)
    fenced = "```json\n[{\"source_sentence\": \"A fact stated here plainly.\"}]\n```"
    monkeypatch.setattr(
        "gemynd.ingest.extractors.openai_claim_adapter.urllib.request.urlopen",
        _mock_chat_response(fenced),
    )
    rows = adapter.extract_claims("A fact stated here plainly for extraction purposes.")
    assert rows and rows[0]["source_sentence"].startswith("A fact")


def test_short_paragraph_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAICompatibleClaimAdapter(base_url="http://localhost:8000/v1", config=None)
    called = {"n": 0}

    def _should_not_call(*_a, **_k):  # noqa: ANN002, ANN003
        called["n"] += 1
        raise AssertionError("network should not be hit for short input")

    monkeypatch.setattr(
        "gemynd.ingest.extractors.openai_claim_adapter.urllib.request.urlopen",
        _should_not_call,
    )
    assert adapter.extract_claims("too short") == []
    assert called["n"] == 0


def test_network_error_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAICompatibleClaimAdapter(base_url="http://localhost:8000/v1", config=None)

    def _raise(*_a, **_k):  # noqa: ANN002, ANN003
        raise OSError("connection refused")

    monkeypatch.setattr(
        "gemynd.ingest.extractors.openai_claim_adapter.urllib.request.urlopen",
        _raise,
    )
    assert adapter.extract_claims("A long enough paragraph to pass the minimum length gate.") == []


def test_response_format_schema_shape() -> None:
    rf = _build_claim_response_format(None)
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]["schema"]
    # Root must be an array — that's what _parse_response accepts.
    assert schema["type"] == "array"
    claim = schema["items"]
    assert claim["required"] == ["source_sentence"]
    assert claim["properties"]["epistemic_status"]["enum"] == [
        "observed", "estimated", "uncertain", "reported"
    ]


def _claim_props(profile):
    return _build_claim_response_format(None, profile=profile)["json_schema"]["schema"]["items"]["properties"]


def test_schema_profiles_field_sets() -> None:
    full = _claim_props("full")
    trim = _claim_props("trim")
    min_epi = _claim_props("min_epi")
    bare = _claim_props("min")
    # full carries the verbose fields
    assert {"normalized_sentence", "extraction_confidence", "evidence_start", "notes"} <= set(full)
    # trim drops char offsets + notes but keeps normalized/confidence/date
    assert "evidence_start" not in trim and "notes" not in trim
    assert {"normalized_sentence", "extraction_confidence", "claim_date"} <= set(trim)
    # min_epi keeps only source_sentence, claim_type, epistemic_status, claim_links
    assert set(min_epi) == {"source_sentence", "claim_type", "epistemic_status", "claim_links"}
    # its links are lean: surface_form + entity_type_hint only
    assert set(min_epi["claim_links"]["items"]["properties"]) == {"surface_form", "entity_type_hint"}
    # bare 'min' additionally drops epistemic_status
    assert "epistemic_status" not in bare


def test_schema_profile_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_LLM_SCHEMA_PROFILE", "min_epi")
    props = _build_claim_response_format(None)["json_schema"]["schema"]["items"]["properties"]
    assert set(props) == {"source_sentence", "claim_type", "epistemic_status", "claim_links"}


def test_unknown_schema_profile_falls_back_to_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_LLM_SCHEMA_PROFILE", "bogus")
    props = _build_claim_response_format(None)["json_schema"]["schema"]["items"]["properties"]
    assert "extraction_confidence" in props  # fell back to full


def test_json_schema_sent_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIM_LLM_JSON_SCHEMA", raising=False)
    monkeypatch.delenv("CLAIM_LLM_EXTRA_BODY", raising=False)
    adapter = OpenAICompatibleClaimAdapter(base_url="http://localhost:8000/v1", config=None)
    fake, captured = _capture_request()
    monkeypatch.setattr(
        "gemynd.ingest.extractors.openai_claim_adapter.urllib.request.urlopen", fake
    )
    adapter.extract_claims("A long enough paragraph to pass the minimum length gate.")
    assert captured["body"]["response_format"]["json_schema"]["name"] == "claim_extraction"


def test_json_schema_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_LLM_JSON_SCHEMA", "0")
    monkeypatch.delenv("CLAIM_LLM_EXTRA_BODY", raising=False)
    adapter = OpenAICompatibleClaimAdapter(base_url="http://localhost:8000/v1", config=None)
    fake, captured = _capture_request()
    monkeypatch.setattr(
        "gemynd.ingest.extractors.openai_claim_adapter.urllib.request.urlopen", fake
    )
    adapter.extract_claims("A long enough paragraph to pass the minimum length gate.")
    assert "response_format" not in captured["body"]


def test_extra_body_env_merged_last(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_LLM_EXTRA_BODY", '{"reasoning_effort": "none"}')
    monkeypatch.delenv("CLAIM_LLM_JSON_SCHEMA", raising=False)
    adapter = OpenAICompatibleClaimAdapter(base_url="http://localhost:8000/v1", config=None)
    fake, captured = _capture_request()
    monkeypatch.setattr(
        "gemynd.ingest.extractors.openai_claim_adapter.urllib.request.urlopen", fake
    )
    adapter.extract_claims("A long enough paragraph to pass the minimum length gate.")
    assert captured["body"]["reasoning_effort"] == "none"


def test_retries_without_response_format_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIM_LLM_JSON_SCHEMA", raising=False)
    monkeypatch.delenv("CLAIM_LLM_EXTRA_BODY", raising=False)
    adapter = OpenAICompatibleClaimAdapter(base_url="http://localhost:8000/v1", config=None)
    bodies: list[dict] = []

    @contextmanager
    def _fake(request, timeout=None):  # noqa: ANN001
        body = json.loads(request.data.decode("utf-8"))
        bodies.append(body)
        if "response_format" in body:  # first attempt: simulate 400
            raise OSError("400 Bad Request")
        out = json.dumps(
            {"choices": [{"message": {"content": json.dumps([{"source_sentence": "Recovered claim sentence here."}])}}]}
        ).encode("utf-8")
        yield io.BytesIO(out)

    monkeypatch.setattr(
        "gemynd.ingest.extractors.openai_claim_adapter.urllib.request.urlopen", _fake
    )
    rows = adapter.extract_claims("A long enough paragraph to pass the minimum length gate.")
    assert len(bodies) == 2  # tried with schema, then retried without
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
    assert rows and rows[0]["source_sentence"].startswith("Recovered")


def test_factory_falls_back_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIM_LLM_BASE_URL", raising=False)
    from gemynd.ingest.extractors.claim_extractor import NullLLMAdapter

    adapter = try_create_openai_adapter(config=None)
    assert isinstance(adapter, NullLLMAdapter)


def test_factory_builds_cached_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setenv("CLAIM_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("CLAIM_LLM_MODEL", "qwen-test")
    from gemynd.ingest.extractors.claim_cache import CachedClaimAdapter

    adapter = try_create_openai_adapter(config=None, cache_path=str(tmp_path / "c.sqlite"))
    assert isinstance(adapter, CachedClaimAdapter)

# ---------------------------------------------------------------------------
# SDK-backed adapter (OpenAICompatClaimAdapter / create_claim_llm_adapter)
# ---------------------------------------------------------------------------


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
