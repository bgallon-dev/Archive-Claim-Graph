"""Tests for the OpenAI-compatible (open-source / self-hosted) claim adapter."""
from __future__ import annotations

import io
import json
from contextlib import contextmanager

import pytest

from gemynd.ingest.extractors.openai_claim_adapter import (
    OpenAICompatibleClaimAdapter,
    _build_claim_response_format,
    _normalize_endpoint,
    try_create_openai_adapter,
)


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
