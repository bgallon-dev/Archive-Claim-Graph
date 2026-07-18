"""Tests for gemynd.shared.openai_compat — the query-time LLM provider seam.

The adapter must be a drop-in for the Anthropic Messages surface that
synthesis.py / llm_cypher.py / MeteredAnthropicClient consume, backed by an
OpenAI /v1/chat/completions server.  All HTTP is faked at urllib level.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from gemynd.shared.openai_compat import (
    OpenAICompatibleMessagesClient,
    create_query_llm_client,
    query_llm_provider,
)
from gemynd.shared.settings import Settings


class _FakeResponse:
    """Context-manager + line-iterable stand-in for urlopen's response."""

    def __init__(self, body: bytes = b"", lines: list[bytes] | None = None):
        self._body = body
        self._lines = lines or []
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _capture_urlopen(monkeypatch, responses):
    """Patch urlopen; record (url, body) per call; pop responses in order."""
    calls: list[tuple[str, dict]] = []

    def fake_urlopen(request, timeout=None):
        calls.append((request.full_url, json.loads(request.data.decode("utf-8"))))
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


_CREATE_PAYLOAD = {
    "model": "qwen-2.5-7b",
    "choices": [{"message": {"content": "MATCH (n) RETURN n"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 9},
}


class TestCreate:
    def test_maps_anthropic_surface(self, monkeypatch):
        calls = _capture_urlopen(monkeypatch, [
            _FakeResponse(body=json.dumps(_CREATE_PAYLOAD).encode())])
        client = OpenAICompatibleMessagesClient(
            "http://localhost:8000/v1", model="qwen-2.5-7b")
        msg = client.messages.create(
            model="qwen-2.5-7b", max_tokens=1024, system="You write Cypher.",
            messages=[{"role": "user", "content": "list nodes"}])

        url, body = calls[0]
        assert url == "http://localhost:8000/v1/chat/completions"
        assert body["messages"][0] == {"role": "system",
                                       "content": "You write Cypher."}
        assert body["messages"][1]["content"] == "list nodes"
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.0
        assert "stream" not in body

        assert msg.content[0].text == "MATCH (n) RETURN n"
        assert msg.usage.input_tokens == 120
        assert msg.usage.output_tokens == 9
        assert msg.stop_reason == "end_turn"
        assert msg.model == "qwen-2.5-7b"

    def test_length_maps_to_max_tokens(self, monkeypatch):
        payload = dict(_CREATE_PAYLOAD)
        payload["choices"] = [{"message": {"content": "x"},
                               "finish_reason": "length"}]
        _capture_urlopen(monkeypatch, [
            _FakeResponse(body=json.dumps(payload).encode())])
        client = OpenAICompatibleMessagesClient("http://h/v1", model="m")
        msg = client.messages.create(
            model="m", max_tokens=5, system="", messages=[])
        assert msg.stop_reason == "max_tokens"


def _sse(events: list[dict]) -> list[bytes]:
    lines = [b": keep-alive", b""]
    lines += [b"data: " + json.dumps(e).encode() for e in events]
    lines.append(b"data: [DONE]")
    return lines


_STREAM_EVENTS = [
    {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
    {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 2}},
]


class TestStream:
    def test_streams_chunks_and_final_message(self, monkeypatch):
        calls = _capture_urlopen(monkeypatch, [
            _FakeResponse(lines=_sse(_STREAM_EVENTS))])
        client = OpenAICompatibleMessagesClient("http://h/v1", model="m")
        with client.messages.stream(
                model="m", max_tokens=64, system="s",
                messages=[{"role": "user", "content": "hi"}],
                timeout=30.0) as stream:
            chunks = list(stream.text_stream)
            final = stream.get_final_message()

        assert calls[0][1]["stream"] is True
        assert calls[0][1]["stream_options"] == {"include_usage": True}
        assert chunks == ["Hel", "lo"]
        assert final.content[0].text == "Hello"
        assert final.usage.input_tokens == 40
        assert final.usage.output_tokens == 2
        assert final.stop_reason == "end_turn"

    def test_final_message_drains_unconsumed_stream(self, monkeypatch):
        _capture_urlopen(monkeypatch, [
            _FakeResponse(lines=_sse(_STREAM_EVENTS))])
        client = OpenAICompatibleMessagesClient("http://h/v1", model="m")
        with client.messages.stream(
                model="m", max_tokens=64, system="", messages=[]) as stream:
            final = stream.get_final_message()  # no prior iteration
        assert final.content[0].text == "Hello"

    def test_retries_without_stream_options_on_400(self, monkeypatch):
        err = urllib.error.HTTPError(
            "http://h", 400, "bad request", None, io.BytesIO(b""))
        calls = _capture_urlopen(monkeypatch, [
            err, _FakeResponse(lines=_sse(_STREAM_EVENTS))])
        client = OpenAICompatibleMessagesClient("http://h/v1", model="m")
        with client.messages.stream(
                model="m", max_tokens=64, system="", messages=[]) as stream:
            final = stream.get_final_message()
        assert "stream_options" not in calls[1][1]
        assert final.content[0].text == "Hello"


class TestProviderDispatch:
    def _settings(self, **kw) -> Settings:
        return Settings(**kw)

    def test_default_is_anthropic(self):
        assert query_llm_provider(self._settings()) == "anthropic"

    def test_base_url_selects_openai(self):
        s = self._settings(synthesis_llm_base_url="http://localhost:8000/v1")
        assert query_llm_provider(s) == "openai_compatible"

    def test_explicit_provider_wins(self):
        s = self._settings(synthesis_llm_provider="ollama")
        assert query_llm_provider(s) == "ollama"

    def test_unknown_provider_falls_back_to_anthropic(self):
        s = self._settings(synthesis_llm_provider="wat")
        assert query_llm_provider(s) == "anthropic"

    def test_extra_body_merged_into_request(self, monkeypatch):
        calls = _capture_urlopen(monkeypatch, [
            _FakeResponse(body=json.dumps(_CREATE_PAYLOAD).encode())])
        client = OpenAICompatibleMessagesClient(
            "http://h/v1", model="m",
            extra_body={"reasoning_effort": "low"})
        client.messages.create(model="m", max_tokens=8, system="", messages=[])
        assert calls[0][1]["reasoning_effort"] == "low"

    def test_factory_parses_extra_body_env(self):
        s = self._settings(
            synthesis_llm_base_url="http://h/v1",
            synthesis_llm_extra_body='{"reasoning_effort": "low"}')
        client, _ = create_query_llm_client(s)
        assert client.messages.extra_body == {"reasoning_effort": "low"}

    def test_factory_openai_path(self):
        s = self._settings(
            synthesis_llm_base_url="http://localhost:8000/v1",
            synthesis_llm_model="qwen-2.5-7b")
        client, model = create_query_llm_client(s)
        assert isinstance(client, OpenAICompatibleMessagesClient)
        assert model == "qwen-2.5-7b"
        assert client.messages.endpoint == "http://localhost:8000/v1/chat/completions"


class TestMeteringCompatibility:
    def test_metered_wrapper_records_openai_usage(self, monkeypatch):
        from gemynd.shared.token_tracker import MeteredAnthropicClient

        _capture_urlopen(monkeypatch, [
            _FakeResponse(body=json.dumps(_CREATE_PAYLOAD).encode())])

        class _SpyLogger:
            def __init__(self):
                self.records = []

            def enqueue(self, record):
                self.records.append(record)

        raw = OpenAICompatibleMessagesClient("http://h/v1", model="m")
        spy = _SpyLogger()
        metered = MeteredAnthropicClient(raw, spy, caller="llm_cypher")
        msg = metered.messages.create(
            model="m", max_tokens=8, system="", messages=[])
        assert msg.content[0].text == "MATCH (n) RETURN n"
        assert len(spy.records) == 1
        rec = spy.records[0]
        assert rec.caller == "llm_cypher"
        assert rec.input_tokens == 120
        assert rec.output_tokens == 9
        assert rec.model == "qwen-2.5-7b"


class TestClaimIdRepair:
    """Citation-typo repair (synthesis.py): local models drop characters
    when copying long claim ids; repair only on a UNIQUE subsequence match."""

    VALID = {
        "run_x:claim:011:81af2f29f2892519",
        "run_x:claim:011:fc6f179ea2d6b9c8",
        "run_x:claim:009:3eda7178ee62df1a",
    }

    def test_dropped_characters_repaired(self):
        from gemynd.retrieval.synthesis import _match_claim_id
        assert _match_claim_id(
            "run_x:claim:011:81af2f2892519", self.VALID
        ) == "run_x:claim:011:81af2f29f2892519"

    def test_inserted_characters_repaired(self):
        from gemynd.retrieval.synthesis import _match_claim_id
        assert _match_claim_id(
            "run_x:claim:009:3eda71788ee62df1aa", self.VALID
        ) == "run_x:claim:009:3eda7178ee62df1a"

    def test_ambiguous_and_garbage_rejected(self):
        from gemynd.retrieval.synthesis import _match_claim_id
        assert _match_claim_id("zzz", self.VALID) is None
        # prefix common to two ids -> ambiguous -> no repair
        assert _match_claim_id("run_x:claim:011:", self.VALID) is None

    def test_repair_mutates_sentences_and_outline(self):
        from gemynd.retrieval.synthesis import _repair_claim_ids

        class _B:
            def __init__(self, c):
                self.claim_id = c

        raw = {
            "answer_sentences": [
                {"text": "t",
                 "supporting_claim_ids": ["run_x:claim:011:81af2f2892519"],
                 "is_inferred": False},
            ],
            "section_outline": [
                {"heading": "h",
                 "claim_ids": ["run_x:claim:011:81af2f2892519"]},
            ],
        }
        notes = _repair_claim_ids(raw, [_B(c) for c in self.VALID])
        assert len(notes) == 2
        assert raw["answer_sentences"][0]["supporting_claim_ids"] == [
            "run_x:claim:011:81af2f29f2892519"]
        assert raw["section_outline"][0]["claim_ids"] == [
            "run_x:claim:011:81af2f29f2892519"]


class TestSynthesisEngineDispatch:
    def test_engine_uses_openai_client_when_configured(self, monkeypatch):
        monkeypatch.setenv("SYNTHESIS_LLM_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("SYNTHESIS_LLM_MODEL", "qwen-2.5-7b")
        from gemynd.retrieval.synthesis import SynthesisEngine

        engine = SynthesisEngine()
        assert isinstance(engine._client, OpenAICompatibleMessagesClient)
        assert engine._model == "qwen-2.5-7b"

    def test_engine_default_stays_anthropic(self, monkeypatch):
        monkeypatch.delenv("SYNTHESIS_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("SYNTHESIS_LLM_PROVIDER", raising=False)
        import gemynd.retrieval.synthesis as syn

        if syn._anthropic_pkg is None:
            pytest.skip("anthropic sdk not installed")
        engine = syn.SynthesisEngine(api_key="test-key")
        assert not isinstance(engine._client, OpenAICompatibleMessagesClient)
        assert engine._model == Settings.from_env().synthesis_model
