"""OpenAI-compatible synthesis client (LM Studio path) — mock-based.

No LM Studio required: a fake OpenAI client feeds canned streaming chunks so we
can assert the adapter maps onto the Anthropic `.messages.stream()` surface the
engine expects. The real end-to-end (against LM Studio) was verified manually;
see DEMO.md / the LM Studio notes.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gemynd.retrieval.synthesis_openai import (
    OpenAICompatSynthesisClient,
    _OpenAIStream,
)


def _chunk(content=None, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )]
    )


class _FakeOpenAI:
    """Minimal stand-in for openai.OpenAI: records the create() kwargs and
    replays a fixed list of chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return iter(self._chunks)


def test_stream_yields_content_and_skips_empty_deltas():
    chunks = [_chunk("Hello"), _chunk(None), _chunk(" world"), _chunk("", "stop")]
    s = _OpenAIStream(iter(chunks))
    assert list(s.text_stream) == ["Hello", " world"]


def test_get_final_message_maps_length_to_max_tokens():
    s = _OpenAIStream(iter([_chunk("x", "length")]))
    list(s.text_stream)  # drain so finish_reason is captured
    assert s.get_final_message().stop_reason == "max_tokens"


def test_get_final_message_defaults_to_end_turn():
    s = _OpenAIStream(iter([_chunk("x", "stop")]))
    list(s.text_stream)
    assert s.get_final_message().stop_reason == "end_turn"


def test_stream_prepends_system_and_converts_messages():
    fake = _FakeOpenAI([_chunk("{}", "stop")])
    client = OpenAICompatSynthesisClient(client=fake)
    with client.messages.stream(
        model="local-model", max_tokens=123, system="SYS",
        messages=[{"role": "user", "content": "USER"}], timeout=42.0,
    ) as stream:
        text = "".join(stream.text_stream)
    assert text == "{}"
    kw = fake.last_kwargs
    assert kw["model"] == "local-model"
    assert kw["max_tokens"] == 123
    assert kw["stream"] is True
    assert kw["timeout"] == 42.0
    # system becomes the first message; user follows
    assert kw["messages"][0] == {"role": "system", "content": "SYS"}
    assert kw["messages"][1] == {"role": "user", "content": "USER"}


def test_missing_api_key_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="no API key"):
        OpenAICompatSynthesisClient(base_url="http://localhost:1234/v1", api_key="")


def test_engine_selects_openai_client_and_model_for_local_provider(monkeypatch):
    """provider=lmstudio wires the OpenAI-compatible client + the local model,
    with no Anthropic client and no token metering — verifiable without LM Studio
    (constructing openai.OpenAI does not open a connection)."""
    monkeypatch.setenv("SYNTHESIS_PROVIDER", "lmstudio")
    monkeypatch.setenv("SYNTHESIS_OPENAI_MODEL", "google/gemma-4-e4b")
    monkeypatch.setenv("LMSTUDIO_KEY", "sk-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")

    from gemynd.retrieval.synthesis import SynthesisEngine

    eng = SynthesisEngine(max_tokens=1000)
    assert eng._provider == "lmstudio"
    assert eng._model == "google/gemma-4-e4b"
    assert isinstance(eng._client, OpenAICompatSynthesisClient)


def test_engine_local_provider_errors_without_a_model(monkeypatch):
    monkeypatch.setenv("SYNTHESIS_PROVIDER", "lmstudio")
    monkeypatch.setenv("SYNTHESIS_OPENAI_MODEL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LMSTUDIO_KEY", "sk-test-key")
    from gemynd.retrieval.synthesis import SynthesisEngine
    with pytest.raises(RuntimeError, match="no model is set"):
        SynthesisEngine(max_tokens=1000)


def test_resolve_local_key_prefers_canonical_then_falls_back(monkeypatch):
    from gemynd.retrieval.synthesis import _resolve_local_key
    from gemynd.shared.settings import Settings

    for name in ("SYNTHESIS_OPENAI_API_KEY", "LMSTUDIO_KEY", "LMSUDIO_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    s = Settings()  # openai_api_key = ""
    assert _resolve_local_key(s) == ""

    monkeypatch.setenv("OPENAI_API_KEY", "from-openai")
    assert _resolve_local_key(s) == "from-openai"
    monkeypatch.setenv("LMSTUDIO_KEY", "from-lmstudio")
    assert _resolve_local_key(s) == "from-lmstudio"  # canonical wins over OPENAI_API_KEY


def test_default_provider_is_anthropic(monkeypatch):
    """Absent SYNTHESIS_PROVIDER, nothing changes — the engine stays on Anthropic."""
    monkeypatch.delenv("SYNTHESIS_PROVIDER", raising=False)
    from gemynd.shared.settings import Settings
    assert Settings.from_env().synthesis_provider == "anthropic"
