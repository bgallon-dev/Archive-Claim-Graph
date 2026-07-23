"""OpenAI-compatible synthesis client (LM Studio, Ollama, vLLM, OpenAI cloud).

`SynthesisEngine` talks to Anthropic through a small slice of the SDK surface:

    with client.messages.stream(model=, max_tokens=, system=, messages=, timeout=) as s:
        for chunk in s.text_stream: ...
        final = s.get_final_message()   # -> .stop_reason

This module presents that *exact* surface backed by the OpenAI Chat Completions
API, so a local OpenAI-compatible server can drive synthesis with no change to
the engine's streaming/parse/validate logic. It is only ever used for a
non-Anthropic provider (LM Studio et al.) — Anthropic synthesis still goes
through the real SDK + MeteredAnthropicClient.

Why an adapter and not the OpenAI SDK directly in the engine: keeping the engine
provider-agnostic means the JSON-envelope parsing, citation validation, retry,
and PII redaction stay in one place and cannot drift between providers.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Iterator

_log = logging.getLogger(__name__)

try:
    import openai as _openai_pkg
except ImportError:  # pragma: no cover - optional dependency
    _openai_pkg = None  # type: ignore[assignment]

_DEFAULT_BASE_URL = "http://localhost:1234/v1"


class _OpenAIStream:
    """Wraps an OpenAI streaming completion to expose Anthropic's stream shape."""

    def __init__(self, raw_stream: Any) -> None:
        self._raw = raw_stream
        self._finish_reason: str | None = None

    @property
    def text_stream(self) -> Iterator[str]:
        for chunk in self._raw:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            # finish_reason lands on the final chunk; remember it for stop_reason.
            fr = getattr(choice, "finish_reason", None)
            if fr:
                self._finish_reason = fr
            delta = getattr(choice, "delta", None)
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                yield piece

    def get_final_message(self) -> SimpleNamespace:
        # The engine only reads .stop_reason, and only compares it to
        # "max_tokens". Map OpenAI's "length" to that; everything else is a
        # normal completion as far as the engine is concerned.
        stop = "max_tokens" if self._finish_reason == "length" else "end_turn"
        return SimpleNamespace(stop_reason=stop)


class _StreamContext:
    """Context manager returned by ``messages.stream(...)``."""

    def __init__(self, client: Any, *, model: str, max_tokens: int,
                 system: str, messages: list[dict], timeout: float | None) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._messages = messages
        self._timeout = timeout
        self._raw: Any = None

    def __enter__(self) -> _OpenAIStream:
        # Anthropic keeps `system` separate; OpenAI takes it as the first
        # message. The engine only ever sends string-content user turns.
        oai_messages = [{"role": "system", "content": self._system}]
        for m in self._messages:
            oai_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        self._raw = self._client.chat.completions.create(**kwargs)
        return _OpenAIStream(self._raw)

    def __exit__(self, *exc: Any) -> bool:
        close = getattr(self._raw, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        return False


class _Messages:
    def __init__(self, client: Any) -> None:
        self._client = client

    def stream(self, *, model: str, max_tokens: int, system: str,
               messages: list[dict], timeout: float | None = None, **_ignored: Any) -> _StreamContext:
        return _StreamContext(
            self._client, model=model, max_tokens=max_tokens,
            system=system, messages=messages, timeout=timeout,
        )


class OpenAICompatSynthesisClient:
    """Drop-in for the Anthropic client's ``.messages.stream(...)`` slice,
    backed by an OpenAI-compatible endpoint (LM Studio by default).

    A raw client can be injected for testing (``client=``); otherwise one is
    built from *base_url* / *api_key*.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            if _openai_pkg is None:  # pragma: no cover
                raise RuntimeError(
                    "openai package is not installed. Install with: pip install openai"
                )
            resolved_base = base_url or _DEFAULT_BASE_URL
            # LM Studio requires a real Bearer token; there is no usable default,
            # so surface a clear error rather than a confusing 401 mid-query.
            if not api_key:
                raise RuntimeError(
                    "SYNTHESIS_PROVIDER is set to a local/OpenAI provider but no API "
                    "key was given. Set OPENAI_API_KEY to an LM Studio API token "
                    "(LM Studio → Developer → API tokens), or disable auth in LM Studio."
                )
            self._client = _openai_pkg.OpenAI(base_url=resolved_base, api_key=api_key)
        self.messages = _Messages(self._client)
