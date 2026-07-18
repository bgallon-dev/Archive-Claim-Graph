"""OpenAI-compatible chat client with an Anthropic-Messages-shaped surface.

The query-time LLM consumers (``retrieval/synthesis.py``,
``retrieval/llm_cypher.py``) speak the Anthropic Messages interface:

    client.messages.create(model=, max_tokens=, system=, messages=) -> msg
    client.messages.stream(...) as stream:                # context manager
        stream.text_stream                                # chunk iterator
        stream.get_final_message()                        # msg

``OpenAICompatibleMessagesClient`` reproduces exactly that surface over any
server speaking the OpenAI ``/v1/chat/completions`` API — vLLM, Ollama,
LM Studio, llama.cpp's server, TGI, or OpenAI itself — so both consumers
and the ``MeteredAnthropicClient`` usage-tracking wrapper work unchanged
(the returned message carries ``.content[0].text``, ``.usage.input_tokens``
/ ``.output_tokens``, ``.stop_reason``, ``.model``).

Provider dispatch mirrors ``ingest/extractors/llm_adapter_factory.py``:
``SYNTHESIS_LLM_PROVIDER`` explicit wins; otherwise a set
``SYNTHESIS_LLM_BASE_URL`` selects the OpenAI-compatible path; otherwise
Anthropic (existing default).  Configuration comes from ``Settings``
(``synthesis_llm_*`` fields), never read from ``os.environ`` here.

Note on cost metering: local models are priced by ``token_pricing.yaml``'s
default block unless a ``TOKEN_PRICE_<MODEL>_INPUT/_OUTPUT`` override sets
them to zero — token COUNTS stay accurate either way.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from gemynd.shared.settings import Settings

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

_STOP_REASON_MAP = {"stop": "end_turn", "length": "max_tokens"}


def _normalize_endpoint(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


# ---------------------------------------------------------------------------
# Anthropic-shaped response objects
# ---------------------------------------------------------------------------

@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _TextBlock:
    text: str


@dataclass
class _Message:
    content: list[_TextBlock]
    usage: _Usage
    stop_reason: str
    model: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class _SSEStream:
    """Context manager mimicking ``anthropic.Anthropic().messages.stream()``.

    Consumes the OpenAI server-sent-event stream; per-chunk reads keep the
    connection alive past any cumulative wall-clock timeout, matching why
    synthesis moved to streaming in the first place.
    """

    def __init__(self, response: Any, model: str) -> None:
        self._response = response
        self._model = model
        self._parts: list[str] = []
        self._finish_reason: str | None = None
        self._usage = _Usage()
        self._gen: Iterator[str] | None = None

    def __enter__(self) -> "_SSEStream":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._response.close()

    @property
    def text_stream(self) -> Iterator[str]:
        if self._gen is None:
            self._gen = self._iter_deltas()
        return self._gen

    def _iter_deltas(self) -> Iterator[str]:
        for raw_line in self._response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[len("data:"):].strip()
            if payload_text == "[DONE]":
                break
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            usage = payload.get("usage")
            if usage:
                self._usage = _Usage(
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                )
            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]
            delta = (choice.get("delta") or {}).get("content")
            if delta:
                self._parts.append(delta)
                yield delta

    def get_final_message(self) -> _Message:
        for _ in self.text_stream:  # drain anything the caller didn't consume
            pass
        return _Message(
            content=[_TextBlock(text="".join(self._parts))],
            usage=self._usage,
            stop_reason=_STOP_REASON_MAP.get(
                self._finish_reason or "stop", self._finish_reason or "end_turn"),
            model=self._model,
        )


@dataclass
class _OpenAIMessages:
    endpoint: str
    model: str
    api_key: str
    timeout: float
    temperature: float
    extra_body: dict = field(default_factory=dict)

    def _request(self, body: dict, timeout: float | None) -> Any:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        return urllib.request.urlopen(
            request, timeout=timeout if timeout is not None else self.timeout)

    def _body(self, *, model: str | None, max_tokens: int, system: str,
              messages: list[dict], **_ignored: Any) -> dict:
        chat = [{"role": "system", "content": system}] if system else []
        chat.extend(messages)
        body = {
            "model": model or self.model,
            "messages": chat,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        # Server-specific knobs (e.g. {"reasoning_effort": "low"} to curb a
        # reasoning model's thinking budget) — merged last so they win.
        body.update(self.extra_body)
        return body

    def create(self, *, timeout: float | None = None, **kwargs: Any) -> _Message:
        body = self._body(**kwargs)
        with self._request(body, timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choice = (payload.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        usage = payload.get("usage") or {}
        finish = choice.get("finish_reason") or "stop"
        return _Message(
            content=[_TextBlock(text=text)],
            usage=_Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
            stop_reason=_STOP_REASON_MAP.get(finish, finish),
            model=payload.get("model") or self.model,
        )

    def stream(self, *, timeout: float | None = None, **kwargs: Any) -> _SSEStream:
        body = self._body(**kwargs)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        try:
            response = self._request(body, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            # Older servers reject stream_options; retry without it (usage
            # then simply reports 0 — counts, not correctness).
            try:
                exc.close()
            except Exception:
                pass
            body.pop("stream_options", None)
            response = self._request(body, timeout)
        return _SSEStream(response, body["model"])


class OpenAICompatibleMessagesClient:
    """Drop-in for ``anthropic.Anthropic`` limited to the ``messages`` API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout: float = 120.0,
        temperature: float = 0.0,
        extra_body: dict | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("OpenAICompatibleMessagesClient requires a base_url")
        self.messages = _OpenAIMessages(
            endpoint=_normalize_endpoint(base_url),
            model=model or "local-model",
            api_key=api_key or "not-needed",
            timeout=timeout,
            temperature=temperature,
            extra_body=dict(extra_body or {}),
        )


# ---------------------------------------------------------------------------
# Provider dispatch (mirrors llm_adapter_factory)
# ---------------------------------------------------------------------------

def query_llm_provider(settings: "Settings") -> str:
    """Which backend serves query-time LLM calls (synthesis + LLM-Cypher)."""
    explicit = (settings.synthesis_llm_provider or "").strip().lower()
    if explicit:
        if explicit != "anthropic" and explicit not in _OPENAI_PROVIDERS:
            _log.warning(
                "Unknown SYNTHESIS_LLM_PROVIDER=%r — defaulting to anthropic",
                explicit)
            return "anthropic"
        return explicit
    if settings.synthesis_llm_base_url:
        return "openai_compatible"
    return "anthropic"


def create_query_llm_client(
    settings: "Settings",
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> tuple[Any, str]:
    """Build the raw query-time messages client per the configured provider.

    Returns ``(client, model_id)``.  Callers wrap the client in
    ``MeteredAnthropicClient`` themselves when a token logger is available.
    Raises ``RuntimeError`` when the Anthropic path is selected but the SDK
    is not installed (matching SynthesisEngine's existing behavior).
    """
    resolved_timeout = timeout if timeout is not None else settings.synthesis_timeout_seconds
    provider = query_llm_provider(settings)
    if provider in _OPENAI_PROVIDERS:
        model = settings.synthesis_llm_model or "local-model"
        extra_body: dict = {}
        if settings.synthesis_llm_extra_body:
            try:
                extra_body = json.loads(settings.synthesis_llm_extra_body)
            except json.JSONDecodeError:
                _log.warning(
                    "SYNTHESIS_LLM_EXTRA_BODY is not valid JSON — ignored")
        _log.info("Query LLM provider: %s (OpenAI-compatible, model=%s)",
                  provider, model)
        client = OpenAICompatibleMessagesClient(
            base_url=settings.synthesis_llm_base_url,
            model=model,
            api_key=settings.synthesis_llm_api_key,
            timeout=resolved_timeout,
            extra_body=extra_body,
        )
        return client, model

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "anthropic package is not installed. Install with: "
            "pip install -e .[retrieval] — or set SYNTHESIS_LLM_BASE_URL to "
            "use an OpenAI-compatible server instead."
        ) from exc
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=resolved_timeout,
        max_retries=(max_retries if max_retries is not None
                     else settings.synthesis_max_retries),
    )
    return client, settings.synthesis_model
