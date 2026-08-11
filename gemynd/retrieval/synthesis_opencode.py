"""OpenCode synthesis client (server API and CLI transports).

`SynthesisEngine` talks to models through a small slice of the Anthropic SDK
surface:

    with client.messages.stream(model=, max_tokens=, system=, messages=, timeout=) as s:
        for chunk in s.text_stream: ...
        final = s.get_final_message()   # -> .stop_reason

This module implements that surface backed by the opencode agent platform,
either a long-running `opencode serve` HTTP server or a per-call
`opencode run` CLI subprocess. The engine's JSON-envelope parsing, citation
validation, retry logic, and PII redaction stay untouched.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from gemynd.shared.token_tracker import TokenUsageRecord

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

class _TransportError(RuntimeError):
    """Raised when the opencode server returns a non-2xx response or is unreachable."""


def _basic_auth_header(username: str, password: str) -> dict[str, str]:
    if not password:
        return {}
    creds = f"{username}:{password}".encode("utf-8")
    return {"Authorization": f"Basic {base64.b64encode(creds).decode('ascii')}"}


def _model_ref(model: str) -> dict[str, str] | None:
    """Convert ``provider/model-id`` to opencode's ModelRef object.

    OpenCode expects ``{providerID, modelID}`` in PromptInput. The empty string
    or a value without a slash is treated as unset.
    """
    if not model:
        return None
    if "/" not in model:
        return None
    provider_id, model_id = model.split("/", 1)
    return {"providerID": provider_id, "modelID": model_id}


def _request(
    method: str,
    url: str,
    path: str,
    body: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> Any:
    """Make a JSON request to the opencode server and return the parsed body."""
    full_url = url.rstrip("/") + path
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=True).encode("utf-8")
    req_headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(full_url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise _TransportError(
            f"opencode server {method} {path} failed with HTTP {exc.code}: {body_text}"
        ) from exc
    except urllib.error.URLError as exc:
        raise _TransportError(
            f"opencode server unreachable at {full_url}: {exc.reason}"
        ) from exc


# ---------------------------------------------------------------------------
# Server transport
# ---------------------------------------------------------------------------

class _ServerTransport:
    """Default urllib-based transport for `opencode serve`."""

    def __init__(self, username: str, password: str) -> None:
        self._username = username
        self._password = password

    def __call__(
        self,
        method: str,
        url: str,
        path: str,
        body: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        merged_headers = _basic_auth_header(self._username, self._password)
        if headers:
            merged_headers.update(headers)
        return _request(method, url, path, body, merged_headers)


class _ServerStreamContext:
    """Context manager returned by the server-mode ``messages.stream(...)``."""

    def __init__(
        self,
        transport: Callable[..., Any],
        url: str,
        username: str,
        password: str,
        session_id: str,
        message_id: str,
        model: str,
        agent: str,
        system: str,
        user_message: str,
        timeout: float | None,
        token_logger: Any | None,
    ) -> None:
        self._transport = transport
        self._url = url
        self._username = username
        self._password = password
        self._session_id = session_id
        self._message_id = message_id
        self._model = model
        self._agent = agent
        self._system = system
        self._user_message = user_message
        self._timeout = timeout
        self._token_logger = token_logger
        self._stop_reason = "end_turn"
        self._tokens: dict[str, Any] | None = None

    def __enter__(self) -> "_ServerStreamContext":
        """Send the prompt and return the streaming context."""
        headers = _basic_auth_header(self._username, self._password)
        body: dict[str, Any] = {
            "agent": self._agent,
            "system": self._system,
            # OpenCode's PromptInput.tools is a Record<string, boolean>, not an array.
            "tools": {},
            "messageID": self._message_id,
            "parts": [{"type": "text", "text": self._user_message}],
        }
        model_ref = _model_ref(self._model)
        if model_ref is not None:
            body["model"] = model_ref
        self._transport(
            "POST", self._url, f"/session/{self._session_id}/prompt_async", body, headers
        )
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Best-effort session cleanup."""
        headers = _basic_auth_header(self._username, self._password)
        try:
            self._transport(
                "DELETE", self._url, f"/session/{self._session_id}", None, headers
            )
        except Exception:
            _log.debug(
                "failed to delete opencode session %s", self._session_id, exc_info=True
            )
        return False

    def _message_list(self) -> list[dict[str, Any]]:
        """Fetch the session message list; tolerate both bare-list and {"messages": [...]} shapes."""
        headers = _basic_auth_header(self._username, self._password)
        resp = self._transport(
            "GET", self._url, f"/session/{self._session_id}/message", None, headers
        )
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict):
            return list(resp.get("messages") or [])
        return []

    @staticmethod
    def _assistant_text(msg: dict[str, Any]) -> str:
        parts = msg.get("parts") or []
        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text") or "")
        return "".join(texts)

    @staticmethod
    def _is_complete(msg: dict[str, Any]) -> bool:
        info = msg.get("info") or {}
        # Modern opencode messages use `finish` (e.g. "stop"/"length") and
        # `time.completed` instead of the older boolean `complete` flag.
        if info.get("finish"):
            return True
        if (info.get("time") or {}).get("completed"):
            return True
        if info.get("complete") is True:
            return True
        # Fallback: text parts that carry an end timestamp are finished.
        for part in (msg.get("parts") or []):
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("time") or {}
                if isinstance(t, dict) and t.get("end"):
                    return True
        return False

    @staticmethod
    def _extract_stop_reason(msg: dict[str, Any]) -> str:
        info = msg.get("info") or {}
        reason = info.get("finish") or info.get("reason") or info.get("stopReason") or ""
        if reason in ("max_tokens", "length"):
            return "max_tokens"
        return "end_turn"

    @staticmethod
    def _extract_tokens(msg: dict[str, Any]) -> dict[str, Any] | None:
        info = msg.get("info") or {}
        return info.get("tokens")

    def _abort(self) -> None:
        headers = _basic_auth_header(self._username, self._password)
        try:
            self._transport(
                "POST", self._url, f"/session/{self._session_id}/abort", {}, headers
            )
        except Exception:
            _log.debug(
                "failed to abort opencode session %s", self._session_id, exc_info=True
            )

    def _record_tokens(self) -> None:
        if self._token_logger is None or self._tokens is None:
            return
        try:
            record = TokenUsageRecord(
                caller="synthesis",
                model=self._model or "opencode",
                input_tokens=int(self._tokens.get("input") or 0),
                output_tokens=int(self._tokens.get("output") or 0),
            )
            self._token_logger.enqueue(record)
        except Exception:
            _log.debug("failed to record opencode token usage", exc_info=True)

    @property
    def text_stream(self) -> Iterator[str]:
        """Poll the session message list and yield only-new assistant text deltas."""
        deadline = time.time() + self._timeout if self._timeout else None
        last_len = 0
        while True:
            if deadline and time.time() > deadline:
                self._abort()
                raise RuntimeError(
                    f"opencode synthesis timed out after {self._timeout}s "
                    f"(session {self._session_id})"
                )

            messages = self._message_list()
            assistant_msgs = [
                m for m in messages if (m.get("info") or {}).get("role") == "assistant"
            ]
            if assistant_msgs:
                msg = assistant_msgs[-1]
                text = self._assistant_text(msg)
                if len(text) > last_len:
                    yield text[last_len:]
                    last_len = len(text)
                if self._is_complete(msg):
                    self._stop_reason = self._extract_stop_reason(msg)
                    self._tokens = self._extract_tokens(msg)
                    break

            time.sleep(0.5)

        self._record_tokens()

    def get_final_message(self) -> SimpleNamespace:
        return SimpleNamespace(stop_reason=self._stop_reason)


# ---------------------------------------------------------------------------
# CLI transport
# ---------------------------------------------------------------------------

class _CliStreamContext:
    """Context manager returned by the CLI-mode ``messages.stream(...)``."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        model: str,
        token_logger: Any | None,
        timeout: float | None,
    ) -> None:
        self._process = process
        self._model = model
        self._token_logger = token_logger
        self._timeout = timeout
        self._stop_reason = "end_turn"
        self._tokens: dict[str, Any] | None = None
        self._stderr_parts: list[str] = []
        self._timed_out = False
        self._timer: threading.Timer | None = None

    def __enter__(self) -> "_CliStreamContext":
        if self._timeout:

            def _expire() -> None:
                self._timed_out = True
                if self._process.poll() is None:
                    self._process.terminate()

            self._timer = threading.Timer(self._timeout, _expire)
            self._timer.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._timer is not None:
            self._timer.cancel()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        return False

    def _stderr_tail(self) -> str:
        return "".join(self._stderr_parts)[-2000:]

    def _record_tokens(self) -> None:
        if self._token_logger is None or self._tokens is None:
            return
        try:
            record = TokenUsageRecord(
                caller="synthesis",
                model=self._model or "opencode",
                input_tokens=int(self._tokens.get("input") or 0),
                output_tokens=int(self._tokens.get("output") or 0),
            )
            self._token_logger.enqueue(record)
        except Exception:
            _log.debug("failed to record opencode token usage", exc_info=True)

    @property
    def text_stream(self) -> Iterator[str]:
        accumulated: list[str] = []
        stdout = self._process.stdout
        assert stdout is not None
        for raw_line in iter(stdout.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "text":
                part = event.get("part") or {}
                delta = part.get("text") or ""
                if delta:
                    accumulated.append(delta)
                    yield delta
            elif etype == "step_finish":
                # Real `opencode run --format json` nests the payload under
                # ``event["part"]``. Fall back to the top-level shape for tests.
                payload = event.get("part") or event
                self._tokens = payload.get("tokens") or {}
                reason = payload.get("reason") or ""
                if reason in ("max_tokens", "length"):
                    self._stop_reason = "max_tokens"

        stderr = self._process.stderr
        if stderr is not None:
            try:
                err = stderr.read().decode("utf-8", errors="replace")
                self._stderr_parts.append(err)
            except Exception:
                pass

        if self._timer is not None:
            self._timer.cancel()

        rc = self._process.wait()
        self._record_tokens()

        if rc != 0 and not accumulated:
            if self._timed_out:
                raise RuntimeError(
                    f"opencode run timed out after {self._timeout}s"
                )
            raise RuntimeError(
                f"opencode run exited with code {rc}. stderr: {self._stderr_tail()}"
            )

    def get_final_message(self) -> SimpleNamespace:
        return SimpleNamespace(stop_reason=self._stop_reason)


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class OpencodeSynthesisClient:
    """Drop-in for the Anthropic client's ``.messages.stream()`` slice, backed
    by the opencode agent platform.

    Parameters
    ----------
    mode:
        ``"server"`` (default) connects to a running ``opencode serve``;
        ``"cli"`` spawns a subprocess per call.
    url, username, password:
        ``opencode serve`` endpoint and HTTP basic auth credentials.
    model:
        Model identifier forwarded to opencode (e.g. ``"anthropic/claude-sonnet-4-6"``),
        or ``""`` to let opencode use its configured default.
    agent:
        Agent name passed to ``opencode run`` / the server message body.
    binary, dir:
        ``opencode`` executable path and working directory for CLI mode.
    timeout:
        Per-call timeout in seconds.
    token_logger:
        Optional ``TokenUsageLogger`` for recording token usage.
    transport:
        Optional callable for server-mode tests with signature
        ``(method, url, path, body, headers) -> json``.
    """

    def __init__(
        self,
        mode: str = "server",
        url: str = "http://127.0.0.1:4096",
        username: str = "opencode",
        password: str = "",
        model: str = "",
        agent: str = "synthesis",
        binary: str = "opencode",
        dir: str = "",
        timeout: float | None = None,
        token_logger: Any | None = None,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._mode = mode.strip().lower()
        self._url = url
        self._username = username
        self._password = password
        self._model = model
        self._agent = agent
        self._binary = binary
        self._dir = dir
        self._timeout = timeout
        self._token_logger = token_logger
        if self._mode == "server":
            self._transport: Callable[..., Any] = transport or _ServerTransport(
                username, password
            )
        else:
            self._transport = transport  # type: ignore[assignment]
        self.messages = _Messages(self)

    def check(self) -> None:
        """Fail-fast: verify the opencode backend is reachable/available.

        The engine calls this at construction so configuration errors surface
        at startup rather than mid-query.
        """
        if self._mode == "server":
            _request("GET", self._url, "/global/health", None, {}, timeout=5.0)
            return

        # CLI mode: ensure the binary resolves.
        resolved = shutil.which(self._binary)
        if not resolved:
            raise RuntimeError(
                f"SYNTHESIS_PROVIDER=opencode (mode=cli) but the opencode binary "
                f"is not on PATH: {self._binary!r}. Install opencode or set "
                f"SYNTHESIS_OPENCODE_BINARY to the full path."
            )

    def _create_session(self) -> str:
        if self._mode != "server":
            raise RuntimeError("create_session is only valid in server mode")
        resp = self._transport("POST", self._url, "/session", {})
        if isinstance(resp, dict) and "id" in resp:
            return str(resp["id"])
        raise _TransportError(f"opencode /session returned unexpected shape: {resp!r}")


class _Messages:
    def __init__(self, client: OpencodeSynthesisClient) -> None:
        self._client = client

    def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
        timeout: float | None = None,
        **_ignored: Any,
    ) -> Any:
        """Implement the Anthropic ``messages.stream`` surface.

        The *model* kwarg comes from the engine's ``_call_and_parse`` and is
        intentionally ignored for transport construction (see B1 in the plan).
        The client uses its own configured ``self._client._model`` so that an
        unset model is forwarded as omitted, not replaced by the Anthropic
        default string.
        """
        client = self._client
        user_message = messages[0].get("content", "") if messages else ""
        effective_timeout = timeout if timeout is not None else client._timeout

        if client._mode == "server":
            session_id = client._create_session()
            message_id = f"msg-{os.urandom(8).hex()}"
            return _ServerStreamContext(
                transport=client._transport,
                url=client._url,
                username=client._username,
                password=client._password,
                session_id=session_id,
                message_id=message_id,
                model=client._model,
                agent=client._agent,
                system=system,
                user_message=user_message,
                timeout=effective_timeout,
                token_logger=client._token_logger,
            )
            


        # CLI mode
        resolved = shutil.which(client._binary) or client._binary
        argv = [resolved, "run", "--format", "json", "--agent", client._agent]
        if client._model:
            argv.extend(["--model", client._model])
        if client._dir:
            argv.extend(["--dir", client._dir])
        if sys.platform == "win32" and resolved.lower().endswith((".cmd", ".bat")):
            argv = ["cmd", "/c", resolved] + argv[1:]

        env = os.environ.copy()
        env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
        if max_tokens:
            env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(max_tokens)

        creationflags = 0
        if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        prompt = f"{system}\n\n{user_message}"
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=client._dir or None,
                env=env,
                creationflags=creationflags,
                text=False,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to start opencode run: {exc}") from exc

        try:
            assert process.stdin is not None
            process.stdin.write(prompt.encode("utf-8"))
            process.stdin.close()
        except Exception as exc:
            process.terminate()
            raise RuntimeError(f"failed to write prompt to opencode run: {exc}") from exc

        return _CliStreamContext(
            process=process,
            model=client._model,
            token_logger=client._token_logger,
            timeout=effective_timeout,
        )
