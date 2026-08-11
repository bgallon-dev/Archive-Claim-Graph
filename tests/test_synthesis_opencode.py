"""Opencode synthesis provider tests — mock-based.

No opencode binary or server is required. Fake transports and subprocess mocks
exercise the server and CLI transports plus the engine wiring.
"""
from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Server transport
# ---------------------------------------------------------------------------


def _make_server_client(**kwargs: Any) -> Any:
    from gemynd.retrieval.synthesis_opencode import OpencodeSynthesisClient

    defaults = {
        "mode": "server",
        "url": "http://test",
        "username": "u",
        "password": "p",
        "model": "",
        "agent": "synthesis",
    }
    defaults.update(kwargs)
    return OpencodeSynthesisClient(**defaults)


def test_server_transport_polls_and_yields_deltas(monkeypatch):
    """Poll-based server transport yields exactly-once deltas and records tokens."""
    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.time.sleep", lambda _x: None
    )
    calls = []
    message_responses = [
        [],
        [{"info": {"role": "assistant", "complete": False}, "parts": []}],
        [
            {
                "info": {"role": "assistant", "complete": False},
                "parts": [{"type": "text", "text": "Hel"}],
            }
        ],
        [
            {
                "info": {"role": "assistant", "complete": False},
                "parts": [{"type": "text", "text": "Hello"}],
            }
        ],
        [
            {
                "info": {
                    "role": "assistant",
                    "complete": True,
                    "reason": "stop",
                    "tokens": {"input": 10, "output": 5},
                },
                "parts": [{"type": "text", "text": "Hello world"}],
            }
        ],
    ]

    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        calls.append((method, path))
        if path == "/global/health":
            return {"healthy": True}
        if path == "/session":
            return {"id": "ses-1"}
        if path == "/session/ses-1/prompt_async":
            return None
        if path == "/session/ses-1/message":
            return message_responses.pop(0)
        if path == "/session/ses-1/abort":
            return None
        if path == "/session/ses-1":
            return None
        raise AssertionError(f"unexpected path: {path}")

    logger = MagicMock()
    client = _make_server_client(transport=transport, token_logger=logger)
    with client.messages.stream(
        model="ignored-by-client",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        deltas = list(stream.text_stream)

    assert deltas == ["Hel", "lo", " world"]
    assert stream.get_final_message().stop_reason == "end_turn"
    assert ("DELETE", "/session/ses-1") in calls

    assert logger.enqueue.called
    record = logger.enqueue.call_args[0][0]
    assert record.caller == "synthesis"
    assert record.model == "opencode"
    assert record.input_tokens == 10
    assert record.output_tokens == 5

    prompt_call = [c for c in calls if c[0] == "POST" and c[1] == "/session/ses-1/prompt_async"]
    assert prompt_call


def test_server_transport_omits_model_when_unset():
    calls = []

    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        calls.append((method, path, body))
        if path == "/session":
            return {"id": "ses-1"}
        if path.endswith("/prompt_async"):
            return None
        if path.endswith("/message"):
            return [
                {
                    "info": {"role": "assistant", "complete": True},
                    "parts": [{"type": "text", "text": "{}"}],
                }
            ]
        return None

    client = _make_server_client(model="", transport=transport)
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        list(stream.text_stream)

    prompt_body = [c[2] for c in calls if c[0] == "POST" and c[1].endswith("/prompt_async")][0]
    assert "model" not in prompt_body
    assert prompt_body["agent"] == "synthesis"
    assert prompt_body["system"] == "SYS"
    assert prompt_body["tools"] == {}
    assert prompt_body["parts"] == [{"type": "text", "text": "USER"}]
    assert prompt_body["messageID"].startswith("msg")



def test_server_transport_forwards_model_when_set():
    calls = []

    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        calls.append((method, path, body))
        if path == "/session":
            return {"id": "ses-1"}
        if path.endswith("/prompt_async"):
            return None
        if path.endswith("/message"):
            return [
                {
                    "info": {"role": "assistant", "complete": True},
                    "parts": [{"type": "text", "text": "{}"}],
                }
            ]
        return None

    client = _make_server_client(model="anthropic/claude-sonnet-4-6", transport=transport)
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        list(stream.text_stream)

    prompt_body = [c[2] for c in calls if c[0] == "POST" and c[1].endswith("/prompt_async")][0]
    assert prompt_body["model"] == {"providerID": "anthropic", "modelID": "claude-sonnet-4-6"}


def test_server_transport_single_chunk_fallback(monkeypatch):
    """When the server only materialises text at completion, yield it as one chunk."""
    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.time.sleep", lambda _x: None
    )

    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        if path == "/session":
            return {"id": "ses-1"}
        if path.endswith("/prompt_async"):
            return None
        if path.endswith("/message"):
            return [
                {
                    "info": {"role": "assistant", "complete": True, "reason": "stop"},
                    "parts": [{"type": "text", "text": "{"}],
                }
            ]
        return None

    client = _make_server_client(transport=transport)
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        chunks = list(stream.text_stream)
    assert chunks == ["{"]


def test_server_transport_timeout_aborts(monkeypatch):
    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.time.sleep", lambda _x: None
    )

    class _Time:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls <= 2:
                return 100.0
            return 111.0  # exceeds deadline of 100 + 10

    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.time.time", _Time()
    )

    calls = []

    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        calls.append((method, path))
        if path == "/session":
            return {"id": "ses-1"}
        if path.endswith("/prompt_async"):
            return None
        if path.endswith("/message"):
            return []
        if path.endswith("/abort"):
            return None
        raise AssertionError(f"unexpected path: {path}")

    client = _make_server_client(transport=transport)
    with pytest.raises(RuntimeError, match="timed out"):
        with client.messages.stream(
            model="ignored",
            max_tokens=100,
            system="SYS",
            messages=[{"role": "user", "content": "USER"}],
            timeout=10,
        ) as stream:
            list(stream.text_stream)

    assert ("POST", "/session/ses-1/abort") in calls


def test_server_transport_max_tokens_stop_reason(monkeypatch):
    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.time.sleep", lambda _x: None
    )

    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        if path == "/session":
            return {"id": "ses-1"}
        if path.endswith("/prompt_async"):
            return None
        if path.endswith("/message"):
            return [
                {
                    "info": {"role": "assistant", "complete": True, "reason": "max_tokens"},
                    "parts": [{"type": "text", "text": "x"}],
                }
            ]
        return None

    client = _make_server_client(transport=transport)
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        list(stream.text_stream)
    assert stream.get_final_message().stop_reason == "max_tokens"


def test_server_transport_finish_stop_reason(monkeypatch):
    """Real AssistantMessage uses info.finish (e.g. 'length') for stop reason."""
    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.time.sleep", lambda _x: None
    )

    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        if path == "/session":
            return {"id": "ses-1"}
        if path.endswith("/prompt_async"):
            return None
        if path.endswith("/message"):
            return [
                {
                    "info": {
                        "role": "assistant",
                        "finish": "length",
                        "tokens": {"input": 1, "output": 1},
                    },
                    "parts": [{"type": "text", "text": "x"}],
                }
            ]
        return None

    client = _make_server_client(transport=transport)
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        list(stream.text_stream)
    assert stream.get_final_message().stop_reason == "max_tokens"


def test_server_transport_time_completed_completion(monkeypatch):
    """Real AssistantMessage completion may only set info.time.completed."""
    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.time.sleep", lambda _x: None
    )

    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        if path == "/session":
            return {"id": "ses-1"}
        if path.endswith("/prompt_async"):
            return None
        if path.endswith("/message"):
            return [
                {
                    "info": {"role": "assistant", "time": {"completed": 1234567890}},
                    "parts": [{"type": "text", "text": "done"}],
                }
            ]
        return None

    client = _make_server_client(transport=transport)
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        chunks = list(stream.text_stream)
    assert chunks == ["done"]


def test_server_check_raises_on_unreachable():
    def transport(method: str, url: str, path: str, body: Any = None, headers: Any = None):
        raise RuntimeError("boom")

    client = _make_server_client(transport=transport)
    with pytest.raises(RuntimeError):
        client.check()


# ---------------------------------------------------------------------------
# CLI transport
# ---------------------------------------------------------------------------


def _make_cli_client(**kwargs: Any) -> Any:
    from gemynd.retrieval.synthesis_opencode import OpencodeSynthesisClient

    defaults = {
        "mode": "cli",
        "model": "",
        "agent": "synthesis",
        "binary": "opencode",
    }
    defaults.update(kwargs)
    return OpencodeSynthesisClient(**defaults)


class _FakeProcess:
    def __init__(self, stdout_lines=None, rc=0):
        self.stdout_lines = stdout_lines or []
        self.rc = rc
        self.terminated = False
        self.killed = False
        self._waited = False
        self.stdin = MagicMock()
        data = b"\n".join(json.dumps(line).encode() for line in self.stdout_lines)
        if data:
            data += b"\n"
        self.stdout = io.BytesIO(data)
        self.stderr = io.BytesIO(b"")

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def poll(self):
        if self.terminated or self.killed or self._waited:
            return self.rc
        return None

    def wait(self, timeout=None):
        self._waited = True
        return self.rc


def test_cli_transport_argv_and_stdin(monkeypatch):
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        proc = _FakeProcess(rc=0)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/opencode")
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    client = _make_cli_client()
    with client.messages.stream(
        model="ignored",
        max_tokens=123,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        list(stream.text_stream)

    assert captured["argv"] == ["/usr/bin/opencode", "run", "--format", "json", "--agent", "synthesis"]
    assert captured["env"]["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    assert captured["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "123"
    written = captured["proc"].stdin.write.call_args[0][0]
    assert b"SYS" in written
    assert b"USER" in written
    captured["proc"].stdin.close.assert_called_once()


def test_cli_transport_forwards_model_and_dir(monkeypatch):
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["cwd"] = kwargs.get("cwd")
        proc = _FakeProcess(rc=0)
        return proc

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/opencode")
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    client = _make_cli_client(model="anthropic/claude-sonnet-4-6", dir="/tmp/demo")
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ):
        pass

    assert "--model" in captured["argv"]
    assert "anthropic/claude-sonnet-4-6" in captured["argv"]
    assert "--dir" in captured["argv"]
    assert "/tmp/demo" in captured["argv"]
    assert captured["cwd"] == "/tmp/demo"


def test_cli_transport_jsonl_deltas_and_tokens(monkeypatch):
    logger = MagicMock()
    stdout_lines = [
        {"type": "text", "part": {"text": "Hel"}},
        {"type": "text", "part": {"text": "lo"}},
        {"type": "step_finish", "part": {"reason": "stop", "tokens": {"input": 7, "output": 2}}},
    ]

    def fake_popen(argv, **kwargs):
        return _FakeProcess(stdout_lines=stdout_lines, rc=0)

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/opencode")
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    client = _make_cli_client(token_logger=logger)
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        deltas = list(stream.text_stream)

    assert deltas == ["Hel", "lo"]
    assert stream.get_final_message().stop_reason == "end_turn"
    record = logger.enqueue.call_args[0][0]
    assert record.input_tokens == 7
    assert record.output_tokens == 2


def test_cli_transport_max_tokens_stop_reason(monkeypatch):
    stdout_lines = [
        {"type": "text", "part": {"text": "x"}},
        {"type": "step_finish", "part": {"reason": "max_tokens", "tokens": {"input": 1, "output": 1}}},
    ]

    def fake_popen(argv, **kwargs):
        return _FakeProcess(stdout_lines=stdout_lines, rc=0)

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/opencode")
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    client = _make_cli_client()
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ) as stream:
        list(stream.text_stream)

    assert stream.get_final_message().stop_reason == "max_tokens"


def test_cli_transport_error_event_raises(monkeypatch):
    stdout_lines = [
        {"type": "error", "message": "model not found"},
    ]

    def fake_popen(argv, **kwargs):
        return _FakeProcess(stdout_lines=stdout_lines, rc=1)

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/opencode")
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    client = _make_cli_client()
    with pytest.raises(RuntimeError, match="exited with code 1"):
        with client.messages.stream(
            model="ignored",
            max_tokens=100,
            system="SYS",
            messages=[{"role": "user", "content": "USER"}],
            timeout=10,
        ) as stream:
            list(stream.text_stream)


def test_cli_transport_timeout_terminates_process(monkeypatch):
    """Simulate timeout by making threading.Timer fire immediately."""

    class _ImmediateTimer:
        def __init__(self, interval, function, args=None, kwargs=None):
            self.interval = interval
            self.function = function
            self.args = args or ()
            self.kwargs = kwargs or {}

        def start(self):
            self.function(*self.args, **self.kwargs)

        def cancel(self):
            pass

    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.threading.Timer", _ImmediateTimer
    )

    proc = _FakeProcess(stdout_lines=[], rc=1)

    def fake_popen(argv, **kwargs):
        return proc

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/opencode")
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    client = _make_cli_client()
    with pytest.raises(RuntimeError, match="timed out"):
        with client.messages.stream(
            model="ignored",
            max_tokens=100,
            system="SYS",
            messages=[{"role": "user", "content": "USER"}],
            timeout=10,
        ) as stream:
            list(stream.text_stream)

    assert proc.terminated


def test_cli_transport_cmd_shim_on_windows(monkeypatch):
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        proc = _FakeProcess(rc=0)
        return proc

    monkeypatch.setattr("shutil.which", lambda _b: r"C:\Users\x\opencode.cmd")
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("gemynd.retrieval.synthesis_opencode.sys.platform", "win32")
    monkeypatch.setattr(
        "gemynd.retrieval.synthesis_opencode.subprocess.CREATE_NO_WINDOW",
        0x08000000,
    )

    client = _make_cli_client()
    with client.messages.stream(
        model="ignored",
        max_tokens=100,
        system="SYS",
        messages=[{"role": "user", "content": "USER"}],
        timeout=10,
    ):
        pass

    assert captured["argv"][:3] == ["cmd", "/c", r"C:\Users\x\opencode.cmd"]


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def test_engine_selects_opencode_client_and_model(monkeypatch):
    monkeypatch.setenv("SYNTHESIS_PROVIDER", "opencode")
    monkeypatch.setenv("SYNTHESIS_OPENCODE_MODE", "cli")
    monkeypatch.setenv("SYNTHESIS_OPENCODE_MODEL", "")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/opencode")

    from gemynd.retrieval.synthesis import SynthesisEngine
    from gemynd.retrieval.synthesis_opencode import OpencodeSynthesisClient

    eng = SynthesisEngine(max_tokens=1000)
    assert eng._provider == "opencode"
    assert eng._model == ""
    assert isinstance(eng._client, OpencodeSynthesisClient)
    # Full system prompt (not the compact local one).
    assert "specialist researcher" in eng._system_prompt


def test_engine_opencode_forwards_model_when_set(monkeypatch):
    monkeypatch.setenv("SYNTHESIS_PROVIDER", "opencode")
    monkeypatch.setenv("SYNTHESIS_OPENCODE_MODE", "cli")
    monkeypatch.setenv("SYNTHESIS_OPENCODE_MODEL", "anthropic/claude-sonnet-4-6")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/opencode")

    from gemynd.retrieval.synthesis import SynthesisEngine

    eng = SynthesisEngine(max_tokens=1000)
    assert eng._model == "anthropic/claude-sonnet-4-6"


def test_engine_opencode_missing_binary_raises(monkeypatch):
    monkeypatch.setenv("SYNTHESIS_PROVIDER", "opencode")
    monkeypatch.setenv("SYNTHESIS_OPENCODE_MODE", "cli")
    monkeypatch.setattr("shutil.which", lambda _b: None)

    from gemynd.retrieval.synthesis import SynthesisEngine

    with pytest.raises(RuntimeError, match="opencode binary"):
        SynthesisEngine(max_tokens=1000)


def test_default_provider_is_anthropic(monkeypatch):
    """Absent SYNTHESIS_PROVIDER, the engine stays on Anthropic."""
    monkeypatch.delenv("SYNTHESIS_PROVIDER", raising=False)

    from gemynd.shared.settings import Settings

    assert Settings.from_env().synthesis_provider == "anthropic"
