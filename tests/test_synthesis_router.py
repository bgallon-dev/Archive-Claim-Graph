"""Unit tests for Layer 3 synthesis router (provider switch + fallback chain).

Covers:
  • Chain resolution: requested / default / fallback ordering and dedup.
  • Fallback-on-failure: an engine that raises falls through to the next.
  • Missing engines (unavailable providers) are skipped, not errored.
  • provider_used tagging on success and on fallback.
  • available_providers / default_provider surfaces for the UI.
  • build_synthesis_router: per-provider engines, best-effort construction,
    distinct per-engine max_tokens for local providers, default-fail warning,
    zero-available startup raise.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gemynd.retrieval.synthesis import SynthesisEngine
from gemynd.retrieval.synthesis_router import (
    SynthesisRouter,
    build_synthesis_router,
    _local_max_tokens,
)


class _FakeEngine:
    """Minimal stand-in for SynthesisEngine with an injectable behaviour."""

    def __init__(self, provider: str, *, fail: bool = False, value: str = "ok") -> None:
        self._provider = provider
        self._fail = fail
        self._value = value

    def synthesise(self, **kwargs) -> str:
        if self._fail:
            raise RuntimeError(f"{self._provider} engine exploded")
        return f"{self._value}:{self._provider}"

    def complete_json(self, **kwargs) -> dict:
        if self._fail:
            raise RuntimeError(f"{self._provider} engine exploded")
        return {"provider": self._provider}


def _make_router(
    engines: dict[str, object],
    default: str = "anthropic",
    fallback: list[str] | None = None,
) -> SynthesisRouter:
    return SynthesisRouter(
        engines=engines,  # type: ignore[arg-type]
        default_provider=default,
        fallback_providers=fallback or [],
    )


class TestChainResolution:
    def test_default_provider_used_when_none_requested(self) -> None:
        router = _make_router(
            {"anthropic": _FakeEngine("anthropic"), "opencode": _FakeEngine("opencode")},
            default="anthropic",
        )
        result, used = router.synthesise()
        assert used == "anthropic"
        assert result == "ok:anthropic"

    def test_auto_means_default(self) -> None:
        router = _make_router(
            {"anthropic": _FakeEngine("anthropic")},
            default="anthropic",
        )
        result, used = router.synthesise(provider="auto")
        assert used == "anthropic"

    def test_explicit_provider_wins(self) -> None:
        router = _make_router(
            {"anthropic": _FakeEngine("anthropic"), "opencode": _FakeEngine("opencode")},
            default="anthropic",
        )
        result, used = router.synthesise(provider="opencode")
        assert used == "opencode"
        assert result == "ok:opencode"

    def test_chain_dedups_duplicates(self) -> None:
        router = _make_router(
            {"anthropic": _FakeEngine("anthropic")},
            default="anthropic",
            fallback=["anthropic", "opencode"],
        )
        # opencode is in the fallback list but has no engine — skipped silently.
        result, used = router.synthesise(provider="opencode")
        assert used == "anthropic"

    def test_requested_provider_fails_falls_back_to_default(self) -> None:
        router = _make_router(
            {
                "anthropic": _FakeEngine("anthropic"),
                "opencode": _FakeEngine("opencode", fail=True),
            },
            default="anthropic",
            fallback=["opencode"],
        )
        result, used = router.synthesise(provider="opencode")
        assert used == "anthropic"
        assert result == "ok:anthropic"

    def test_default_fails_falls_back_to_configured_fallback(self) -> None:
        router = _make_router(
            {
                "anthropic": _FakeEngine("anthropic", fail=True),
                "opencode": _FakeEngine("opencode"),
            },
            default="anthropic",
            fallback=["opencode"],
        )
        result, used = router.synthesise()
        assert used == "opencode"
        assert result == "ok:opencode"

    def test_all_engines_fail_reraises_last_exception(self) -> None:
        router = _make_router(
            {
                "anthropic": _FakeEngine("anthropic", fail=True),
                "opencode": _FakeEngine("opencode", fail=True),
            },
            default="anthropic",
            fallback=["opencode"],
        )
        with pytest.raises(RuntimeError, match="opencode engine exploded"):
            router.synthesise()

    def test_no_available_engines_raises_clear_error(self) -> None:
        router = _make_router({}, default="anthropic", fallback=["opencode"])
        with pytest.raises(RuntimeError, match="No synthesis provider is available"):
            router.synthesise()

    def test_unavailable_requested_provider_skipped_not_keyerror(self) -> None:
        router = _make_router(
            {"anthropic": _FakeEngine("anthropic")},
            default="anthropic",
        )
        result, used = router.synthesise(provider="lmstudio")
        assert used == "anthropic"

    def test_structured_completion_uses_requested_provider(self) -> None:
        router = _make_router(
            {"anthropic": _FakeEngine("anthropic"), "opencode": _FakeEngine("opencode")}
        )
        result, used = router.complete_json(
            provider="opencode", system_prompt="system", user_message="user"
        )
        assert result == {"provider": "opencode"}
        assert used == "opencode"

    def test_structured_completion_falls_back(self) -> None:
        router = _make_router(
            {
                "anthropic": _FakeEngine("anthropic", fail=True),
                "opencode": _FakeEngine("opencode"),
            },
            fallback=["opencode"],
        )
        result, used = router.complete_json(
            system_prompt="system", user_message="user"
        )
        assert result == {"provider": "opencode"}
        assert used == "opencode"


class TestProviderSurfacing:
    def test_available_providers_default_first(self) -> None:
        router = _make_router(
            {"anthropic": _FakeEngine("anthropic"), "opencode": _FakeEngine("opencode")},
            default="anthropic",
            fallback=["opencode"],
        )
        assert router.available_providers == ["anthropic", "opencode"]

    def test_available_providers_only_built_engines(self) -> None:
        router = _make_router(
            {"anthropic": _FakeEngine("anthropic")},
            default="anthropic",
            fallback=["opencode"],
        )
        assert router.available_providers == ["anthropic"]

    def test_default_provider_property(self) -> None:
        router = _make_router({"opencode": _FakeEngine("opencode")}, default="opencode")
        assert router.default_provider == "opencode"


class TestLocalMaxTokens:
    def test_local_cap_applies(self) -> None:
        assert _local_max_tokens(24000, 1200) == 1200
        assert _local_max_tokens(8192, 1200) == 1200

    def test_local_floor_respects_minimum(self) -> None:
        assert _local_max_tokens(500, 1200) == 500

    def test_local_cap_never_below_256(self) -> None:
        # A tiny configured cap is floored to 256 when max_tokens allows it.
        assert _local_max_tokens(24000, 50) == 256


class TestBuildSynthesisRouter:
    @patch("gemynd.retrieval.synthesis_router.SynthesisEngine")
    def test_builds_default_engine_only_without_fallbacks(self, engine_cls) -> None:
        settings = MagicMock()
        settings.synthesis_provider = "anthropic"
        settings.synthesis_fallback_providers = []
        router = build_synthesis_router(settings)
        assert router.default_provider == "anthropic"
        assert router.available_providers == ["anthropic"]
        engine_cls.assert_called_once()
        # Non-local provider must receive the full max_tokens budget.
        assert engine_cls.call_args.kwargs["max_tokens"] == 8192
        assert engine_cls.call_args.kwargs["provider"] == "anthropic"

    @patch("gemynd.retrieval.synthesis_router.SynthesisEngine")
    def test_builds_default_and_fallback_engines(self, engine_cls) -> None:
        settings = MagicMock()
        settings.synthesis_provider = "opencode"
        settings.synthesis_fallback_providers = ["anthropic"]
        router = build_synthesis_router(settings)
        assert router.default_provider == "opencode"
        assert router.available_providers == ["opencode", "anthropic"]
        providers = [c.kwargs["provider"] for c in engine_cls.call_args_list]
        assert providers == ["opencode", "anthropic"]

    @patch("gemynd.retrieval.synthesis_router.SynthesisEngine")
    def test_local_fallback_receives_capped_max_tokens(self, engine_cls) -> None:
        settings = MagicMock()
        settings.synthesis_provider = "anthropic"
        settings.synthesis_fallback_providers = ["lmstudio"]
        router = build_synthesis_router(settings, max_tokens=24000)
        assert router.available_providers == ["anthropic", "lmstudio"]
        caps = {c.kwargs["provider"]: c.kwargs["max_tokens"] for c in engine_cls.call_args_list}
        assert caps["anthropic"] == 24000
        assert caps["lmstudio"] == 1200

    @patch("gemynd.retrieval.synthesis_router.SynthesisEngine")
    def test_failed_construction_marks_provider_unavailable(self, engine_cls) -> None:
        def _fail(**kwargs):
            raise RuntimeError("boom")
        engine_cls.side_effect = _fail
        settings = MagicMock()
        settings.synthesis_provider = "opencode"
        settings.synthesis_fallback_providers = ["anthropic"]
        with pytest.raises(RuntimeError, match="No synthesis provider could be initialized"):
            build_synthesis_router(settings)

    @patch("gemynd.retrieval.synthesis_router.SynthesisEngine")
    def test_default_failure_with_fallback_starts_and_warns(self, engine_cls) -> None:
        def _select(**kwargs):
            if kwargs["provider"] == "opencode":
                raise RuntimeError("opencode down")
            return MagicMock()
        engine_cls.side_effect = _select
        settings = MagicMock()
        settings.synthesis_provider = "opencode"
        settings.synthesis_fallback_providers = ["anthropic"]
        with patch("gemynd.retrieval.synthesis_router._log.warning") as warn:
            router = build_synthesis_router(settings)
        assert router.available_providers == ["anthropic"]
        assert router.default_provider == "anthropic"
        warning_messages = " ".join(str(call) for call in warn.call_args_list)
        assert "DEFAULT SYNTHESIS PROVIDER" in warning_messages

    def test_close_delegates_to_engines(self) -> None:
        engine = _FakeEngine("anthropic")
        router = _make_router({"anthropic": engine})
        engine.close = MagicMock()  # type: ignore[attr-defined]
        router.close()
        engine.close.assert_called_once()


class TestSettingsParse:
    def test_fallback_providers_parsed_from_env(self, monkeypatch) -> None:
        from gemynd.shared.settings import Settings

        monkeypatch.setenv("SYNTHESIS_FALLBACK_PROVIDERS", " anthropic ,opencode, ")
        settings = Settings.from_env()
        assert settings.synthesis_fallback_providers == ["anthropic", "opencode"]

    def test_fallback_providers_empty_by_default(self, monkeypatch) -> None:
        from gemynd.shared.settings import Settings

        monkeypatch.delenv("SYNTHESIS_FALLBACK_PROVIDERS", raising=False)
        assert Settings.from_env().synthesis_fallback_providers == []


def test_provider_override_param_selects_opencode_branch() -> None:
    """SynthesisEngine(provider=...) overrides the env default provider."""
    settings = MagicMock()
    settings.synthesis_provider = "anthropic"
    settings.synthesis_timeout_seconds = 60.0
    settings.synthesis_opencode_mode = "server"
    settings.synthesis_opencode_url = "http://127.0.0.1:4096"
    settings.synthesis_opencode_username = "opencode"
    settings.synthesis_opencode_password = ""
    settings.synthesis_opencode_model = ""
    settings.synthesis_opencode_agent = "synthesis"
    settings.synthesis_opencode_binary = "opencode"
    settings.synthesis_opencode_dir = ""
    client_cls = MagicMock()
    with (
        patch.dict("os.environ", {"SYNTHESIS_PROVIDER": "anthropic"}, clear=False),
        patch("gemynd.retrieval.synthesis._Settings.from_env", return_value=settings),
        patch("gemynd.retrieval.synthesis_opencode.OpencodeSynthesisClient", client_cls),
    ):
        eng = SynthesisEngine(provider="opencode", max_tokens=1000)
    client_cls.assert_called_once()
    assert eng._provider == "opencode"
