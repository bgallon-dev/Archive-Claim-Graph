"""Layer 3 — synthesis provider router with fallback chain.

Holds one :class:`SynthesisEngine` per configured provider and routes each
query to the requested provider (or the default), falling through an ordered
chain of backup providers when an engine call fails.

Providers whose engine could not be built at startup (e.g. the opencode serve
endpoint is down, the Anthropic key is missing, or the local model is unset)
are recorded as unavailable and skipped by routing — never treated as errors.
If the *default* provider is unavailable but a fallback built, a prominent
startup warning is emitted so a dead primary is not silently masked.
"""
from __future__ import annotations

import logging
from typing import Any

from gemynd.retrieval.synthesis import SynthesisEngine

_log = logging.getLogger(__name__)

_LOCAL_PROVIDERS = {"lmstudio", "openai", "local"}


class SynthesisRouter:
    """Route synthesis requests across pre-built engines with fallback.

    Parameters
    ----------
    engines:
        provider_id -> ready-to-call SynthesisEngine. Only successfully built
        engines are present; a provider absent from this dict is unavailable.
    default_provider:
        Provider used when the caller does not request one explicitly
        (``provider=None`` or ``"auto"``).
    fallback_providers:
        Ordered backup providers tried after the default when an engine call
        raises. Entries not present in *engines* are skipped silently.
    """

    def __init__(
        self,
        engines: dict[str, SynthesisEngine],
        default_provider: str,
        fallback_providers: list[str],
    ) -> None:
        self._engines = dict(engines)
        self._default = default_provider
        self._fallback = list(fallback_providers)

    @property
    def available_providers(self) -> list[str]:
        """Provider IDs of successfully-built engines, in default-first order."""
        ordered = [self._default] + self._fallback
        seen: set[str] = set()
        out: list[str] = []
        for pid in ordered:
            if pid in seen or pid not in self._engines:
                continue
            seen.add(pid)
            out.append(pid)
        for pid in sorted(self._engines):
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out

    @property
    def default_provider(self) -> str:
        return self._default

    def _resolve_chain(self, provider: str | None) -> list[str]:
        """Ordered, deduped list of provider IDs to try for this request.

        The requested provider (or the default when the caller asks for
        ``None``/``"auto"``) leads, followed by the default (if different) and
        then the configured fallbacks. Only providers with a built engine are
        included; unavailable ones are skipped rather than errored.
        """
        requested = provider if provider and provider != "auto" else self._default
        chain: list[str] = []
        for pid in (requested, self._default, *self._fallback):
            if pid in chain:
                continue
            if pid in self._engines:
                chain.append(pid)
        return chain

    def synthesise(
        self,
        provider: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Run synthesis, falling back through the chain on failure.

        Returns ``(result, provider_used)``. On success *provider_used* is the
        provider that produced the result. If every engine in the chain raises,
        the last exception is re-raised so the web layer can surface its typed
        fallback response.
        """
        chain = self._resolve_chain(provider)
        if not chain:
            raise RuntimeError(
                "No synthesis provider is available. Configure at least one of "
                "SYNTHESIS_PROVIDER / SYNTHESIS_FALLBACK_PROVIDERS with a working "
                "engine (Anthropic key, opencode serve, or local model)."
            )
        last_exc: Exception | None = None
        for pid in chain:
            try:
                result = self._engines[pid].synthesise(**kwargs)
                if pid != self._default:
                    _log.info("synthesis answered via %s (default %s)", pid, self._default)
                return result, pid
            except Exception as exc:  # noqa: BLE001 - chain fallback by design
                last_exc = exc
                remaining = [p for p in chain if p != pid]
                _log.warning(
                    "synthesis provider %s failed (%s: %s); %s",
                    pid,
                    type(exc).__name__,
                    exc,
                    "trying next provider(s)" if remaining else "no fallback left",
                )
        assert last_exc is not None
        raise last_exc

    def complete_json(
        self,
        provider: str | None = None,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 900,
    ) -> tuple[dict[str, Any], str]:
        """Run a structured completion through the normal provider chain."""
        chain = self._resolve_chain(provider)
        if not chain:
            raise RuntimeError("No synthesis provider is available for structured completion.")
        last_exc: Exception | None = None
        for pid in chain:
            try:
                result = self._engines[pid].complete_json(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_tokens,
                )
                return result, pid
            except Exception as exc:  # noqa: BLE001 - provider fallback by design
                last_exc = exc
                _log.warning(
                    "structured completion provider %s failed (%s); trying fallback",
                    pid,
                    type(exc).__name__,
                )
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        """Close any engines that need cleanup (no-op today)."""
        for engine in self._engines.values():
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - best-effort shutdown
                    _log.debug("engine close failed", exc_info=True)


def _local_max_tokens(max_tokens: int, local_cap: int) -> int:
    """Cap response length for local/OpenAI-compatible providers."""
    return min(max_tokens, max(256, local_cap))


def build_synthesis_router(
    settings: Any,
    *,
    api_key: str | None = None,
    max_tokens: int = 8192,
    synthesis_context: str | None = None,
    timeout: float | None = None,
    token_logger: Any | None = None,
    local_max_tokens: int = 1200,
) -> SynthesisRouter:
    """Build one engine per configured provider, best-effort.

    The default provider (``settings.synthesis_provider``) and each entry in
    ``settings.synthesis_fallback_providers`` get their own engine. Construction
    failures are logged and that provider marked unavailable. Local providers
    receive a capped ``max_tokens``; all others the full budget. Raises at
    startup only when *no* engine could be built.
    """
    providers = [settings.synthesis_provider or "anthropic"]
    for pid in settings.synthesis_fallback_providers or []:
        if pid not in providers:
            providers.append(pid)

    engines: dict[str, SynthesisEngine] = {}
    for pid in providers:
        engine_max_tokens = (
            _local_max_tokens(max_tokens, local_max_tokens)
            if pid in _LOCAL_PROVIDERS
            else max_tokens
        )
        try:
            engines[pid] = SynthesisEngine(
                api_key=api_key,
                max_tokens=engine_max_tokens,
                synthesis_context=synthesis_context,
                timeout=timeout,
                token_logger=token_logger,
                provider=pid,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort per provider
            _log.warning(
                "synthesis provider %s unavailable: %s: %s",
                pid,
                type(exc).__name__,
                exc,
            )

    if not engines:
        raise RuntimeError(
            "No synthesis provider could be initialized. Check the Anthropic API "
            "key, opencode serve/binary, or local model configuration."
        )

    default = settings.synthesis_provider or "anthropic"
    if default not in engines:
        _log.warning(
            "!! DEFAULT SYNTHESIS PROVIDER %r UNAVAILABLE; running on fallback "
            "provider(s) %s",
            default,
            ", ".join(sorted(engines)),
        )

    return SynthesisRouter(
        engines=engines,
        default_provider=default if default in engines else next(iter(engines)),
        fallback_providers=settings.synthesis_fallback_providers or [],
    )
