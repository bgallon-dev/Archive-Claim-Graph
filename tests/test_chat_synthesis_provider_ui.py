"""chat.html synthesis-provider UI tests.

Verifies the settings-panel provider radio group renders conditionally from the
server-supplied context, the request body includes the selected provider, and
the "answered via" note renders only when the response stamps a provider.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from jinja2 import ChoiceLoader, Environment, FileSystemLoader

from gemynd.retrieval.web.app import _build_synthesis_providers, _PROVIDER_LABELS


ROOT = Path(__file__).resolve().parents[1]


class _Req:
    cookies = {}


def _env() -> Environment:
    return Environment(
        loader=ChoiceLoader([
            FileSystemLoader(str(ROOT / "gemynd" / "retrieval" / "web" / "templates")),
            FileSystemLoader(str(ROOT / "gemynd" / "shared_templates")),
        ]),
        autoescape=True,
    )


def _render(**overrides) -> str:
    ctx = {
        "request": _Req(),
        "composite_members": [],
        "corpus_year_min": None,
        "corpus_year_max": None,
        "featured_questions": [],
        "synthesis_providers": [],
        "synthesis_default_provider": None,
        "user_role": "admin",
        "user_display_name": "Admin",
        "user_initials": "A",
        "is_admin": True,
    }
    ctx.update(overrides)
    return _env().get_template("chat.html").render(**ctx)


def test_provider_radios_render_when_providers_available() -> None:
    html = _render(
        synthesis_providers=[
            {"value": "anthropic", "label": "Anthropic"},
            {"value": "opencode", "label": "OpenCode"},
        ],
        synthesis_default_provider="anthropic",
    )
    assert 'name="synthesis_provider"' in html
    assert 'value="auto"' in html
    assert 'value="anthropic"' in html
    assert 'value="opencode"' in html
    # Auto is pre-checked and names the server default.
    assert 'value="auto" checked' in html
    assert "Auto (anthropic)" in html


def test_provider_radios_hidden_when_none_available() -> None:
    html = _render(synthesis_providers=[], synthesis_default_provider=None)
    # The JS selector string is always present; the *group* must be absent.
    assert "Synthesis provider" not in html
    assert "Auto (anthropic)" not in html


def test_request_body_includes_synthesis_provider() -> None:
    template = (
        ROOT / "gemynd" / "retrieval" / "web" / "templates" / "chat.html"
    ).read_text(encoding="utf-8")
    assert "synthesis_provider: synthesis_provider" in template
    # The JS reads the checked radio, defaulting to null when absent.
    assert 'input[name="synthesis_provider"]:checked' in template
    assert "var synthesis_provider = providerSel ? providerSel.value : null;" in template


def test_answered_via_note_renders_only_when_provider_used() -> None:
    template = (
        ROOT / "gemynd" / "retrieval" / "web" / "templates" / "chat.html"
    ).read_text(encoding="utf-8")
    # The note is gated on the stamped field.
    assert "if (data.synthesis_provider_used)" in template
    assert "answered via " in template
    assert "synth-provider-note" in template


def test_build_synthesis_providers_labels_known_providers() -> None:
    providers = _build_synthesis_providers(["anthropic", "opencode"])
    labels = {p["value"]: p["label"] for p in providers}
    assert labels == {"anthropic": "Anthropic", "opencode": "OpenCode"}
    assert _PROVIDER_LABELS["lmstudio"] == "LM Studio"
