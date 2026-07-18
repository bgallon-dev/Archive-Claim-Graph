"""Render checks for the help page and feedback UI.

Templates render with ``request=None`` here (codebase convention), which is
exactly the case that has broken base-template changes before — any
``request.`` reference must be null-safe.
"""
from __future__ import annotations

from jinja2 import ChoiceLoader, Environment, FileSystemLoader


def _env() -> Environment:
    return Environment(
        loader=ChoiceLoader([
            FileSystemLoader("gemynd/retrieval/web/templates"),
            FileSystemLoader("gemynd/shared_templates"),
        ]),
        autoescape=True,
    )


def _base_ctx(**over):
    ctx = {
        "request": None,
        "user_role": "researcher",
        "user_display_name": "tester",
        "user_initials": "TE",
        "is_admin": False,
    }
    ctx.update(over)
    return ctx


class TestHelpPage:
    def test_renders(self):
        html = _env().get_template("help.html").render(**_base_ctx(active_page="help"))
        assert "Reading an answer" in html

    def test_defines_the_terms_a_tester_actually_sees(self):
        html = _env().get_template("help.html").render(**_base_ctx())
        for term in ("observed", "estimated", "uncertain", "reported"):
            assert term in html, f"epistemic status {term!r} undefined in help"
        assert "Confidence" in html
        assert "Claim type" in html

    def test_distinguishes_confidence_from_truth(self):
        """The most common misreading; the page must address it explicitly."""
        html = _env().get_template("help.html").render(**_base_ctx())
        assert "not whether the underlying fact is true" in html

    def test_explains_the_three_empty_result_cases(self):
        html = _env().get_template("help.html").render(**_base_ctx())
        assert "Didn&#39;t search" in html or "Didn't search" in html
        assert "found nothing" in html
        assert "ranked lower" in html


class TestFeedbackWidget:
    def test_renders_with_null_request(self):
        """Base template is rendered with request=None across the suite."""
        html = _env().get_template("fragments/feedback_widget.html").render(request=None)
        assert 'hx-post="/feedback"' in html
        assert 'name="message"' in html
        assert 'name="page_path"' in html

    def test_offers_every_stored_category(self):
        from gemynd.retrieval.feedback_store import FEEDBACK_CATEGORIES

        html = _env().get_template("fragments/feedback_widget.html").render(request=None)
        for cat in FEEDBACK_CATEGORIES:
            assert f'value="{cat}"' in html, f"category {cat!r} missing from the form"

    def test_included_in_base_shell(self):
        """Reachable from every authenticated page, not just one."""
        html = _env().get_template("help.html").render(**_base_ctx())
        assert 'hx-post="/feedback"' in html
        assert 'href="/help"' in html


class TestFeedbackList:
    def test_empty_state_is_explained_not_blank(self):
        html = _env().get_template("feedback_list.html").render(
            **_base_ctx(is_admin=True, rows=[], store_available=True)
        )
        assert "No reports yet" in html

    def test_unconfigured_store_is_surfaced(self):
        html = _env().get_template("feedback_list.html").render(
            **_base_ctx(is_admin=True, rows=[], store_available=False)
        )
        assert "isn&#39;t configured" in html or "isn't configured" in html
        assert "FEEDBACK_DB" in html

    def test_rows_render(self):
        rows = [{
            "feedback_id": "f-1",
            "created_at": "2026-07-17T14:03:22+00:00",
            "user_email": "archivist@example.org",
            "user_role": "researcher",
            "category": "wrong_answer",
            "message": "Cited a document that doesn't mention it.",
            "page_path": "/",
            "resolved": 0,
        }]
        html = _env().get_template("feedback_list.html").render(
            **_base_ctx(is_admin=True, rows=rows, store_available=True)
        )
        assert "archivist@example.org" in html
        assert "wrong answer" in html  # underscores humanised
        assert "2026-07-17 14:03" in html

    def test_message_is_escaped(self):
        rows = [{
            "feedback_id": "f-1",
            "created_at": "2026-07-17T14:03:22+00:00",
            "user_email": "t@example.org",
            "user_role": "researcher",
            "category": "other",
            "message": "<script>alert('xss')</script>",
            "page_path": "/",
            "resolved": 0,
        }]
        html = _env().get_template("feedback_list.html").render(
            **_base_ctx(is_admin=True, rows=rows, store_available=True)
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
