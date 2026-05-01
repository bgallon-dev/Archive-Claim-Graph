from __future__ import annotations

from jinja2 import ChoiceLoader, Environment, FileSystemLoader

from gemynd.retrieval.web.app import _build_history_context


def _template_env() -> Environment:
    return Environment(
        loader=ChoiceLoader([
            FileSystemLoader("gemynd/retrieval/web/templates"),
            FileSystemLoader("gemynd/shared_templates"),
        ]),
        autoescape=True,
    )


def test_history_saved_rerun_url_preserves_filters() -> None:
    ctx = _build_history_context(
        rows=[],
        total=0,
        saved=[
            {
                "saved_id": "abc",
                "label": "Bounded",
                "query_text": "elk near Cheney",
                "bucket": "analytical",
                "year_min": 1890,
                "year_max": 1910,
                "conversation_id": "conv_123",
                "created_by": "tester",
                "created_at": "2026-04-29T12:00:00",
            }
        ],
        q="",
        bucket="",
        limit=50,
        offset=0,
    )

    assert ctx["saved"][0]["srerun_url"] == (
        "/?q=elk+near+Cheney&bucket=analytical&year_min=1890&year_max=1910"
        "&conversation_id=conv_123"
    )


def test_history_pagination_exposes_five_page_skips() -> None:
    ctx = _build_history_context(
        rows=[],
        total=1247,
        saved=[],
        q="ducks",
        bucket="hybrid",
        limit=50,
        offset=550,
    )

    pagination = ctx["pagination"]
    assert pagination["has_skip_back"] is True
    assert pagination["has_skip_forward"] is True
    assert "offset=300" in pagination["skip_back_url"]
    assert "offset=800" in pagination["skip_forward_url"]
    assert "q=ducks" in pagination["skip_forward_url"]
    assert "bucket=hybrid" in pagination["skip_forward_url"]


def test_review_queue_template_marks_sensitivity_actions() -> None:
    html = _template_env().get_template("admin_review.html").render(
        request=None,
        active_page="review",
        user_initials="U",
        user_display_name="User",
        user_role="admin",
        total_queued=4,
        detector_count=2,
        sensitivity_count=2,
        queue_filters=[("ocr_entity", "OCR / Entity", 2)],
        active_filter="ocr_entity",
        accepted_today=1,
        rejected_today=0,
        deferred_today=0,
        review_trend=[
            {"day": "2026-04-28", "total": 2, "height_pct": 50},
            {"day": "2026-04-29", "total": 4, "height_pct": 100},
        ],
        new_queued_today=3,
        proposal_cards=[
            {
                "proposal_id": "prop_1",
                "ic_label": "PII exposure",
                "is_sensitivity": True,
                "badge_cls": "badge-red",
                "priority": 0.9,
                "confidence": 0.95,
                "impact_size": 991,
                "description": "PII pattern detected.",
                "diagnostic": "Matched term: <strong>phone</strong>",
                "target_summary": {
                    "total": 991,
                    "counts": [{"label": "mention:suppressed", "count": 991}],
                },
                "review_objects": {
                    "total": 991,
                    "remaining": 990,
                    "items": [
                        {
                            "kind": "Sensitive claim",
                            "label": "phone",
                            "detail": "pii_exposure",
                            "snippet": "Contact [REDACTED] for permit details.",
                            "meta": "claim_1",
                        }
                    ],
                },
                "needs_bulk_confirm": True,
                "actionable": True,
                "can_reject": True,
                "can_defer": True,
            }
        ],
    )

    assert "Sensitivity (2)" in html
    assert 'hx-confirm="This will modify 991 targets in the corpus. Continue?"' in html
    assert "btn-accept-sensitive" in html
    assert "Matched term:" in html
    assert "Objects to review" in html
    assert "Contact [REDACTED] for permit details." in html
    assert "Impacts 991 targets" in html
    assert "New queued today: 3" in html


def test_review_detail_template_previews_large_target_sets() -> None:
    html = _template_env().get_template("admin_review_detail.html").render(
        request=None,
        active_page="review",
        user_initials="U",
        user_display_name="User",
        user_role="admin",
        proposal_id="prop_1234567890",
        issue_class="header_contamination",
        issue_class_label="Header contamination",
        proposal_type="suppress_mention",
        status="queued",
        confidence="0.85",
        priority_score="0.90",
        impact_size=991,
        description="header_contamination",
        is_active=True,
        is_sensitivity=False,
        needs_bulk_confirm=True,
        source_comparison=None,
        review_objects={
            "total": 991,
            "remaining": 990,
            "items": [
                {
                    "kind": "Mention",
                    "label": "header_contamination",
                    "detail": "page 4",
                    "snippet": "Annual report header repeated before the real paragraph.",
                    "meta": "run_20260314T155430Z_0f2abc",
                }
            ],
        },
        targets=[
            {
                "target_kind": "mention",
                "target_id": "run_20260314T155430Z_0f2abc",
                "target_role": "suppressed",
                "exists_in_snapshot": True,
            }
        ],
        target_summary={
            "total": 991,
            "remaining": 990,
            "counts": [{"label": "mention:suppressed", "count": 991}],
        },
        revisions=[],
        events=[],
        patch_json="{}",
        evidence_json="{}",
        error_root_causes=[],
        error_root_cause_labels={},
        error_types={},
        error_type_labels={},
        default_root_cause="",
        default_error_types=[],
    )

    assert "Review objects" in html
    assert "Annual report header repeated before the real paragraph." in html
    assert "990 additional review objects represented" in html
    assert "Showing 1 of 991" in html
    assert "990 additional targets omitted" in html
    assert 'hx-confirm="This will modify 991 targets in the corpus. Continue?"' in html
