from __future__ import annotations

from jinja2 import ChoiceLoader, Environment, FileSystemLoader

from gemynd.retrieval.web.app import _build_gaps_context


def _ctx(
    *,
    temporal: list[dict] | None = None,
    entity_depth: list[dict] | None = None,
    geo_coverage: list[dict] | None = None,
    claim_types: list[dict] | None = None,
    per_corpus_expected_shares: dict[str, dict[str, float]] | None = None,
    per_corpus_claim_types: list[dict] | None = None,
) -> dict:
    return _build_gaps_context(
        temporal=temporal or [],
        entity_depth=entity_depth or [],
        geo_coverage=geo_coverage or [],
        claim_types=claim_types or [],
        conv_gaps=[],
        per_corpus_expected_shares=per_corpus_expected_shares or {},
        per_corpus_claim_types=per_corpus_claim_types or [],
    )


def test_gaps_context_builds_year_level_temporal_timeline() -> None:
    ctx = _ctx(
        temporal=[
            {"year": 1900, "doc_count": 10, "claim_count": 20},
            {"year": 1902, "doc_count": 1, "claim_count": 3},
            {"year": 1904, "doc_count": 20, "claim_count": 30},
        ],
    )

    years = ctx["temporal_timeline"]["years"]
    assert [row["year"] for row in years] == [1900, 1901, 1902, 1903, 1904]
    assert [row["doc_count"] for row in years] == [10, 0, 1, 0, 20]
    assert [row["is_gap"] for row in years] == [False, True, True, True, False]
    assert ctx["temporal_gap_threshold"] == 2.0
    assert ctx["year_span"] == "1900-1904"


def test_gaps_context_enriches_thin_coverage_with_median_fraction() -> None:
    ctx = _ctx(
        entity_depth=[
            {
                "entity_type": "Person",
                "name": "A",
                "primary_claim_count": 1,
                "coverage_median": 20,
            },
            {
                "entity_type": "Person",
                "name": "B",
                "primary_claim_count": 3,
                "coverage_median": 20,
            },
        ],
        geo_coverage=[
            {
                "entity_type": "Place",
                "name": "C",
                "location_specific_claims": 2,
                "coverage_median": 10,
            }
        ],
    )

    _, entity_rows = ctx["entity_depth_by_type"][0]
    assert ctx["entity_depth_median"] == 20.0
    assert entity_rows[0]["claim_count_label"] == "1 claim"
    assert entity_rows[0]["median_pct"] == 5
    assert entity_rows[1]["claim_count_label"] == "3 claims"
    assert entity_rows[1]["median_pct"] == 15
    assert ctx["geo_coverage_median"] == 10.0
    assert ctx["geo_coverage"][0]["median_pct"] == 20


def test_gaps_context_topic_rows_include_bar_values_and_delta() -> None:
    ctx = _ctx(
        claim_types=[
            {"claim_type": "habitat", "count": 10},
            {"claim_type": "policy", "count": 90},
        ],
        per_corpus_expected_shares={
            "turnbull": {"habitat": 0.4, "policy": 0.6},
        },
        per_corpus_claim_types=[
            {"institution_id": "turnbull", "claim_type": "habitat", "count": 10},
            {"institution_id": "turnbull", "claim_type": "policy", "count": 90},
        ],
    )

    habitat = ctx["topic_rows"][0]
    assert habitat["claim_type"] == "policy"
    assert habitat["actual_width_pct"] == 100.0
    assert habitat["expected_width_pct"] == 66.7
    assert habitat["delta_pct"] == 30.0
    assert habitat["status"] == "ok"

    underrepresented = ctx["topic_rows"][1]
    assert underrepresented["claim_type"] == "habitat"
    assert underrepresented["actual_width_pct"] == 25.0
    assert underrepresented["expected_width_pct"] == 100.0
    assert underrepresented["delta_pct"] == -30.0
    assert underrepresented["status"] == "thin"


def test_gaps_context_aligns_per_corpus_comparison_and_uses_worst_status() -> None:
    ctx = _ctx(
        per_corpus_expected_shares={
            "spokane": {"habitat": 0.4, "policy": 0.6},
            "turnbull": {"habitat": 0.4, "policy": 0.6},
        },
        per_corpus_claim_types=[
            {"institution_id": "spokane", "claim_type": "habitat", "count": 40},
            {"institution_id": "spokane", "claim_type": "policy", "count": 60},
            {"institution_id": "turnbull", "claim_type": "habitat", "count": 10},
            {"institution_id": "turnbull", "claim_type": "policy", "count": 90},
        ],
    )

    comparison = ctx["per_corpus_comparison"]
    assert [c["institution_id"] for c in comparison["corpora"]] == ["spokane", "turnbull"]

    habitat = next(row for row in comparison["rows"] if row["claim_type"] == "habitat")
    assert [cell["actual_pct"] for cell in habitat["cells"]] == [40.0, 10.0]
    assert [cell["expected_pct"] for cell in habitat["cells"]] == [40.0, 40.0]
    assert habitat["status"] == "thin"


def test_gaps_template_renders_with_enriched_context() -> None:
    env = Environment(
        loader=ChoiceLoader([
            FileSystemLoader("gemynd/retrieval/web/templates"),
            FileSystemLoader("gemynd/shared_templates"),
        ]),
        autoescape=True,
    )
    ctx = _ctx(
        temporal=[
            {"year": 1900, "doc_count": 10, "claim_count": 20},
            {"year": 1902, "doc_count": 1, "claim_count": 3},
        ],
        entity_depth=[
            {
                "entity_type": "Person",
                "name": "A",
                "primary_claim_count": 1,
                "coverage_median": 20,
            }
        ],
        claim_types=[{"claim_type": "habitat", "count": 10}],
        per_corpus_expected_shares={
            "spokane": {"habitat": 0.4},
            "turnbull": {"habitat": 0.4},
        },
        per_corpus_claim_types=[
            {"institution_id": "spokane", "claim_type": "habitat", "count": 10},
            {"institution_id": "turnbull", "claim_type": "habitat", "count": 10},
        ],
    )

    html = env.get_template("gaps.html").render(
        request=None,
        active_page="gaps",
        user_initials="U",
        user_display_name="User",
        user_role="admin",
        **ctx,
    )

    assert "Query signal gaps" in html
    assert "Per-corpus topical balance" in html
    assert "5% of median" in html


def test_gaps_template_renders_with_legacy_context_shape() -> None:
    env = Environment(
        loader=ChoiceLoader([
            FileSystemLoader("gemynd/retrieval/web/templates"),
            FileSystemLoader("gemynd/shared_templates"),
        ]),
        autoescape=True,
    )

    html = env.get_template("gaps.html").render(
        request=None,
        active_page="gaps",
        user_initials="U",
        user_display_name="User",
        user_role="admin",
        temporal_gaps=[],
        year_span="1900-1902",
        entity_depth_by_type=[
            ("Person", [{"name": "A", "primary_claim_count": 1}]),
        ],
        geo_coverage=[
            {
                "name": "Cheney",
                "entity_type": "Place",
                "location_specific_claims": 2,
            }
        ],
        show_unclassified_banner=False,
        unclassified_pct=0,
        topic_rows=[
            {
                "claim_type": "habitat",
                "actual_pct": 10.0,
                "expected_pct": 40.0,
                "status": "thin",
            }
        ],
        per_corpus_panels=[],
        conv_gaps=[],
    )

    assert "Entity depth gaps" in html
    assert "1 claim" in html
    assert "2-3 claims" in html
    assert "-30.0%" in html
