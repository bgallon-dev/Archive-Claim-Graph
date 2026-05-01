from __future__ import annotations

import pytest

from gemynd.retrieval.web.app import _build_stats_context


def _context(
    *,
    overview: dict | None = None,
    entity_types: list[dict] | None = None,
    temporal: list[dict] | None = None,
    confidence: dict | None = None,
) -> dict:
    return _build_stats_context(
        overview or {
            "total_docs": 2,
            "earliest_year": 1900,
            "latest_year": 1902,
            "total_pages": 4,
            "donor_restricted_count": 1,
        },
        doc_types=[],
        claim_types=[],
        entity_types=entity_types if entity_types is not None else [{"entity_type": "Person", "count": 3}],
        temporal=temporal if temporal is not None else [],
        confidence=confidence or {
            "total_claims": 20,
            "avg_confidence": 0.88,
            "high_count": 16,
            "medium_count": 3,
            "low_count": 1,
            "uncertain_epistemic_count": 2,
        },
    )


def test_stats_context_keeps_annual_rows_for_short_spans() -> None:
    ctx = _context(
        temporal=[
            {"year": 1900, "doc_count": 5},
            {"year": 1902, "doc_count": 10},
        ],
    )

    assert ctx["temporal_max_count"] == 10
    assert ctx["temporal_max_label"] == "1902"
    assert [(r["label"], r["count"], r["width_pct"], r["is_gap"]) for r in ctx["temporal_rows"]] == [
        ("1900", 5, 50, False),
        ("1901", 0, 0, True),
        ("1902", 10, 100, False),
    ]
    assert all(row["segments"] == [] for row in ctx["temporal_rows"])


def test_stats_context_aggregates_long_spans_with_year_segments() -> None:
    ctx = _context(
        temporal=[
            {"year": 1890, "doc_count": 5},
            {"year": 1903, "doc_count": 10},
            {"year": 1909, "doc_count": 30},
            {"year": 1921, "doc_count": 20},
        ],
    )

    rows = ctx["temporal_rows"]
    assert [(r["label"], r["count"], r["width_pct"], r["is_gap"]) for r in rows] == [
        ("1890s", 5, 12, False),
        ("1900s", 40, 100, False),
        ("1910s", 0, 0, True),
        ("1920s", 20, 50, False),
    ]
    assert ctx["temporal_max_count"] == 40
    assert ctx["temporal_max_label"] == "1900s"

    segments = rows[1]["segments"]
    assert len(segments) == 10
    assert segments[3] == {"year": 1903, "count": 10, "opacity": "0.48"}
    assert segments[4] == {"year": 1904, "count": 0, "opacity": "0.00"}
    assert segments[9] == {"year": 1909, "count": 30, "opacity": "1.00"}


def test_stats_context_formats_density_and_preserves_donor_count() -> None:
    ctx = _context()

    assert ctx["donor_restricted"] == 1
    assert ctx["claims_per_doc"] == "10.0"
    assert ctx["entities_per_doc"] == "1.5"


def test_stats_context_density_handles_zero_documents() -> None:
    ctx = _context(
        overview={
            "total_docs": 0,
            "earliest_year": None,
            "latest_year": None,
            "total_pages": 0,
            "donor_restricted_count": 0,
        },
        entity_types=[{"entity_type": "Person", "count": 7}],
        confidence={"total_claims": 5, "high_count": 3, "medium_count": 2, "low_count": 0},
    )

    assert ctx["claims_per_doc"] == "0.0"
    assert ctx["entities_per_doc"] == "0.0"


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        ({"total_claims": 0, "high_count": 0, "medium_count": 0, "low_count": 0}, "No active claims"),
        ({"total_claims": 100, "high_count": 82, "medium_count": 13, "low_count": 5}, "Strong local"),
        ({"total_claims": 100, "high_count": 50, "medium_count": 40, "low_count": 10}, "Mixed local"),
        ({"total_claims": 100, "high_count": 45, "medium_count": 25, "low_count": 30}, "Elevated low-confidence"),
    ],
)
def test_stats_context_confidence_qualifier(confidence: dict, expected: str) -> None:
    ctx = _context(confidence=confidence)

    assert expected in ctx["confidence_qualifier"]
    assert [row["label"] for row in ctx["confidence_rows"]] == ["High", "Medium", "Low"]
