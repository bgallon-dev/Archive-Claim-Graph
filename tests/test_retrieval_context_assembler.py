"""Unit tests for Layer 2B: provenance context assembler.

Tests cover:
  • _row_to_block conversion from raw executor rows.
  • _serialise_block output format.
  • Budget cap (max_claims truncation).
  • Entity-anchored vs. fulltext retrieval cascade selection.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gemynd.retrieval.context_assembler import (
    ProvenanceContextAssembler,
    _bounded_query_limit,
    _build_retrieval_summary,
    _compute_top_co_occurring_entities,
    _derive_structural_view,
    _display_doc_title,
    _extract_exact_date,
    _infer_claim_types,
    _is_ocr_garbage_title,
    _row_to_block,
    _scrub_doc_list_titles,
    _select_retrieval_plan,
    _serialise_block,
    _stratified_budget_cap,
    _temporally_diversified_take,
)
from gemynd.retrieval.models import EntityContext, ProvenanceBlock, ResolvedEntity


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_raw_row(
    claim_id: str = "c-001",
    claim_type: str = "POPULATION_TREND",
    confidence: float = 0.85,
    epistemic: str = "observed",
    sentence: str = "Mallard numbers increased.",
    doc_title: str = "Annual Report 1942",
    year: int | None = 1942,
    species: str | None = "Mallard",
    institution_id: str = "turnbull",
) -> dict:
    return {
        "c": {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "extraction_confidence": confidence,
            "certainty": epistemic,
            "source_sentence": sentence,
        },
        "d": {
            "title": doc_title,
            "date_start": "1942-01-01",
            "date_end": "1942-12-31",
            "institution_id": institution_id,
        },
        "pg": {"page_number": 3},
        "para": {"paragraph_id": "para-001"},
        "obs": {"observation_type": "species_count", "observation_id": "obs-001"},
        "sp": {"name": species} if species else {},
        "y": {"year": year} if year else {},
        "measurements": [],
    }


def _make_block(**kwargs) -> ProvenanceBlock:
    defaults = dict(
        doc_title="Annual Report 1942",
        doc_date_start="1942-01-01",
        doc_date_end="1942-12-31",
        page_number=3,
        paragraph_id="para-001",
        claim_id="c-001",
        claim_type="POPULATION_TREND",
        extraction_confidence=0.85,
        epistemic_status="observed",
        source_sentence="Mallard numbers increased.",
        observation_type="species_count",
        species_name="Mallard",
        year=1942,
        measurements=[],
        institution_id="turnbull",
    )
    defaults.update(kwargs)
    return ProvenanceBlock(**defaults)


# ---------------------------------------------------------------------------
# _row_to_block
# ---------------------------------------------------------------------------

class TestRowToBlock:
    def test_basic_conversion(self):
        row = _make_raw_row()
        block = _row_to_block(row)
        assert block is not None
        assert block.claim_id == "c-001"
        assert block.claim_type == "POPULATION_TREND"
        assert block.extraction_confidence == 0.85
        assert block.epistemic_status == "observed"
        assert block.source_sentence == "Mallard numbers increased."
        assert block.doc_title == "Annual Report 1942"
        assert block.page_number == 3
        assert block.year == 1942
        assert block.species_name == "Mallard"

    def test_missing_claim_id_returns_none(self):
        row = _make_raw_row()
        row["c"] = {}
        assert _row_to_block(row) is None

    def test_none_claim_dict_returns_none(self):
        row = _make_raw_row()
        row["c"] = None
        assert _row_to_block(row) is None

    def test_measurements_parsed(self):
        row = _make_raw_row()
        row["measurements"] = [
            {"name": "duck_count", "numeric_value": 42.0, "unit": "individuals", "approximate": False}
        ]
        block = _row_to_block(row)
        assert block is not None
        assert len(block.measurements) == 1
        assert block.measurements[0]["name"] == "duck_count"
        assert block.measurements[0]["value"] == 42.0

    def test_none_measurements_list(self):
        row = _make_raw_row()
        row["measurements"] = None
        block = _row_to_block(row)
        assert block is not None
        assert block.measurements == []

    def test_series_populated_from_document(self):
        row = _make_raw_row()
        row["d"]["series"] = "BLM GLO Records"
        block = _row_to_block(row)
        assert block is not None
        assert block.series == "BLM GLO Records"

    def test_series_none_when_absent(self):
        row = _make_raw_row()
        block = _row_to_block(row)
        assert block is not None
        assert block.series is None


# ---------------------------------------------------------------------------
# OCR garbage detection / display title fallback
# ---------------------------------------------------------------------------

class TestOcrGarbageDetection:
    def test_clean_title_not_garbage(self):
        assert not _is_ocr_garbage_title("Annual Report 1942")
        assert not _is_ocr_garbage_title("The History of Spokane")

    def test_empty_title_not_garbage(self):
        assert not _is_ocr_garbage_title("")

    def test_spaced_caps_flagged(self):
        assert _is_ocr_garbage_title("G O N TIN U E D h o")
        assert _is_ocr_garbage_title("X Y Z something")

    def test_mostly_uppercase_flagged(self):
        assert _is_ocr_garbage_title("LANDS LANDSI LANDSI")
        assert _is_ocr_garbage_title("THE WOMlEslN'S CORNER")

    def test_runaway_length_flagged(self):
        long_title = "A" + "a" * 130
        assert _is_ocr_garbage_title(long_title)


class TestDisplayDocTitle:
    def test_clean_title_returned_verbatim(self):
        block = _make_block(doc_title="Annual Report 1942")
        assert _display_doc_title(block) == "Annual Report 1942"

    def test_garbage_title_falls_back_to_institution(self):
        block = _make_block(
            doc_title="G O N TIN U E D h o",
            institution_id="spokane_newspaper",
        )
        result = _display_doc_title(block)
        assert "G O N TIN" not in result
        assert "spokane_newspaper" in result
        assert "OCR degraded" in result

    def test_empty_title_with_institution(self):
        block = _make_block(doc_title="", institution_id="turnbull")
        result = _display_doc_title(block)
        assert "turnbull" in result
        assert "OCR degraded" in result

    def test_garbage_title_no_institution_uses_placeholder(self):
        block = _make_block(
            doc_title="LANDS LANDSI LANDSI",
            institution_id=None,
        )
        result = _display_doc_title(block)
        assert "LANDS" not in result
        assert "unavailable" in result

    def test_serialise_block_uses_display_title(self):
        block = _make_block(
            doc_title="G O N TIN U E D h o",
            institution_id="spokane_newspaper",
        )
        text = _serialise_block(block)
        # The garbage title must NOT appear in the synthesis context
        assert "G O N TIN" not in text
        # Structured fallback must
        assert "spokane_newspaper" in text
        assert "OCR degraded" in text

    def test_serialise_block_keeps_clean_title(self):
        block = _make_block(doc_title="The History of Spokane")
        text = _serialise_block(block)
        assert "The History of Spokane" in text


# ---------------------------------------------------------------------------
# _serialise_block
# ---------------------------------------------------------------------------

class TestSerialiseBlock:
    def test_basic_structure(self):
        block = _make_block()
        text = _serialise_block(block)
        assert "DOCUMENT:" in text
        assert "Annual Report 1942" in text
        assert "CONFIDENCE_TIER: HIGH (0.85)" in text
        assert "PAGE: 3" in text
        assert "PARAGRAPH: para-001" in text
        assert "CLAIM_ID: c-001" in text
        assert "CLAIM [POPULATION_TREND" in text
        assert "epistemic=observed" in text
        assert "Mallard numbers increased." in text
        # Confidence no longer appears inline in the CLAIM bracket
        claim_bracket = text.split("CLAIM [")[1].split("]")[0]
        assert "confidence=" not in claim_bracket

    def test_claim_id_rendered_matches_block_field(self):
        """Regression: synthesis validator compares model citations against
        block.claim_id; the rendered context must surface the same id verbatim
        so the model can copy it. Pre-fix the only id-shaped token in the
        rendered text was paragraph_id, so the model cited that and validation
        rejected every citation."""
        block = _make_block(
            paragraph_id="doc_db3544e3b57e69d9:paragraph:00004:p0001",
            claim_id="run_abc123:claim:001:def456",
        )
        text = _serialise_block(block)
        assert "CLAIM_ID: run_abc123:claim:001:def456" in text
        # And the CLAIM_ID line must precede the CLAIM line so the model
        # associates them when reading top-to-bottom.
        assert text.index("CLAIM_ID:") < text.index("CLAIM [")

    def test_retrieved_via_present_when_traversal_types_set(self):
        block = _make_block(traversal_rel_types=["SUBJECT_OF"])
        text = _serialise_block(block)
        assert "RETRIEVED_VIA: SUBJECT_OF" in text

    def test_retrieved_via_absent_when_no_traversal_types(self):
        block = _make_block(traversal_rel_types=[])
        text = _serialise_block(block)
        assert "RETRIEVED_VIA" not in text

    def test_confidence_tier_medium(self):
        block = _make_block(extraction_confidence=0.75)
        text = _serialise_block(block)
        assert "CONFIDENCE_TIER: MEDIUM (0.75)" in text

    def test_confidence_tier_low(self):
        block = _make_block(extraction_confidence=0.50)
        text = _serialise_block(block)
        assert "CONFIDENCE_TIER: LOW (0.50)" in text

    def test_observation_section(self):
        block = _make_block()
        text = _serialise_block(block)
        assert "OBSERVATION [species_count]" in text
        assert "species=Mallard" in text
        assert "year=1942" in text

    def test_measurement_in_output(self):
        block = _make_block(
            measurements=[{"name": "count", "value": 120, "unit": "birds", "approximate": False}]
        )
        text = _serialise_block(block)
        assert "MEASUREMENT: count=120 birds" in text

    def test_approximate_measurement_flag(self):
        block = _make_block(
            measurements=[{"name": "count", "value": 100, "unit": "birds", "approximate": True}]
        )
        text = _serialise_block(block)
        assert "approximate=True" in text

    def test_no_observation(self):
        block = _make_block(observation_type=None, species_name=None, year=None)
        text = _serialise_block(block)
        assert "OBSERVATION" not in text


# ---------------------------------------------------------------------------
# Budget cap
# ---------------------------------------------------------------------------

class TestBudgetCap:
    def test_cypher_fetch_limit_is_capped_for_large_context_budget(self):
        assert _bounded_query_limit(1000, 3) == 250

    def test_conversational_budget_respected(self):
        mock_executor = MagicMock()
        # Return 20 identical rows (above default budget of 10).
        rows = [_make_raw_row(claim_id=f"c-{i:03d}") for i in range(20)]
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(
            executor=mock_executor,
            budget_conversational=10,
            budget_hybrid=4,
        )
        entity_ctx = EntityContext()  # no resolved entities → fulltext path
        blocks, _ = assembler.assemble("describe habitat", entity_ctx)
        assert len(blocks) <= 10

    def test_hybrid_budget_respected(self):
        mock_executor = MagicMock()
        rows = [_make_raw_row(claim_id=f"c-{i:03d}") for i in range(20)]
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(
            executor=mock_executor,
            budget_conversational=10,
            budget_hybrid=4,
        )
        entity_ctx = EntityContext()
        blocks, _ = assembler.assemble("describe and count", entity_ctx, is_hybrid=True)
        assert len(blocks) <= 4

    def test_ranked_by_confidence(self):
        mock_executor = MagicMock()
        rows = [
            _make_raw_row(claim_id="low", confidence=0.4),
            _make_raw_row(claim_id="high", confidence=0.95),
            _make_raw_row(claim_id="mid", confidence=0.7),
        ]
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(
            executor=mock_executor,
            budget_conversational=3,
        )
        entity_ctx = EntityContext()
        blocks, _ = assembler.assemble("query", entity_ctx)
        assert blocks[0].claim_id == "high"
        assert blocks[-1].claim_id == "low"

    def test_neo4j_resource_error_degrades_to_empty_context(self):
        mock_executor = MagicMock()
        mock_executor.run.side_effect = RuntimeError(
            "Neo.TransientError.General.OutOfMemoryError: failed to allocate"
        )
        assembler = ProvenanceContextAssembler(executor=mock_executor)
        entity_ctx = EntityContext(
            resolved=[ResolvedEntity("mallard", "sp-mallard", "Species", 0.95, "REFERS_TO")]
        )

        blocks, context_text = assembler.assemble("tell me about mallard", entity_ctx)

        assert blocks == []
        assert context_text == ""
        assert assembler._last_retrieval_path == "empty"


# ---------------------------------------------------------------------------
# Retrieval path selection
# ---------------------------------------------------------------------------

class TestRetrievalCascade:
    def test_entity_anchored_path_used_when_resolved(self):
        from gemynd.core.graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY

        mock_executor = MagicMock()
        mock_executor.run.return_value = []

        assembler = ProvenanceContextAssembler(executor=mock_executor)
        entity_ctx = EntityContext(
            resolved=[ResolvedEntity("mallard", "sp-mallard", "Species", 0.95, "REFERS_TO")]
        )
        assembler.assemble("query", entity_ctx)
        called_queries = [call[0][0] for call in mock_executor.run.call_args_list]
        assert any(ENTITY_ANCHORED_CLAIMS_QUERY in q for q in called_queries)

    def test_fulltext_path_used_when_unresolved(self):
        from gemynd.core.graph.cypher import FULLTEXT_CLAIMS_QUERY

        mock_executor = MagicMock()
        mock_executor.run.return_value = []

        assembler = ProvenanceContextAssembler(executor=mock_executor)
        entity_ctx = EntityContext(unresolved=["unknownbird"])
        assembler.assemble("query about unknownbird", entity_ctx)
        called_queries = [call[0][0] for call in mock_executor.run.call_args_list]
        assert any(FULLTEXT_CLAIMS_QUERY in q for q in called_queries)


# ---------------------------------------------------------------------------
# Routing strategy selection
# ---------------------------------------------------------------------------

class TestInferClaimTypes:
    def test_habitat_keyword_maps_to_types(self):
        result = _infer_claim_types("what are the habitat conditions?")
        assert result is not None
        assert "habitat_condition" in result
        assert "weather_observation" in result

    def test_population_keyword_maps_to_types(self):
        result = _infer_claim_types("show population trends for mallard")
        assert result is not None
        assert "population_estimate" in result
        assert "species_presence" in result

    def test_no_signal_returns_none(self):
        assert _infer_claim_types("tell me about the refuge") is None

    def test_multiple_keywords_merged(self):
        result = _infer_claim_types("habitat and population data")
        assert result is not None
        assert "habitat_condition" in result
        assert "population_estimate" in result


class TestSelectRetrievalStrategy:
    def _make_entity_ctx(self, n: int = 0) -> EntityContext:
        resolved = [
            ResolvedEntity(f"entity-{i}", f"id-{i}", "Species", 0.9, "REFERS_TO")
            for i in range(n)
        ]
        return EntityContext(resolved=resolved)

    @staticmethod
    def _primary(plan):
        """Return the first (template, params) pair of the first attempt.

        The router now returns a relaxation chain of (label, [(template,
        params), ...]) attempts; these tests only care about the primary.
        """
        assert plan, "expected at least one attempt"
        label, queries = plan[0]
        assert queries, f"attempt {label!r} has no queries"
        return queries[0]

    def test_temporal_route_when_year_and_no_entity_with_anchor(self):
        from gemynd.core.graph.cypher import build_temporal_with_anchor_query

        anchor_cypher = build_temporal_with_anchor_query("Refuge", "ABOUT_REFUGE")
        plan = _select_retrieval_plan(
            "duck counts", self._make_entity_ctx(0), year_min=1950, year_max=1960, budget=10,
            anchor_temporal_plans=[("refuge_abc", anchor_cypher)],
        )
        template, params = self._primary(plan)
        assert template is anchor_cypher
        assert params["year_min"] == 1950
        assert params["year_max"] == 1960
        assert params["anchor_id"] == "refuge_abc"

    def test_temporal_route_when_year_and_no_entity_no_anchor(self):
        from gemynd.core.graph.cypher import TEMPORAL_CLAIMS_QUERY

        plan = _select_retrieval_plan(
            "duck counts", self._make_entity_ctx(0), year_min=1950, year_max=1960, budget=10,
        )
        template, params = self._primary(plan)
        assert template == TEMPORAL_CLAIMS_QUERY
        assert params["year_min"] == 1950

    def test_multi_entity_route_when_two_entities(self):
        from gemynd.core.graph.cypher import MULTI_ENTITY_CLAIMS_QUERY

        plan = _select_retrieval_plan(
            "compare mallard and teal", self._make_entity_ctx(2),
            year_min=None, year_max=None, budget=10
        )
        template, params = self._primary(plan)
        assert template == MULTI_ENTITY_CLAIMS_QUERY
        assert len(params["entity_ids"]) == 2

    def test_entity_plus_claim_type_route(self):
        from gemynd.core.graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY

        plan = _select_retrieval_plan(
            "habitat conditions for mallard", self._make_entity_ctx(1),
            year_min=None, year_max=None, budget=10
        )
        template, params = self._primary(plan)
        assert template == ENTITY_ANCHORED_CLAIMS_QUERY
        # Over-fetch limit (budget * 3) distinguishes this from plain entity route
        assert params["limit"] == 30

    def test_claim_type_only_route_uses_fulltext_then_structural_fallback(self):
        from gemynd.core.graph.cypher import (
            CLAIM_TYPE_SCOPED_QUERY,
            FULLTEXT_CLAIMS_BY_TYPE_QUERY,
        )

        plan = _select_retrieval_plan(
            "describe all population estimates", self._make_entity_ctx(0),
            year_min=None, year_max=None, budget=10
        )
        template, params = self._primary(plan)
        assert template == FULLTEXT_CLAIMS_BY_TYPE_QUERY
        assert "population" in params["search_text"].lower()
        assert "population_estimate" in (params["claim_types"] or [])
        fallback_template, fallback_params = plan[1][1][0]
        assert fallback_template == CLAIM_TYPE_SCOPED_QUERY
        assert "population_estimate" in (fallback_params["claim_types"] or [])

    def test_source_scoped_topic_uses_fulltext_before_browse(self):
        from gemynd.core.graph.cypher import FULLTEXT_CLAIMS_SCOPED_QUERY

        plan = _select_retrieval_plan(
            "What did newspapers report about labor between 1908 and 1909?",
            self._make_entity_ctx(0),
            year_min=1908,
            year_max=1909,
            budget=12,
            intent_map={"labor": ["labor_action"]},
            doc_types=["newspaper_page", "newspaper_article"],
        )
        label, queries = plan[0]
        template, params = queries[0]
        assert label == "scoped_fulltext"
        assert template == FULLTEXT_CLAIMS_SCOPED_QUERY
        assert params["claim_types"] == ["labor_action"]
        assert params["doc_types"] == ["newspaper_page", "newspaper_article"]
        assert params["year_min"] == 1908
        assert params["year_max"] == 1909
        assert params["candidate_limit"] > params["limit"]

    def test_date_scoped_topic_uses_fulltext_before_temporal_scan(self):
        from gemynd.core.graph.cypher import FULLTEXT_CLAIMS_SCOPED_QUERY

        plan = _select_retrieval_plan(
            "What infrastructure projects changed Spokane between 1900 and 1910?",
            self._make_entity_ctx(1),
            year_min=1900,
            year_max=1910,
            budget=12,
            intent_map={
                "infrastructure": ["infrastructure_project", "civic_development"]
            },
            anchor_document_plans={"id-0": "unused-in-this-plan"},
        )
        label, queries = plan[0]
        template, params = queries[0]
        assert label == "scoped_fulltext"
        assert template == FULLTEXT_CLAIMS_SCOPED_QUERY
        assert params["doc_types"] is None
        assert params["series"] is None
        assert params["year_min"] == 1900
        assert params["year_max"] == 1910

    def test_named_event_uses_fulltext_before_dropping_event_anchor(self):
        from gemynd.core.graph.cypher import FULLTEXT_CLAIMS_BY_TYPE_QUERY

        entity_ctx = EntityContext(resolved=[
            ResolvedEntity(
                "Free Speech Fight",
                "event-free-speech",
                "Event",
                1.0,
                "REFERS_TO",
            )
        ])
        plan = _select_retrieval_plan(
            "Give me a timeline of the Free Speech Fight",
            entity_ctx,
            year_min=None,
            year_max=None,
            budget=12,
            intent_map={"free speech": ["free_speech_event", "labor_legal_action"]},
        )

        label, queries = plan[0]
        template, params = queries[0]
        assert label == "event_topic:fulltext"
        assert template == FULLTEXT_CLAIMS_BY_TYPE_QUERY
        assert set(params["claim_types"]) == {
            "free_speech_event", "labor_legal_action",
        }

    def test_named_event_with_year_uses_scoped_fulltext(self):
        from gemynd.core.graph.cypher import FULLTEXT_CLAIMS_SCOPED_QUERY

        entity_ctx = EntityContext(resolved=[
            ResolvedEntity(
                "Free Speech Fight",
                "event-free-speech",
                "Event",
                1.0,
                "REFERS_TO",
            )
        ])
        plan = _select_retrieval_plan(
            "Who was named in the 1909 Free Speech Fight?",
            entity_ctx,
            year_min=1909,
            year_max=1909,
            budget=12,
            intent_map={"free speech": ["free_speech_event", "labor_legal_action"]},
        )

        label, queries = plan[0]
        template, params = queries[0]
        assert label == "event_topic:fulltext"
        assert template == FULLTEXT_CLAIMS_SCOPED_QUERY
        assert params["year_min"] == 1909
        assert params["year_max"] == 1909

    def test_explicit_day_routes_before_broad_temporal_query(self):
        from gemynd.core.graph.cypher import EXACT_DATE_CLAIMS_QUERY

        plan = _select_retrieval_plan(
            "What happened in Spokane on November 3rd, 1909?",
            self._make_entity_ctx(1),
            year_min=1909,
            year_max=1909,
            budget=12,
        )

        label, queries = plan[0]
        template, params = queries[0]
        assert label == "exact_date"
        assert template == EXACT_DATE_CLAIMS_QUERY
        assert params["target_date"] == "1909-11-03"

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("November 3rd, 1909", "1909-11-03"),
            ("Nov. 3, 1909", "1909-11-03"),
            ("3 November 1909", "1909-11-03"),
            ("1909-11-03", "1909-11-03"),
            ("during November 1909", None),
            ("on February 30, 1909", None),
        ],
    )
    def test_exact_date_extraction(self, query, expected):
        assert _extract_exact_date(query) == expected

    def test_compound_source_scoped_topic_balances_fulltext_facets(self):
        from gemynd.core.graph.cypher import FULLTEXT_CLAIMS_SCOPED_QUERY

        plan = _select_retrieval_plan(
            "Compare newspaper mining labor disputes and railroad labor disputes "
            "between 1905 and 1915.",
            self._make_entity_ctx(2),
            year_min=1905,
            year_max=1915,
            budget=12,
            intent_map={
                "mining": ["mining_activity", "labor_action"],
                "railroad": ["railroad_activity", "freight_rate_dispute"],
                "labor": ["labor_action", "worker_condition"],
            },
            doc_types=["newspaper_page"],
        )

        label, queries = plan[0]
        assert label == "scoped_fulltext:facets"
        assert len(queries) == 2
        assert all(template == FULLTEXT_CLAIMS_SCOPED_QUERY for template, _ in queries)
        assert all(params["limit"] == 6 for _, params in queries)
        assert " AND " in queries[0][1]["search_text"]
        assert " AND " in queries[1][1]["search_text"]
        assert queries[0][1]["claim_types"] == [
            "mining_activity", "labor_action", "worker_condition",
        ]
        assert queries[1][1]["claim_types"] == [
            "railroad_activity", "freight_rate_dispute",
            "labor_action", "worker_condition",
        ]

    def test_single_entity_fallback_no_claim_type(self):
        from gemynd.core.graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY

        plan = _select_retrieval_plan(
            "tell me about the refuge", self._make_entity_ctx(1),
            year_min=None, year_max=None, budget=10
        )
        template, params = self._primary(plan)
        assert template == ENTITY_ANCHORED_CLAIMS_QUERY
        # Entity-only attempts now use the shared over-fetch limit.
        assert params["limit"] == 30

    def test_fulltext_fallback_when_nothing_matches(self):
        from gemynd.core.graph.cypher import FULLTEXT_CLAIMS_QUERY

        plan = _select_retrieval_plan(
            "tell me about the refuge", self._make_entity_ctx(0),
            year_min=None, year_max=None, budget=10
        )
        template, params = self._primary(plan)
        assert template == FULLTEXT_CLAIMS_QUERY
        # Fulltext now extracts content-bearing terms joined with OR.
        assert "refuge" in params["search_text"].lower()
        assert " OR " in params["search_text"]


class TestCrossCorpusOverlapCascade:
    """`_select_retrieval_plan` must route cross-corpus overlap queries through
    CROSS_CORPUS_OVERLAP_*_QUERY before falling back to standard cascade modes."""

    def _make_entity_ctx(self, n: int = 0) -> EntityContext:
        resolved = [
            ResolvedEntity(f"entity-{i}", f"id-{i}", "Person", 0.9, "REFERS_TO")
            for i in range(n)
        ]
        return EntityContext(resolved=resolved)

    def test_overlap_attempt_appears_first_when_signal_set(self):
        from gemynd.core.graph.cypher import (
            CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY,
            CROSS_CORPUS_OVERLAP_PATENT_QUERY,
        )
        plan = _select_retrieval_plan(
            "individuals named in newspaper coverage who appear in both records",
            self._make_entity_ctx(0),
            year_min=None, year_max=None, budget=20,
            institution_ids=["spokane_newspaper", "turnbull"],
            cross_corpus_overlap=True,
        )
        assert plan, "expected at least one cascade attempt"
        label, queries = plan[0]
        assert label.startswith("cross_corpus_overlap:")
        templates = {q[0] for q in queries}
        assert CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY in templates
        assert CROSS_CORPUS_OVERLAP_PATENT_QUERY in templates

    def test_overlap_skipped_when_signal_unset(self):
        """Without the cross_corpus_overlap flag, the cascade must not include
        overlap attempts even when two institution_ids are configured."""
        from gemynd.core.graph.cypher import (
            CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY,
            CROSS_CORPUS_OVERLAP_PATENT_QUERY,
        )
        plan = _select_retrieval_plan(
            "describe wetlands",
            self._make_entity_ctx(0),
            year_min=None, year_max=None, budget=20,
            institution_ids=["spokane_newspaper", "turnbull"],
            cross_corpus_overlap=False,
        )
        all_templates: set[str] = set()
        for _label, queries in plan:
            for tpl, _params in queries:
                all_templates.add(tpl)
        assert CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY not in all_templates
        assert CROSS_CORPUS_OVERLAP_PATENT_QUERY not in all_templates

    def test_overlap_skipped_when_single_corpus(self):
        """Cross-corpus overlap requires ≥2 institution_ids; single-corpus
        deployments must fall through to the standard cascade."""
        from gemynd.core.graph.cypher import (
            CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY,
        )
        plan = _select_retrieval_plan(
            "individuals appearing in both records",
            self._make_entity_ctx(0),
            year_min=None, year_max=None, budget=20,
            institution_ids=["spokane_newspaper"],
            cross_corpus_overlap=True,
        )
        all_templates = {q[0] for _, qs in plan for q in qs}
        assert CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY not in all_templates

    def test_overlap_params_carry_corpus_pair_and_claim_types(self):
        """The overlap newspaper-side query must receive both corpus_a/b plus
        any claim_type filter inferred from the query vocabulary (e.g. labor)."""
        from gemynd.core.graph.cypher import CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY
        plan = _select_retrieval_plan(
            "labor disputes individuals in both newspaper and patents",
            self._make_entity_ctx(0),
            year_min=None, year_max=None, budget=20,
            institution_ids=["spokane_newspaper", "turnbull"],
            cross_corpus_overlap=True,
        )
        _label, queries = plan[0]
        newspaper_q = next(
            (q for q in queries if q[0] == CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY),
            None,
        )
        assert newspaper_q is not None
        _, params = newspaper_q
        assert params["corpus_a"] == "spokane_newspaper"
        assert params["corpus_b"] == "turnbull"
        # Permitted-levels must be threaded through so access control still applies.
        assert "permitted_levels" in params


class TestDuplicateEntityIdDedup:
    def test_duplicate_ids_select_entity_anchored_not_multi_entity(self):
        """Two surface forms resolving to the same entity_id should not
        trigger the MULTI_ENTITY path."""
        from gemynd.core.graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY

        resolved = [
            ResolvedEntity("Turnbull Refuge", "refuge_abc", "Refuge", 1.0, "REFERS_TO"),
            ResolvedEntity("refuge", "refuge_abc", "Refuge", 0.88, "REFERS_TO"),
        ]
        entity_ctx = EntityContext(resolved=resolved)
        plan = _select_retrieval_plan(
            "what wildlife observations at Turnbull Refuge",
            entity_ctx, year_min=None, year_max=None, budget=20,
        )
        _label, queries = plan[0]
        template, params = queries[0]
        assert template == ENTITY_ANCHORED_CLAIMS_QUERY
        assert params["entity_id"] == "refuge_abc"

    def test_assembler_uses_entity_anchored_for_duplicate_ids(self):
        """End-to-end: duplicate entity_ids must not route to MULTI_ENTITY."""
        from gemynd.core.graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY

        mock_executor = MagicMock()
        mock_executor.run.return_value = [_make_raw_row()]

        assembler = ProvenanceContextAssembler(executor=mock_executor)
        entity_ctx = EntityContext(
            resolved=[
                ResolvedEntity("Turnbull Refuge", "refuge_abc", "Refuge", 1.0, "REFERS_TO"),
                ResolvedEntity("refuge", "refuge_abc", "Refuge", 0.88, "REFERS_TO"),
            ]
        )
        assembler.assemble("wildlife at Turnbull Refuge", entity_ctx)
        called_query = mock_executor.run.call_args[0][0]
        assert called_query == ENTITY_ANCHORED_CLAIMS_QUERY


# ---------------------------------------------------------------------------
# Temporal diversification
# ---------------------------------------------------------------------------

class TestTemporalDiversification:
    def _blocks(self, years: list[int | None], base_conf: float = 0.80) -> list[ProvenanceBlock]:
        return [
            _make_block(claim_id=f"c-{i}", year=y, extraction_confidence=base_conf - i * 0.001)
            for i, y in enumerate(years)
        ]

    def test_diverse_years_sampled(self):
        years = [1938, 1939, 1940, 1941, 1942] * 2  # 10 blocks, 5 years
        blocks = self._blocks(years)
        result = _temporally_diversified_take(blocks, 5)
        result_years = {b.year for b in result}
        assert result_years == {1938, 1939, 1940, 1941, 1942}

    def test_single_year_falls_through(self):
        blocks = self._blocks([1942] * 10)
        result = _temporally_diversified_take(blocks, 5)
        assert len(result) == 5
        # Falls through to top-N; original order (desc confidence) preserved
        assert result == blocks[:5]

    def test_below_min_buckets_falls_through(self):
        # 4 distinct years < min_year_buckets=5
        years = [1938, 1939, 1940, 1941] * 3
        blocks = self._blocks(years)
        result = _temporally_diversified_take(blocks, 6, min_year_buckets=5)
        assert result == blocks[:6]

    def test_none_year_bucketed_separately(self):
        # Mix of dated and undated — must not crash
        years = [1938, 1939, 1940, 1941, 1942, None, None]
        blocks = self._blocks(years)
        result = _temporally_diversified_take(blocks, 5)
        assert 1 <= len(result) <= 5

    def test_proportional_distribution(self):
        # 8 blocks from 1940, 2 from 1945 + 3 more years to exceed min_year_buckets
        years = [1940] * 8 + [1941] * 1 + [1942] * 1 + [1943] * 1 + [1945] * 2
        blocks = self._blocks(years)
        result = _temporally_diversified_take(blocks, 6)
        year_counts = {}
        for b in result:
            year_counts[b.year] = year_counts.get(b.year, 0) + 1
        # 1940 should get more slots than 1945 (proportional to its share)
        assert year_counts.get(1940, 0) > year_counts.get(1945, 0)


# ---------------------------------------------------------------------------
# Stratified budget cap
# ---------------------------------------------------------------------------

class TestStratifiedBudgetCap:
    def _corpus_blocks(
        self,
        institution_id: str,
        n: int,
        base_conf: float,
        years: list[int] | None = None,
        id_prefix: str | None = None,
    ) -> list[ProvenanceBlock]:
        prefix = id_prefix or institution_id
        return [
            _make_block(
                claim_id=f"{prefix}-{i}",
                institution_id=institution_id,
                extraction_confidence=base_conf - i * 0.001,
                year=(years[i % len(years)] if years else 1942),
            )
            for i in range(n)
        ]

    def test_equal_split_two_corpora(self):
        corp_a = self._corpus_blocks("corp_a", 30, base_conf=0.90)
        corp_b = self._corpus_blocks("corp_b", 30, base_conf=0.70)
        all_blocks = corp_a + corp_b
        all_blocks.sort(key=lambda b: b.extraction_confidence, reverse=True)

        result = _stratified_budget_cap(all_blocks, 20)
        assert len(result) == 20
        a_count = sum(1 for b in result if b.institution_id == "corp_a")
        b_count = sum(1 for b in result if b.institution_id == "corp_b")
        # Both corpora represented — without stratification corp_b would get 0
        assert a_count >= 9
        assert b_count >= 9

    def test_single_corpus_delegates(self):
        blocks = self._corpus_blocks("only_corp", 30, base_conf=0.80)
        result = _stratified_budget_cap(blocks, 10)
        assert len(result) == 10
        assert all(b.institution_id == "only_corp" for b in result)

    def test_below_threshold_corpus_not_stratified(self):
        # corp_b has only 3 blocks (< min_corpus_threshold=5): no stratification
        corp_a = self._corpus_blocks("corp_a", 30, base_conf=0.80)
        corp_b = self._corpus_blocks("corp_b", 3, base_conf=0.70)
        all_blocks = corp_a + corp_b
        all_blocks.sort(key=lambda b: b.extraction_confidence, reverse=True)

        result = _stratified_budget_cap(all_blocks, 10, min_corpus_threshold=5)
        assert len(result) == 10
        # Falls through to temporal-only path; corp_b claims may or may not appear
        b_count = sum(1 for b in result if b.institution_id == "corp_b")
        a_count = sum(1 for b in result if b.institution_id == "corp_a")
        assert a_count >= b_count  # single-year fallback picks top-10 by confidence

    def test_remainder_to_largest_corpus(self):
        corp_a = self._corpus_blocks("corp_a", 20, base_conf=0.80)
        corp_b = self._corpus_blocks("corp_b", 20, base_conf=0.70)
        all_blocks = corp_a + corp_b
        all_blocks.sort(key=lambda b: b.extraction_confidence, reverse=True)

        result = _stratified_budget_cap(all_blocks, 11)
        assert len(result) == 11
        a_count = sum(1 for b in result if b.institution_id == "corp_a")
        b_count = sum(1 for b in result if b.institution_id == "corp_b")
        # Largest corpus (corp_a, highest conf → first after sort) gets 6; other gets 5
        assert abs(a_count - b_count) == 1

    def test_temporal_applied_within_each_corpus(self):
        years = [1938, 1939, 1940, 1941, 1942, 1943]
        corp_a = self._corpus_blocks("corp_a", 12, base_conf=0.85, years=years)
        corp_b = self._corpus_blocks("corp_b", 12, base_conf=0.75, years=years)
        all_blocks = corp_a + corp_b
        all_blocks.sort(key=lambda b: b.extraction_confidence, reverse=True)

        result = _stratified_budget_cap(all_blocks, 12)
        a_years = {b.year for b in result if b.institution_id == "corp_a"}
        b_years = {b.year for b in result if b.institution_id == "corp_b"}
        # Each corpus should contribute multiple years (temporal diversification fired)
        assert len(a_years) >= 2
        assert len(b_years) >= 2


# ---------------------------------------------------------------------------
# Retrieval summary
# ---------------------------------------------------------------------------

class TestRetrievalSummary:
    def _make_selected(
        self,
        specs: list[tuple[str, int, int]],  # (institution_id, year, count)
    ) -> list[ProvenanceBlock]:
        blocks: list[ProvenanceBlock] = []
        i = 0
        for institution_id, year, count in specs:
            for _ in range(count):
                blocks.append(_make_block(claim_id=f"c-{i}", institution_id=institution_id, year=year))
                i += 1
        return blocks

    def test_no_header_when_not_truncated(self):
        blocks = self._make_selected([("turnbull", 1942, 5)])
        assert _build_retrieval_summary(blocks, blocks) == ""

    def test_header_shows_total_and_selected(self):
        candidates = self._make_selected([("turnbull", 1942, 100)])
        selected = candidates[:30]
        summary = _build_retrieval_summary(candidates, selected)
        assert "30 of 100" in summary

    def test_multi_corpus_year_range(self):
        spokane = self._make_selected([("spokane_newspaper", 1902, 5), ("spokane_newspaper", 1910, 10)])
        turnbull = self._make_selected([("turnbull", 1937, 5), ("turnbull", 2007, 10)])
        candidates = spokane + turnbull + self._make_selected([("spokane_newspaper", 1905, 100)])
        selected = spokane + turnbull
        summary = _build_retrieval_summary(candidates, selected)
        assert "spokane_newspaper" in summary
        assert "turnbull" in summary
        assert "1902" in summary
        assert "1937" in summary


class TestClaimTypePostFilter:
    def test_post_filter_applied_for_entity_plus_claim_type(self):
        mock_executor = MagicMock()
        rows = [
            _make_raw_row(claim_id="c-001", claim_type="habitat_condition"),
            _make_raw_row(claim_id="c-002", claim_type="population_estimate"),
            _make_raw_row(claim_id="c-003", claim_type="habitat_condition"),
        ]
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(executor=mock_executor, budget_conversational=10)
        entity_ctx = EntityContext(
            resolved=[ResolvedEntity("mallard", "sp-mallard", "Species", 0.95, "REFERS_TO")]
        )
        blocks, _ = assembler.assemble("habitat conditions for mallard", entity_ctx)
        returned_types = {b.claim_type for b in blocks}
        assert returned_types == {"habitat_condition"}

    def test_post_filter_falls_back_when_all_filtered(self):
        mock_executor = MagicMock()
        # All rows have a claim_type not matching "habitat" vocabulary
        rows = [
            _make_raw_row(claim_id="c-001", claim_type="management_action"),
            _make_raw_row(claim_id="c-002", claim_type="management_action"),
        ]
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(executor=mock_executor, budget_conversational=10)
        entity_ctx = EntityContext(
            resolved=[ResolvedEntity("mallard", "sp-mallard", "Species", 0.95, "REFERS_TO")]
        )
        # "habitat" triggers claim_type filter but no rows match — should fall back to all rows
        blocks, _ = assembler.assemble("habitat conditions for mallard", entity_ctx)
        assert len(blocks) == 2


# ---------------------------------------------------------------------------
# End-to-end assembler integration — exercises the diagnostic failure modes
# through the full assemble() pipeline rather than the helpers in isolation.
# ---------------------------------------------------------------------------

class TestAssemblerIntegration:
    def test_cross_corpus_blending_both_corpora_represented(self):
        """The Spokane/Turnbull pathology: high-conf corpus would crowd out lower-conf one.

        Pre-fix: 30 Spokane Press claims at 0.82 + 30 Turnbull claims at 0.72, budget=20
        → 20 Spokane, 0 Turnbull. The diagnostic baseline observed exactly this.
        Post-fix: stratified split surfaces both corpora.
        """
        spokane = [
            _make_raw_row(
                claim_id=f"sp-{i:03d}",
                confidence=0.82 - i * 0.001,
                institution_id="spokane_newspaper",
                year=1905,
                doc_title=f"Spokane Press {i}",
            )
            for i in range(30)
        ]
        turnbull = [
            _make_raw_row(
                claim_id=f"tb-{i:03d}",
                confidence=0.72 - i * 0.001,
                institution_id="turnbull",
                year=1950,
                doc_title=f"Refuge Report {i}",
            )
            for i in range(30)
        ]
        mock_executor = MagicMock()
        mock_executor.run.return_value = spokane + turnbull

        assembler = ProvenanceContextAssembler(executor=mock_executor, budget_conversational=20)
        blocks, _ = assembler.assemble("query", EntityContext())

        spokane_count = sum(1 for b in blocks if b.institution_id == "spokane_newspaper")
        turnbull_count = sum(1 for b in blocks if b.institution_id == "turnbull")
        assert spokane_count >= 8, f"spokane_newspaper underrepresented: {spokane_count}"
        assert turnbull_count >= 8, f"turnbull excluded by confidence sort: {turnbull_count}"
        assert spokane_count + turnbull_count == 20

    def test_temporal_diversification_high_volume_entity(self):
        """The mallard pathology: 383 active claims spanning 1938–1963, but the answer covered
        only 1939 and 1997 because tied confidence produced near-arbitrary year selection.

        Synthetic version: 27 claims spanning 9 years at near-identical confidence.
        Without temporal diversification, top-by-confidence would cluster on whichever year
        sorted first within ties.
        """
        years = [1938, 1940, 1942, 1945, 1948, 1950, 1955, 1960, 1963]
        rows = [
            _make_raw_row(
                claim_id=f"m-{i:03d}",
                confidence=0.78 - i * 0.0001,
                year=year,
                institution_id="turnbull",
            )
            for i, year in enumerate(years * 3)
        ]
        mock_executor = MagicMock()
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(executor=mock_executor, budget_conversational=10)
        blocks, _ = assembler.assemble("query", EntityContext())

        result_years = {b.year for b in blocks}
        decades = {y // 10 for y in result_years if y is not None}
        assert len(result_years) >= 5, f"clustered on too few years: {result_years}"
        assert len(decades) >= 3, f"missed decade coverage: {sorted(decades)}"

    def test_retrieval_summary_header_present_when_truncated(self):
        rows = [
            _make_raw_row(
                claim_id=f"c-{i:03d}",
                institution_id="turnbull" if i < 25 else "spokane_newspaper",
                year=1942 if i < 25 else 1905,
            )
            for i in range(50)
        ]
        mock_executor = MagicMock()
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(executor=mock_executor, budget_conversational=10)
        _, context_text = assembler.assemble("query", EntityContext())

        assert context_text.startswith("[RETRIEVAL SUMMARY:")
        assert "10 of 50" in context_text
        # Both corpora named in the breakdown when both surface in selection
        first_line = context_text.split("\n")[0]
        assert "turnbull" in first_line or "spokane_newspaper" in first_line

    def test_no_summary_header_when_all_candidates_fit(self):
        rows = [_make_raw_row(claim_id=f"c-{i:03d}") for i in range(5)]
        mock_executor = MagicMock()
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(executor=mock_executor, budget_conversational=10)
        _, context_text = assembler.assemble("query", EntityContext())

        assert not context_text.startswith("[RETRIEVAL SUMMARY:")

    def test_garbage_doc_title_replaced_in_synthesis_context(self):
        """OCR-degraded titles must not leak into the context_text seen by the synthesis engine.

        Pre-fix (verified by the followup diagnostic): _serialise_block used raw block.doc_title,
        which sent strings like "G ON TIN UE D p A G E Q n Z" to the LLM. The chat UI's
        _citation_label() filter never reached the synthesis path.
        """
        rows = [
            _make_raw_row(
                claim_id="garbage-001",
                doc_title="G O N TIN U E D h o",
                institution_id="spokane_newspaper",
            ),
        ]
        mock_executor = MagicMock()
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(executor=mock_executor)
        _, context_text = assembler.assemble("query", EntityContext())

        assert "G O N TIN" not in context_text
        assert "spokane_newspaper" in context_text
        assert "OCR degraded" in context_text


# ---------------------------------------------------------------------------
# Tier 1 list operations: scrub OCR-garbage titles before they reach the UI
# ---------------------------------------------------------------------------

class TestScrubDocListTitles:
    def test_clean_titles_pass_through_unchanged(self):
        rows = [
            {"doc_id": "d1", "title": "Annual Report 1942", "claim_count": 5},
            {"doc_id": "d2", "title": "The History of Spokane", "claim_count": 3},
        ]
        out = _scrub_doc_list_titles(rows)
        assert out[0]["title"] == "Annual Report 1942"
        assert out[1]["title"] == "The History of Spokane"
        assert "title_raw" not in out[0]

    def test_garbage_title_replaced_with_none(self):
        rows = [{"doc_id": "d1", "title": "G O N TIN U E D h o", "claim_count": 5}]
        out = _scrub_doc_list_titles(rows)
        assert out[0]["title"] is None
        # Original preserved for archivist follow-up.
        assert out[0]["title_raw"] == "G O N TIN U E D h o"

    def test_all_caps_garbage_replaced(self):
        rows = [{"doc_id": "d1", "title": "SPOKANE TELLS NO HARD LUCK HISTORY"}]
        out = _scrub_doc_list_titles(rows)
        assert out[0]["title"] is None
        assert out[0]["title_raw"] == "SPOKANE TELLS NO HARD LUCK HISTORY"

    def test_empty_title_unchanged(self):
        rows = [{"doc_id": "d1", "title": "", "claim_count": 0}]
        out = _scrub_doc_list_titles(rows)
        # Empty input is not OCR garbage; left alone (frontend handles missing).
        assert out[0]["title"] == ""
        assert "title_raw" not in out[0]

    def test_does_not_mutate_input_rows(self):
        rows = [{"doc_id": "d1", "title": "G O N TIN U E D"}]
        original_title = rows[0]["title"]
        _scrub_doc_list_titles(rows)
        assert rows[0]["title"] == original_title  # input untouched


# ---------------------------------------------------------------------------
# _derive_structural_view
# ---------------------------------------------------------------------------

def _make_resolved_entity(
    surface_form: str = "Turnbull",
    entity_id: str = "ent-001",
    entity_type: str = "Refuge",
) -> ResolvedEntity:
    return ResolvedEntity(
        surface_form=surface_form,
        entity_id=entity_id,
        entity_type=entity_type,
        resolution_confidence=0.95,
        resolution_relation="REFERS_TO",
    )


class TestDeriveStructuralView:
    def test_returns_none_below_threshold(self):
        blocks = [_make_block(claim_id=f"c-{i}", paragraph_id=f"p-{i}") for i in range(4)]
        ctx = EntityContext(resolved=[_make_resolved_entity()])
        sv = _derive_structural_view(
            blocks=blocks,
            entity_context=ctx,
            top_co_entities=[],
            quarantined_count=0,
            ocr_dropped=0,
        )
        assert sv is None

    def test_basic_shape_with_ten_blocks(self):
        blocks = [
            _make_block(
                claim_id=f"c-{i}",
                paragraph_id=f"p-{i % 4}",
                doc_title=f"Doc {i % 3}",
                year=1900 + i,
                claim_type=("OBSERVATION" if i % 2 == 0 else "EVENT"),
                extraction_confidence=(0.95 if i < 3 else 0.75 if i < 7 else 0.5),
                series=("Series A" if i < 6 else "Series B"),
                epistemic_status=("asserted" if i < 8 else "uncertain"),
            )
            for i in range(10)
        ]
        ctx = EntityContext(resolved=[_make_resolved_entity()])
        sv = _derive_structural_view(
            blocks=blocks,
            entity_context=ctx,
            top_co_entities=[],
            quarantined_count=0,
            ocr_dropped=0,
        )
        assert sv is not None
        assert sv.claim_count.value == 10
        assert sv.claim_count.template == "How many claims mention Turnbull?"
        assert sv.claim_count.tier1_op == "count_claims_for_entity"
        assert sv.document_count.value == 3
        assert sv.paragraph_count.value == 4
        assert sv.year_range.value == "1900–1909"
        # Confidence dist: 3 high (0.95), 4 medium (0.75), 3 low (0.5).
        assert sv.confidence_dist == {"high": 3, "medium": 4, "low": 3}
        # Year buckets all in the 1900 decade.
        assert sv.year_buckets == [{"decade": 1900, "n": 10}]
        # claim_types: 5 each.
        ct_names = [ct["name"] for ct in sv.top_claim_types]
        assert set(ct_names) == {"OBSERVATION", "EVENT"}
        # series: A (6) ranks before B (4).
        assert [s["name"] for s in sv.top_series] == ["Series A", "Series B"]
        # Epistemic counter.
        assert sv.epistemic_dist == {"asserted": 8, "uncertain": 2}

    def test_single_doc_single_year_degrades_gracefully(self):
        blocks = [
            _make_block(
                claim_id=f"c-{i}",
                paragraph_id=f"p-{i}",
                doc_title="Same Doc",
                year=1942,
            )
            for i in range(6)
        ]
        ctx = EntityContext(resolved=[_make_resolved_entity()])
        sv = _derive_structural_view(
            blocks=blocks,
            entity_context=ctx,
            top_co_entities=[],
            quarantined_count=0,
            ocr_dropped=0,
        )
        assert sv is not None
        assert sv.document_count.value == 1
        assert sv.year_range.value == "1942–1942"
        assert sv.year_buckets == [{"decade": 1940, "n": 6}]

    def test_no_resolved_entity_drops_anchor_templates(self):
        blocks = [_make_block(claim_id=f"c-{i}", paragraph_id=f"p-{i}") for i in range(5)]
        ctx = EntityContext(resolved=[])
        sv = _derive_structural_view(
            blocks=blocks,
            entity_context=ctx,
            top_co_entities=[],
            quarantined_count=0,
            ocr_dropped=0,
        )
        assert sv is not None
        # Anchor-bound cells: template is None, not the unfilled "{anchor}".
        assert sv.claim_count.template is None
        assert sv.document_count.template is None
        assert sv.year_range.template is None
        assert sv.claim_count.tier1_op is None

    def test_omits_empty_optional_cells(self):
        blocks = [
            _make_block(claim_id=f"c-{i}", paragraph_id=f"p-{i}", series=None)
            for i in range(5)
        ]
        ctx = EntityContext(resolved=[_make_resolved_entity()])
        sv = _derive_structural_view(
            blocks=blocks,
            entity_context=ctx,
            top_co_entities=[],
            quarantined_count=0,
            ocr_dropped=0,
        )
        assert sv is not None
        assert sv.top_series == []
        # Quarantined cell present with value=0; frontend gates the render.
        assert sv.quarantined is not None
        assert sv.quarantined.value == 0
        # value=0 → no template (clicking would be meaningless).
        assert sv.quarantined.template is None

    def test_top_co_entities_passthrough_with_templates_added(self):
        blocks = [_make_block(claim_id=f"c-{i}", paragraph_id=f"p-{i}") for i in range(5)]
        ctx = EntityContext(resolved=[_make_resolved_entity()])
        co = [
            {"name": "J.P. Graves", "entity_id": "ent-graves", "entity_type": "Person", "n": 4},
            {"name": "Spokane Press", "entity_id": "ent-press", "entity_type": "Organization", "n": 3},
        ]
        sv = _derive_structural_view(
            blocks=blocks,
            entity_context=ctx,
            top_co_entities=co,
            quarantined_count=0,
            ocr_dropped=0,
        )
        assert sv is not None
        assert len(sv.top_entities) == 2
        assert sv.top_entities[0]["name"] == "J.P. Graves"
        assert sv.top_entities[0]["template"] == "Who appears with J.P. Graves?"
        assert sv.top_entities[0]["tier1_op"] == "co_occurring_entities_for_entity"
        assert sv.top_entities[1]["template"] == "Who appears with Spokane Press?"

    def test_quarantined_and_ocr_dropped_passthrough(self):
        blocks = [_make_block(claim_id=f"c-{i}", paragraph_id=f"p-{i}") for i in range(5)]
        ctx = EntityContext(resolved=[_make_resolved_entity()])
        sv = _derive_structural_view(
            blocks=blocks,
            entity_context=ctx,
            top_co_entities=[],
            quarantined_count=3,
            ocr_dropped=7,
        )
        assert sv is not None
        assert sv.quarantined is not None
        assert sv.quarantined.value == 3
        assert sv.quarantined.template == "Show quarantined claims for this query"
        assert sv.ocr_dropped is not None
        assert sv.ocr_dropped.value == 7


class TestComputeTopCoOccurringEntities:
    def test_happy_path_returns_normalised_rows(self):
        executor = MagicMock()
        executor.run.return_value = [
            {"entity_id": "ent-graves", "name": "J.P. Graves", "entity_type": "Person", "n": 4},
            {"entity_id": "ent-press", "name": "Spokane Press", "entity_type": "Organization", "n": 3},
        ]
        out = _compute_top_co_occurring_entities(
            executor=executor,
            claim_ids=["c-1", "c-2"],
            exclude_entity_ids=frozenset(["ent-anchor"]),
            institution_ids=["turnbull"],
            permitted_levels=["public"],
            limit=5,
        )
        assert len(out) == 2
        assert out[0] == {
            "name": "J.P. Graves",
            "entity_id": "ent-graves",
            "entity_type": "Person",
            "n": 4,
        }
        # Verify Cypher params shape — caller relies on these keys.
        executor.run.assert_called_once()
        query, params = executor.run.call_args[0]
        assert "e.entity_type" not in query
        assert "e.canonical_name" not in query
        assert "e.surface_form" not in query
        assert "[label IN labels(e) WHERE label <> 'Entity'][0] AS entity_type" in query
        assert params["claim_ids"] == ["c-1", "c-2"]
        assert params["exclude_entity_ids"] == ["ent-anchor"]
        assert params["limit"] == 5

    def test_empty_claim_ids_short_circuits(self):
        executor = MagicMock()
        out = _compute_top_co_occurring_entities(
            executor=executor,
            claim_ids=[],
            exclude_entity_ids=frozenset(),
            institution_ids=None,
            permitted_levels=None,
        )
        assert out == []
        executor.run.assert_not_called()

    def test_executor_failure_returns_empty(self):
        executor = MagicMock()
        executor.run.side_effect = RuntimeError("neo4j unreachable")
        out = _compute_top_co_occurring_entities(
            executor=executor,
            claim_ids=["c-1"],
            exclude_entity_ids=frozenset(),
            institution_ids=None,
            permitted_levels=None,
        )
        assert out == []  # graceful degradation, never raises

    def test_drops_rows_with_missing_name_or_id(self):
        executor = MagicMock()
        executor.run.return_value = [
            {"entity_id": "ent-1", "name": "Valid", "entity_type": "Person", "n": 2},
            {"entity_id": "ent-2", "name": None, "entity_type": "Person", "n": 1},
            {"entity_id": None, "name": "No ID", "entity_type": "Person", "n": 1},
        ]
        out = _compute_top_co_occurring_entities(
            executor=executor,
            claim_ids=["c-1"],
            exclude_entity_ids=frozenset(),
            institution_ids=None,
            permitted_levels=None,
        )
        assert len(out) == 1
        assert out[0]["name"] == "Valid"
