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
    _infer_claim_types,
    _row_to_block,
    _select_retrieval_strategy,
    _serialise_block,
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
) -> dict:
    return {
        "c": {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "extraction_confidence": confidence,
            "certainty": epistemic,
            "source_sentence": sentence,
        },
        "d": {"title": doc_title, "date_start": "1942-01-01", "date_end": "1942-12-31"},
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
        assert "CLAIM [POPULATION_TREND" in text
        assert "epistemic=observed" in text
        assert "Mallard numbers increased." in text
        # Confidence no longer appears inline in the CLAIM bracket
        claim_bracket = text.split("CLAIM [")[1].split("]")[0]
        assert "confidence=" not in claim_bracket

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

    def test_temporal_route_when_year_and_no_entity_with_anchor(self):
        from gemynd.core.graph.cypher import TEMPORAL_CLAIMS_QUERY_WITH_REFUGE

        template, params = _select_retrieval_strategy(
            "duck counts", self._make_entity_ctx(0), year_min=1950, year_max=1960, budget=10,
            anchor_entity_id="refuge_abc",
        )
        assert template == TEMPORAL_CLAIMS_QUERY_WITH_REFUGE
        assert params["year_min"] == 1950
        assert params["year_max"] == 1960
        assert params["refuge_id"] == "refuge_abc"

    def test_temporal_route_when_year_and_no_entity_no_anchor(self):
        from gemynd.core.graph.cypher import TEMPORAL_CLAIMS_QUERY

        template, params = _select_retrieval_strategy(
            "duck counts", self._make_entity_ctx(0), year_min=1950, year_max=1960, budget=10,
        )
        assert template == TEMPORAL_CLAIMS_QUERY
        assert params["year_min"] == 1950

    def test_multi_entity_route_when_two_entities(self):
        from gemynd.core.graph.cypher import MULTI_ENTITY_CLAIMS_QUERY

        template, params = _select_retrieval_strategy(
            "compare mallard and teal", self._make_entity_ctx(2),
            year_min=None, year_max=None, budget=10
        )
        assert template == MULTI_ENTITY_CLAIMS_QUERY
        assert len(params["entity_ids"]) == 2

    def test_entity_plus_claim_type_route(self):
        from gemynd.core.graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY

        template, params = _select_retrieval_strategy(
            "habitat conditions for mallard", self._make_entity_ctx(1),
            year_min=None, year_max=None, budget=10
        )
        assert template == ENTITY_ANCHORED_CLAIMS_QUERY
        # Over-fetch limit (budget * 3) distinguishes this from plain entity route
        assert params["limit"] == 30

    def test_claim_type_only_route_when_no_entity(self):
        from gemynd.core.graph.cypher import CLAIM_TYPE_SCOPED_QUERY

        template, params = _select_retrieval_strategy(
            "describe all population estimates", self._make_entity_ctx(0),
            year_min=None, year_max=None, budget=10
        )
        assert template == CLAIM_TYPE_SCOPED_QUERY
        assert "population_estimate" in (params["claim_types"] or [])

    def test_single_entity_fallback_no_claim_type(self):
        from gemynd.core.graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY

        template, params = _select_retrieval_strategy(
            "tell me about the refuge", self._make_entity_ctx(1),
            year_min=None, year_max=None, budget=10
        )
        assert template == ENTITY_ANCHORED_CLAIMS_QUERY
        assert params["limit"] == 20  # budget * 2 (not the over-fetch * 3)

    def test_fulltext_fallback_when_nothing_matches(self):
        from gemynd.core.graph.cypher import FULLTEXT_CLAIMS_QUERY

        template, params = _select_retrieval_strategy(
            "tell me about the refuge", self._make_entity_ctx(0),
            year_min=None, year_max=None, budget=10
        )
        assert template == FULLTEXT_CLAIMS_QUERY
        assert params["search_text"] == "tell me about the refuge"


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
        template, params = _select_retrieval_strategy(
            "what wildlife observations at Turnbull Refuge",
            entity_ctx, year_min=None, year_max=None, budget=20,
        )
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
