"""Unit tests for Layer 2A: Cypher query builder.

These tests verify that:
  1. Query template strings contain the expected parameter placeholders.
  2. The query builder calls the executor with correct parameter dicts.
  3. AnalyticalResult objects are structured correctly.

No live Neo4j connection is required — the executor is mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag_pipeline.graph.cypher import (
    HABITAT_CONDITION_QUERY,
    PROVENANCE_CHAIN_QUERY,
    SPECIES_TREND_QUERY,
)
from graphrag_pipeline.retrieval.query_builder import CypherQueryBuilder


# ---------------------------------------------------------------------------
# Template string checks
# ---------------------------------------------------------------------------

class TestQueryTemplates:
    def test_species_trend_has_species_id_param(self):
        assert "$species_id" in SPECIES_TREND_QUERY

    def test_species_trend_has_year_params(self):
        assert "$year_min" in SPECIES_TREND_QUERY
        assert "$year_max" in SPECIES_TREND_QUERY

    def test_habitat_condition_has_habitat_id_param(self):
        assert "$habitat_id" in HABITAT_CONDITION_QUERY

    def test_provenance_chain_has_claim_id_param(self):
        assert "$claim_id" in PROVENANCE_CHAIN_QUERY

    def test_templates_reference_latest_run(self):
        for template in (SPECIES_TREND_QUERY, HABITAT_CONDITION_QUERY, PROVENANCE_CHAIN_QUERY):
            assert "ExtractionRun" in template
            assert "max(r.run_timestamp)" in template


# ---------------------------------------------------------------------------
# Query builder behaviour (mocked executor)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_executor():
    executor = MagicMock()
    executor.run.return_value = []
    return executor


@pytest.fixture()
def builder(mock_executor):
    return CypherQueryBuilder(executor=mock_executor)


class TestSpeciesTrend:
    def test_calls_executor_with_correct_params(self, builder, mock_executor):
        builder.species_trend("sp-mallard", year_min=1940, year_max=1950)
        mock_executor.run.assert_called_once_with(
            SPECIES_TREND_QUERY,
            {"species_id": "sp-mallard", "year_min": 1940, "year_max": 1950},
        )

    def test_returns_analytical_result(self, builder):
        result = builder.species_trend("sp-mallard")
        assert result.query_name == "species_trend"
        assert "year" in result.columns
        assert "observation_count" in result.columns
        assert isinstance(result.rows, list)

    def test_none_year_bounds_passed_through(self, builder, mock_executor):
        builder.species_trend("sp-mallard")
        _, kwargs = mock_executor.run.call_args
        params = mock_executor.run.call_args[0][1]
        assert params["year_min"] is None
        assert params["year_max"] is None


class TestHabitatConditions:
    def test_calls_executor_with_correct_params(self, builder, mock_executor):
        builder.habitat_conditions("h-pothole", year_min=1945, year_max=1960)
        mock_executor.run.assert_called_once_with(
            HABITAT_CONDITION_QUERY,
            {"habitat_id": "h-pothole", "year_min": 1945, "year_max": 1960},
        )

    def test_returns_analytical_result(self, builder):
        result = builder.habitat_conditions("h-pothole")
        assert result.query_name == "habitat_conditions"
        assert "habitat" in result.columns


class TestProvenanceChain:
    def test_calls_executor_with_claim_id(self, builder, mock_executor):
        builder.provenance_chain("claim-abc")
        mock_executor.run.assert_called_once_with(
            PROVENANCE_CHAIN_QUERY,
            {"claim_id": "claim-abc"},
        )

    def test_returns_list(self, builder):
        result = builder.provenance_chain("claim-abc")
        assert isinstance(result, list)


class TestAnalyticalResultToSummary:
    def test_to_summary_text_no_rows(self):
        from graphrag_pipeline.retrieval.models import AnalyticalResult
        result = AnalyticalResult(query_name="species_trend", columns=["year", "count"], rows=[])
        assert "no results" in result.to_summary_text()

    def test_to_summary_text_with_rows(self):
        from graphrag_pipeline.retrieval.models import AnalyticalResult
        result = AnalyticalResult(
            query_name="species_trend",
            columns=["year", "count"],
            rows=[{"year": 1942, "count": 5}],
        )
        text = result.to_summary_text()
        assert "1942" in text
        assert "5" in text
