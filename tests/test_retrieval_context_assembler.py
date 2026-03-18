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

from graphrag_pipeline.retrieval.context_assembler import (
    ProvenanceContextAssembler,
    _row_to_block,
    _serialise_block,
)
from graphrag_pipeline.retrieval.models import EntityContext, ProvenanceBlock, ResolvedEntity


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
        assert "PAGE: 3" in text
        assert "PARAGRAPH: para-001" in text
        assert "CLAIM [POPULATION_TREND" in text
        assert "confidence=0.85" in text
        assert "epistemic=observed" in text
        assert "Mallard numbers increased." in text

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
        from graphrag_pipeline.graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY

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
        from graphrag_pipeline.graph.cypher import FULLTEXT_CLAIMS_QUERY

        mock_executor = MagicMock()
        mock_executor.run.return_value = []

        assembler = ProvenanceContextAssembler(executor=mock_executor)
        entity_ctx = EntityContext(unresolved=["unknownbird"])
        assembler.assemble("query about unknownbird", entity_ctx)
        called_queries = [call[0][0] for call in mock_executor.run.call_args_list]
        assert any(FULLTEXT_CLAIMS_QUERY in q for q in called_queries)
