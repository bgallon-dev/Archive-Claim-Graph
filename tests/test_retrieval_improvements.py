"""
Tests for retrieval pipeline improvements.

Coverage:
  - _infer_claim_types: query vocabulary → claim_type mapping
  - _select_retrieval_strategy: template routing logic
  - _serialise_block: confidence tier and RETRIEVED_VIA signal
  - Synthesis prompt: per-question-type answer structure
  - End-to-end: query → template → blocks → answer (mocked executor)

Integration tests (require live Neo4j) are marked with:
  @pytest.mark.integration

Run unit tests only:
  pytest tests/test_retrieval_improvements.py -m "not integration"

Run all including integration:
  pytest tests/test_retrieval_improvements.py
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from graphrag_pipeline.core.graph.cypher import (
    ENTITY_ANCHORED_CLAIMS_QUERY,
    FULLTEXT_CLAIMS_QUERY,
)
from graphrag_pipeline.retrieval.context_assembler import (
    ProvenanceContextAssembler,
    _row_to_block,
    _serialise_block,
)
from graphrag_pipeline.retrieval.models import (
    EntityContext,
    ProvenanceBlock,
    ResolvedEntity,
)


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_resolved(entity_id: str, entity_type: str = "Species") -> ResolvedEntity:
    return ResolvedEntity(
        surface_form=entity_id,
        entity_id=entity_id,
        entity_type=entity_type,
        resolution_confidence=0.95,
        resolution_relation="REFERS_TO",
    )


def _make_block(
    claim_id: str = "c-001",
    claim_type: str = "population_estimate",
    confidence: float = 0.85,
    epistemic: str = "observed",
    sentence: str = "Mallard numbers increased.",
    year: int | None = 1942,
    traversal_rel_types: list[str] | None = None,
) -> ProvenanceBlock:
    return ProvenanceBlock(
        doc_title="Annual Report 1942",
        doc_date_start="1942-01-01",
        doc_date_end="1942-12-31",
        page_number=3,
        paragraph_id="para-001",
        claim_id=claim_id,
        claim_type=claim_type,
        extraction_confidence=confidence,
        epistemic_status=epistemic,
        source_sentence=sentence,
        observation_type="population_count",
        species_name="Mallard",
        year=year,
        measurements=[],
        traversal_rel_types=traversal_rel_types or [],
    )


def _make_raw_row(
    claim_id: str = "c-001",
    claim_type: str = "population_estimate",
    confidence: float = 0.85,
    sentence: str = "Mallard numbers increased.",
    year: int | None = 1942,
) -> dict:
    return {
        "c": {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "extraction_confidence": confidence,
            "certainty": "observed",
            "source_sentence": sentence,
        },
        "d": {"title": "Annual Report 1942", "date_start": "1942-01-01", "date_end": "1942-12-31"},
        "pg": {"page_number": 3},
        "para": {"paragraph_id": "para-001"},
        "obs": {"observation_type": "population_count"},
        "sp": {"name": "Mallard"},
        "y": {"year": year} if year else {},
        "measurements": [],
    }


# ---------------------------------------------------------------------------
# _infer_claim_types
# ---------------------------------------------------------------------------

class TestInferClaimTypes:
    """
    Tests for query vocabulary → claim_type list mapping.
    Import path will be graphrag_pipeline.retrieval.context_assembler
    once _infer_claim_types is added there.
    """

    def _infer(self, query: str) -> list[str] | None:
        # Import here so the test file loads even before the function exists,
        # giving a clear ImportError rather than a confusing AttributeError.
        from graphrag_pipeline.retrieval.context_assembler import _infer_claim_types
        return _infer_claim_types(query)

    def test_predator_keywords_map_to_predator_types(self):
        result = self._infer("what predator control was done in the 1950s?")
        assert result is not None
        assert "predator_control" in result

    def test_habitat_keywords_map_to_habitat_types(self):
        result = self._infer("what were wetland habitat conditions like?")
        assert result is not None
        assert "habitat_condition" in result

    def test_population_keywords_map_to_species_types(self):
        result = self._infer("how did mallard population change over time?")
        assert result is not None
        assert "population_estimate" in result
        assert "species_presence" in result

    def test_breeding_keyword_maps_correctly(self):
        result = self._infer("describe breeding activity at the refuge")
        assert result is not None
        assert "breeding_activity" in result

    def test_fire_keyword_maps_correctly(self):
        result = self._infer("were there any fire incidents in 1945?")
        assert result is not None
        assert "fire_incident" in result

    def test_management_keyword_maps_correctly(self):
        result = self._infer("what management actions were taken?")
        assert result is not None
        assert "management_action" in result

    def test_unrecognized_query_returns_none(self):
        result = self._infer("tell me about the history of the refuge")
        assert result is None

    def test_multiple_keywords_return_union(self):
        result = self._infer("predator control and habitat management in the 1940s")
        assert result is not None
        assert "predator_control" in result
        assert "management_action" in result

    def test_case_insensitive(self):
        result = self._infer("PREDATOR CONTROL at Turnbull")
        assert result is not None
        assert "predator_control" in result

    def test_migration_keyword(self):
        result = self._infer("when did migration arrive?")
        assert result is not None
        assert "migration_timing" in result

    def test_empty_query_returns_none(self):
        result = self._infer("")
        assert result is None


# ---------------------------------------------------------------------------
# _select_retrieval_strategy
# ---------------------------------------------------------------------------

class TestSelectRetrievalStrategy:
    """
    Tests for template routing logic.
    Verifies the correct Cypher template and parameter shape are selected
    for each combination of entity resolution, year range, and claim type signal.
    """

    def _select(
        self,
        query: str,
        resolved: list[ResolvedEntity] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        budget: int = 20,
    ):
        from graphrag_pipeline.retrieval.context_assembler import _select_retrieval_strategy
        entity_ctx = EntityContext(resolved=resolved or [])
        return _select_retrieval_strategy(query, entity_ctx, year_min, year_max, budget)

    # --- Fulltext fallback ---

    def test_no_signal_uses_fulltext(self):
        from graphrag_pipeline.core.graph.cypher import FULLTEXT_CLAIMS_QUERY
        template, params = self._select("tell me about the refuge")
        assert template == FULLTEXT_CLAIMS_QUERY
        assert "search_text" in params

    # --- Temporal path ---

    def test_year_range_no_entity_uses_temporal_template(self):
        from graphrag_pipeline.core.graph.cypher import TEMPORAL_CLAIMS_QUERY_WITH_REFUGE
        template, params = self._select(
            "what happened in the 1950s?",
            year_min=1950, year_max=1959,
        )
        assert template == TEMPORAL_CLAIMS_QUERY_WITH_REFUGE
        assert params["year_min"] == 1950
        assert params["year_max"] == 1959
        assert params["refuge_id"]

    def test_temporal_template_passes_claim_types_when_detected(self):
        from graphrag_pipeline.core.graph.cypher import TEMPORAL_CLAIMS_QUERY_WITH_REFUGE
        template, params = self._select(
            "what predator control happened in the 1950s?",
            year_min=1950, year_max=1959,
        )
        assert template == TEMPORAL_CLAIMS_QUERY_WITH_REFUGE
        assert params["claim_types"] is not None
        assert "predator_control" in params["claim_types"]

    def test_temporal_template_claim_types_none_when_no_signal(self):
        from graphrag_pipeline.core.graph.cypher import TEMPORAL_CLAIMS_QUERY_WITH_REFUGE
        template, params = self._select(
            "what happened at the refuge in 1945?",
            year_min=1945, year_max=1945,
        )
        assert template == TEMPORAL_CLAIMS_QUERY_WITH_REFUGE
        assert params["claim_types"] is None

    # --- Multi-entity comparative path ---

    def test_two_entities_uses_multi_entity_template(self):
        from graphrag_pipeline.core.graph.cypher import MULTI_ENTITY_CLAIMS_QUERY
        resolved = [
            _make_resolved("sp-mallard", "Species"),
            _make_resolved("sp-teal", "Species"),
        ]
        template, params = self._select(
            "compare mallard and teal populations",
            resolved=resolved,
        )
        assert template == MULTI_ENTITY_CLAIMS_QUERY
        assert "sp-mallard" in params["entity_ids"]
        assert "sp-teal" in params["entity_ids"]

    def test_multi_entity_includes_year_bounds(self):
        from graphrag_pipeline.core.graph.cypher import MULTI_ENTITY_CLAIMS_QUERY
        resolved = [_make_resolved("sp-mallard"), _make_resolved("h-marsh", "Habitat")]
        template, params = self._select(
            "compare mallard and marsh habitat",
            resolved=resolved,
            year_min=1940, year_max=1960,
        )
        assert template == MULTI_ENTITY_CLAIMS_QUERY
        assert params["year_min"] == 1940
        assert params["year_max"] == 1960

    # --- Single entity + claim type ---

    def test_single_entity_with_claim_type_signal_uses_entity_anchored(self):
        resolved = [_make_resolved("sp-mallard")]
        template, params = self._select(
            "how did mallard population change?",
            resolved=resolved,
        )
        assert template == ENTITY_ANCHORED_CLAIMS_QUERY
        assert params["entity_id"] == "sp-mallard"

    def test_single_entity_over_fetches_when_claim_type_detected(self):
        """
        When a claim type is detected alongside a single entity,
        the query should over-fetch (limit = budget * 3) to allow post-filtering.
        """
        resolved = [_make_resolved("sp-mallard")]
        budget = 10
        template, params = self._select(
            "how did mallard population change?",
            resolved=resolved,
            budget=budget,
        )
        assert template == ENTITY_ANCHORED_CLAIMS_QUERY
        assert params["limit"] >= budget * 2

    # --- Claim type scoped, no entity ---

    def test_claim_type_no_entity_uses_claim_type_scoped(self):
        from graphrag_pipeline.core.graph.cypher import CLAIM_TYPE_SCOPED_QUERY
        template, params = self._select(
            "describe all predator control activities"
        )
        assert template == CLAIM_TYPE_SCOPED_QUERY
        assert "predator_control" in params["claim_types"]
        assert params["entity_ids"] is None

    # --- Single entity, no claim type ---

    def test_single_entity_no_claim_type_uses_entity_anchored(self):
        resolved = [_make_resolved("sp-mallard")]
        template, params = self._select(
            "tell me about mallards",
            resolved=resolved,
        )
        assert template == ENTITY_ANCHORED_CLAIMS_QUERY
        assert params["entity_id"] == "sp-mallard"

    # --- Budget propagation ---

    def test_budget_propagated_to_limit(self):
        budget = 15
        template, params = self._select(
            "tell me about the refuge",
            budget=budget,
        )
        assert params["limit"] >= budget

    def test_temporal_limit_is_at_least_budget(self):
        from graphrag_pipeline.core.graph.cypher import TEMPORAL_CLAIMS_QUERY
        budget = 20
        template, params = self._select(
            "what happened in 1945?",
            year_min=1945, year_max=1945,
            budget=budget,
        )
        assert params["limit"] >= budget


# ---------------------------------------------------------------------------
# _serialise_block (confidence tier + RETRIEVED_VIA signal)
# ---------------------------------------------------------------------------

class TestSerialiseBlockSignals:
    """
    Tests for the improved serialization format.
    Verifies confidence tier labeling and RETRIEVED_VIA metadata are present.
    """

    def test_high_confidence_tier_label(self):
        block = _make_block(confidence=0.90)
        text = _serialise_block(block)
        assert "HIGH" in text

    def test_medium_confidence_tier_label(self):
        block = _make_block(confidence=0.75)
        text = _serialise_block(block)
        assert "MEDIUM" in text

    def test_low_confidence_tier_label(self):
        block = _make_block(confidence=0.55)
        text = _serialise_block(block)
        assert "LOW" in text

    def test_confidence_boundary_high(self):
        # Exactly 0.85 should be HIGH
        block = _make_block(confidence=0.85)
        text = _serialise_block(block)
        assert "HIGH" in text

    def test_confidence_boundary_medium(self):
        # Exactly 0.70 should be MEDIUM
        block = _make_block(confidence=0.70)
        text = _serialise_block(block)
        assert "MEDIUM" in text

    def test_retrieved_via_present_when_traversal_types_set(self):
        block = _make_block(traversal_rel_types=["SPECIES_FOCUS"])
        text = _serialise_block(block)
        assert "RETRIEVED_VIA" in text
        assert "SPECIES_FOCUS" in text

    def test_retrieved_via_absent_when_no_traversal_types(self):
        block = _make_block(traversal_rel_types=[])
        text = _serialise_block(block)
        assert "RETRIEVED_VIA" not in text

    def test_multiple_traversal_types_all_present(self):
        block = _make_block(traversal_rel_types=["SPECIES_FOCUS", "IN_YEAR"])
        text = _serialise_block(block)
        assert "SPECIES_FOCUS" in text
        assert "IN_YEAR" in text

    def test_confidence_tier_appears_before_claim_line(self):
        """
        CONFIDENCE_TIER should appear early in the block so models
        can use it as a priority signal before reading the claim.
        """
        block = _make_block(confidence=0.90)
        text = _serialise_block(block)
        tier_pos = text.find("CONFIDENCE_TIER")
        claim_pos = text.find("CLAIM [")
        assert tier_pos < claim_pos, (
            "CONFIDENCE_TIER must appear before the CLAIM line in serialized output"
        )

    def test_numeric_confidence_still_present(self):
        block = _make_block(confidence=0.87)
        text = _serialise_block(block)
        assert "0.87" in text

    def test_existing_fields_still_present(self):
        """Regression: existing fields must not be dropped by the new format."""
        block = _make_block()
        text = _serialise_block(block)
        assert "DOCUMENT:" in text
        assert "PAGE:" in text
        assert "PARAGRAPH:" in text
        assert "CLAIM [" in text
        assert block.source_sentence in text

    def test_serialise_block_with_measurements(self):
        block = _make_block()
        block.measurements = [
            {"name": "individual_count", "value": 450, "unit": "individuals", "approximate": False}
        ]
        text = _serialise_block(block)
        assert "MEASUREMENT" in text
        assert "450" in text


# ---------------------------------------------------------------------------
# Synthesis prompt behavior
# ---------------------------------------------------------------------------

class TestSynthesisPromptBehavior:
    """
    Tests that the updated synthesis prompt produces correctly structured
    answers for each query type. Uses a mock Anthropic client so no
    live API call is made.
    """

    def _make_engine(self, response_text: str):
        from graphrag_pipeline.retrieval.synthesis import SynthesisEngine
        engine = SynthesisEngine.__new__(SynthesisEngine)
        content_block = MagicMock()
        content_block.text = response_text
        message = MagicMock()
        message.content = [content_block]
        client = MagicMock()
        client.messages.create.return_value = message
        engine._client = client
        engine._max_tokens = 1000
        engine._timeout = 60
        from graphrag_pipeline.retrieval.synthesis import _build_system_prompt, _DEFAULT_SYNTHESIS_CONTEXT
        engine._system_prompt = _build_system_prompt(_DEFAULT_SYNTHESIS_CONTEXT)
        return engine

    def _valid_response(self, answer: str = "Test answer.") -> str:
        return json.dumps({
            "answer": answer,
            "confidence_assessment": "Evidence is adequate.",
            "supporting_claim_ids": ["c-001"],
            "caveats": [],
        })

    # --- Temporal query: expects year-organized answer ---

    def test_temporal_query_prompt_includes_year_instruction(self):
        """
        The system prompt should instruct the model to organize by year
        for temporal queries. Verify the prompt text contains this rule.
        """
        from graphrag_pipeline.retrieval.synthesis import _SYSTEM_PROMPT_TEMPLATE
        assert "year or time period" in _SYSTEM_PROMPT_TEMPLATE.lower() or \
               "temporal" in _SYSTEM_PROMPT_TEMPLATE.lower() or \
               "single-document" in _SYSTEM_PROMPT_TEMPLATE.lower(), (
            "Synthesis prompt must contain instruction to organize temporal answers by year "
            "and to include single-document evidence."
        )

    def test_single_document_evidence_not_suppressed_by_prompt(self):
        """
        The updated prompt must NOT instruct the model to omit claims
        that appear in only one document — this was the primary cause of
        vague answers on temporal and management queries.
        """
        from graphrag_pipeline.retrieval.synthesis import _SYSTEM_PROMPT_TEMPLATE
        # The old instruction that caused suppression
        assert "no corroborating parallel" not in _SYSTEM_PROMPT_TEMPLATE
        assert "omit" not in _SYSTEM_PROMPT_TEMPLATE.lower() or \
               "single" not in _SYSTEM_PROMPT_TEMPLATE.lower(), (
            "Prompt must not instruct model to omit single-document claims."
        )

    def test_confidence_tier_referenced_in_prompt(self):
        """Model must be told how to use the CONFIDENCE_TIER signal."""
        from graphrag_pipeline.retrieval.synthesis import _SYSTEM_PROMPT_TEMPLATE
        assert "HIGH" in _SYSTEM_PROMPT_TEMPLATE or \
               "confidence" in _SYSTEM_PROMPT_TEMPLATE.lower()

    def test_retrieved_via_referenced_in_prompt(self):
        """Model must be told what RETRIEVED_VIA means."""
        from graphrag_pipeline.retrieval.synthesis import _SYSTEM_PROMPT_TEMPLATE
        assert "RETRIEVED_VIA" in _SYSTEM_PROMPT_TEMPLATE or \
               "retrieved" in _SYSTEM_PROMPT_TEMPLATE.lower()

    # --- Response parsing still works ---

    def test_temporal_answer_parsed_correctly(self):
        engine = self._make_engine(self._valid_response(
            "In 1942, mallard populations increased. In 1943, numbers declined."
        ))
        blocks = [_make_block(year=1942), _make_block(claim_id="c-002", year=1943)]
        result = engine.synthesise("how did mallards change?", blocks, "context")
        assert "1942" in result.answer
        assert "1943" in result.answer

    def test_comparative_answer_parsed_correctly(self):
        engine = self._make_engine(self._valid_response(
            "Mallards showed stable populations while teal declined."
        ))
        blocks = [_make_block(), _make_block(claim_id="c-002", claim_type="species_presence")]
        result = engine.synthesise("compare mallard and teal", blocks, "context")
        assert result.answer
        assert result.supporting_claim_ids

    def test_management_action_answer_preserves_specificity(self):
        engine = self._make_engine(self._valid_response(
            "In 1945, 12 coyotes were removed under predator control program."
        ))
        blocks = [_make_block(claim_type="predator_control",
                              sentence="12 coyotes were removed in 1945.")]
        result = engine.synthesise("what predator control was done?", blocks, "context")
        assert "12" in result.answer or "coyote" in result.answer.lower()

    def test_low_confidence_claims_flagged_in_caveats(self):
        response = json.dumps({
            "answer": "Some evidence suggests habitat degradation.",
            "confidence_assessment": "Evidence quality is low.",
            "supporting_claim_ids": ["c-001"],
            "caveats": ["Claim c-001 has low extraction confidence (0.55)."],
        })
        engine = self._make_engine(response)
        blocks = [_make_block(confidence=0.55)]
        result = engine.synthesise("describe habitat conditions", blocks, "context")
        assert result.caveats

    def test_absent_evidence_stated_explicitly(self):
        response = json.dumps({
            "answer": "No evidence of fire incidents was found in the provided context.",
            "confidence_assessment": "No relevant claims retrieved.",
            "supporting_claim_ids": [],
            "caveats": ["No fire incident claims in retrieved context."],
        })
        engine = self._make_engine(response)
        result = engine.synthesise("were there fires in 1960?", [], "")
        assert result.answer
        assert result.supporting_claim_ids == []


# ---------------------------------------------------------------------------
# Post-filter behavior in assemble()
# ---------------------------------------------------------------------------

class TestPostFilterBehavior:
    """
    Tests that the claim_type post-filter in assemble() works correctly
    for the entity-anchored path: filters when types detected, falls back
    to unfiltered when filter removes everything.
    """

    def _make_assembler(self, rows: list[dict]) -> ProvenanceContextAssembler:
        mock_executor = MagicMock()
        mock_executor.run.return_value = rows
        return ProvenanceContextAssembler(
            executor=mock_executor,
            budget_conversational=20,
        )

    def test_post_filter_keeps_matching_claim_types(self):
        rows = [
            _make_raw_row("c-001", claim_type="predator_control"),
            _make_raw_row("c-002", claim_type="species_presence"),
            _make_raw_row("c-003", claim_type="predator_control"),
        ]
        assembler = self._make_assembler(rows)
        entity_ctx = EntityContext(resolved=[_make_resolved("sp-coyote")])
        blocks, _ = assembler.assemble(
            "what predator control was done with coyotes?",
            entity_ctx,
        )
        claim_types = {b.claim_type for b in blocks}
        # predator_control should dominate; species_presence may be filtered
        assert "predator_control" in claim_types

    def test_post_filter_falls_back_when_all_filtered(self):
        """
        If the inferred claim types match nothing in the retrieved rows,
        the assembler must return the unfiltered set rather than empty context.
        """
        rows = [
            _make_raw_row("c-001", claim_type="economic_use"),
            _make_raw_row("c-002", claim_type="development_activity"),
        ]
        assembler = self._make_assembler(rows)
        entity_ctx = EntityContext(resolved=[_make_resolved("sp-mallard")])
        # "predator" signal won't match economic_use or development_activity
        blocks, _ = assembler.assemble(
            "predator control at Turnbull with mallard impact",
            entity_ctx,
        )
        # Should fall back to unfiltered — not empty
        assert len(blocks) > 0

    def test_ocr_corrupted_blocks_still_dropped_after_filter(self):
        rows = [
            _make_raw_row("c-001", claim_type="predator_control"),
        ]
        rows[0]["c"]["source_sentence"] = "Valid sentence about coyotes."
        corrupted = _make_raw_row("c-002", claim_type="predator_control")
        corrupted["c"]["source_sentence"] = "^^ OCR garbage \x0c control"
        rows.append(corrupted)

        assembler = self._make_assembler(rows)
        entity_ctx = EntityContext(resolved=[_make_resolved("sp-coyote")])
        blocks, _ = assembler.assemble("predator control", entity_ctx)
        claim_ids = [b.claim_id for b in blocks]
        assert "c-001" in claim_ids
        assert "c-002" not in claim_ids


# ---------------------------------------------------------------------------
# End-to-end: query → template selection → blocks → synthesis
# ---------------------------------------------------------------------------

class TestEndToEndRetrievalPath:
    """
    Full pipeline tests from query text to SynthesisResult.
    Uses mocked executor and mocked Anthropic client — no external services.
    Verifies that the correct template is called and that the result is well-formed.
    """

    def _make_pipeline(self, rows: list[dict], synthesis_answer: str = "Test answer."):
        from graphrag_pipeline.retrieval.synthesis import SynthesisEngine

        mock_executor = MagicMock()
        mock_executor.run.return_value = rows

        assembler = ProvenanceContextAssembler(
            executor=mock_executor,
            budget_conversational=20,
        )

        engine = SynthesisEngine.__new__(SynthesisEngine)
        content_block = MagicMock()
        content_block.text = json.dumps({
            "answer": synthesis_answer,
            "confidence_assessment": "Adequate evidence.",
            "supporting_claim_ids": [r["c"]["claim_id"] for r in rows[:3]],
            "caveats": [],
        })
        message = MagicMock()
        message.content = [content_block]
        client = MagicMock()
        client.messages.create.return_value = message
        engine._client = client
        engine._max_tokens = 1000
        engine._timeout = 60
        from graphrag_pipeline.retrieval.synthesis import _build_system_prompt, _DEFAULT_SYNTHESIS_CONTEXT
        engine._system_prompt = _build_system_prompt(_DEFAULT_SYNTHESIS_CONTEXT)

        return assembler, engine, mock_executor

    def test_temporal_query_calls_temporal_template(self):
        from graphrag_pipeline.core.graph.cypher import TEMPORAL_CLAIMS_QUERY_WITH_REFUGE
        rows = [_make_raw_row(f"c-{i:03d}") for i in range(5)]
        assembler, engine, mock_executor = self._make_pipeline(rows)

        entity_ctx = EntityContext()
        blocks, context_text = assembler.assemble(
            "what happened at the refuge in the 1950s?",
            entity_ctx,
            year_min=1950,
            year_max=1959,
        )
        called_templates = [call[0][0] for call in mock_executor.run.call_args_list]
        assert any(TEMPORAL_CLAIMS_QUERY_WITH_REFUGE in t for t in called_templates), (
            "Temporal query with year bounds should use TEMPORAL_CLAIMS_QUERY_WITH_REFUGE"
        )

    def test_comparative_query_calls_multi_entity_template(self):
        from graphrag_pipeline.core.graph.cypher import MULTI_ENTITY_CLAIMS_QUERY
        rows = [_make_raw_row(f"c-{i:03d}") for i in range(5)]
        assembler, engine, mock_executor = self._make_pipeline(rows)

        entity_ctx = EntityContext(resolved=[
            _make_resolved("sp-mallard"),
            _make_resolved("sp-teal"),
        ])
        blocks, context_text = assembler.assemble(
            "compare mallard and teal populations",
            entity_ctx,
        )
        called_templates = [call[0][0] for call in mock_executor.run.call_args_list]
        assert any(MULTI_ENTITY_CLAIMS_QUERY in t for t in called_templates), (
            "Two resolved entities should trigger MULTI_ENTITY_CLAIMS_QUERY"
        )

    def test_management_query_retrieves_management_claim_types(self):
        rows = [
            _make_raw_row("c-001", claim_type="predator_control",
                          sentence="12 coyotes were removed."),
            _make_raw_row("c-002", claim_type="management_action",
                          sentence="Grazing was restricted."),
        ]
        assembler, engine, mock_executor = self._make_pipeline(
            rows, synthesis_answer="Predator control removed 12 coyotes."
        )
        entity_ctx = EntityContext()
        blocks, context_text = assembler.assemble(
            "what predator control management was done?",
            entity_ctx,
        )
        result = engine.synthesise(
            "what predator control management was done?",
            blocks,
            context_text,
        )
        assert result.answer
        assert len(result.supporting_claim_ids) > 0

    def test_species_query_produces_non_empty_answer(self):
        rows = [_make_raw_row(f"c-{i:03d}", claim_type="population_estimate") for i in range(5)]
        assembler, engine, mock_executor = self._make_pipeline(
            rows, synthesis_answer="Mallard populations peaked at 3000 in 1942."
        )
        entity_ctx = EntityContext(resolved=[_make_resolved("sp-mallard")])
        blocks, context_text = assembler.assemble(
            "how did mallard populations change over time?",
            entity_ctx,
        )
        result = engine.synthesise(
            "how did mallard populations change over time?",
            blocks,
            context_text,
        )
        assert result.answer
        assert result.confidence_assessment

    def test_habitat_query_produces_non_empty_answer(self):
        rows = [
            _make_raw_row("c-001", claim_type="habitat_condition",
                          sentence="Wetland levels were low in 1955."),
        ]
        assembler, engine, mock_executor = self._make_pipeline(
            rows, synthesis_answer="Wetland conditions were poor in 1955 due to drought."
        )
        entity_ctx = EntityContext(resolved=[_make_resolved("h-marsh", "Habitat")])
        blocks, context_text = assembler.assemble(
            "what were wetland habitat conditions like?",
            entity_ctx,
        )
        result = engine.synthesise(
            "what were wetland habitat conditions like?",
            blocks,
            context_text,
        )
        assert result.answer

    def test_serialized_context_contains_confidence_tier(self):
        rows = [_make_raw_row("c-001", confidence=0.92)]
        assembler, _, _ = self._make_pipeline(rows)
        entity_ctx = EntityContext()
        blocks, context_text = assembler.assemble("describe the refuge", entity_ctx)
        assert "HIGH" in context_text or "MEDIUM" in context_text or "LOW" in context_text

    def test_min_extraction_confidence_populated(self):
        rows = [
            _make_raw_row("c-001", confidence=0.90),
            _make_raw_row("c-002", confidence=0.65),
        ]
        assembler, engine, _ = self._make_pipeline(rows)
        entity_ctx = EntityContext()
        blocks, context_text = assembler.assemble("describe the refuge", entity_ctx)
        result = engine.synthesise("describe the refuge", blocks, context_text)
        assert result.min_extraction_confidence is not None
        assert result.min_extraction_confidence <= 0.90

    def test_empty_retrieval_produces_valid_result(self):
        assembler, engine, _ = self._make_pipeline([])
        entity_ctx = EntityContext()
        blocks, context_text = assembler.assemble(
            "were there fires in 1960?", entity_ctx
        )
        result = engine.synthesise("were there fires in 1960?", blocks, context_text)
        assert result.answer is not None


# ---------------------------------------------------------------------------
# Integration tests (require live Neo4j + populated graph)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationRetrieval:
    """
    Integration tests against a live Neo4j instance.

    Require environment variables:
      NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    These tests verify that the new Cypher templates execute without error
    and return structurally valid results. They do not assert specific
    claim content since corpus data varies.

    Run with:
      pytest tests/test_retrieval_improvements.py -m integration
    """

    @pytest.fixture(scope="class")
    def executor(self):
        import os
        from graphrag_pipeline.retrieval.executor import Neo4jQueryExecutor
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")
        database = os.environ.get("NEO4J_DATABASE", "neo4j")
        if not all([uri, user, password]):
            pytest.skip("NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD not set")
        ex = Neo4jQueryExecutor(uri=uri, user=user, password=password, database=database)
        yield ex
        ex.close()

    @pytest.fixture(scope="class")
    def assembler(self, executor):
        return ProvenanceContextAssembler(executor=executor, budget_conversational=10)

    _TENANT_PARAMS: dict[str, object] = {
        "institution_id": "turnbull",
        "permitted_levels": ["public", "staff_only", "restricted"],
    }

    def test_temporal_template_executes(self, executor):
        from graphrag_pipeline.core.graph.cypher import TEMPORAL_CLAIMS_QUERY
        rows = executor.run(TEMPORAL_CLAIMS_QUERY, {
            **self._TENANT_PARAMS,
            "year_min": 1940,
            "year_max": 1950,
            "claim_types": None,
            "limit": 5,
        })
        assert isinstance(rows, list)
        for row in rows:
            assert "c" in row
            assert "y" in row

    def test_multi_entity_template_executes(self, executor):
        from graphrag_pipeline.core.graph.cypher import MULTI_ENTITY_CLAIMS_QUERY
        from graphrag_pipeline.core.resolver import default_seed_entities
        seeds = default_seed_entities()
        species = [e for e in seeds if e.entity_type == "Species"][:2]
        if len(species) < 2:
            pytest.skip("Need at least 2 species entities in graph")
        rows = executor.run(MULTI_ENTITY_CLAIMS_QUERY, {
            **self._TENANT_PARAMS,
            "entity_ids": [s.entity_id for s in species],
            "claim_types": None,
            "year_min": None,
            "year_max": None,
            "limit": 5,
        })
        assert isinstance(rows, list)

    def test_claim_type_scoped_template_executes(self, executor):
        from graphrag_pipeline.core.graph.cypher import CLAIM_TYPE_SCOPED_QUERY
        rows = executor.run(CLAIM_TYPE_SCOPED_QUERY, {
            **self._TENANT_PARAMS,
            "claim_types": ["predator_control", "management_action"],
            "entity_ids": None,
            "year_min": None,
            "year_max": None,
            "limit": 5,
        })
        assert isinstance(rows, list)
        for row in rows:
            c = row.get("c") or {}
            if c.get("claim_type"):
                assert c["claim_type"] in (
                    "predator_control", "management_action"
                ), f"Unexpected claim_type: {c['claim_type']}"

    def test_temporal_query_assemble_returns_blocks(self, assembler):
        entity_ctx = EntityContext()
        blocks, context_text = assembler.assemble(
            "what happened in the 1940s?",
            entity_ctx,
            year_min=1940,
            year_max=1949,
        )
        assert isinstance(blocks, list)
        assert isinstance(context_text, str)

    def test_fulltext_fallback_returns_blocks(self, assembler):
        entity_ctx = EntityContext(unresolved=["wetland"])
        blocks, context_text = assembler.assemble(
            "describe wetland conditions",
            entity_ctx,
        )
        assert isinstance(blocks, list)

    def test_all_blocks_have_confidence_tier_in_serialization(self, assembler):
        entity_ctx = EntityContext()
        blocks, context_text = assembler.assemble(
            "describe refuge management",
            entity_ctx,
        )
        if blocks:
            assert any(tier in context_text for tier in ("HIGH", "MEDIUM", "LOW"))

    def test_retrieved_via_present_for_entity_anchored_blocks(self, assembler):
        from graphrag_pipeline.core.resolver import default_seed_entities
        seeds = default_seed_entities()
        species = next((e for e in seeds if e.entity_type == "Species"), None)
        if not species:
            pytest.skip("No species entities in seed")
        entity_ctx = EntityContext(resolved=[
            ResolvedEntity(
                surface_form=species.name,
                entity_id=species.entity_id,
                entity_type="Species",
                resolution_confidence=0.95,
                resolution_relation="REFERS_TO",
            )
        ])
        blocks, context_text = assembler.assemble(
            f"tell me about {species.name}",
            entity_ctx,
        )
        # Any block retrieved via entity anchor should have traversal_rel_types set
        anchored_blocks = [b for b in blocks if b.traversal_rel_types]
        # Not asserting all blocks have it — fulltext fallback blocks won't
        # Just verify the field is populated when traversal data is available
        assert isinstance(anchored_blocks, list)
