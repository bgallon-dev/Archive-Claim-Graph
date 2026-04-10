"""Unit tests for Layer 3: synthesis engine.

Uses a deterministic mock Anthropic client so no live API call is made.
Tests verify:
  • JSON response is parsed into SynthesisResult correctly.
  • min_extraction_confidence is computed from supporting claims.
  • Markdown fence stripping works.
  • ValueError is raised for non-JSON model output.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gemynd.retrieval.models import AnalyticalResult, ProvenanceBlock, SynthesisResult


def _make_block(claim_id: str, confidence: float) -> ProvenanceBlock:
    return ProvenanceBlock(
        doc_title="Report",
        doc_date_start=None,
        doc_date_end=None,
        page_number=1,
        paragraph_id="para-001",
        claim_id=claim_id,
        claim_type="POPULATION_TREND",
        extraction_confidence=confidence,
        epistemic_status="observed",
        source_sentence="Mallards increased.",
    )


def _make_mock_client(response_text: str):
    """Build a mock anthropic.Anthropic client that returns *response_text*."""
    content_block = MagicMock()
    content_block.text = response_text
    message = MagicMock()
    message.content = [content_block]
    client = MagicMock()
    client.messages.create.return_value = message
    return client


_VALID_RESPONSE = json.dumps({
    "answer": "Mallard populations increased substantially.",
    "confidence_assessment": "Evidence is strong with high extraction confidence.",
    "supporting_claim_ids": ["c-001", "c-002"],
    "caveats": [],
})

_RESPONSE_WITH_FENCE = f"```json\n{_VALID_RESPONSE}\n```"


class TestSynthesise:
    def _engine(self, response_text: str):
        from gemynd.retrieval.synthesis import SynthesisEngine
        engine = SynthesisEngine.__new__(SynthesisEngine)
        engine._client = _make_mock_client(response_text)
        engine._max_tokens = 1000
        engine._timeout = 60
        return engine

    def test_basic_parse(self):
        engine = self._engine(_VALID_RESPONSE)
        blocks = [_make_block("c-001", 0.9), _make_block("c-002", 0.7)]
        result = engine.synthesise("How many mallards?", blocks, "context text")
        assert result.answer == "Mallard populations increased substantially."
        assert result.confidence_assessment != ""
        assert result.supporting_claim_ids == ["c-001", "c-002"]
        assert isinstance(result.caveats, list)

    def test_min_confidence_computed(self):
        engine = self._engine(_VALID_RESPONSE)
        blocks = [_make_block("c-001", 0.9), _make_block("c-002", 0.7)]
        result = engine.synthesise("query", blocks, "context")
        assert result.min_extraction_confidence == pytest.approx(0.7, abs=1e-4)

    def test_min_confidence_from_all_blocks_not_just_cited(self):
        response = json.dumps({
            "answer": "answer",
            "confidence_assessment": "ok",
            "supporting_claim_ids": ["c-999"],
            "caveats": [],
        })
        engine = self._engine(response)
        blocks = [_make_block("c-001", 0.9)]
        result = engine.synthesise("query", blocks, "context")
        assert result.min_extraction_confidence == pytest.approx(0.9, abs=1e-4)

    def test_markdown_fence_stripped(self):
        engine = self._engine(_RESPONSE_WITH_FENCE)
        blocks = [_make_block("c-001", 0.9)]
        result = engine.synthesise("query", blocks, "context")
        assert result.answer == "Mallard populations increased substantially."

    def test_non_json_raises_value_error(self):
        engine = self._engine("This is plain text not JSON.")
        with pytest.raises(ValueError, match="non-JSON"):
            engine.synthesise("query", [], "context")

    def test_analytical_result_forwarded(self):
        engine = self._engine(_VALID_RESPONSE)
        analytical = AnalyticalResult(
            query_name="species_trend",
            columns=["year", "count"],
            rows=[{"year": 1942, "count": 5}],
        )
        result = engine.synthesise("query", [], "context", analytical_result=analytical)
        assert result.analytical_result is analytical

    def test_anthropic_client_called_with_correct_model(self):
        from gemynd.retrieval.synthesis import MODEL
        engine = self._engine(_VALID_RESPONSE)
        engine.synthesise("query", [], "context")
        call_kwargs = engine._client.messages.create.call_args[1]
        assert call_kwargs["model"] == MODEL

    def test_caveats_present_in_response(self):
        response = json.dumps({
            "answer": "Answer",
            "confidence_assessment": "Low confidence.",
            "supporting_claim_ids": [],
            "caveats": ["Data may be estimated."],
        })
        engine = self._engine(response)
        result = engine.synthesise("query", [], "context")
        assert "Data may be estimated." in result.caveats
