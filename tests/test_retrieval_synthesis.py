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


class _FakeStreamCM:
    """Fake Anthropic stream context manager.

    Yields the configured ``text_chunks`` when ``text_stream`` is iterated,
    then ``get_final_message()`` returns a stub with the configured stop_reason
    and usage values — mirroring the real SDK's streaming surface.
    """

    def __init__(
        self,
        text_chunks: list[str],
        stop_reason: str = "end_turn",
        input_tokens: int = 100,
        output_tokens: int = 200,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._text_chunks = text_chunks
        self._stop_reason = stop_reason
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._model = model

    def __enter__(self):
        self.text_stream = iter(self._text_chunks)
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        msg = MagicMock()
        msg.stop_reason = self._stop_reason
        msg.model = self._model
        msg.usage.input_tokens = self._input_tokens
        msg.usage.output_tokens = self._output_tokens
        return msg


def _make_mock_client(response_text: str, stop_reason: str = "end_turn"):
    """Build a mock anthropic.Anthropic client whose ``messages.stream()``
    yields *response_text* as a single chunk.
    """
    client = MagicMock()
    client.messages.stream.return_value = _FakeStreamCM(
        text_chunks=[response_text],
        stop_reason=stop_reason,
    )
    return client


_VALID_RESPONSE = json.dumps({
    "answer": "Mallard populations increased substantially.",
    "confidence_assessment": "Evidence is strong with high extraction confidence.",
    "supporting_claim_ids": ["c-001", "c-002"],
    "caveats": [],
})

_RESPONSE_WITH_FENCE = f"```json\n{_VALID_RESPONSE}\n```"


class TestSynthesise:
    def _engine(self, response_text: str, stop_reason: str = "end_turn"):
        from gemynd.retrieval.synthesis import SynthesisEngine
        engine = SynthesisEngine.__new__(SynthesisEngine)
        engine._client = _make_mock_client(response_text, stop_reason=stop_reason)
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

    def test_non_json_falls_back_to_prose_answer(self):
        """When the model returns prose instead of JSON, the synthesis engine
        wraps the raw text into a SynthesisResult rather than raising — the
        prose is still useful to the user even if metadata is unavailable.
        Citations fall back to every provenance block in the retrieval context.
        """
        prose = ("Based on the archival records, mallards appear in 23 claims "
                 "from 1938 to 1963.")
        engine = self._engine(prose)
        blocks = [_make_block("c-001", 0.9), _make_block("c-002", 0.7)]
        result = engine.synthesise("query", blocks, "context")
        # The raw prose is preserved as the answer.
        assert result.answer == prose
        # Citations fall back to all provided provenance blocks.
        assert set(result.supporting_claim_ids) == {"c-001", "c-002"}
        # A caveat explains why metadata is incomplete.
        assert any("prose rather than" in c.lower() for c in result.caveats)

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
        call_kwargs = engine._client.messages.stream.call_args[1]
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

    def test_stream_max_tokens_raises_value_error(self):
        engine = self._engine(_VALID_RESPONSE, stop_reason="max_tokens")
        with pytest.raises(ValueError, match="truncated"):
            engine.synthesise("query", [], "context")

    def test_prose_then_fenced_json_recovered(self):
        """The model sometimes outputs explanatory prose, then a markdown
        fence around the JSON. The lenient extractor should find and parse
        the embedded JSON object rather than falling through to the
        prose-wrapper fallback."""
        model_output = (
            "The current query is identical to the prior query. The answer "
            "below consolidates both retrievals.\n\n"
            "```json\n"
            "{\"answer\": \"Mallards increased.\", "
            "\"confidence_assessment\": \"Strong evidence.\", "
            "\"supporting_claim_ids\": [\"c-001\"], "
            "\"caveats\": []}\n"
            "```\n"
        )
        engine = self._engine(model_output)
        blocks = [_make_block("c-001", 0.9)]
        result = engine.synthesise("query", blocks, "context")
        # Should parse the embedded JSON, not fall back to prose-wrap.
        assert result.answer == "Mallards increased."
        assert result.confidence_assessment == "Strong evidence."
        assert result.supporting_claim_ids == ["c-001"]
        # No "prose rather than" caveat since JSON was successfully extracted.
        assert not any("prose rather than" in c.lower() for c in result.caveats)

    def test_extract_first_json_object_helper(self):
        """The helper finds balanced top-level JSON objects, ignoring braces
        inside string literals.
        """
        from gemynd.retrieval.synthesis import _extract_first_json_object

        # Direct JSON
        assert _extract_first_json_object('{"a": 1}') == '{"a": 1}'
        # JSON embedded in prose
        result = _extract_first_json_object("preamble\n```json\n{\"k\": \"v\"}\n```\ntrailing")
        assert result == '{"k": "v"}'
        # Nested objects — returns the outermost balanced object
        result = _extract_first_json_object('{"a": {"b": 1}}')
        assert result == '{"a": {"b": 1}}'
        # Strings containing braces don't break the counter
        result = _extract_first_json_object('{"x": "hello {world}"}')
        assert result == '{"x": "hello {world}"}'
        # Escaped quotes inside strings are handled
        result = _extract_first_json_object('{"x": "say \\"hi\\""}')
        assert result == '{"x": "say \\"hi\\""}'
        # Empty / no-brace input
        assert _extract_first_json_object("") is None
        assert _extract_first_json_object("just prose") is None
        # Unbalanced returns None
        assert _extract_first_json_object('{"a": ') is None

    def test_stream_chunks_concatenated(self):
        """Multi-chunk stream must reassemble into the full JSON response."""
        chunks = ['{"answer": "Multi', '-chunk reply.", "confidence_assessment": "',
                  'ok", "supporting_claim_ids": [], "caveats": []}']
        client = MagicMock()
        client.messages.stream.return_value = _FakeStreamCM(text_chunks=chunks)
        from gemynd.retrieval.synthesis import SynthesisEngine
        engine = SynthesisEngine.__new__(SynthesisEngine)
        engine._client = client
        engine._max_tokens = 1000
        engine._timeout = 60
        result = engine.synthesise("query", [], "context")
        assert result.answer == "Multi-chunk reply."
