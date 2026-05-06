"""Unit tests for Layer 3: synthesis engine.

Uses a deterministic mock Anthropic client so no live API call is made.
Tests verify:
  • Structured JSON envelope (organizing_structure, answer_sentences,
    section_outline) is parsed into SynthesisResult correctly.
  • The schema gate rejects uncited assertions and triggers a single retry.
  • Hard-fail (SynthesisSchemaError) raises after a second invalid attempt.
  • Deterministic provenance builder distinguishes searched / contributed /
    deprioritized corpora.
  • Divergence detector flags tight temporal clusters with thematic structure.
  • min_extraction_confidence is computed from all retrieved blocks.
  • Markdown fence stripping and prose-then-JSON recovery still work.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from gemynd.retrieval.models import (
    AnalyticalResult,
    AnswerSentence,
    ProvenanceBlock,
    SynthesisResult,
)


def _make_block(claim_id: str, confidence: float, year: int | None = None,
                institution_id: str | None = None) -> ProvenanceBlock:
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
        year=year,
        institution_id=institution_id,
    )


class _FakeStreamCM:
    """Fake Anthropic stream context manager — yields configured chunks
    when ``text_stream`` is iterated, then ``get_final_message()`` returns
    a stub with the configured stop_reason and usage.  Mirrors the real
    SDK's streaming surface."""

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


def _make_mock_client(response_text: str | list[str], stop_reason: str = "end_turn"):
    """Build a mock anthropic.Anthropic client.

    *response_text* is either a single string (returned for every stream call)
    or a list of strings (one per consecutive stream call — used to test the
    retry path where attempt 1 fails validation and attempt 2 succeeds).
    """
    client = MagicMock()
    if isinstance(response_text, list):
        client.messages.stream.side_effect = [
            _FakeStreamCM(text_chunks=[r], stop_reason=stop_reason)
            for r in response_text
        ]
    else:
        client.messages.stream.return_value = _FakeStreamCM(
            text_chunks=[response_text], stop_reason=stop_reason,
        )
    return client


def _valid_response(
    sentences: list[dict] | None = None,
    structure: str = "thematic",
    section_outline: list[dict] | None = None,
    confidence_assessment: str = "Evidence is strong with high extraction confidence.",
    caveats: list[str] | None = None,
) -> str:
    """Build a schema-valid JSON envelope for the new synthesis schema."""
    if sentences is None:
        sentences = [
            {"text": "Mallard populations increased substantially.",
             "supporting_claim_ids": ["c-001", "c-002"], "is_inferred": False},
        ]
    return json.dumps({
        "organizing_structure": structure,
        "section_outline": section_outline or [],
        "answer_sentences": sentences,
        "confidence_assessment": confidence_assessment,
        "caveats": caveats or [],
    })


_VALID_RESPONSE = _valid_response()
_RESPONSE_WITH_FENCE = f"```json\n{_VALID_RESPONSE}\n```"


class TestSynthesise:
    def _engine(self, response_text, stop_reason: str = "end_turn"):
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
        assert result.organizing_structure == "thematic"
        assert len(result.answer_sentences) == 1
        assert result.answer_sentences[0].supporting_claim_ids == ["c-001", "c-002"]
        assert result.answer_sentences[0].is_inferred is False
        # Flat back-compat list union of all sentence claim_ids.
        assert result.supporting_claim_ids == ["c-001", "c-002"]

    def test_min_confidence_computed(self):
        engine = self._engine(_VALID_RESPONSE)
        blocks = [_make_block("c-001", 0.9), _make_block("c-002", 0.7)]
        result = engine.synthesise("query", blocks, "context")
        assert result.min_extraction_confidence == pytest.approx(0.7, abs=1e-4)

    def test_min_confidence_from_all_blocks_not_just_cited(self):
        # Only c-001 is cited via is_inferred=False, but c-002 is in provenance.
        # min_extraction_confidence should reflect both blocks.
        sentences = [
            {"text": "Cited sentence.",
             "supporting_claim_ids": ["c-001"], "is_inferred": False},
        ]
        engine = self._engine(_valid_response(sentences=sentences))
        blocks = [_make_block("c-001", 0.9), _make_block("c-002", 0.5)]
        result = engine.synthesise("query", blocks, "context")
        assert result.min_extraction_confidence == pytest.approx(0.5, abs=1e-4)

    def test_markdown_fence_stripped(self):
        engine = self._engine(_RESPONSE_WITH_FENCE)
        blocks = [_make_block("c-001", 0.9), _make_block("c-002", 0.7)]
        result = engine.synthesise("query", blocks, "context")
        assert result.answer == "Mallard populations increased substantially."

    def test_pure_prose_hard_fails_after_retry(self):
        """The model returns prose (no JSON) on both attempts — the schema
        gate raises SynthesisSchemaError rather than silently shipping
        uncited content.  This is the explicit design choice: don't trust
        the model to recite, fail loudly when it can't be verified."""
        from gemynd.retrieval.synthesis import SynthesisSchemaError
        prose = "Based on the archival records, mallards appear in 23 claims."
        engine = self._engine([prose, prose])
        blocks = [_make_block("c-001", 0.9)]
        with pytest.raises(SynthesisSchemaError):
            engine.synthesise("query", blocks, "context")

    def test_uncited_sentence_triggers_retry_and_recovers(self):
        """Attempt 1 has a sentence with no citations and is_inferred=False
        (validation failure).  Attempt 2 marks that sentence inferred.  The
        retry path should hand back the second response cleanly."""
        bad = json.dumps({
            "organizing_structure": "thematic",
            "section_outline": [],
            "answer_sentences": [
                {"text": "Unsupported assertion.",
                 "supporting_claim_ids": [], "is_inferred": False},
            ],
            "confidence_assessment": "ok",
            "caveats": [],
        })
        good = _valid_response(sentences=[
            {"text": "Unsupported assertion.",
             "supporting_claim_ids": [], "is_inferred": True},
        ])
        engine = self._engine([bad, good])
        blocks = [_make_block("c-001", 0.9)]
        result = engine.synthesise("query", blocks, "context")
        assert len(result.answer_sentences) == 1
        assert result.answer_sentences[0].is_inferred is True
        # Both stream calls were made.
        assert engine._client.messages.stream.call_count == 2

    def test_unknown_claim_id_hard_fails_after_retry(self):
        """A sentence cites a claim_id not in the provenance blocks — the
        validator rejects it.  When the retry returns the same bad payload,
        SynthesisSchemaError fires."""
        from gemynd.retrieval.synthesis import SynthesisSchemaError
        bad = _valid_response(sentences=[
            {"text": "Bogus.",
             "supporting_claim_ids": ["c-NOT-IN-CONTEXT"], "is_inferred": False},
        ])
        engine = self._engine([bad, bad])
        blocks = [_make_block("c-001", 0.9)]
        with pytest.raises(SynthesisSchemaError) as excinfo:
            engine.synthesise("query", blocks, "context")
        assert any("c-NOT-IN-CONTEXT" in e for e in excinfo.value.validation_errors)

    def test_paragraph_id_cited_as_claim_id_fails_validation(self):
        """Regression for the exact production failure: the rendered context
        block surfaces both a `PARAGRAPH:` value and a (now-added) `CLAIM_ID:`
        value with different formats.  Pre-fix the model only saw the
        paragraph_id and cited it; the validator must still reject that.

        The companion fix is in context_assembler._serialise_block which now
        renders CLAIM_ID alongside PARAGRAPH so the model can pick the right
        one.  This test pins the validator behaviour so any regression that
        re-introduces paragraph_id-as-claim_id fails loudly."""
        from gemynd.retrieval.synthesis import SynthesisSchemaError
        bad = _valid_response(sentences=[
            {"text": "Citing paragraph_id by mistake.",
             "supporting_claim_ids": ["doc_db3544e3b57e69d9:paragraph:00004:p0001"],
             "is_inferred": False},
        ])
        engine = self._engine([bad, bad])
        blocks = [ProvenanceBlock(
            doc_title="Spokane Press",
            doc_date_start=None, doc_date_end=None,
            page_number=1,
            paragraph_id="doc_db3544e3b57e69d9:paragraph:00004:p0001",
            claim_id="run_abc:claim:001:def456",
            claim_type="labor_action",
            extraction_confidence=0.9,
            epistemic_status="observed",
            source_sentence="Strike at the foundry.",
            institution_id="spokane_newspaper",
        )]
        with pytest.raises(SynthesisSchemaError) as excinfo:
            engine.synthesise("query", blocks, "context")
        assert any(
            "doc_db3544e3b57e69d9:paragraph:00004:p0001" in e
            for e in excinfo.value.validation_errors
        )

    def test_correctly_cited_claim_id_passes_validation(self):
        """Companion to the regression above: when the model copies the
        rendered CLAIM_ID verbatim (the post-fix expected behaviour) the
        validator must accept it."""
        good = _valid_response(sentences=[
            {"text": "Properly cited.",
             "supporting_claim_ids": ["run_abc:claim:001:def456"],
             "is_inferred": False},
        ])
        engine = self._engine(good)
        blocks = [ProvenanceBlock(
            doc_title="Spokane Press",
            doc_date_start=None, doc_date_end=None,
            page_number=1,
            paragraph_id="doc_db3544e3b57e69d9:paragraph:00004:p0001",
            claim_id="run_abc:claim:001:def456",
            claim_type="labor_action",
            extraction_confidence=0.9,
            epistemic_status="observed",
            source_sentence="Strike at the foundry.",
            institution_id="spokane_newspaper",
        )]
        result = engine.synthesise("query", blocks, "context")
        assert len(result.answer_sentences) == 1
        assert result.answer_sentences[0].supporting_claim_ids == [
            "run_abc:claim:001:def456"
        ]

    def test_system_prompt_instructs_model_to_use_claim_id(self):
        """The system prompt must explicitly direct the model to cite the
        CLAIM_ID field rather than the PARAGRAPH value.  This is the prompt-
        side half of the claim_id rendering fix."""
        from gemynd.retrieval.synthesis import (
            _DEFAULT_SYNTHESIS_CONTEXT,
            _build_system_prompt,
        )
        prompt = _build_system_prompt(_DEFAULT_SYNTHESIS_CONTEXT)
        assert "CLAIM_ID:" in prompt
        # The prompt must call out the failure mode it is preventing.
        assert "PARAGRAPH" in prompt
        assert "supporting_claim_ids" in prompt

    def test_invalid_organizing_structure_rejected(self):
        from gemynd.retrieval.synthesis import SynthesisSchemaError
        bad = json.dumps({
            "organizing_structure": "narrative",  # not in the closed enum
            "section_outline": [],
            "answer_sentences": [
                {"text": "x", "supporting_claim_ids": ["c-001"], "is_inferred": False},
            ],
            "confidence_assessment": "ok",
            "caveats": [],
        })
        engine = self._engine([bad, bad])
        blocks = [_make_block("c-001", 0.9)]
        with pytest.raises(SynthesisSchemaError):
            engine.synthesise("query", blocks, "context")

    def test_analytical_result_forwarded(self):
        engine = self._engine(_valid_response(sentences=[
            {"text": "x", "supporting_claim_ids": [], "is_inferred": True},
        ]))
        analytical = AnalyticalResult(
            query_name="species_trend",
            columns=["year", "count"],
            rows=[{"year": 1942, "count": 5}],
        )
        result = engine.synthesise("query", [], "context", analytical_result=analytical)
        assert result.analytical_result is analytical

    def test_anthropic_client_called_with_correct_model(self):
        from gemynd.retrieval.synthesis import MODEL
        engine = self._engine(_valid_response(sentences=[
            {"text": "x", "supporting_claim_ids": [], "is_inferred": True},
        ]))
        engine._timeout = 12
        engine.synthesise("query", [], "context")
        call_kwargs = engine._client.messages.stream.call_args[1]
        assert call_kwargs["model"] == MODEL
        assert call_kwargs["timeout"] == 12

    def test_caveats_present_in_response(self):
        response = _valid_response(
            sentences=[{"text": "x", "supporting_claim_ids": [], "is_inferred": True}],
            caveats=["Data may be estimated."],
        )
        engine = self._engine(response)
        result = engine.synthesise("query", [], "context")
        assert "Data may be estimated." in result.caveats

    def test_stream_max_tokens_raises_value_error(self):
        engine = self._engine(_VALID_RESPONSE, stop_reason="max_tokens")
        with pytest.raises(ValueError, match="truncated"):
            engine.synthesise("query", [], "context")

    def test_prose_then_fenced_json_recovered(self):
        """The lenient extractor should find an embedded JSON object and
        skip the leading prose."""
        embedded = _valid_response()
        model_output = (
            "The current query is identical to the prior query. The answer "
            "below consolidates both retrievals.\n\n"
            "```json\n" + embedded + "\n```\n"
        )
        engine = self._engine(model_output)
        blocks = [_make_block("c-001", 0.9), _make_block("c-002", 0.7)]
        result = engine.synthesise("query", blocks, "context")
        assert result.answer == "Mallard populations increased substantially."
        assert result.confidence_assessment != ""

    def test_extract_first_json_object_helper(self):
        from gemynd.retrieval.synthesis import _extract_first_json_object
        assert _extract_first_json_object('{"a": 1}') == '{"a": 1}'
        result = _extract_first_json_object("preamble\n```json\n{\"k\": \"v\"}\n```\ntrailing")
        assert result == '{"k": "v"}'
        result = _extract_first_json_object('{"a": {"b": 1}}')
        assert result == '{"a": {"b": 1}}'
        result = _extract_first_json_object('{"x": "hello {world}"}')
        assert result == '{"x": "hello {world}"}'
        result = _extract_first_json_object('{"x": "say \\"hi\\""}')
        assert result == '{"x": "say \\"hi\\""}'
        assert _extract_first_json_object("") is None
        assert _extract_first_json_object("just prose") is None
        assert _extract_first_json_object('{"a": ') is None

    def test_stream_chunks_concatenated(self):
        """Multi-chunk stream must reassemble into the full JSON response."""
        full = _VALID_RESPONSE
        midpoint = len(full) // 2
        chunks = [full[:midpoint], full[midpoint:]]
        client = MagicMock()
        client.messages.stream.return_value = _FakeStreamCM(text_chunks=chunks)
        from gemynd.retrieval.synthesis import SynthesisEngine
        engine = SynthesisEngine.__new__(SynthesisEngine)
        engine._client = client
        engine._max_tokens = 1000
        engine._timeout = 60
        blocks = [_make_block("c-001", 0.9), _make_block("c-002", 0.7)]
        result = engine.synthesise("query", blocks, "context")
        assert result.answer == "Mallard populations increased substantially."


class TestRetrievalProvenance:
    """Deterministic provenance builder — never depends on the model."""

    def test_searched_minus_contributed_marks_deprioritized(self):
        from gemynd.retrieval.synthesis import _build_retrieval_provenance
        blocks = [
            _make_block("c-001", 0.9, year=1909, institution_id="turnbull"),
            _make_block("c-002", 0.8, year=1910, institution_id="turnbull"),
            _make_block("c-003", 0.7, year=1911, institution_id="spokane"),
        ]
        prov = _build_retrieval_provenance(
            searched=["turnbull", "spokane", "glo"], provenance_blocks=blocks,
        )
        assert prov.searched == ["turnbull", "spokane", "glo"]
        contributed_ids = {c.institution_id for c in prov.contributed}
        assert contributed_ids == {"turnbull", "spokane"}
        deprioritized_ids = {d.institution_id for d in prov.deprioritized}
        assert deprioritized_ids == {"glo"}
        assert prov.deprioritized[0].reason == "no matches"

    def test_per_corpus_year_range_populated(self):
        from gemynd.retrieval.synthesis import _build_retrieval_provenance
        blocks = [
            _make_block("c-001", 0.9, year=1909, institution_id="turnbull"),
            _make_block("c-002", 0.8, year=1915, institution_id="turnbull"),
        ]
        prov = _build_retrieval_provenance(searched=["turnbull"], provenance_blocks=blocks)
        turnbull = next(c for c in prov.contributed if c.institution_id == "turnbull")
        assert turnbull.block_count == 2
        assert turnbull.year_min == 1909
        assert turnbull.year_max == 1915

    def test_no_searched_arg_falls_back_to_contributed_only(self):
        """When the caller doesn't pass searched_institution_ids, the preamble
        can't distinguish 'didn't search' from 'searched and found nothing' —
        it just lists what contributed."""
        from gemynd.retrieval.synthesis import _build_retrieval_provenance
        blocks = [_make_block("c-001", 0.9, institution_id="turnbull")]
        prov = _build_retrieval_provenance(searched=None, provenance_blocks=blocks)
        assert prov.searched == ["turnbull"]
        assert prov.deprioritized == []


class TestStructureDivergence:
    """Server-side detector that catches the Free Speech Fight failure mode:
    procedural temporal arc decomposed into parallel topical buckets."""

    def test_tight_temporal_cluster_with_thematic_structure_flagged(self):
        from gemynd.retrieval.synthesis import _detect_structure_divergence
        blocks = [
            _make_block(f"c-{i}", 0.9, year=1909 + i % 2, institution_id="spokane")
            for i in range(6)
        ]
        warning = _detect_structure_divergence("thematic", blocks)
        assert warning is not None
        assert warning.declared_structure == "thematic"
        assert warning.detected_signal == "tight_temporal_cluster"

    def test_chronological_choice_no_divergence(self):
        from gemynd.retrieval.synthesis import _detect_structure_divergence
        blocks = [
            _make_block(f"c-{i}", 0.9, year=1909 + i % 2, institution_id="spokane")
            for i in range(6)
        ]
        # Same data — but the model picked chronological. No divergence.
        assert _detect_structure_divergence("chronological", blocks) is None
        assert _detect_structure_divergence("causal", blocks) is None
        assert _detect_structure_divergence("mixed", blocks) is None

    def test_wide_temporal_span_thematic_no_divergence(self):
        """Blocks span > 5 years → genuinely thematic territory; no warning."""
        from gemynd.retrieval.synthesis import _detect_structure_divergence
        blocks = [
            _make_block("c-1", 0.9, year=1900),
            _make_block("c-2", 0.9, year=1920),
            _make_block("c-3", 0.9, year=1950),
            _make_block("c-4", 0.9, year=1970),
        ]
        assert _detect_structure_divergence("thematic", blocks) is None

    def test_too_few_blocks_no_divergence(self):
        """Below the 4-block threshold there isn't enough signal to cry
        divergence — keep silent rather than nag on small-result queries."""
        from gemynd.retrieval.synthesis import _detect_structure_divergence
        blocks = [_make_block("c-1", 0.9, year=1909),
                  _make_block("c-2", 0.9, year=1910)]
        assert _detect_structure_divergence("thematic", blocks) is None


class TestEndToEnd:
    """Wire the validator + provenance + divergence path together end-to-end."""

    def _engine(self, response_text):
        from gemynd.retrieval.synthesis import SynthesisEngine
        engine = SynthesisEngine.__new__(SynthesisEngine)
        engine._client = _make_mock_client(response_text)
        engine._max_tokens = 1000
        engine._timeout = 60
        return engine

    def test_provenance_and_divergence_attached_to_result(self):
        """Tight temporal cluster + thematic structure should attach a
        DivergenceWarning to the SynthesisResult; provenance preamble
        should reflect searched-vs-contributed delta."""
        sentences = [
            {"text": "Some statement.",
             "supporting_claim_ids": ["c-1"], "is_inferred": False},
        ]
        response = _valid_response(sentences=sentences, structure="thematic")
        engine = self._engine(response)
        blocks = [
            _make_block(f"c-{i}", 0.9, year=1909 + i % 2, institution_id="spokane")
            for i in range(1, 6)
        ]
        result = engine.synthesise(
            "query", blocks, "context",
            searched_institution_ids=["spokane", "glo"],
        )
        assert result.retrieval_provenance is not None
        assert result.retrieval_provenance.searched == ["spokane", "glo"]
        deprioritized_ids = {d.institution_id for d in result.retrieval_provenance.deprioritized}
        assert "glo" in deprioritized_ids
        assert result.structure_divergence is not None
        assert result.structure_divergence.detected_signal == "tight_temporal_cluster"
