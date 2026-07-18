"""Short citation handles (C1, C2, …) and their resolution back to claim ids.

Why this exists: claim ids are ~55 characters of opaque run prefix plus hash.
Measured on gemma-4-e4b, every citation in every answer was truncated to the
run prefix — which is ambiguous, because several claims share one run, so
_match_claim_id correctly refuses to repair it. The prose and the reasoning
were right; only the transcription failed, and that made synthesis unusable.

Handles remove the failure mode without guessing: they are positional over
the same ordered block list the context was built from, so resolution is
exact rather than fuzzy.
"""
from __future__ import annotations

import pytest

from gemynd.retrieval.context_assembler import _serialise_block, cite_token_for_index
from gemynd.retrieval.models import ProvenanceBlock
from gemynd.retrieval.synthesis import (
    _repair_claim_ids,
    _resolve_citation_tokens,
    _validate_response,
)

# Shape of a real id: run prefix + claim ordinal + hash.
_ID_A = "run_20260318T014232Z_08efed72:claim:098:0b8380b4ec6cf15a"
_ID_B = "run_20260318T014232Z_08efed72:claim:104:2f7c1a9e5b3d8c40"  # same run!
_ID_C = "run_20260714T060514557827Z_9b4fca8d:claim:016:6a5c3c4fb56fb7f7"


def _block(claim_id: str, sentence: str = "A claim.") -> ProvenanceBlock:
    return ProvenanceBlock(
        doc_title="Narrative Report 1939",
        doc_date_start="1939-01-01",
        doc_date_end="1939-12-31",
        page_number=3,
        paragraph_id="para-001",
        claim_id=claim_id,
        claim_type="predator_control",
        extraction_confidence=0.9,
        epistemic_status="observed",
        source_sentence=sentence,
        observation_type=None,
        species_name=None,
        year=1939,
        measurements=[],
        institution_id="turnbull",
    )


@pytest.fixture()
def blocks() -> list[ProvenanceBlock]:
    return [_block(_ID_A), _block(_ID_B), _block(_ID_C)]


class TestTokenFormat:
    def test_is_one_based(self):
        assert cite_token_for_index(0) == "C1"
        assert cite_token_for_index(2) == "C3"

    def test_appears_in_serialised_block(self):
        text = _serialise_block(_block(_ID_A), cite_token="C1")
        assert "CITE: C1" in text
        assert f"CLAIM_ID: {_ID_A}" in text  # full id still offered

    def test_omitted_when_not_supplied(self):
        assert "CITE:" not in _serialise_block(_block(_ID_A))


class TestResolution:
    def test_resolves_handles_to_claim_ids(self, blocks):
        raw = {"answer_sentences": [
            {"text": "x", "supporting_claim_ids": ["C1"], "is_inferred": False},
            {"text": "y", "supporting_claim_ids": ["C3"], "is_inferred": False},
        ]}
        notes = _resolve_citation_tokens(raw, blocks)
        assert len(notes) == 2
        assert raw["answer_sentences"][0]["supporting_claim_ids"] == [_ID_A]
        assert raw["answer_sentences"][1]["supporting_claim_ids"] == [_ID_C]

    def test_disambiguates_claims_sharing_a_run(self, blocks):
        """The case fuzzy repair cannot handle: C1 and C2 share a run prefix."""
        raw = {"answer_sentences": [
            {"text": "x", "supporting_claim_ids": ["C1", "C2"], "is_inferred": False},
        ]}
        _resolve_citation_tokens(raw, blocks)
        assert raw["answer_sentences"][0]["supporting_claim_ids"] == [_ID_A, _ID_B]

    @pytest.mark.parametrize("emitted", ["C2", "c2", "[2]", "2", " C2 ", "(2)"])
    def test_tolerates_shapes_models_actually_emit(self, blocks, emitted):
        raw = {"answer_sentences": [
            {"text": "x", "supporting_claim_ids": [emitted], "is_inferred": False},
        ]}
        _resolve_citation_tokens(raw, blocks)
        assert raw["answer_sentences"][0]["supporting_claim_ids"] == [_ID_B]

    def test_resolves_section_outline_too(self, blocks):
        raw = {"section_outline": [{"heading": "H", "claim_ids": ["C1", "C3"]}]}
        _resolve_citation_tokens(raw, blocks)
        assert raw["section_outline"][0]["claim_ids"] == [_ID_A, _ID_C]

    def test_real_claim_ids_pass_through_untouched(self, blocks):
        """The Anthropic path cites full ids and must be unaffected."""
        raw = {"answer_sentences": [
            {"text": "x", "supporting_claim_ids": [_ID_A], "is_inferred": False},
        ]}
        assert _resolve_citation_tokens(raw, blocks) == []
        assert raw["answer_sentences"][0]["supporting_claim_ids"] == [_ID_A]

    def test_out_of_range_handle_is_left_for_the_validator(self, blocks):
        """Never invent a mapping — an unknown handle must still be rejected."""
        raw = {"answer_sentences": [
            {"text": "x", "supporting_claim_ids": ["C99"], "is_inferred": False},
        ]}
        _resolve_citation_tokens(raw, blocks)
        assert raw["answer_sentences"][0]["supporting_claim_ids"] == ["C99"]
        assert _validate_response(
            {"organizing_structure": "thematic", **raw}, blocks
        ), "an unresolvable handle must fail validation"

    def test_non_citation_strings_untouched(self, blocks):
        raw = {"answer_sentences": [
            {"text": "x", "supporting_claim_ids": ["not-an-id"], "is_inferred": False},
        ]}
        _resolve_citation_tokens(raw, blocks)
        assert raw["answer_sentences"][0]["supporting_claim_ids"] == ["not-an-id"]


class TestEndToEnd:
    def test_handle_answer_passes_validation(self, blocks):
        """The exact failure observed with e4b, now surviving the gate."""
        raw = {
            "organizing_structure": "chronological",
            "answer_sentences": [
                {"text": "In 1939 coyotes were taken monthly.",
                 "supporting_claim_ids": ["C1"], "is_inferred": False},
                {"text": "A later quarter recorded none.",
                 "supporting_claim_ids": ["C2"], "is_inferred": False},
            ],
            "section_outline": [{"heading": "Predator control", "claim_ids": ["C1", "C2"]}],
        }
        _resolve_citation_tokens(raw, blocks)
        _repair_claim_ids(raw, blocks)
        assert _validate_response(raw, blocks) == []

    def test_truncated_run_prefix_still_rejected(self, blocks):
        """Regression: the raw e4b failure mode must not silently 'work'.

        A bare run prefix names a document, not a claim, and two of these
        blocks share one. Resolution must not guess between them.
        """
        raw = {
            "organizing_structure": "chronological",
            "answer_sentences": [
                {"text": "x", "supporting_claim_ids": ["run_20260318T014232Z_08efed72"],
                 "is_inferred": False},
            ],
        }
        _resolve_citation_tokens(raw, blocks)
        _repair_claim_ids(raw, blocks)
        assert _validate_response(raw, blocks), "ambiguous prefix must not resolve"


class TestInlineCitationStripping:
    """Models sometimes append the citation to the prose as well as the field.

    The UI renders citations as pills from `supporting_claim_ids`, so an
    inline "(C8)" shows the reader a raw identifier they should never see.
    e4b does this; Sonnet does not.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("700 Canada geese came into the refuge (C8).",
         "700 Canada geese came into the refuge."),
        ("Two sources agree (C1, C9).", "Two sources agree."),
        ("Coyotes were taken monthly [C3]", "Coyotes were taken monthly"),
        (f"A claim ({_ID_A}).", "A claim."),
        ("Mid-sentence (C2) citation kept readable.",
         "Mid-sentence citation kept readable."),
    ])
    def test_strips_citation_parentheticals(self, raw, expected):
        from gemynd.retrieval.synthesis import _strip_inline_citations
        assert _strip_inline_citations(raw) == expected

    @pytest.mark.parametrize("text", [
        "Water levels fell (an unusually dry August).",
        "The refuge (established 1937) held 900 mallards.",
        "Counts were approximate (see caveats).",
        "American coot (Fulica americana) are not waterfowl.",
        "The figure is ~2000 acres (approximate).",
    ])
    def test_leaves_ordinary_parentheticals_alone(self, text):
        """The rule must be narrow — real prose often uses parentheses."""
        from gemynd.retrieval.synthesis import _strip_inline_citations
        assert _strip_inline_citations(text) == text

    def test_applied_during_coercion(self):
        from gemynd.retrieval.synthesis import _coerce_answer_sentences
        out = _coerce_answer_sentences([
            {"text": "Coyotes were taken monthly (C1).",
             "supporting_claim_ids": [_ID_A], "is_inferred": False},
        ])
        assert out[0].text == "Coyotes were taken monthly."
        # Stripping display text must not disturb the citation field.
        assert out[0].supporting_claim_ids == [_ID_A]
