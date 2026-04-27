"""Tests for the cross-corpus NER enrichment pass."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gemynd.ingest.enrichment.cross_corpus_ner import (
    _abbreviation_forms,
    _build_expanded_records,
    _collapse_spaced_chars,
    _extract_candidates,
    _is_ocr_corrupted,
    _last_token_anchor,
    _score_candidates,
)
from gemynd.core.models import EntityRecord
from scripts.generate_ocr_variants import OCRConfusionMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grantee(name: str, entity_id: str | None = None) -> EntityRecord:
    from gemynd.core.ids import make_entity_id
    from gemynd.core.resolver import normalize_name
    eid = entity_id or make_entity_id("Person", normalize_name(name))
    return EntityRecord(
        entity_id=eid,
        entity_type="Person",
        name=name,
        normalized_form=normalize_name(name),
    )


def _empty_matrix() -> OCRConfusionMatrix:
    return OCRConfusionMatrix(pairs={})


def _builtin_matrix() -> OCRConfusionMatrix:
    return OCRConfusionMatrix.from_sources()


# ---------------------------------------------------------------------------
# OCR corruption detection
# ---------------------------------------------------------------------------

def test_is_ocr_corrupted_garbage():
    assert _is_ocr_corrupted("foo\x0bbar")
    assert _is_ocr_corrupted("foo ^ bar baz")


def test_is_ocr_corrupted_spaced_chars():
    assert _is_ocr_corrupted("G O N T I N u E D h o w e v e r")


def test_is_ocr_corrupted_clean():
    assert not _is_ocr_corrupted("Edward B Copeland filed a patent in 1910.")


# ---------------------------------------------------------------------------
# Spaced-char collapse
# ---------------------------------------------------------------------------

def test_collapse_all_caps_name():
    # Space-separated single letters collapse into one run — word boundaries
    # cannot be inferred from the characters alone, so the result is a single
    # token.  Scoring relies on SequenceMatcher to recover the similarity.
    assert _collapse_spaced_chars("E D W A R D C O P E L A N D") == "EDWARDCOPELAND"


def test_collapse_partial_name():
    result = _collapse_spaced_chars("E D W A R D Copeland")
    assert "EDWARD" in result
    assert "Copeland" in result


def test_collapse_no_op_on_clean():
    text = "Edward Copeland"
    assert _collapse_spaced_chars(text) == text


# ---------------------------------------------------------------------------
# Abbreviation forms
# ---------------------------------------------------------------------------

def test_abbreviation_forms_three_part():
    forms = _abbreviation_forms("Edward B Copeland")
    assert "Edward Copeland" in forms
    assert "E Copeland" in forms
    # Last-name-only forms are intentionally excluded
    assert "Copeland" not in forms


def test_abbreviation_forms_two_part():
    forms = _abbreviation_forms("John Smith")
    assert "John Smith" in forms
    assert "J Smith" in forms
    # Last-name-only forms are intentionally excluded
    assert "Smith" not in forms


def test_abbreviation_forms_single_token():
    assert _abbreviation_forms("Copeland") == []


# ---------------------------------------------------------------------------
# Variant expansion
# ---------------------------------------------------------------------------

def test_variant_expansion_rn_m():
    """rn→m confusion: 'Turnbull' should produce 'tumbull' variant."""
    grantee = _make_grantee("Will Turnbull")
    expanded = _build_expanded_records([grantee], _builtin_matrix(), min_ocr_prob=0.08)
    norms = {e.normalized_form for e in expanded}
    assert any("tumbull" in n for n in norms), f"Expected 'tumbull' variant in {norms}"


def test_variant_expansion_includes_canonical():
    grantee = _make_grantee("Edward B Copeland")
    expanded = _build_expanded_records([grantee], _empty_matrix(), min_ocr_prob=0.08)
    norms = {e.normalized_form for e in expanded}
    assert "edward b copeland" in norms
    assert "edward copeland" in norms
    # Last-name-only excluded
    assert "copeland" not in norms


def test_no_single_token_in_expanded():
    """All expanded records must be multi-word (no single-token false positives)."""
    grantee = _make_grantee("Alexander Johnson")
    expanded = _build_expanded_records([grantee], _builtin_matrix(), min_ocr_prob=0.08)
    for rec in expanded:
        assert " " in rec.normalized_form, f"Single-token form leaked: {rec.normalized_form!r}"


# ---------------------------------------------------------------------------
# Last-token anchor
# ---------------------------------------------------------------------------

def test_last_token_anchor_passes_exact():
    assert _last_token_anchor("george cook", "george cook") is True


def test_last_token_anchor_passes_near():
    # "cooke"/"cook" — raw SequenceMatcher ratio = 2*4/(5+4) = 0.889 >= 0.84
    assert _last_token_anchor("george cooke", "george cook") is True


def test_last_token_anchor_rejects_different_surname():
    # "frost"/"ross" — raw ratio ≈ 0.50 < 0.75
    assert _last_token_anchor("george frost", "george ross") is False


def test_last_token_anchor_rejects_anderson_sanders():
    # "anderson"/"sanders" — raw SequenceMatcher ratio = 0.800 < 0.84 threshold
    assert _last_token_anchor("john anderson", "john sanders") is False


def test_last_token_anchor_rejects_ander_sanders():
    # "ander"/"sanders" — raw ratio = 2*5/(5+7) = 0.833 < 0.84 threshold
    # Prevents "John Ander" (OCR truncation) from matching "John Sanders"
    assert _last_token_anchor("john ander", "john sanders") is False


def test_last_token_anchor_passes_initials():
    # Last token "a" is a single-char initial — treated as wildcard
    assert _last_token_anchor("edward a", "edward anderson") is True


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def test_extract_candidates_finds_two_word_name():
    text = "The patent was granted to Edward Copeland in the county."
    candidates = _extract_candidates(text)
    surfaces = [c[0] for c in candidates]
    assert "Edward Copeland" in surfaces


def test_extract_candidates_requires_two_tokens():
    """Single capitalized words must not be returned as candidates."""
    text = "Copeland arrived from Chicago."
    candidates = _extract_candidates(text)
    surfaces = [c[0] for c in candidates]
    assert "Copeland" not in surfaces


def test_extract_candidates_skips_stopwords():
    text = "The Spokane Washington land deed was filed."
    candidates = _extract_candidates(text)
    surfaces = [c[0] for c in candidates]
    assert "Spokane" not in surfaces
    assert "Washington" not in surfaces


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_score_exact_match_refers_to():
    grantee = _make_grantee("Edward Copeland")
    expanded = _build_expanded_records([grantee], _empty_matrix(), min_ocr_prob=0.08)
    candidates = [("Edward Copeland", 0, 15)]
    matches = _score_candidates(candidates, expanded, ocr_suspect=False)
    assert len(matches) == 1
    assert matches[0].relation_type == "REFERS_TO"
    assert matches[0].score >= 0.85


def test_score_different_surname_rejected():
    """'George Frost' must not match 'George Ross' — different surnames."""
    grantee = _make_grantee("George Ross")
    expanded = _build_expanded_records([grantee], _empty_matrix(), min_ocr_prob=0.08)
    candidates = [("George Frost", 0, 12)]
    matches = _score_candidates(candidates, expanded, ocr_suspect=False)
    assert len(matches) == 0


def test_score_spaced_char_name_collapses_and_matches():
    """'E D W A R D C O P E L A N D' collapses and scores as REFERS_TO."""
    grantee = _make_grantee("Edward Copeland")
    expanded = _build_expanded_records([grantee], _empty_matrix(), min_ocr_prob=0.08)
    # Surface as a spaced-char OCR failure
    surface = "E D W A R D C O P E L A N D"
    candidates = [(surface, 0, len(surface))]
    matches = _score_candidates(candidates, expanded, ocr_suspect=False)
    assert len(matches) == 1
    assert matches[0].relation_type == "REFERS_TO"
    assert matches[0].ocr_suspect is True


def test_score_no_match_below_threshold():
    grantee = _make_grantee("Edward Copeland")
    expanded = _build_expanded_records([grantee], _empty_matrix(), min_ocr_prob=0.08)
    candidates = [("John Williams", 0, 13)]
    matches = _score_candidates(candidates, expanded, ocr_suspect=False)
    assert len(matches) == 0


def test_score_initial_mismatch_capped_to_possibly():
    """'Sam Johnson' must not REFERS_TO 'A Johnson' — first initial 's' ≠ 'a'."""
    grantee = _make_grantee("A Johnson")
    expanded = _build_expanded_records([grantee], _empty_matrix(), min_ocr_prob=0.08)
    candidates = [("Sam Johnson", 0, 11)]
    matches = _score_candidates(candidates, expanded, ocr_suspect=False)
    assert len(matches) == 1
    assert matches[0].relation_type == "POSSIBLY_REFERS_TO"


def test_score_initial_match_allows_refers_to():
    """'Hugh Allen' correctly REFERS_TO 'H Allen' — first initial 'h' matches."""
    grantee = _make_grantee("H Allen")
    expanded = _build_expanded_records([grantee], _empty_matrix(), min_ocr_prob=0.08)
    candidates = [("Hugh Allen", 0, 10)]
    matches = _score_candidates(candidates, expanded, ocr_suspect=False)
    assert len(matches) == 1
    assert matches[0].relation_type == "REFERS_TO"
