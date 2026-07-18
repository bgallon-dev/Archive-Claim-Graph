"""Invariants that guard the "no silent Turnbull fallback" property.

These tests pin the domain-agnosticism refactor: a run must name its domain, and
config-absent paths fall back to empty/raise — never to Turnbull's ontology. They
also lock the measurement-alternation parity (singular *and* plural forms) that a
naive config migration would have silently dropped.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gemynd.core.domain_config import load_domain_config
from gemynd.core.models import ClaimRecord
from gemynd.core.resolver import default_seed_entities
from gemynd.ingest.concept_assigner import assign_concepts
from gemynd.ingest.extractors.anthropic_claim_adapter import _build_system_prompt
from gemynd.ingest.extractors.mention_extractor import RuleBasedMentionExtractor
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR

_NEWSPAPER_DIR = Path(__file__).parent.parent / "gemynd" / "resources_newspaper"

_TURNBULL_LABELS = {"Refuge", "Species", "Habitat", "SurveyMethod"}
_TURNBULL_RELATIONS = {"SPECIES_FOCUS", "HABITAT_FOCUS", "METHOD_FOCUS"}
_TURNBULL_CONCEPTS = {
    "concept_nesting_success", "concept_breeding_success", "concept_drought_condition",
    "concept_habitat_condition", "concept_survey_result", "concept_seasonal_condition",
}


# ── Fail-loud: a run must name its domain ──────────────────────────────────────

def test_load_domain_config_rejects_none() -> None:
    with pytest.raises(ValueError, match="explicit resources_dir"):
        load_domain_config(None)


def test_default_seed_entities_rejects_none() -> None:
    with pytest.raises(ValueError, match="explicit resources_dir"):
        default_seed_entities(None)


# ── Newspaper config carries none of Turnbull's taxonomy ───────────────────────

def test_newspaper_config_has_no_turnbull_taxonomy() -> None:
    config = load_domain_config(_NEWSPAPER_DIR)
    # No ecological entity labels.
    assert not (_TURNBULL_LABELS & set(config.entity_labels))
    # No Refuge->Place (or any) categorial coercion declared.
    assert config.categorial_label_map == {}
    # Rule-based claim-link roles are empty (rule layer is a no-op; GLiNER/LLM
    # do newspaper linking). Crucially, no Species/Habitat/SurveyMethod roles.
    role_types = {r.entity_type for r in config.claim_link_roles}
    assert not (_TURNBULL_LABELS & role_types)
    # No Turnbull ecological concept rules leaked in.
    concept_ids = {rule[0] for rule in config.concept_rules}
    assert not (_TURNBULL_CONCEPTS & concept_ids)
    # No Turnbull measurement units (hay_cut/coveys_count/wood_killed_cords).
    measurement_names = {name for _pat, name, _unit in config.measurement_pattern_specs}
    assert "hay_cut" not in measurement_names
    assert "coveys_count" not in measurement_names


# ── Empty config paths produce empty, not Turnbull ─────────────────────────────

def test_concept_assignment_empty_without_config() -> None:
    claim = ClaimRecord(
        claim_id="claim_x",
        run_id="run_x",
        paragraph_id="p1",
        claim_type="breeding_activity",
        source_sentence="The pairs nested successfully and reared young.",
        normalized_sentence="the pairs nested successfully and reared young.",
        certainty="certain",
        extraction_confidence=0.9,
    )
    # No config → no concept rules → no assignments (never the ecological set).
    assert assign_concepts(claim, config=None) == []


def test_build_system_prompt_newspaper_has_no_ecological_relations() -> None:
    config = load_domain_config(_NEWSPAPER_DIR)
    prompt = _build_system_prompt(config)
    assert "SUBJECT_OF_CLAIM" in prompt
    assert "ORGANIZATION_FOCUS" in prompt
    for leaked in _TURNBULL_RELATIONS:
        assert leaked not in prompt, f"{leaked} leaked into newspaper prompt"


def test_mention_extractor_uses_domain_lexicon_not_turnbull() -> None:
    config = load_domain_config(_NEWSPAPER_DIR)
    extractor = RuleBasedMentionExtractor(config=config)
    # Newspaper seed vocabulary must not include Turnbull species terms.
    assert "mallard" not in extractor._lexicon_hints
    assert "coyote" not in extractor._lexicon_hints


# ── Measurement alternation parity (the reviewer's #1 hazard) ──────────────────

def test_measurement_patterns_preserve_singular_and_plural() -> None:
    config = load_domain_config(TURNBULL_RESOURCES_DIR)
    by_name = {name: pat for pat, name, _unit in config.measurement_pattern_specs}
    # Each scan pattern must match BOTH the singular and plural unit form — a
    # naive (\d+)\s*(<plural-key>) rebuild would have dropped the singular.
    assert by_name["acres"].search("burned 5 acre today")
    assert by_name["acres"].search("burned 5 acres today")
    assert by_name["rainfall"].search("2 inch of rain")
    assert by_name["rainfall"].search("2 inches of rain")
    assert by_name["coveys_count"].search("saw 3 covy")
    assert by_name["coveys_count"].search("saw 3 covies")
    assert by_name["hay_cut"].search("cut 10 ton")
    assert by_name["hay_cut"].search("cut 10 tons")


def test_spelling_review_vocab_is_domain_not_turnbull() -> None:
    from gemynd.ingest.spelling_review import _vocab_from_config

    vocab = _vocab_from_config(load_domain_config(_NEWSPAPER_DIR))
    # Spelling-review "known term" judgement keys to the newspaper vocabulary,
    # not Turnbull's ecological species terms. (The newspaper legitimately knows
    # "refuge" — it covers the refuge's pre-history — so that is NOT a leak.)
    assert "mallard" not in vocab.seed_reference_terms
    assert "mallard" not in vocab.lexicon_hints
    assert "mallard" not in vocab.domain_reference_terms


def test_unit_right_re_in_sync_with_patterns() -> None:
    config = load_domain_config(TURNBULL_RESOURCES_DIR)
    rr = config.measurement_unit_right_re
    assert rr is not None
    # Range right-context recognizes the same units (legacy alternation form).
    assert rr.match("acres")
    assert rr.match("acre")
    assert rr.match("inch")
    assert rr.match("inches")
    assert rr.match("covies")
    # Preserved quirk: the right-context regex used `covies?`, which never
    # matched the bare singular "covy". Lock it so a future edit can't silently
    # broaden range classification.
    assert rr.match("covy") is None
