"""Smoke tests for the resource loader — verifies all files parse correctly
and return the expected shapes, without testing pipeline logic."""
from __future__ import annotations

import re
from pathlib import Path

from gemynd.shared.resource_loader import (
    load_claim_role_policy,
    load_claim_type_patterns,
    load_domain_profile,
    load_measurement_species,
    load_measurement_units,
    load_ocr_correction_map,
    load_ocr_corrections,
    load_seed_entity_rows,
    load_spelling_reference_terms,
)


# ── Domain profile ────────────────────────────────────────────────────────────

def test_domain_profile_loads() -> None:
    profile = load_domain_profile()
    assert profile["domain"] == "turnbull_refuge_archive"
    assert "resources" in profile


# ── Seed entities ─────────────────────────────────────────────────────────────

def test_seed_entity_rows_nonempty() -> None:
    rows = load_seed_entity_rows()
    assert len(rows) > 20


def test_seed_entity_rows_have_required_columns() -> None:
    rows = load_seed_entity_rows()
    for row in rows:
        assert "entity_type" in row
        assert "name" in row
        assert "prop_key" in row
        assert "prop_value" in row


def test_seed_entities_cover_expected_types() -> None:
    rows = load_seed_entity_rows()
    types_present = {r["entity_type"] for r in rows}
    for expected in ("Species", "Habitat", "Place", "Refuge", "Activity", "SurveyMethod"):
        assert expected in types_present, f"Missing entity type: {expected}"


def test_seed_entities_include_extra_activity_aliases() -> None:
    """grazing/suppression/public relations/planting aliases moved from code."""
    rows = load_seed_entity_rows()
    activity_names = {r["name"] for r in rows if r["entity_type"] == "Activity"}
    for alias in ("grazing", "suppression", "public relations", "planting"):
        assert alias in activity_names, f"Missing activity alias: {alias!r}"


def test_seed_entities_survey_methods_present() -> None:
    rows = load_seed_entity_rows()
    method_names = {r["name"] for r in rows if r["entity_type"] == "SurveyMethod"}
    assert "aerial survey" in method_names
    assert "banding" in method_names


# ── Claim type patterns ───────────────────────────────────────────────────────

def test_claim_type_patterns_nonempty() -> None:
    patterns = load_claim_type_patterns()
    assert len(patterns) >= 10


def test_claim_type_patterns_returns_compiled_patterns() -> None:
    patterns = load_claim_type_patterns()
    for claim_type, compiled, weight in patterns:
        assert isinstance(claim_type, str)
        assert hasattr(compiled, "findall"), "pattern should be a compiled regex"
        assert isinstance(weight, float)
        assert weight > 0


def test_claim_type_patterns_cover_known_types() -> None:
    patterns = load_claim_type_patterns()
    types_covered = {ct for ct, _, _ in patterns}
    for expected in ("predator_control", "fire_incident", "species_presence", "breeding_activity"):
        assert expected in types_covered


def test_claim_type_patterns_are_case_insensitive() -> None:
    patterns = load_claim_type_patterns()
    _, pattern, _ = next(p for p in patterns if p[0] == "predator_control")
    assert pattern.search("A COYOTE was trapped")
    assert pattern.search("a coyote was trapped")


# ── Claim role policy ─────────────────────────────────────────────────────────

def test_claim_role_policy_nonempty() -> None:
    policy = load_claim_role_policy()
    assert len(policy) >= 20


def test_claim_role_policy_keys_are_tuples() -> None:
    policy = load_claim_role_policy()
    for key, value in policy.items():
        claim_type, entity_type = key
        assert isinstance(claim_type, str)
        assert isinstance(entity_type, str)
        assert isinstance(value, str)


def test_claim_role_policy_known_entry() -> None:
    policy = load_claim_role_policy()
    assert policy[("predator_control", "Species")] == "MANAGEMENT_TARGET"
    assert policy[("habitat_condition", "Habitat")] == "HABITAT_FOCUS"
    assert policy[("population_estimate", "SurveyMethod")] == "METHOD_FOCUS"


# ── Measurement units ─────────────────────────────────────────────────────────

def test_measurement_units_nonempty() -> None:
    units = load_measurement_units()
    assert len(units) >= 10


def test_measurement_units_values_are_name_unit_tuples() -> None:
    units = load_measurement_units()
    for key, (name, unit) in units.items():
        assert isinstance(key, str)
        assert isinstance(name, str)
        assert isinstance(unit, str)


def test_measurement_units_cover_known_entries() -> None:
    units = load_measurement_units()
    assert units["acres"] == ("acres", "acres")
    assert units["cords"] == ("wood_killed_cords", "cords")
    assert units["covies"] == ("coveys_count", "coveys")


# ── Measurement species ───────────────────────────────────────────────────────

def test_measurement_species_has_type_hints() -> None:
    species = load_measurement_species()
    assert "type_hints" in species
    assert len(species["type_hints"]) >= 15


def test_measurement_species_has_immediate_patterns() -> None:
    species = load_measurement_species()
    assert "immediate_patterns" in species
    assert len(species["immediate_patterns"]) >= 10


def test_measurement_species_type_hints_map_to_categories() -> None:
    species = load_measurement_species()
    hints = species["type_hints"]
    assert hints["mallard"] == "waterfowl"
    assert hints["coyote"] == "predator"
    assert hints["deer"] == "ungulate"
    assert hints["nest"] == "breeding"


def test_measurement_species_immediate_patterns_build_valid_regex() -> None:
    species = load_measurement_species()
    pattern = re.compile(
        r"^\s*(" + "|".join(species["immediate_patterns"]) + r")\b",
        re.IGNORECASE,
    )
    assert pattern.match(" mallards ")
    assert pattern.match("coyotes")
    assert pattern.match("foxes")


# ── OCR corrections ───────────────────────────────────────────────────────────

def test_ocr_corrections_is_frozenset() -> None:
    corrections = load_ocr_corrections()
    assert isinstance(corrections, frozenset)


def test_ocr_corrections_nonempty() -> None:
    corrections = load_ocr_corrections()
    assert len(corrections) >= 3


def test_ocr_corrections_contain_known_errors() -> None:
    corrections = load_ocr_corrections()
    assert "tumbull" in corrections
    assert "emgman" in corrections


def test_ocr_correction_map_contains_known_suggestions() -> None:
    corrections = load_ocr_correction_map()
    assert corrections["tumbull"] == "turnbull"
    assert corrections["emgman"] == "engman"


def test_spelling_reference_terms_cover_domain_vocabulary() -> None:
    terms = load_spelling_reference_terms()
    for expected in ("turnbull", "mallard", "acres", "suppression", "haying"):
        assert expected in terms
