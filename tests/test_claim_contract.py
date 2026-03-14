from graphrag_pipeline.claim_contract import (
    ALLOWED_CLAIM_TYPES,
    CLAIM_LOCATION_RELATION,
    CLAIM_ENTITY_RELATIONS,
    OBSERVATION_ELIGIBLE_TYPES,
    UNCLASSIFIED_TYPE,
    validate_claim_link_relation,
    validate_claim_type,
)


def test_valid_type_passes_through() -> None:
    assert validate_claim_type("population_estimate") == "population_estimate"
    assert validate_claim_type("economic_use") == "economic_use"
    assert validate_claim_type("fire_incident") == "fire_incident"


def test_unclassified_type_passes_through() -> None:
    assert validate_claim_type(UNCLASSIFIED_TYPE) == UNCLASSIFIED_TYPE


def test_legacy_renames() -> None:
    assert validate_claim_type("wildlife_count") == "population_estimate"
    assert validate_claim_type("weather_condition") == "weather_observation"


def test_unknown_type_coerces_to_unclassified() -> None:
    assert validate_claim_type("completely_made_up_type") == UNCLASSIFIED_TYPE
    assert validate_claim_type("") == UNCLASSIFIED_TYPE


def test_unclassified_not_observation_eligible() -> None:
    assert UNCLASSIFIED_TYPE not in OBSERVATION_ELIGIBLE_TYPES


def test_observation_eligible_subset_of_allowed() -> None:
    # Every eligible type must be in the allowed set (legacy wildlife_count excluded
    # from ALLOWED but still in ELIGIBLE for compat — that is intentional).
    non_legacy_eligible = OBSERVATION_ELIGIBLE_TYPES - {"wildlife_count"}
    assert non_legacy_eligible <= ALLOWED_CLAIM_TYPES


def test_allowed_types_includes_fallback() -> None:
    assert UNCLASSIFIED_TYPE in ALLOWED_CLAIM_TYPES


def test_validate_claim_link_relation_accepts_new_roles_and_occurrence_location() -> None:
    for relation_type in CLAIM_ENTITY_RELATIONS | {CLAIM_LOCATION_RELATION}:
        assert validate_claim_link_relation(relation_type) == relation_type


def test_validate_claim_link_relation_rejects_legacy_about_for_new_emission() -> None:
    assert validate_claim_link_relation("ABOUT") is None
    assert validate_claim_link_relation("ABOUT", allow_legacy=True) == "ABOUT"
