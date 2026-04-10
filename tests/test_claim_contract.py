from gemynd.core.claim_contract import (
    ALLOWED_CLAIM_TYPES,
    CLAIM_LOCATION_RELATION,
    CLAIM_ENTITY_RELATIONS,
    UNCLASSIFIED_TYPE,
    validate_claim_link_relation,
    validate_claim_type,
)


def test_valid_type_passes_through_when_allowed_set_provided() -> None:
    allowed = frozenset({"population_estimate", "economic_use", UNCLASSIFIED_TYPE})
    assert validate_claim_type("population_estimate", allowed=allowed) == "population_estimate"
    assert validate_claim_type("economic_use", allowed=allowed) == "economic_use"


def test_unclassified_type_passes_through() -> None:
    assert validate_claim_type(UNCLASSIFIED_TYPE) == UNCLASSIFIED_TYPE


def test_legacy_renames_applied_when_rename_map_provided() -> None:
    allowed = frozenset({"population_estimate", UNCLASSIFIED_TYPE})
    renames = {"wildlife_count": "population_estimate"}
    assert validate_claim_type("wildlife_count", allowed=allowed, renames=renames) == "population_estimate"


def test_unknown_type_coerces_to_unclassified() -> None:
    assert validate_claim_type("completely_made_up_type") == UNCLASSIFIED_TYPE
    assert validate_claim_type("") == UNCLASSIFIED_TYPE


def test_allowed_types_includes_fallback() -> None:
    assert UNCLASSIFIED_TYPE in ALLOWED_CLAIM_TYPES


def test_module_level_allowed_set_is_sentinel_only() -> None:
    """Module-level ALLOWED_CLAIM_TYPES is a sentinel; domain sets come from config."""
    assert ALLOWED_CLAIM_TYPES == frozenset({UNCLASSIFIED_TYPE})


def test_validate_claim_link_relation_accepts_new_roles_and_occurrence_location() -> None:
    for relation_type in CLAIM_ENTITY_RELATIONS | {CLAIM_LOCATION_RELATION}:
        assert validate_claim_link_relation(relation_type) == relation_type


def test_validate_claim_link_relation_rejects_about() -> None:
    assert validate_claim_link_relation("ABOUT") is None
