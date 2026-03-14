from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

CONTRACT_VERSION: str = "v2"
UNCLASSIFIED_TYPE: str = "unclassified_assertion"
LEGACY_ABOUT_RELATION: str = "ABOUT"
CLAIM_LOCATION_RELATION: str = "OCCURRED_AT"

CLAIM_ENTITY_RELATION_PRECEDENCE: tuple[str, ...] = (
    "SPECIES_FOCUS",
    "HABITAT_FOCUS",
    "METHOD_FOCUS",
    "MANAGEMENT_TARGET",
    "LOCATION_FOCUS",
    "SUBJECT_OF_CLAIM",
    "TOPIC_OF_CLAIM",
)

CLAIM_ENTITY_RELATIONS: frozenset[str] = frozenset(CLAIM_ENTITY_RELATION_PRECEDENCE)
EXTRACTOR_CLAIM_LINK_RELATIONS: frozenset[str] = frozenset(CLAIM_ENTITY_RELATIONS | {CLAIM_LOCATION_RELATION})
ALL_CLAIM_LINK_RELATIONS: frozenset[str] = frozenset(EXTRACTOR_CLAIM_LINK_RELATIONS | {LEGACY_ABOUT_RELATION})

RELATION_TO_ENTITY_TYPE_HINTS: dict[str, frozenset[str]] = {
    "SPECIES_FOCUS": frozenset({"Species"}),
    "HABITAT_FOCUS": frozenset({"Habitat"}),
    "METHOD_FOCUS": frozenset({"SurveyMethod"}),
    "MANAGEMENT_TARGET": frozenset({"Activity", "Species"}),
    "LOCATION_FOCUS": frozenset({"Place", "Refuge"}),
    "SUBJECT_OF_CLAIM": frozenset({"Person", "Organization", "Place", "Refuge"}),
    "TOPIC_OF_CLAIM": frozenset({"Activity", "Person", "Organization", "Place", "Refuge"}),
    CLAIM_LOCATION_RELATION: frozenset({"Place", "Refuge"}),
}

# All claim types recognised by the pipeline, including the coercion fallback.
ALLOWED_CLAIM_TYPES: frozenset[str] = frozenset({
    "population_estimate",
    "species_presence",
    "species_absence",
    "breeding_activity",
    "habitat_condition",
    "management_action",
    "migration_timing",
    "weather_observation",
    "predator_control",
    "fire_incident",
    "economic_use",
    "development_activity",
    "public_contact",
    UNCLASSIFIED_TYPE,
})

# Claim types that produce an Observation node during semantic extraction.
OBSERVATION_ELIGIBLE_TYPES: frozenset[str] = frozenset({
    "population_estimate",
    "species_presence",
    "species_absence",
    "breeding_activity",
    "habitat_condition",
    "migration_timing",
    "weather_observation",
    "predator_control",
    # Legacy name kept for backward compat with older serialized bundles.
    "wildlife_count",
})

# Deterministic mapping from claim_type to observation_type.
CLAIM_TYPE_TO_OBSERVATION_TYPE: dict[str, str] = {
    "population_estimate": "population_count",
    "wildlife_count": "population_count",  # legacy compat
    "species_presence": "presence_record",
    "species_absence": "presence_record",
    "breeding_activity": "nesting_record",
    "migration_timing": "migration_record",
    "weather_observation": "weather_record",
    "habitat_condition": "habitat_record",
    "predator_control": "predator_record",
}

# Deterministic mapping from claim_type to event_type (the Neo4j label).
CLAIM_TYPE_TO_EVENT_TYPE: dict[str, str] = {
    "population_estimate":  "SurveyEvent",
    "species_presence":     "SurveyEvent",
    "species_absence":      "SurveyEvent",
    "breeding_activity":    "BreedingEvent",
    "migration_timing":     "MigrationEvent",
    "fire_incident":        "FireEvent",
    "predator_control":     "ManagementEvent",
    "management_action":    "ManagementEvent",
    "habitat_condition":    "HabitatEvent",
    "weather_observation":  "WeatherEvent",
    "economic_use":         "ManagementEvent",
    "development_activity": "ManagementEvent",
    "public_contact":       "ManagementEvent",
    # "unclassified_assertion" intentionally excluded — not a real event
}

# All claim types that should produce an Event node.
EVENT_ELIGIBLE_TYPES: frozenset[str] = frozenset(CLAIM_TYPE_TO_EVENT_TYPE.keys())

# Legacy rename map: old extractor output → current canonical type.
_LEGACY_RENAMES: dict[str, str] = {
    "wildlife_count": "population_estimate",
    "weather_condition": "weather_observation",
}


def validate_claim_type(claim_type: str) -> str:
    """Return a valid claim type, coercing unknown types to UNCLASSIFIED_TYPE."""
    if claim_type in ALLOWED_CLAIM_TYPES:
        return claim_type
    renamed = _LEGACY_RENAMES.get(claim_type)
    if renamed:
        return renamed
    return UNCLASSIFIED_TYPE


def validate_claim_link_relation(relation_type: str, *, allow_legacy: bool = False) -> str | None:
    cleaned = relation_type.strip().upper()
    if cleaned in EXTRACTOR_CLAIM_LINK_RELATIONS:
        return cleaned
    if allow_legacy and cleaned == LEGACY_ABOUT_RELATION:
        return cleaned
    return None


def claim_relation_priority(relation_type: str) -> int:
    try:
        return CLAIM_ENTITY_RELATION_PRECEDENCE.index(relation_type)
    except ValueError:
        return len(CLAIM_ENTITY_RELATION_PRECEDENCE)


def entity_type_allowed_for_relation(relation_type: str, entity_type: str | None) -> bool:
    if entity_type is None:
        return True
    allowed = RELATION_TO_ENTITY_TYPE_HINTS.get(relation_type)
    if not allowed:
        return True
    return entity_type in allowed


@lru_cache(maxsize=1)
def _load_compatibility_data() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Load and cache claim_relation_compatibility.yaml."""
    from .resource_loader import load_claim_relation_compatibility
    data = load_claim_relation_compatibility()
    compatibility: dict[str, dict[str, str]] = data.get("compatibility") or {}
    preferred: dict[str, list[str]] = data.get("preferred_entity_types") or {}
    return compatibility, preferred


def get_relation_compatibility(claim_type: str, relation_type: str) -> str:
    """Return 'strong', 'weak', or 'forbidden' for a claim_type × relation_type pair.

    Defaults to 'weak' for any unlisted pair so new relation types degrade
    gracefully rather than silently passing as strong.
    """
    compatibility, _ = _load_compatibility_data()
    return compatibility.get(claim_type, {}).get(relation_type, "weak")


def get_preferred_entity_types(claim_type: str) -> list[str]:
    """Return the ordered list of preferred entity types for a claim_type.

    Earlier positions are more preferred.  Returns an empty list when no
    preference is defined (caller should apply no reranking).
    """
    _, preferred = _load_compatibility_data()
    return list(preferred.get(claim_type, []))
