from __future__ import annotations

from functools import lru_cache

CONTRACT_VERSION: str = "v2"
UNCLASSIFIED_TYPE: str = "unclassified_assertion"
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

# Minimal fallback allow-list used only when a caller fails to pass the
# DomainConfig-derived set. The canonical allowed set lives on
# ``DomainConfig.allowed_claim_types`` and is derived from the derivation
# registry at load time.
ALLOWED_CLAIM_TYPES: frozenset[str] = frozenset({UNCLASSIFIED_TYPE})

# Observation/event eligibility and claim_type → derivation-type mappings are
# loaded per domain from derivation_registry.yaml via DomainConfig. No
# module-level fallback — callers must route through DerivationContext.

# Legacy rename map: old extractor output → current canonical type.
# Empty by default; domain-specific renames come from DomainConfig.legacy_renames.
_LEGACY_RENAMES: dict[str, str] = {}


def validate_claim_type(
    claim_type: str,
    *,
    allowed: frozenset[str] | None = None,
    renames: dict[str, str] | None = None,
) -> str:
    """Return a valid claim type, coercing unknown types to UNCLASSIFIED_TYPE.

    When *allowed* and *renames* are provided (from DomainConfig), they
    replace the module-level constants.
    """
    _allowed = allowed if allowed is not None else ALLOWED_CLAIM_TYPES
    _renames = renames if renames is not None else _LEGACY_RENAMES
    if claim_type in _allowed:
        return claim_type
    renamed = _renames.get(claim_type)
    if renamed:
        return renamed
    return UNCLASSIFIED_TYPE


def validate_claim_link_relation(
    relation_type: str,
    *,
    valid_relations: frozenset[str] | None = None,
) -> str | None:
    """Return cleaned relation type if valid, else None.

    When *valid_relations* is provided (from DomainConfig), it replaces
    the module-level EXTRACTOR_CLAIM_LINK_RELATIONS.
    """
    cleaned = relation_type.strip().upper()
    _valid = valid_relations if valid_relations is not None else EXTRACTOR_CLAIM_LINK_RELATIONS
    if cleaned in _valid:
        return cleaned
    return None


def claim_relation_priority(
    relation_type: str,
    *,
    precedence: tuple[str, ...] | None = None,
) -> int:
    """Return the priority index for a relation type (lower = higher priority).

    When *precedence* is provided (from DomainConfig), it replaces the
    module-level CLAIM_ENTITY_RELATION_PRECEDENCE.
    """
    _prec = precedence if precedence is not None else CLAIM_ENTITY_RELATION_PRECEDENCE
    try:
        return _prec.index(relation_type)
    except ValueError:
        return len(_prec)


def entity_type_allowed_for_relation(
    relation_type: str,
    entity_type: str | None,
    *,
    hints: dict[str, frozenset[str]] | None = None,
) -> bool:
    """Check whether an entity type is valid for a given relation type.

    When *hints* is provided (from DomainConfig), it replaces the
    module-level RELATION_TO_ENTITY_TYPE_HINTS.
    """
    if entity_type is None:
        return True
    _hints = hints if hints is not None else RELATION_TO_ENTITY_TYPE_HINTS
    allowed = _hints.get(relation_type)
    if not allowed:
        return True
    return entity_type in allowed


@lru_cache(maxsize=1)
def _load_compatibility_data() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Load and cache claim_relation_compatibility.yaml."""
    from gemynd.shared.resource_loader import load_claim_relation_compatibility
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
