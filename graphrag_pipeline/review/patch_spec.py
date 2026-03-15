"""Typed patch spec mini-language – schema definitions and validation."""
from __future__ import annotations

from typing import Any

from .models import (
    DERIVATION_KINDS,
    EVIDENCE_BASIS_VALUES,
    PROPOSAL_TYPES,
    SUPPRESSION_REASONS,
    SUPPRESSION_SCOPES,
)

PATCH_SPEC_SCHEMA_VERSION = "v1"

# Allowed claim-entity relation types for link operations
ALLOWED_CLAIM_ENTITY_RELATIONS: frozenset[str] = frozenset({
    "SPECIES_FOCUS",
    "HABITAT_FOCUS",
    "METHOD_FOCUS",
    "MANAGEMENT_TARGET",
    "LOCATION_FOCUS",
    "SUBJECT_OF_CLAIM",
    "TOPIC_OF_CLAIM",
})

ALIAS_MODES: frozenset[str] = frozenset({"preserve_aliases", "none"})

# Allowed exclusion reasons (detector-owned bounded enum)
EXCLUSION_REASONS: frozenset[str] = frozenset({
    "method_overtrigger",
    "missing_species_focus",
    "missing_event_location",
    "boilerplate_claim",
    "header_artifact",
})


class PatchSpecValidationError(Exception):
    """Raised when a patch_spec fails schema validation."""


def _require_keys(spec: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(spec.keys())
    if missing:
        raise PatchSpecValidationError(f"{label}: missing required keys {sorted(missing)}")


def _reject_unknown_keys(spec: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(spec.keys()) - allowed
    if unknown:
        raise PatchSpecValidationError(f"{label}: unknown keys {sorted(unknown)}")


def _require_type(spec: dict[str, Any], key: str, expected_type: type, label: str) -> None:
    if key in spec and not isinstance(spec[key], expected_type):
        raise PatchSpecValidationError(
            f"{label}: '{key}' must be {expected_type.__name__}, got {type(spec[key]).__name__}"
        )


def _require_enum(spec: dict[str, Any], key: str, allowed: frozenset[str], label: str) -> None:
    if key in spec and spec[key] not in allowed:
        raise PatchSpecValidationError(
            f"{label}: '{key}' must be one of {sorted(allowed)}, got {spec[key]!r}"
        )


def _require_sorted_unique_strings(spec: dict[str, Any], key: str, label: str) -> None:
    val = spec.get(key)
    if not isinstance(val, list):
        raise PatchSpecValidationError(f"{label}: '{key}' must be a list")
    if not all(isinstance(v, str) for v in val):
        raise PatchSpecValidationError(f"{label}: '{key}' must contain only strings")
    if val != sorted(val):
        raise PatchSpecValidationError(f"{label}: '{key}' must be sorted")
    if len(val) != len(set(val)):
        raise PatchSpecValidationError(f"{label}: '{key}' must contain unique values")


# ---------------------------------------------------------------------------
# Per-proposal-type validators
# ---------------------------------------------------------------------------

def _validate_merge_entities(spec: dict[str, Any]) -> None:
    label = "merge_entities"
    required = {"schema_version", "proposal_type", "canonical_entity_id", "merge_entity_ids", "canonical_name"}
    optional = {"alias_mode"}
    _require_keys(spec, required, label)
    _reject_unknown_keys(spec, required | optional, label)
    _require_type(spec, "canonical_entity_id", str, label)
    _require_type(spec, "canonical_name", str, label)
    _require_sorted_unique_strings(spec, "merge_entity_ids", label)
    if spec["canonical_entity_id"] in spec["merge_entity_ids"]:
        raise PatchSpecValidationError(f"{label}: canonical_entity_id must not be in merge_entity_ids")
    if "alias_mode" in spec:
        _require_enum(spec, "alias_mode", ALIAS_MODES, label)


def _validate_create_alias(spec: dict[str, Any]) -> None:
    label = "create_alias"
    required = {"schema_version", "proposal_type", "canonical_entity_id", "alias_entity_id", "canonical_name"}
    optional = {"alias_name"}
    _require_keys(spec, required, label)
    _reject_unknown_keys(spec, required | optional, label)
    _require_type(spec, "canonical_entity_id", str, label)
    _require_type(spec, "alias_entity_id", str, label)
    _require_type(spec, "canonical_name", str, label)
    if spec["canonical_entity_id"] == spec["alias_entity_id"]:
        raise PatchSpecValidationError(f"{label}: canonical_entity_id must differ from alias_entity_id")
    if "alias_name" in spec:
        _require_type(spec, "alias_name", str, label)


def _validate_suppress_mention(spec: dict[str, Any]) -> None:
    label = "suppress_mention"
    required = {"schema_version", "proposal_type", "mention_ids", "suppression_reason", "scope"}
    _require_keys(spec, required, label)
    _reject_unknown_keys(spec, required, label)
    _require_sorted_unique_strings(spec, "mention_ids", label)
    if not spec["mention_ids"]:
        raise PatchSpecValidationError(f"{label}: mention_ids must not be empty")
    _require_enum(spec, "suppression_reason", SUPPRESSION_REASONS, label)
    _require_enum(spec, "scope", SUPPRESSION_SCOPES, label)


def _validate_relabel_claim_link(spec: dict[str, Any]) -> None:
    label = "relabel_claim_link"
    required = {
        "schema_version", "proposal_type", "claim_id", "entity_id",
        "old_relation_type", "new_relation_type", "evidence_basis",
    }
    _require_keys(spec, required, label)
    _reject_unknown_keys(spec, required, label)
    _require_type(spec, "claim_id", str, label)
    _require_type(spec, "entity_id", str, label)
    _require_type(spec, "old_relation_type", str, label)
    _require_type(spec, "new_relation_type", str, label)
    if spec["old_relation_type"] == spec["new_relation_type"]:
        raise PatchSpecValidationError(f"{label}: old_relation_type must differ from new_relation_type")
    _require_enum(spec, "evidence_basis", EVIDENCE_BASIS_VALUES, label)


def _validate_add_claim_entity_link(spec: dict[str, Any]) -> None:
    label = "add_claim_entity_link"
    required = {
        "schema_version", "proposal_type", "claim_id", "entity_id",
        "relation_type", "evidence_basis",
    }
    _require_keys(spec, required, label)
    _reject_unknown_keys(spec, required, label)
    _require_type(spec, "claim_id", str, label)
    _require_type(spec, "entity_id", str, label)
    _require_enum(spec, "relation_type", ALLOWED_CLAIM_ENTITY_RELATIONS, label)
    _require_enum(spec, "evidence_basis", EVIDENCE_BASIS_VALUES, label)


def _validate_add_claim_location_link(spec: dict[str, Any]) -> None:
    label = "add_claim_location_link"
    required = {
        "schema_version", "proposal_type", "claim_id", "entity_id",
        "relation_type", "evidence_basis",
    }
    _require_keys(spec, required, label)
    _reject_unknown_keys(spec, required, label)
    _require_type(spec, "claim_id", str, label)
    _require_type(spec, "entity_id", str, label)
    if spec.get("relation_type") != "OCCURRED_AT":
        raise PatchSpecValidationError(f"{label}: relation_type must be 'OCCURRED_AT'")
    _require_enum(spec, "evidence_basis", EVIDENCE_BASIS_VALUES, label)


def _validate_exclude_claim_from_derivation(spec: dict[str, Any]) -> None:
    label = "exclude_claim_from_derivation"
    required = {"schema_version", "proposal_type", "claim_id", "derivation_kind", "reason"}
    _require_keys(spec, required, label)
    _reject_unknown_keys(spec, required, label)
    _require_type(spec, "claim_id", str, label)
    _require_enum(spec, "derivation_kind", DERIVATION_KINDS, label)
    _require_enum(spec, "reason", EXCLUSION_REASONS, label)


_VALIDATORS: dict[str, Any] = {
    "merge_entities": _validate_merge_entities,
    "create_alias": _validate_create_alias,
    "suppress_mention": _validate_suppress_mention,
    "relabel_claim_link": _validate_relabel_claim_link,
    "add_claim_entity_link": _validate_add_claim_entity_link,
    "add_claim_location_link": _validate_add_claim_location_link,
    "exclude_claim_from_derivation": _validate_exclude_claim_from_derivation,
}


def validate_patch_spec(spec: dict[str, Any]) -> None:
    """Validate a typed patch spec against its schema.

    Raises PatchSpecValidationError on any schema violation.
    """
    if not isinstance(spec, dict):
        raise PatchSpecValidationError("patch_spec must be a dict")

    if "schema_version" not in spec:
        raise PatchSpecValidationError("patch_spec missing required 'schema_version'")
    if spec["schema_version"] != PATCH_SPEC_SCHEMA_VERSION:
        raise PatchSpecValidationError(
            f"Unsupported schema_version {spec['schema_version']!r}; expected {PATCH_SPEC_SCHEMA_VERSION!r}"
        )

    proposal_type = spec.get("proposal_type")
    if proposal_type not in PROPOSAL_TYPES:
        raise PatchSpecValidationError(
            f"Unknown proposal_type {proposal_type!r}; allowed: {sorted(PROPOSAL_TYPES)}"
        )

    validator = _VALIDATORS.get(proposal_type)
    if validator:
        validator(spec)


def make_patch_spec(proposal_type: str, **kwargs: Any) -> dict[str, Any]:
    """Build a patch_spec dict with schema_version and proposal_type pre-filled."""
    spec = {"schema_version": PATCH_SPEC_SCHEMA_VERSION, "proposal_type": proposal_type}
    spec.update(kwargs)
    return spec
