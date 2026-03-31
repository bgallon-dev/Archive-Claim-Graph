"""Unified domain configuration object.

``DomainConfig`` aggregates all domain resources (seed entities, patterns,
measurement units, OCR corrections, derivation registry, etc.) into a single
dataclass that can be loaded once and threaded through the extraction pipeline.

``ClaimDerivationSpec`` is a frozen typed wrapper for derivation registry
entries, replacing raw dicts produced by ``load_derivation_registry()``.

Usage::

    from graphrag_pipeline.core.domain_config import load_domain_config

    config = load_domain_config(resources_dir)
    extractor = RuleBasedClaimExtractor(config=config)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from graphrag_pipeline.core.models import EntityRecord

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClaimDerivationSpec:
    """Typed, frozen representation of one derivation registry entry.

    Replaces the raw ``dict`` returned by ``load_derivation_registry()`` so
    that ``build_derivation_contexts()`` can accept either form.
    """

    observation_type: str | None
    event_type: str | None
    required_entities: tuple[str, ...]
    optional_entities: tuple[str, ...]


@dataclass(slots=True)
class DomainConfig:
    """All domain resources bundled into one place.

    Loaded by :func:`load_domain_config` and optionally passed to extractor
    and resolver constructors.  When ``config`` is provided to a constructor
    it replaces the module-level cached loaders for that instance.
    """

    seed_entities: list[EntityRecord]
    claim_type_patterns: list[tuple[str, re.Pattern, float]]
    claim_role_policy: dict[tuple[str, str], str]
    measurement_units: dict[str, tuple[str, str]]
    measurement_species: dict[str, Any]
    ocr_corrections: frozenset[str]
    ocr_correction_map: dict[str, str]
    negative_lexicon: frozenset[str]
    preferred_entity_types: dict[str, list[str]]
    compatibility_matrix: dict[str, dict[str, str]]
    derivation_registry: dict[str, ClaimDerivationSpec]
    domain_anchor: dict[str, Any] | None
    year_validation: dict[str, Any] | None
    synthesis_context: str

    # Phase 5 — externalized domain schema fields
    claim_entity_relation_precedence: tuple[str, ...]
    claim_entity_relations: frozenset[str]
    relation_to_entity_type_hints: dict[str, frozenset[str]]
    claim_location_relation: str
    entity_labels: frozenset[str]
    legacy_renames: dict[str, str]
    allowed_claim_types: frozenset[str]
    observation_eligible_types: frozenset[str]
    event_eligible_types: frozenset[str]
    concept_rules: list[tuple[str, frozenset[str], re.Pattern, float]]
    query_intent_to_claim_types: dict[str, list[str]]

    @property
    def claim_entity_relation_cypher(self) -> str:
        """Cypher IN-list literal for claim-entity relation types."""
        return ", ".join(f"'{r}'" for r in self.claim_entity_relation_precedence)

    @property
    def extractor_claim_link_relations(self) -> frozenset[str]:
        """All relation types valid in extractor claim links."""
        return frozenset(self.claim_entity_relations | {self.claim_location_relation})


def load_domain_config(resources_dir: Path | None = None) -> DomainConfig:
    """Load every domain resource file and return a validated :class:`DomainConfig`.

    All imports are deferred to avoid circular-import issues at module load
    time.  The function is safe to call multiple times; caching at call-site
    is left to the caller.
    """
    from graphrag_pipeline.core.resolver import default_seed_entities
    from graphrag_pipeline.shared.resource_loader import (
        load_claim_relation_compatibility,
        load_claim_role_policy,
        load_claim_type_patterns,
        load_concept_rules,
        load_derivation_registry,
        load_domain_profile,
        load_domain_schema,
        load_measurement_species,
        load_measurement_units,
        load_negative_entities,
        load_ocr_correction_map,
        load_ocr_corrections,
        load_query_intent,
    )

    profile = load_domain_profile(resources_dir)
    compat = load_claim_relation_compatibility(resources_dir)

    raw_registry = load_derivation_registry(resources_dir)
    registry: dict[str, ClaimDerivationSpec] = {
        ct: ClaimDerivationSpec(
            observation_type=entry.get("observation_type"),
            event_type=entry.get("event_type"),
            required_entities=tuple(entry.get("required_entities") or []),
            optional_entities=tuple(entry.get("optional_entities") or []),
        )
        for ct, entry in raw_registry.items()
    }

    # Domain schema: graph vocabulary (relation types, entity labels, etc.)
    schema = load_domain_schema(resources_dir)
    claim_type_patterns = load_claim_type_patterns(resources_dir)

    relation_entries = schema.get("claim_entity_relations") or []
    relation_precedence = tuple(e["relation"] for e in relation_entries)
    relation_hints: dict[str, frozenset[str]] = {
        e["relation"]: frozenset(e.get("entity_types") or [])
        for e in relation_entries
    }
    schema_location_rel = schema.get("claim_location_relation", "OCCURRED_AT")
    location_entity_types = schema.get("claim_location_entity_types") or []
    if location_entity_types:
        relation_hints[schema_location_rel] = frozenset(location_entity_types)

    entity_labels = frozenset(schema.get("entity_labels") or [])
    legacy_renames = schema.get("legacy_renames") or {}

    # Derive allowed_claim_types from patterns + registry + unclassified sentinel.
    from graphrag_pipeline.core.claim_contract import UNCLASSIFIED_TYPE
    pattern_types = {ct for ct, _, _ in claim_type_patterns}
    registry_types = set(registry.keys())
    allowed_claim_types = frozenset(
        pattern_types | registry_types | set(legacy_renames.values()) | {UNCLASSIFIED_TYPE}
    )

    # Derive eligible types from derivation registry.
    obs_eligible = frozenset(
        ct for ct, spec in registry.items() if spec.observation_type
    )
    evt_eligible = frozenset(
        ct for ct, spec in registry.items() if spec.event_type
    )

    config = DomainConfig(
        seed_entities=default_seed_entities(resources_dir),
        claim_type_patterns=claim_type_patterns,
        claim_role_policy=load_claim_role_policy(resources_dir),
        measurement_units=load_measurement_units(resources_dir),
        measurement_species=load_measurement_species(resources_dir),
        ocr_corrections=load_ocr_corrections(resources_dir),
        ocr_correction_map=load_ocr_correction_map(resources_dir),
        negative_lexicon=load_negative_entities(resources_dir),
        preferred_entity_types=compat.get("preferred_entity_types", {}),
        compatibility_matrix=compat.get("compatibility", {}),
        derivation_registry=registry,
        domain_anchor=profile.get("document_anchor"),
        year_validation=profile.get("year_validation"),
        synthesis_context=profile.get("synthesis_context", ""),
        claim_entity_relation_precedence=relation_precedence,
        claim_entity_relations=frozenset(relation_precedence),
        relation_to_entity_type_hints=relation_hints,
        claim_location_relation=schema_location_rel,
        entity_labels=entity_labels,
        legacy_renames=legacy_renames,
        allowed_claim_types=allowed_claim_types,
        observation_eligible_types=obs_eligible,
        event_eligible_types=evt_eligible,
        concept_rules=load_concept_rules(resources_dir),
        query_intent_to_claim_types=load_query_intent(resources_dir),
    )
    _validate_config(config)
    return config


def _validate_config(config: DomainConfig) -> None:
    """Emit warnings for cross-resource inconsistencies.  Never raises."""
    from graphrag_pipeline.core.claim_contract import UNCLASSIFIED_TYPE

    pattern_claim_types = {ct for ct, _, _ in config.claim_type_patterns}
    seed_entity_types = {e.entity_type for e in config.seed_entities}

    # Legacy renames and the unclassified fallback intentionally lack patterns.
    _skip = set(config.legacy_renames.keys()) | {UNCLASSIFIED_TYPE}

    for ct in config.derivation_registry:
        if ct not in pattern_claim_types and ct not in _skip:
            _log.warning(
                "derivation_registry entry %r has no matching claim_type_pattern", ct
            )

    for ct, types in config.preferred_entity_types.items():
        for et in types:
            if et not in seed_entity_types:
                _log.warning(
                    "preferred_entity_types[%r] references unknown entity type %r", ct, et
                )

    for (ct, _et) in config.claim_role_policy:
        if ct not in pattern_claim_types:
            _log.warning(
                "claim_role_policy entry %r has no matching claim_type_pattern", ct
            )
