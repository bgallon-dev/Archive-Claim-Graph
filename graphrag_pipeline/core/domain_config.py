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
        load_derivation_registry,
        load_domain_profile,
        load_measurement_species,
        load_measurement_units,
        load_negative_entities,
        load_ocr_correction_map,
        load_ocr_corrections,
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

    config = DomainConfig(
        seed_entities=default_seed_entities(resources_dir),
        claim_type_patterns=load_claim_type_patterns(resources_dir),
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
    )
    _validate_config(config)
    return config


def _validate_config(config: DomainConfig) -> None:
    """Emit warnings for cross-resource inconsistencies.  Never raises."""
    pattern_claim_types = {ct for ct, _, _ in config.claim_type_patterns}
    seed_entity_types = {e.entity_type for e in config.seed_entities}

    for ct in config.derivation_registry:
        if ct not in pattern_claim_types:
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
