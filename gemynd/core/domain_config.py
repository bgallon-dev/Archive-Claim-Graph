"""Unified domain configuration object.

``DomainConfig`` aggregates all domain resources (seed entities, patterns,
measurement units, OCR corrections, derivation registry, etc.) into a single
dataclass that can be loaded once and threaded through the extraction pipeline.

``ClaimDerivationSpec`` is a frozen typed wrapper for derivation registry
entries, replacing raw dicts produced by ``load_derivation_registry()``.

Usage::

    from gemynd.core.domain_config import load_domain_config

    config = load_domain_config(resources_dir)
    extractor = RuleBasedClaimExtractor(config=config)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gemynd.core.models import EntityRecord

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

    # Claim sentence validator — domain-specific finite-verb pattern
    validator_verb_re: re.Pattern[str] | None = None
    # Heading detection override — None means use default; set to a compiled
    # pattern to customise, or to a pattern that never matches to disable.
    validator_heading_re: re.Pattern[str] | None = None

    # Retrieval-layer fields
    institution_id: str = ""
    expected_claim_shares: dict[str, float] = field(default_factory=dict)

    @property
    def extraction_stopwords(self) -> frozenset[str]:
        """Domain-specific stopwords derived from document_anchor fields."""
        words: set[str] = set()
        if self.domain_anchor:
            for kw in self.domain_anchor.get("title_keywords", []):
                words.add(kw.lower())
            nf = self.domain_anchor.get("normalized_form", "")
            words.update(tok.lower() for tok in nf.split() if len(tok) > 2)
        return frozenset(words)

    @property
    def anchor_entity_id(self) -> str | None:
        """Entity ID of the primary anchor from seed_entities."""
        if not self.domain_anchor:
            return None
        anchor_type = self.domain_anchor.get("entity_type")
        anchor_norm = self.domain_anchor.get("normalized_form", "").lower()
        for e in self.seed_entities:
            if e.entity_type == anchor_type and anchor_norm in e.normalized_form:
                return e.entity_id
        return None

    @property
    def anchor_entity_type(self) -> str | None:
        """Entity type label for the domain anchor (e.g. 'Refuge')."""
        if self.domain_anchor:
            return self.domain_anchor.get("entity_type")
        return None

    @property
    def anchor_relation(self) -> str | None:
        """Document→anchor relation type (e.g. 'ABOUT_REFUGE').

        Read from ``document_anchor.relation`` in ``domain_profile.yaml``.
        ``None`` when the domain has no anchor or the relation is unset.
        """
        if self.domain_anchor:
            return self.domain_anchor.get("relation")
        return None

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
    from gemynd.core.resolver import default_seed_entities
    from gemynd.shared.resource_loader import (
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
        load_validator_verbs,
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
    from gemynd.core.claim_contract import UNCLASSIFIED_TYPE
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

    # Validator verb vocabulary
    verb_words = load_validator_verbs(resources_dir)
    validator_verb_re: re.Pattern[str] | None = None
    if verb_words:
        verb_alt = "|".join(re.escape(v) for v in verb_words)
        try:
            validator_verb_re = re.compile(rf"\b({verb_alt})\b", re.IGNORECASE)
        except re.error:
            _log.warning("validator_verbs produced invalid regex; using default")
            validator_verb_re = None

    # Heading detection override from domain_profile.yaml:
    #   heading_detection: false          → disable heading rejection entirely
    #   heading_detection: "^REGEX$"      → use custom regex
    #   heading_detection: (absent/true)  → use built-in default
    validator_heading_re: re.Pattern[str] | None = None
    heading_cfg = profile.get("heading_detection")
    if heading_cfg is False:
        # Explicitly disabled — use a pattern that never matches.
        validator_heading_re = re.compile(r"(?!)")  # negative lookahead: never matches
    elif isinstance(heading_cfg, str):
        try:
            validator_heading_re = re.compile(heading_cfg)
        except re.error:
            _log.warning("heading_detection regex is invalid; using default")
            validator_heading_re = None

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
        validator_verb_re=validator_verb_re,
        validator_heading_re=validator_heading_re,
        institution_id=profile.get("institution_id", ""),
        expected_claim_shares=profile.get("expected_claim_shares") or {},
    )
    _validate_config(config, resources_dir)
    return config


def _validate_config(
    config: DomainConfig,
    resources_dir: Path | None = None,
) -> list[str]:
    """Check cross-resource consistency.  Returns a list of issue strings.

    Every issue is also emitted as a warning log.  The function never raises.
    """
    from gemynd.core.claim_contract import UNCLASSIFIED_TYPE

    issues: list[str] = []

    def _warn(msg: str) -> None:
        _log.warning(msg)
        issues.append(msg)

    pattern_claim_types = {ct for ct, _, _ in config.claim_type_patterns}
    seed_entity_types = {e.entity_type for e in config.seed_entities}

    # Legacy renames and the unclassified fallback intentionally lack patterns.
    _skip = set(config.legacy_renames.keys()) | {UNCLASSIFIED_TYPE}

    # --- Existing checks ---

    for ct in config.derivation_registry:
        if ct not in pattern_claim_types and ct not in _skip:
            _warn(
                f"derivation_registry entry {ct!r} has no matching claim_type_pattern"
            )

    for ct, types in config.preferred_entity_types.items():
        for et in types:
            if et not in seed_entity_types:
                _warn(
                    f"preferred_entity_types[{ct!r}] references unknown entity type {et!r}"
                )

    for (ct, _et) in config.claim_role_policy:
        if ct not in pattern_claim_types:
            _warn(
                f"claim_role_policy entry {ct!r} has no matching claim_type_pattern"
            )

    if not config.institution_id:
        _warn("domain_profile is missing 'institution_id'")
    if not config.synthesis_context:
        _warn("domain_profile is missing 'synthesis_context'")

    # --- Onboarding checks ---

    # 1. Entity labels vs. seed entity types
    if config.entity_labels:
        labels_without_seeds = config.entity_labels - seed_entity_types
        # Period is auto-created during extraction, not seeded.
        labels_without_seeds -= {"Period"}
        for label in sorted(labels_without_seeds):
            _warn(
                f"entity_labels includes {label!r} but no seed entity has that type"
            )
        seeds_without_labels = seed_entity_types - config.entity_labels
        for et in sorted(seeds_without_labels):
            _warn(
                f"seed_entities contains type {et!r} not listed in entity_labels"
            )

    # 2. Derivation registry covers all pattern-defined claim types
    for ct in sorted(pattern_claim_types):
        if ct not in config.derivation_registry and ct not in _skip:
            _warn(
                f"claim_type_pattern {ct!r} has no derivation_registry entry"
            )

    # 3. Document anchor entity type exists in seed vocabulary
    if config.domain_anchor:
        anchor_type = config.domain_anchor.get("entity_type")
        if anchor_type and anchor_type not in seed_entity_types:
            _warn(
                f"document_anchor entity_type {anchor_type!r} not found in seed_entities"
            )
        if not config.domain_anchor.get("relation"):
            _warn(
                "document_anchor is set but 'relation' is missing — "
                "retrieval cannot build an anchor-scoped temporal query"
            )

    # 4. Sensitivity vocabulary file exists
    if resources_dir is not None:
        from gemynd.shared.resource_loader import _dir
        res = _dir(resources_dir)
        sens_cfg_path = res / "sensitivity_config.yaml"
        if sens_cfg_path.exists():
            try:
                import yaml
                with sens_cfg_path.open(encoding="utf-8") as fh:
                    sens_cfg = yaml.safe_load(fh) or {}
                vocab_file = (
                    sens_cfg.get("indigenous_sensitivity", {})
                    .get("vocabulary_file")
                )
                if vocab_file:
                    vocab_path = res / vocab_file
                    if not vocab_path.exists():
                        _warn(
                            f"sensitivity_config references vocabulary_file "
                            f"{vocab_file!r} but {vocab_path} does not exist"
                        )
            except Exception:
                pass  # YAML parse failure handled elsewhere

    return issues
