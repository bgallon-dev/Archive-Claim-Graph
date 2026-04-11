"""Mutable accumulator flowing through extraction pipeline phases.

``ExtractionState`` replaces the ~20 local variables that previously lived
inside ``extract_semantic()``.  Each phase function reads upstream fields and
writes its own outputs onto the same state object.  The terminal
``to_semantic_bundle()`` method assembles the immutable public output.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig
    from gemynd.core.models import (
        ClaimConceptLinkRecord,
        ClaimEntityLinkRecord,
        ClaimLinkDiagnosticRecord,
        ClaimLocationLinkRecord,
        ClaimPeriodLinkRecord,
        ClaimRecord,
        DocumentAnchorLinkRecord,
        DocumentPeriodLinkRecord,
        DocumentYearLinkRecord,
        EntityHierarchyLinkRecord,
        EntityRecord,
        EntityResolutionRecord,
        EventMeasurementLinkRecord,
        EventObservationLinkRecord,
        EventRecord,
        ExtractionRunRecord,
        MeasurementRecord,
        MentionRecord,
        ObservationMeasurementLinkRecord,
        ObservationRecord,
        SemanticBundle,
        StructureBundle,
        YearRecord,
    )
    from gemynd.ingest.derivation_context import DerivationContext
    from gemynd.ingest.extractors.claim_extractor import ClaimLinkDraft


@dataclass
class ExtractionState:
    """Mutable accumulator flowing through extraction pipeline phases.

    **Invariant:** ``entity_lookup`` must always mirror ``entities``.
    Use :meth:`register_entity` to maintain this.
    """

    # ── Inputs (set once at construction) ──────────────────────────────
    structure: StructureBundle
    config: DomainConfig
    extraction_run: ExtractionRunRecord

    # ── Phase 2: paragraph-level extraction ────────────────────────────
    claims: list[ClaimRecord] = field(default_factory=list)
    measurements: list[MeasurementRecord] = field(default_factory=list)
    mentions: list[MentionRecord] = field(default_factory=list)
    claim_links_by_claim: dict[str, list[ClaimLinkDraft]] = field(default_factory=lambda: defaultdict(list))
    paragraph_texts: dict[str, str] = field(default_factory=dict)
    mentions_by_paragraph: dict[str, list[MentionRecord]] = field(default_factory=lambda: defaultdict(list))
    claims_by_paragraph: dict[str, list[ClaimRecord]] = field(default_factory=lambda: defaultdict(list))
    claim_counter: int = 0
    measurement_counter: int = 0

    # ── Phase 3: entity resolution ─────────────────────────────────────
    entities: list[EntityRecord] = field(default_factory=list)
    entity_resolutions: list[EntityResolutionRecord] = field(default_factory=list)
    entity_lookup: dict[str, EntityRecord] = field(default_factory=dict)
    resolutions_by_mention: dict[str, EntityResolutionRecord] = field(default_factory=dict)

    # ── Phase 4: domain anchor ─────────────────────────────────────────
    doc_anchor_id: str | None = None

    # ── Phase 5: claim-link resolution ─────────────────────────────────
    claim_entity_links: list[ClaimEntityLinkRecord] = field(default_factory=list)
    claim_link_diagnostics: list[ClaimLinkDiagnosticRecord] = field(default_factory=list)
    claim_location_links: list[ClaimLocationLinkRecord] = field(default_factory=list)

    # ── Phase 6: period ────────────────────────────────────────────────
    claim_period_links: list[ClaimPeriodLinkRecord] = field(default_factory=list)
    document_anchor_links: list[DocumentAnchorLinkRecord] = field(default_factory=list)
    document_period_links: list[DocumentPeriodLinkRecord] = field(default_factory=list)

    # ── Phases 7–9: derivation, observations, events ───────────────────
    derivation_contexts: list[DerivationContext] = field(default_factory=list)
    observations: list[ObservationRecord] = field(default_factory=list)
    years: list[YearRecord] = field(default_factory=list)
    obs_measurement_links: list[ObservationMeasurementLinkRecord] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    event_obs_links: list[EventObservationLinkRecord] = field(default_factory=list)
    event_meas_links: list[EventMeasurementLinkRecord] = field(default_factory=list)

    # ── Phases 10–12: year entities, entity hierarchy, concepts ────────
    document_year_links: list[DocumentYearLinkRecord] = field(default_factory=list)
    entity_hierarchy_links: list[EntityHierarchyLinkRecord] = field(default_factory=list)
    claim_concept_links: list[ClaimConceptLinkRecord] = field(default_factory=list)

    # ── Helpers ────────────────────────────────────────────────────────

    def register_entity(self, entity: EntityRecord) -> None:
        """Append *entity* to ``entities`` and update ``entity_lookup``."""
        self.entities.append(entity)
        self.entity_lookup[entity.entity_id] = entity

    def to_semantic_bundle(self) -> SemanticBundle:
        """Assemble the immutable public output."""
        from gemynd.core.models import SemanticBundle

        return SemanticBundle(
            extraction_run=self.extraction_run,
            claims=self.claims,
            measurements=self.measurements,
            mentions=self.mentions,
            entities=self.entities,
            entity_resolutions=self.entity_resolutions,
            claim_entity_links=self.claim_entity_links,
            claim_link_diagnostics=self.claim_link_diagnostics,
            claim_location_links=self.claim_location_links,
            claim_period_links=self.claim_period_links,
            document_anchor_links=self.document_anchor_links,
            document_period_links=self.document_period_links,
            document_signed_by_links=[],
            person_affiliation_links=[],
            observations=self.observations,
            years=self.years,
            observation_measurement_links=self.obs_measurement_links,
            document_year_links=self.document_year_links,
            entity_hierarchy_links=self.entity_hierarchy_links,
            events=self.events,
            event_observation_links=self.event_obs_links,
            event_measurement_links=self.event_meas_links,
            claim_concept_links=self.claim_concept_links,
        )
