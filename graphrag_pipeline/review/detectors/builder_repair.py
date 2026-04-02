"""Builder repair detector.

Emits `relabel_claim_link`, `add_claim_entity_link`, `add_claim_location_link`,
or `exclude_claim_from_derivation` proposals for:
  - missing species focus
  - missing event location
  - method-focus overtrigger cases
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...core.claim_contract import (
    CLAIM_ENTITY_RELATIONS,
    EVENT_ELIGIBLE_TYPES,
    OBSERVATION_ELIGIBLE_TYPES,
    get_relation_compatibility,
)
from ...core.models import (
    ClaimEntityLinkRecord,
    ClaimLinkDiagnosticRecord,
    ClaimLocationLinkRecord,
    ClaimRecord,
    EntityRecord,
    SemanticBundle,
    StructureBundle,
)
from ..ids import make_claim_entity_link_target_id, make_claim_location_link_target_id
from ..models import DetectorProposal, ProposalTarget
from ..patch_spec import make_patch_spec

DETECTOR_NAME = "builder_repair_detector"
DETECTOR_VERSION = "v1"

# Claim types that strongly imply species context
_SPECIES_EXPECTING_TYPES = frozenset({
    "population_estimate", "species_presence", "species_absence",
    "breeding_activity", "migration_timing", "predator_control",
})

# Claim types that strongly imply location context
_LOCATION_EXPECTING_TYPES = frozenset({
    "population_estimate", "species_presence", "species_absence",
    "breeding_activity", "fire_incident", "habitat_condition",
    "management_action", "predator_control",
})

# Claim types where METHOD_FOCUS is weak or forbidden
_METHOD_OVERTRIGGER_TYPES = frozenset({
    "fire_incident", "public_contact", "economic_use",
    "development_activity",
})


def _claim_has_relation(
    claim_id: str,
    relation_type: str,
    links: list[ClaimEntityLinkRecord],
) -> bool:
    return any(
        link.claim_id == claim_id and link.relation_type == relation_type
        for link in links
    )


def _claim_has_location(
    claim_id: str,
    location_links: list[ClaimLocationLinkRecord],
) -> bool:
    return any(link.claim_id == claim_id for link in location_links)


def _find_species_candidates(
    claim: ClaimRecord,
    entities: dict[str, EntityRecord],
    diagnostics_by_claim: dict[str, list[ClaimLinkDiagnosticRecord]],
) -> list[EntityRecord]:
    """Find species entities that were candidates but not linked."""
    candidates: list[EntityRecord] = []
    for diag in diagnostics_by_claim.get(claim.claim_id, []):
        if diag.entity_type_hint == "Species" or diag.relation_type == "SPECIES_FOCUS":
            # Look for species entities that match
            for entity in entities.values():
                if entity.entity_type == "Species" and entity.entity_id not in [
                    e.entity_id for e in candidates
                ]:
                    candidates.append(entity)
    return candidates


def _find_location_candidates(
    claim: ClaimRecord,
    entities: dict[str, EntityRecord],
    diagnostics_by_claim: dict[str, list[ClaimLinkDiagnosticRecord]],
) -> list[EntityRecord]:
    """Find place/refuge entities that could serve as location context."""
    candidates: list[EntityRecord] = []
    for entity in entities.values():
        if entity.entity_type in ("Place", "Refuge") and entity.entity_id not in [
            e.entity_id for e in candidates
        ]:
            candidates.append(entity)
    return candidates


def detect(
    structure: StructureBundle,
    semantic: SemanticBundle,
    snapshot_id: str,
) -> list[DetectorProposal]:
    """Run the builder repair detector."""
    entity_lookup = {e.entity_id: e for e in semantic.entities}
    paragraphs_by_id = {p.paragraph_id: p for p in structure.paragraphs}
    diagnostics_by_claim: dict[str, list[ClaimLinkDiagnosticRecord]] = defaultdict(list)
    for d in semantic.claim_link_diagnostics:
        diagnostics_by_claim[d.claim_id].append(d)

    links_by_claim: dict[str, list[ClaimEntityLinkRecord]] = defaultdict(list)
    for link in semantic.claim_entity_links:
        links_by_claim[link.claim_id].append(link)

    proposals: list[DetectorProposal] = []

    for claim in semantic.claims:
        claim_links = links_by_claim.get(claim.claim_id, [])
        claim_diags = diagnostics_by_claim.get(claim.claim_id, [])

        # --- Missing species focus ---
        if (
            claim.claim_type in _SPECIES_EXPECTING_TYPES
            and not _claim_has_relation(claim.claim_id, "SPECIES_FOCUS", semantic.claim_entity_links)
        ):
            # Check if there's a species entity available in the bundle
            species_candidates = _find_species_candidates(claim, entity_lookup, diagnostics_by_claim)
            if species_candidates:
                # Propose adding the most likely species link
                best_species = species_candidates[0]
                target_id = make_claim_entity_link_target_id(
                    claim.claim_id, "SPECIES_FOCUS", best_species.entity_id,
                )
                targets = [
                    ProposalTarget(
                        proposal_id="",
                        target_kind="claim",
                        target_id=claim.claim_id,
                        target_role="source_claim",
                        exists_in_snapshot=True,
                    ),
                    ProposalTarget(
                        proposal_id="",
                        target_kind="entity",
                        target_id=best_species.entity_id,
                        target_role="target_entity",
                        exists_in_snapshot=True,
                    ),
                    ProposalTarget(
                        proposal_id="",
                        target_kind="claim_entity_link",
                        target_id=target_id,
                        target_role="proposed_link",
                        exists_in_snapshot=False,
                    ),
                ]
                patch = make_patch_spec(
                    "add_claim_entity_link",
                    claim_id=claim.claim_id,
                    entity_id=best_species.entity_id,
                    relation_type="SPECIES_FOCUS",
                    evidence_basis="claim_link_diagnostic",
                )
                para = paragraphs_by_id.get(claim.paragraph_id)
                evidence = {
                    "confidence": 0.65,
                    "issue_class": "missing_species_focus",
                    "impact_size": 3,
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "source_sentence": claim.source_sentence,
                    "normalized_sentence": claim.normalized_sentence,
                    "candidate_species": best_species.to_dict(),
                    "diagnostics": [d.to_dict() for d in claim_diags],
                    "source_file": structure.document.source_file,
                    "paragraph_raw_ocr_text": (para.raw_ocr_text or "")[:2000] if para else "",
                    "paragraph_clean_text": (para.clean_text or "")[:2000] if para else "",
                }
                proposals.append(DetectorProposal(
                    anti_pattern_id="ap_missing_species",
                    issue_class="missing_species_focus",
                    proposal_type="add_claim_entity_link",
                    confidence=0.65,
                    targets=targets,
                    patch_spec=patch,
                    evidence_snapshot=evidence,
                    reasoning_summary={"reason": f"Claim '{claim.claim_id}' ({claim.claim_type}) lacks SPECIES_FOCUS; '{best_species.name}' is a candidate"},
                    detector_name=DETECTOR_NAME,
                    detector_version=DETECTOR_VERSION,
                ))

        # --- Missing event location ---
        if (
            claim.claim_type in _LOCATION_EXPECTING_TYPES
            and not _claim_has_location(claim.claim_id, semantic.claim_location_links)
        ):
            location_candidates = _find_location_candidates(claim, entity_lookup, diagnostics_by_claim)
            if location_candidates:
                best_location = location_candidates[0]
                target_id = make_claim_location_link_target_id(
                    claim.claim_id, best_location.entity_id,
                )
                targets = [
                    ProposalTarget(
                        proposal_id="",
                        target_kind="claim",
                        target_id=claim.claim_id,
                        target_role="source_claim",
                        exists_in_snapshot=True,
                    ),
                    ProposalTarget(
                        proposal_id="",
                        target_kind="entity",
                        target_id=best_location.entity_id,
                        target_role="target_entity",
                        exists_in_snapshot=True,
                    ),
                    ProposalTarget(
                        proposal_id="",
                        target_kind="claim_location_link",
                        target_id=target_id,
                        target_role="proposed_link",
                        exists_in_snapshot=False,
                    ),
                ]
                patch = make_patch_spec(
                    "add_claim_location_link",
                    claim_id=claim.claim_id,
                    entity_id=best_location.entity_id,
                    relation_type="OCCURRED_AT",
                    evidence_basis="document_context",
                )
                para = paragraphs_by_id.get(claim.paragraph_id)
                evidence = {
                    "confidence": 0.60,
                    "issue_class": "missing_event_location",
                    "impact_size": 3,
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "source_sentence": claim.source_sentence,
                    "normalized_sentence": claim.normalized_sentence,
                    "candidate_location": best_location.to_dict(),
                    "source_file": structure.document.source_file,
                    "paragraph_raw_ocr_text": (para.raw_ocr_text or "")[:2000] if para else "",
                    "paragraph_clean_text": (para.clean_text or "")[:2000] if para else "",
                }
                proposals.append(DetectorProposal(
                    anti_pattern_id="ap_missing_location",
                    issue_class="missing_event_location",
                    proposal_type="add_claim_location_link",
                    confidence=0.60,
                    targets=targets,
                    patch_spec=patch,
                    evidence_snapshot=evidence,
                    reasoning_summary={"reason": f"Claim '{claim.claim_id}' ({claim.claim_type}) lacks OCCURRED_AT; '{best_location.name}' is a candidate"},
                    detector_name=DETECTOR_NAME,
                    detector_version=DETECTOR_VERSION,
                ))

        # --- Method overtrigger ---
        method_links = [
            link for link in claim_links
            if link.relation_type == "METHOD_FOCUS"
        ]
        if method_links and claim.claim_type in _METHOD_OVERTRIGGER_TYPES:
            compatibility = get_relation_compatibility(claim.claim_type, "METHOD_FOCUS")
            if compatibility in ("weak", "forbidden"):
                for method_link in method_links:
                    method_entity = entity_lookup.get(method_link.entity_id)
                    targets = [
                        ProposalTarget(
                            proposal_id="",
                            target_kind="claim",
                            target_id=claim.claim_id,
                            target_role="source_claim",
                            exists_in_snapshot=True,
                        ),
                        ProposalTarget(
                            proposal_id="",
                            target_kind="entity",
                            target_id=method_link.entity_id,
                            target_role="target_entity",
                            exists_in_snapshot=True,
                        ),
                    ]
                    patch = make_patch_spec(
                        "exclude_claim_from_derivation",
                        claim_id=claim.claim_id,
                        derivation_kind="observation",
                        reason="method_overtrigger",
                    )
                    para = paragraphs_by_id.get(claim.paragraph_id)
                    evidence = {
                        "confidence": 0.80,
                        "issue_class": "method_overtrigger",
                        "impact_size": 2,
                        "claim_id": claim.claim_id,
                        "claim_type": claim.claim_type,
                        "relation_type": "METHOD_FOCUS",
                        "compatibility": compatibility,
                        "method_entity": method_entity.to_dict() if method_entity else {},
                        "source_sentence": claim.source_sentence,
                        "normalized_sentence": claim.normalized_sentence,
                        "source_file": structure.document.source_file,
                        "paragraph_raw_ocr_text": (para.raw_ocr_text or "")[:2000] if para else "",
                        "paragraph_clean_text": (para.clean_text or "")[:2000] if para else "",
                    }
                    proposals.append(DetectorProposal(
                        anti_pattern_id="ap_method_overtrigger",
                        issue_class="method_overtrigger",
                        proposal_type="exclude_claim_from_derivation",
                        confidence=0.80,
                        targets=targets,
                        patch_spec=patch,
                        evidence_snapshot=evidence,
                        reasoning_summary={"reason": f"METHOD_FOCUS on '{claim.claim_type}' is {compatibility}; claim should be excluded from observation derivation"},
                        detector_name=DETECTOR_NAME,
                        detector_version=DETECTOR_VERSION,
                    ))

    return proposals
