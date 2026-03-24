"""Detection orchestration – runs all detectors, validates, and upserts proposals."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import SemanticBundle, StructureBundle
from .detectors import ocr_entity, junk_mention, builder_repair, sensitivity_monitor
from .ids import (
    SNAPSHOT_SCHEMA_VERSION,
    file_sha256,
    make_proposal_id,
    make_review_run_id,
    make_revision_id,
    make_snapshot_id,
    patch_spec_fingerprint,
)
from .models import (
    DetectorProposal,
    Proposal,
    ProposalRevision,
    ProposalTarget,
    ReviewRun,
)
from .patch_spec import PatchSpecValidationError, validate_patch_spec
from .store import ISSUE_CLASS_TO_ANTI_PATTERN, ReviewStore

_AUTO_ACCEPT_THRESHOLD = 0.90
_AUTO_SUPPRESS_THRESHOLD = 0.50
_AUTO_ACCEPT_PROPOSAL_TYPE = "suppress_mention"
_AUTO_ACCEPT_REASONS: frozenset[str] = frozenset({"ocr_garbage", "short_generic_token"})

_LOW_IMPACT_SUPPRESS_CLASSES: frozenset[str] = frozenset({
    "header_contamination",
    "boilerplate_contamination",
})


def _compute_priority_score(
    confidence: float,
    impact_size: int,
    proposal_type: str = "",
    issue_class: str = "",
) -> float:
    """Compute priority_score weighted by archival impact of the proposal type.

    Ranking intent (highest to lowest):
      1. merge_entities / create_alias — a wrong merge has corpus-wide consequences.
         Formula: 0.5 * confidence + 0.5 * min(1.0, impact_size / 5.0)
      2. missing_species_focus / missing_event_location — retrieval completeness.
         Formula: base + 0.1 flat boost (capped at 1.0)
      3. All other types — default formula.
      4. suppress_mention for header/boilerplate — low consequence, deprioritised.
         Formula: 0.4 * confidence + 0.2 * min(1.0, impact_size / 10.0)  [max ≈ 0.6]
    """
    if proposal_type in ("merge_entities", "create_alias"):
        return min(1.0, 0.5 * confidence + 0.5 * min(1.0, impact_size / 5.0))
    if issue_class in ("missing_species_focus", "missing_event_location"):
        return min(1.0, 0.7 * confidence + 0.3 * min(1.0, impact_size / 10.0) + 0.1)
    if proposal_type == "suppress_mention" and issue_class in _LOW_IMPACT_SUPPRESS_CLASSES:
        return min(1.0, 0.4 * confidence + 0.2 * min(1.0, impact_size / 10.0))
    return min(1.0, 0.7 * confidence + 0.3 * min(1.0, impact_size / 10.0))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_and_filter(
    detector_proposals: list[DetectorProposal],
    snapshot_id: str,
) -> tuple[list[DetectorProposal], list[dict[str, Any]]]:
    """Validate each proposal's patch_spec. Return (valid, rejected_info)."""
    valid: list[DetectorProposal] = []
    rejected: list[dict[str, Any]] = []
    for dp in detector_proposals:
        try:
            validate_patch_spec(dp.patch_spec)
            valid.append(dp)
        except PatchSpecValidationError as e:
            rejected.append({
                "issue_class": dp.issue_class,
                "proposal_type": dp.proposal_type,
                "error": str(e),
            })
    return valid, rejected


def _route_proposal(dp: DetectorProposal) -> tuple[str, str]:
    """Return (status, review_tier) for a validated proposal."""
    if dp.confidence < _AUTO_SUPPRESS_THRESHOLD:
        return "rejected", "auto_suppressed"
    if (
        dp.confidence >= _AUTO_ACCEPT_THRESHOLD
        and dp.proposal_type == _AUTO_ACCEPT_PROPOSAL_TYPE
        and dp.patch_spec.get("suppression_reason") in _AUTO_ACCEPT_REASONS
    ):
        return "accepted_pending_apply", "auto_accepted"
    return "queued", "needs_review"


def run_detection(
    structure: StructureBundle,
    semantic: SemanticBundle,
    store: ReviewStore,
    structure_bundle_path: str,
    semantic_bundle_path: str,
) -> dict[str, Any]:
    """Run all three v1 detectors, validate proposals, and upsert into the store.

    Returns a summary dict with counts and any validation rejections.
    """
    now = _now_iso()

    # Compute snapshot
    struct_sha = file_sha256(structure_bundle_path)
    sem_sha = file_sha256(semantic_bundle_path)
    extraction_run_id = semantic.extraction_run.run_id
    doc_id = structure.document.doc_id

    snapshot_id = make_snapshot_id(
        doc_id, struct_sha, sem_sha,
        SNAPSHOT_SCHEMA_VERSION, extraction_run_id,
    )

    # Create review run
    review_run_id = make_review_run_id(snapshot_id, now)
    review_run = ReviewRun(
        review_run_id=review_run_id,
        snapshot_id=snapshot_id,
        doc_id=doc_id,
        structure_bundle_path=structure_bundle_path,
        semantic_bundle_path=semantic_bundle_path,
        structure_bundle_sha256=struct_sha,
        semantic_bundle_sha256=sem_sha,
        snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
        extraction_run_id=extraction_run_id,
        created_at=now,
    )
    store.save_review_run(review_run)

    # Run detectors
    all_detector_proposals: list[DetectorProposal] = []
    all_detector_proposals.extend(ocr_entity.detect(structure, semantic, snapshot_id))
    all_detector_proposals.extend(junk_mention.detect(structure, semantic, snapshot_id))
    all_detector_proposals.extend(builder_repair.detect(structure, semantic, snapshot_id))
    all_detector_proposals.extend(sensitivity_monitor.detect(structure, semantic, snapshot_id))

    # Validate
    valid_proposals, rejected = _validate_and_filter(all_detector_proposals, snapshot_id)

    # Upsert proposals
    upserted = 0
    tier_counts: dict[str, int] = {"auto_accepted": 0, "auto_suppressed": 0, "needs_review": 0}
    for dp in valid_proposals:
        # Compute deterministic IDs
        target_tuples = [
            (t.target_kind, t.target_id, t.target_role)
            for t in dp.targets
        ]
        proposal_id = make_proposal_id(
            snapshot_id, dp.issue_class, dp.proposal_type, target_tuples, dp.patch_spec,
        )
        impact_size = len(dp.targets)
        priority = _compute_priority_score(dp.confidence, impact_size, dp.proposal_type, dp.issue_class)

        # Resolve anti_pattern_id
        anti_pattern_id = dp.anti_pattern_id or ISSUE_CLASS_TO_ANTI_PATTERN.get(dp.issue_class, "")

        # Route to appropriate tier
        status, review_tier = _route_proposal(dp)
        tier_counts[review_tier] += 1

        # Build evidence snapshot with required fields
        evidence = dict(dp.evidence_snapshot)
        evidence.setdefault("confidence", dp.confidence)
        evidence.setdefault("priority_score", priority)
        evidence.setdefault("impact_size", impact_size)
        evidence.setdefault("issue_class", dp.issue_class)

        # Create proposal
        proposal = Proposal(
            proposal_id=proposal_id,
            review_run_id=review_run_id,
            snapshot_id=snapshot_id,
            anti_pattern_id=anti_pattern_id,
            issue_class=dp.issue_class,
            proposal_type=dp.proposal_type,
            status=status,
            confidence=dp.confidence,
            priority_score=priority,
            impact_size=impact_size,
            created_at=now,
            updated_at=now,
            review_tier=review_tier,
        )

        # Fill proposal_id on targets
        targets = []
        for t in dp.targets:
            targets.append(ProposalTarget(
                proposal_id=proposal_id,
                target_kind=t.target_kind,
                target_id=t.target_id,
                target_role=t.target_role,
                exists_in_snapshot=t.exists_in_snapshot,
            ))

        # Create initial revision
        revision_id = make_revision_id(proposal_id, 1)
        ps_fp = patch_spec_fingerprint(dp.patch_spec)

        revision = ProposalRevision(
            revision_id=revision_id,
            proposal_id=proposal_id,
            revision_number=1,
            revision_kind="generated",
            patch_spec_json=json.dumps(dp.patch_spec, ensure_ascii=True, sort_keys=True),
            patch_spec_fingerprint=ps_fp,
            evidence_snapshot_json=json.dumps(evidence, ensure_ascii=True),
            live_context_json="{}",
            reasoning_summary_json=json.dumps(dp.reasoning_summary, ensure_ascii=True),
            detector_name=dp.detector_name,
            detector_version=dp.detector_version,
            generator_type=dp.generator_type,
            llm_provider=dp.llm_provider,
            llm_model=dp.llm_model,
            validator_version="v1",
            validation_state="valid",
            created_by="system",
            created_at=now,
        )

        proposal.current_revision_id = revision_id
        store.upsert_proposal(proposal, targets)
        store.save_revision(revision)
        upserted += 1

    return {
        "review_run_id": review_run_id,
        "snapshot_id": snapshot_id,
        "doc_id": doc_id,
        "proposals_generated": len(all_detector_proposals),
        "proposals_valid": len(valid_proposals),
        "proposals_rejected": len(rejected),
        "proposals_upserted": upserted,
        "proposals_auto_accepted": tier_counts["auto_accepted"],
        "proposals_auto_suppressed": tier_counts["auto_suppressed"],
        "proposals_needs_review": tier_counts["needs_review"],
        "rejected_details": rejected,
        "counts_by_queue": store.proposal_counts_by_queue(snapshot_id),
        "counts_by_status": store.proposal_counts_by_status(snapshot_id),
        "counts_by_tier": store.proposal_counts_by_tier(snapshot_id),
    }
