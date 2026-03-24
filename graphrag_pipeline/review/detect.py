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


def _compute_priority_score(confidence: float, impact_size: int) -> float:
    """priority_score = min(1.0, 0.7 * confidence + 0.3 * min(1.0, impact_size / 10.0))"""
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
        priority = _compute_priority_score(dp.confidence, impact_size)

        # Resolve anti_pattern_id
        anti_pattern_id = dp.anti_pattern_id or ISSUE_CLASS_TO_ANTI_PATTERN.get(dp.issue_class, "")

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
            status="queued",
            confidence=dp.confidence,
            priority_score=priority,
            impact_size=impact_size,
            created_at=now,
            updated_at=now,
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
        "rejected_details": rejected,
        "counts_by_queue": store.proposal_counts_by_queue(snapshot_id),
        "counts_by_status": store.proposal_counts_by_status(snapshot_id),
    }
