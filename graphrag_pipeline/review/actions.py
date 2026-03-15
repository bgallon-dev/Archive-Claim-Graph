"""Review actions – accept, reject, defer, edit, split."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .ids import (
    make_correction_event_id,
    make_proposal_id,
    make_revision_id,
    patch_spec_fingerprint,
)
from .models import (
    PROPOSAL_STATUSES,
    CorrectionEvent,
    Proposal,
    ProposalRevision,
    ProposalTarget,
)
from .patch_spec import PatchSpecValidationError, validate_patch_spec
from .store import ReviewStore


class ReviewActionError(Exception):
    """Raised when a review action is invalid."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_actionable(proposal: Proposal) -> None:
    if proposal.status not in ("queued", "deferred"):
        raise ReviewActionError(
            f"Proposal {proposal.proposal_id} has status '{proposal.status}' and cannot be acted on"
        )


def accept_proposal(
    store: ReviewStore,
    proposal_id: str,
    reviewer: str,
    reviewer_note: str = "",
) -> CorrectionEvent:
    """Transition a queued/deferred proposal to accepted_pending_apply."""
    proposal = store.get_proposal(proposal_id)
    if not proposal:
        raise ReviewActionError(f"Proposal {proposal_id} not found")
    _require_actionable(proposal)

    now = _now_iso()
    event_id = make_correction_event_id(proposal_id, "accept", now)
    event = CorrectionEvent(
        event_id=event_id,
        proposal_id=proposal_id,
        revision_id=proposal.current_revision_id,
        action="accept",
        reviewer=reviewer,
        reviewer_note=reviewer_note,
        created_at=now,
    )
    store.update_proposal_status(proposal_id, "accepted_pending_apply")
    store.save_correction_event(event)
    return event


def reject_proposal(
    store: ReviewStore,
    proposal_id: str,
    reviewer: str,
    reviewer_note: str = "",
) -> CorrectionEvent:
    """Transition a queued/deferred proposal to rejected."""
    proposal = store.get_proposal(proposal_id)
    if not proposal:
        raise ReviewActionError(f"Proposal {proposal_id} not found")
    _require_actionable(proposal)

    now = _now_iso()
    event_id = make_correction_event_id(proposal_id, "reject", now)
    event = CorrectionEvent(
        event_id=event_id,
        proposal_id=proposal_id,
        revision_id=proposal.current_revision_id,
        action="reject",
        reviewer=reviewer,
        reviewer_note=reviewer_note,
        created_at=now,
    )
    store.update_proposal_status(proposal_id, "rejected")
    store.save_correction_event(event)
    return event


def defer_proposal(
    store: ReviewStore,
    proposal_id: str,
    reviewer: str,
    reviewer_note: str = "",
) -> CorrectionEvent:
    """Set a queued proposal to deferred status."""
    proposal = store.get_proposal(proposal_id)
    if not proposal:
        raise ReviewActionError(f"Proposal {proposal_id} not found")
    if proposal.status not in ("queued",):
        raise ReviewActionError(
            f"Proposal {proposal_id} has status '{proposal.status}'; only 'queued' can be deferred"
        )

    now = _now_iso()
    event_id = make_correction_event_id(proposal_id, "defer", now)
    event = CorrectionEvent(
        event_id=event_id,
        proposal_id=proposal_id,
        revision_id=proposal.current_revision_id,
        action="defer",
        reviewer=reviewer,
        reviewer_note=reviewer_note,
        created_at=now,
    )
    store.update_proposal_status(proposal_id, "deferred")
    store.save_correction_event(event)
    return event


def edit_proposal(
    store: ReviewStore,
    proposal_id: str,
    reviewer: str,
    edited_patch_spec: dict[str, Any],
    reviewer_note: str = "",
) -> tuple[ProposalRevision, CorrectionEvent]:
    """Create an edited revision with bounded edits.

    The edit may not change proposal_type or issue_class.
    """
    proposal = store.get_proposal(proposal_id)
    if not proposal:
        raise ReviewActionError(f"Proposal {proposal_id} not found")
    _require_actionable(proposal)

    # Validate the edited patch spec
    try:
        validate_patch_spec(edited_patch_spec)
    except PatchSpecValidationError as e:
        raise ReviewActionError(f"Invalid edited patch_spec: {e}") from e

    # Enforce: proposal_type may not change
    if edited_patch_spec.get("proposal_type") != proposal.proposal_type:
        raise ReviewActionError("Edit may not change proposal_type")

    now = _now_iso()
    rev_num = store.next_revision_number(proposal_id)
    revision_id = make_revision_id(proposal_id, rev_num)
    ps_fp = patch_spec_fingerprint(edited_patch_spec)

    # Get previous revision for evidence carryover
    prev_rev = store.get_latest_revision(proposal_id)
    evidence_json = prev_rev.evidence_snapshot_json if prev_rev else "{}"

    revision = ProposalRevision(
        revision_id=revision_id,
        proposal_id=proposal_id,
        revision_number=rev_num,
        revision_kind="edited",
        patch_spec_json=json.dumps(edited_patch_spec, ensure_ascii=True, sort_keys=True),
        patch_spec_fingerprint=ps_fp,
        evidence_snapshot_json=evidence_json,
        detector_name=prev_rev.detector_name if prev_rev else "",
        detector_version=prev_rev.detector_version if prev_rev else "",
        generator_type="human_edit",
        validator_version="v1",
        validation_state="valid",
        created_by=reviewer,
        created_at=now,
    )
    store.save_revision(revision)
    store.update_proposal_revision(proposal_id, revision_id)

    event_id = make_correction_event_id(proposal_id, "edit", now)
    event = CorrectionEvent(
        event_id=event_id,
        proposal_id=proposal_id,
        revision_id=revision_id,
        action="edit",
        reviewer=reviewer,
        reviewer_note=reviewer_note,
        created_at=now,
    )
    store.save_correction_event(event)
    return revision, event


def split_proposal(
    store: ReviewStore,
    proposal_id: str,
    reviewer: str,
    child_target_groups: list[list[ProposalTarget]],
    reviewer_note: str = "",
) -> list[Proposal]:
    """Split a proposal into children by narrowing target sets.

    The parent becomes superseded. Each child preserves proposal_type,
    issue_class, and snapshot_id.
    """
    proposal = store.get_proposal(proposal_id)
    if not proposal:
        raise ReviewActionError(f"Proposal {proposal_id} not found")
    _require_actionable(proposal)

    prev_rev = store.get_latest_revision(proposal_id)
    if not prev_rev:
        raise ReviewActionError(f"Proposal {proposal_id} has no revisions")

    original_patch = json.loads(prev_rev.patch_spec_json)
    original_targets = store.get_proposal_targets(proposal_id)
    original_target_ids = {t.target_id for t in original_targets}

    now = _now_iso()
    children: list[Proposal] = []

    for group in child_target_groups:
        # Validate: children can only narrow, not add new targets
        for t in group:
            if t.target_id not in original_target_ids and t.exists_in_snapshot:
                raise ReviewActionError(
                    f"Split child target '{t.target_id}' was not in original proposal"
                )

        target_tuples = [(t.target_kind, t.target_id, t.target_role) for t in group]
        child_proposal_id = make_proposal_id(
            proposal.snapshot_id, proposal.issue_class, proposal.proposal_type,
            target_tuples, original_patch,
        )
        child_impact = len(group)
        child_priority = min(1.0, 0.7 * proposal.confidence + 0.3 * min(1.0, child_impact / 10.0))

        child_proposal = Proposal(
            proposal_id=child_proposal_id,
            review_run_id=proposal.review_run_id,
            snapshot_id=proposal.snapshot_id,
            anti_pattern_id=proposal.anti_pattern_id,
            issue_class=proposal.issue_class,
            proposal_type=proposal.proposal_type,
            status="queued",
            confidence=proposal.confidence,
            priority_score=child_priority,
            impact_size=child_impact,
            created_at=now,
            updated_at=now,
        )

        child_targets = [
            ProposalTarget(
                proposal_id=child_proposal_id,
                target_kind=t.target_kind,
                target_id=t.target_id,
                target_role=t.target_role,
                exists_in_snapshot=t.exists_in_snapshot,
            )
            for t in group
        ]

        rev_num = 1
        child_rev_id = make_revision_id(child_proposal_id, rev_num)
        ps_fp = patch_spec_fingerprint(original_patch)

        child_revision = ProposalRevision(
            revision_id=child_rev_id,
            proposal_id=child_proposal_id,
            revision_number=rev_num,
            revision_kind="split_child",
            patch_spec_json=prev_rev.patch_spec_json,
            patch_spec_fingerprint=ps_fp,
            evidence_snapshot_json=prev_rev.evidence_snapshot_json,
            detector_name=prev_rev.detector_name,
            detector_version=prev_rev.detector_version,
            generator_type="split",
            validator_version="v1",
            validation_state="valid",
            created_by=reviewer,
            created_at=now,
        )

        child_proposal.current_revision_id = child_rev_id
        store.upsert_proposal(child_proposal, child_targets)
        store.save_revision(child_revision)
        children.append(child_proposal)

    # Supersede the parent
    store.update_proposal_status(proposal_id, "superseded")
    event_id = make_correction_event_id(proposal_id, "split", now)
    event = CorrectionEvent(
        event_id=event_id,
        proposal_id=proposal_id,
        revision_id=prev_rev.revision_id,
        action="split",
        reviewer=reviewer,
        reviewer_note=reviewer_note or f"Split into {len(children)} child proposals",
        created_at=now,
    )
    store.save_correction_event(event)

    return children
