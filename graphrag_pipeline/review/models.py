"""Data models for the anti-pattern review layer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Controlled enums
# ---------------------------------------------------------------------------

PROPOSAL_TYPES: frozenset[str] = frozenset({
    "merge_entities",
    "create_alias",
    "suppress_mention",
    "relabel_claim_link",
    "add_claim_entity_link",
    "add_claim_location_link",
    "exclude_claim_from_derivation",
    # Sensitivity monitor proposal types
    "quarantine_claim",
    "quarantine_document",
    "restrict_permanently",
})

ISSUE_CLASSES: frozenset[str] = frozenset({
    "ocr_spelling_variant",
    "duplicate_entity_alias",
    "header_contamination",
    "boilerplate_contamination",
    "short_generic_token",
    "ocr_garbage_mention",
    "missing_species_focus",
    "missing_event_location",
    "method_overtrigger",
    # Sensitivity monitor issue classes
    "pii_exposure",
    "indigenous_sensitivity",
    "living_person_reference",
})

PROPOSAL_STATUSES: frozenset[str] = frozenset({
    "queued",
    "accepted_pending_apply",
    "rejected",
    "deferred",
    "superseded",
})

REVISION_KINDS: frozenset[str] = frozenset({
    "generated",
    "edited",
    "split_child",
})

VALIDATION_STATES: frozenset[str] = frozenset({
    "valid",
    "invalid",
})

REVIEW_ACTIONS: frozenset[str] = frozenset({
    "accept",
    "reject",
    "defer",
    "edit",
    "split",
})

SUPPRESSION_REASONS: frozenset[str] = frozenset({
    "header_contamination",
    "boilerplate_contamination",
    "short_generic_token",
    "ocr_garbage",
})

SUPPRESSION_SCOPES: frozenset[str] = frozenset({
    "semantic_only",
    "full",
})

DERIVATION_KINDS: frozenset[str] = frozenset({
    "observation",
    "event",
    "all",
})

EVIDENCE_BASIS_VALUES: frozenset[str] = frozenset({
    "claim_link_diagnostic",
    "resolved_mention_context",
    "document_context",
})

TARGET_KINDS: frozenset[str] = frozenset({
    "claim",
    "entity",
    "mention",
    "claim_entity_link",
    "claim_location_link",
})


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AntiPatternClass:
    anti_pattern_id: str
    name: str
    description: str
    queue_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewRun:
    review_run_id: str
    snapshot_id: str
    doc_id: str
    structure_bundle_path: str
    semantic_bundle_path: str
    structure_bundle_sha256: str
    semantic_bundle_sha256: str
    snapshot_schema_version: str = "v1"
    extraction_run_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProposalTarget:
    proposal_id: str
    target_kind: str
    target_id: str
    target_role: str
    exists_in_snapshot: bool = True
    reviewer_override: str | None = None  # 'excluded' or None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Proposal:
    proposal_id: str
    review_run_id: str
    snapshot_id: str
    anti_pattern_id: str
    issue_class: str
    proposal_type: str
    status: str = "queued"
    confidence: float = 0.0
    priority_score: float = 0.0
    impact_size: int = 0
    current_revision_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProposalRevision:
    revision_id: str
    proposal_id: str
    revision_number: int
    revision_kind: str
    patch_spec_json: str
    patch_spec_fingerprint: str
    evidence_snapshot_json: str = "{}"
    live_context_json: str = "{}"
    reasoning_summary_json: str = "{}"
    detector_name: str = ""
    detector_version: str = ""
    generator_type: str = "heuristic"
    llm_provider: str | None = None
    llm_model: str | None = None
    validator_version: str = "v1"
    validation_state: str = "valid"
    created_by: str = "system"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CorrectionEvent:
    event_id: str
    proposal_id: str
    revision_id: str
    action: str
    reviewer: str = ""
    reviewer_note: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Detector output container
# ---------------------------------------------------------------------------

@dataclass
class DetectorProposal:
    """Intermediate output from a detector before validation and storage."""
    anti_pattern_id: str
    issue_class: str
    proposal_type: str
    confidence: float
    targets: list[ProposalTarget] = field(default_factory=list)
    patch_spec: dict[str, Any] = field(default_factory=dict)
    evidence_snapshot: dict[str, Any] = field(default_factory=dict)
    reasoning_summary: dict[str, Any] = field(default_factory=dict)
    detector_name: str = ""
    detector_version: str = ""
    generator_type: str = "heuristic"
    llm_provider: str | None = None
    llm_model: str | None = None
