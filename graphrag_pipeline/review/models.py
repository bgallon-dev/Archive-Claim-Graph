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

REVIEW_TIERS: frozenset[str] = frozenset({
    "auto_accepted",
    "auto_suppressed",
    "needs_review",
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

# ---------------------------------------------------------------------------
# Error classification taxonomy (for reviewer feedback / model training)
# ---------------------------------------------------------------------------

ERROR_ROOT_CAUSES: frozenset[str] = frozenset({
    "ocr_confusion",
    "extraction_hallucination",
    "context_misinterpretation",
    "data_quality",
    "system_overtrigger",
    "other",
})

ERROR_TYPES: dict[str, frozenset[str]] = {
    "ocr_confusion": frozenset({
        "char_substitution",
        "char_insertion",
        "char_deletion",
        "word_boundary",
        "ligature_confusion",
        "digit_letter_swap",
    }),
    "extraction_hallucination": frozenset({
        "entity_fabrication",
        "wrong_entity_type",
        "claim_misattribution",
        "measurement_error",
    }),
    "context_misinterpretation": frozenset({
        "header_as_content",
        "caption_as_claim",
        "cross_page_confusion",
        "abbreviation_confusion",
        "coreference_error",
    }),
    "data_quality": frozenset({
        "illegible_source",
        "ambiguous_source",
        "duplicate_in_source",
    }),
    "system_overtrigger": frozenset({
        "false_positive_match",
        "legitimate_mention",
        "correct_link_state",
        "not_sensitive",
    }),
    "other": frozenset({
        "unclassified",
    }),
}

ERROR_ROOT_CAUSE_LABELS: dict[str, str] = {
    "ocr_confusion": "OCR Character Confusion",
    "extraction_hallucination": "Extraction Hallucination",
    "context_misinterpretation": "Context Misinterpretation",
    "data_quality": "Source Data Quality",
    "system_overtrigger": "System Overtrigger",
    "other": "Other",
}

ERROR_TYPE_LABELS: dict[str, str] = {
    "char_substitution": "Character substitution (rn/m, 0/O)",
    "char_insertion": "Extra character inserted",
    "char_deletion": "Character deleted/dropped",
    "word_boundary": "Word incorrectly split or merged",
    "ligature_confusion": "Ligature or diacritical confusion",
    "digit_letter_swap": "Digit/letter confusion (0/O, 1/l)",
    "entity_fabrication": "Entity name not in source text",
    "wrong_entity_type": "Correct name, wrong entity type",
    "claim_misattribution": "Claim attributed to wrong entity",
    "measurement_error": "Numeric value or unit incorrect",
    "header_as_content": "Header/footer treated as content",
    "caption_as_claim": "Caption interpreted as factual claim",
    "cross_page_confusion": "Text from adjacent page misjoined",
    "abbreviation_confusion": "Abbreviation expanded incorrectly",
    "coreference_error": "Reference resolved to wrong entity",
    "illegible_source": "Source text too degraded to read",
    "ambiguous_source": "Source text genuinely ambiguous",
    "duplicate_in_source": "Duplicate content in original",
    "false_positive_match": "Entities are genuinely different",
    "legitimate_mention": "Mention is valid, not junk",
    "correct_link_state": "Existing link configuration correct",
    "not_sensitive": "Content is not actually sensitive",
    "unclassified": "Error type does not fit taxonomy",
}

# Suggested default root cause per issue class
ISSUE_CLASS_DEFAULT_ROOT_CAUSE: dict[str, str] = {
    "ocr_spelling_variant": "ocr_confusion",
    "duplicate_entity_alias": "ocr_confusion",
    "ocr_garbage_mention": "ocr_confusion",
    "header_contamination": "context_misinterpretation",
    "boilerplate_contamination": "context_misinterpretation",
    "short_generic_token": "system_overtrigger",
    "missing_species_focus": "system_overtrigger",
    "missing_event_location": "system_overtrigger",
    "method_overtrigger": "system_overtrigger",
    "pii_exposure": "system_overtrigger",
    "indigenous_sensitivity": "system_overtrigger",
    "living_person_reference": "system_overtrigger",
}


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
    review_tier: str = "needs_review"

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
    error_root_cause: str = ""
    error_type: str = ""
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
