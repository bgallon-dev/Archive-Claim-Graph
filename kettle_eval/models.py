"""EvaluationCase: the unit of active evaluation.

One dataclass with a `case_type` discriminator. Per-type extras live in
`payload`. v0 case types: page_ocr, claim_extraction, entity_resolution,
relation, mention. The set stays open for v1 additions (retrieval, answer,
citation) without a schema bump.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from gemynd.core.models import _BaseRecord

CaseType = Literal[
    "page_ocr",
    "claim_extraction",
    "entity_resolution",
    "relation",
    "mention",
]


# Controlled vocab for ReviewLabel.failure_modes.
#
# Eleven entries cover the seven failure questions from the original manifesto
# (OCR misread, extractor invented a claim, graph merged two people, retrieval
# missed source, synthesis over-claimed, answer dropped uncertainty, citation
# doesn't support sentence) plus four neighboring concerns already named by
# the gemynd review subsystem.
#
# Mirrors the gemynd.review.models.ISSUE_CLASSES pattern: frozen set, validated
# at insert time, enables aggregate failure-mode reporting downstream.
FAILURE_MODES: frozenset[str] = frozenset({
    "ocr_misread",
    "fabricated_claim",
    "wrong_entity_merge",
    "unsupported_citation",
    "overgeneralized_claim",
    "identity_sensitive_misclassification",
    "broken_provenance",
    "missing_uncertainty",
    "wrong_relation_type",
    "wrong_entity_type",
    "other",
})


# Severity scale (0-4 inclusive) for ReviewLabel.severity.
#
# 0 — correct (claim is supported, citation matches, merge is right)
# 1 — minor defect (small OCR typo, no meaning changed)
# 2 — moderate defect (partial entity match, ambiguous date, relation too strong)
# 3 — serious defect (claim not supported by cited source, wrong person merged,
#                     missing uncertainty)
# 4 — critical (fabricated claim, false source support, identity-sensitive
#               misclassification, public-facing historical distortion)
SEVERITY_MIN = 0
SEVERITY_MAX = 4


@dataclass(slots=True)
class EvaluationCase(_BaseRecord):
    case_id: str
    case_family_id: str
    case_type: str
    corpus_id: str
    document_id: str
    object_id: str
    extraction_run_id: str
    pipeline_stage: str
    model_name: str
    # Dedup key for grouping equivalent decisions across runs and documents.
    # See kettle_eval.ids.make_decision_class_id for the signature contract.
    decision_class_id: str = ""
    input_text: str = ""
    page_id: str | None = None
    source_span_id: str | None = None
    source_image_region: dict[str, Any] | None = None
    prompt_hash: str = ""
    config_hash: str = ""
    model_output: dict[str, Any] = field(default_factory=dict)
    risk_features: dict[str, float] = field(default_factory=dict)
    predicted_failure_mean: float = 0.0
    predicted_failure_uncertainty: float = 0.0
    priority_score: float = 0.0
    review_status: str = "unreviewed"
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewLabel(_BaseRecord):
    """A reviewer's adjudication of a single EvaluationCase.

    Severity is on the 0-4 scale (see SEVERITY_MIN/SEVERITY_MAX). Failure
    modes are constrained to the FAILURE_MODES vocab. Multiple labels per
    case are allowed (different reviewers, re-reviews after corrections);
    label_id distinguishes them.

    `corrected_output` is free-shape so a reviewer can supply a
    case-type-appropriate correction — e.g. for an entity_resolution case,
    `{"corrected_entity_id": "...", "corrected_relation_type": "..."}`.
    """

    label_id: str
    case_id: str
    case_family_id: str
    decision_class_id: str
    severity: int
    failure_modes: list[str] = field(default_factory=list)
    reviewer: str = "human"
    corrected_output: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    confidence: float = 1.0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
