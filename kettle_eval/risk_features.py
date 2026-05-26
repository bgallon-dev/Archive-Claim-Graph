"""Risk-feature computation.

Each feature is a value in [0, 1] keyed by case_id. Features come from three
sources:

1. **Provenance gaps** — missing source span, missing page anchor, missing
   evidence offsets. Treated as explicit risk, not absence. For a provenance
   system, missing anchoring is a serious signal.
2. **Derived signals** — direct transforms of confidence/score fields already
   on the records (OCR confidence, extraction confidence, match score).
3. **Detector adapters** — wrap the four existing gemynd.review detectors
   (`ocr_entity`, `junk_mention`, `builder_repair`, `sensitivity_monitor`)
   read-only. Each detector becomes a contributor through the single mapping
   layer `proposal_to_case_targets`.

Compute is bundle-wide and runs once per call. Lazy per-case invocation would
turn the detectors' inherent O(N) scans into O(N^2).
"""
from __future__ import annotations

import logging
import re
from typing import Callable

from gemynd.core.models import (
    ClaimRecord,
    EntityResolutionRecord,
    MentionRecord,
    PageRecord,
    ParagraphRecord,
    SemanticBundle,
    StructureBundle,
)
from gemynd.review.detectors import (
    builder_repair,
    junk_mention,
    ocr_entity,
    sensitivity_monitor,
)
from gemynd.review.models import DetectorProposal

from .cases import build_object_to_case_index
from .models import EvaluationCase

_log = logging.getLogger(__name__)

# Resolution threshold for REFERS_TO (mirror of gemynd.core.resolver.ResolutionPolicy).
_REFERS_TO_THRESHOLD = 0.85
_AMBIGUOUS_LOW = 0.65
_AMBIGUOUS_HIGH = 0.85

_CAUSAL_WORDS = re.compile(
    r"\b(because|caused|due to|resulted in|led to|owing to|on account of)\b",
    re.IGNORECASE,
)
_STRONG_VERBS = re.compile(
    r"\b(proved|established|confirmed|demonstrated|definitively)\b",
    re.IGNORECASE,
)


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _ensure(features: dict[str, dict[str, float]], case_id: str) -> dict[str, float]:
    bucket = features.get(case_id)
    if bucket is None:
        bucket = {}
        features[case_id] = bucket
    return bucket


def compute_features(
    cases: list[EvaluationCase],
    structure: StructureBundle,
    semantic: SemanticBundle,
) -> None:
    """Populate `case.risk_features` in place for every case.

    All values are clipped to [0, 1]. Missing-source features fire to 1.0
    when the field is absent rather than silently defaulting to 0.
    """
    features: dict[str, dict[str, float]] = {case.case_id: {} for case in cases}

    _provenance_gaps(features, cases)
    _derived_signals(features, cases, structure, semantic)
    _heuristic_features(features, cases)
    _doc_type_features(features, cases, structure)

    object_index = build_object_to_case_index(cases)
    _detector_adapter(features, object_index, structure, semantic, junk_mention, "junk_mention_score")
    _detector_adapter(features, object_index, structure, semantic, ocr_entity, "ocr_entity_score")
    _detector_adapter(features, object_index, structure, semantic, builder_repair, "builder_repair_score")
    _detector_adapter(features, object_index, structure, semantic, sensitivity_monitor, "sensitivity_monitor")

    for case in cases:
        bucket = features.get(case.case_id, {})
        case.risk_features = {k: _clip01(v) for k, v in bucket.items()}


# ----------------------------------------------------------------------
# Provenance gap features — missing anchoring is risk, not absence.
# ----------------------------------------------------------------------


def _provenance_gaps(
    features: dict[str, dict[str, float]],
    cases: list[EvaluationCase],
) -> None:
    for case in cases:
        bucket = _ensure(features, case.case_id)
        if case.source_span_id is None:
            bucket["missing_source_span"] = 1.0
        if case.page_id is None:
            bucket["missing_page_anchor"] = 1.0
        if case.case_type == "claim_extraction":
            evidence_start = case.model_output.get("evidence_start")
            evidence_end = case.model_output.get("evidence_end")
            if evidence_start is None or evidence_end is None:
                bucket["missing_evidence_offsets"] = 1.0


# ----------------------------------------------------------------------
# Derived signals — direct transforms of existing confidence/score fields.
# ----------------------------------------------------------------------


def _derived_signals(
    features: dict[str, dict[str, float]],
    cases: list[EvaluationCase],
    structure: StructureBundle,
    semantic: SemanticBundle,
) -> None:
    pages_by_id = {p.page_id: p for p in structure.pages}
    paragraphs_by_id = {p.paragraph_id: p for p in structure.paragraphs}

    for case in cases:
        bucket = _ensure(features, case.case_id)

        if case.case_type == "page_ocr":
            ocr_conf = case.model_output.get("ocr_confidence")
            if ocr_conf is not None:
                bucket["low_ocr_confidence"] = _clip01(1.0 - float(ocr_conf))

        elif case.case_type == "claim_extraction":
            conf = case.model_output.get("extraction_confidence", 0.0)
            bucket["uncertain_extraction"] = _clip01(1.0 - float(conf))
            certainty = case.model_output.get("certainty")
            claim_date = case.model_output.get("claim_date")
            if certainty == "uncertain" and not claim_date:
                bucket["uncertain_date"] = 1.0
            # Evidence-span quality: split into invalid offsets (genuine
            # corruption: negative, inverted, or past-paragraph-end) and
            # overbroad spans (>80% of the paragraph — citation is the
            # whole paragraph, which is rarely what you want). A precise
            # short span inside a long paragraph is GOOD citation and must
            # not fire either feature.
            paragraph = paragraphs_by_id.get(case.source_span_id) if case.source_span_id else None
            ev_start = case.model_output.get("evidence_start")
            ev_end = case.model_output.get("evidence_end")
            if paragraph and ev_start is not None and ev_end is not None:
                para_len = max(paragraph.char_count, 1)
                start = int(ev_start)
                end = int(ev_end)
                if start < 0 or end <= start or end > para_len:
                    bucket["invalid_evidence_offsets"] = 1.0
                else:
                    span_ratio = (end - start) / para_len
                    if span_ratio > 0.80:
                        bucket["overbroad_evidence_span"] = 1.0

            page = pages_by_id.get(case.page_id) if case.page_id else None
            if page and page.ocr_confidence is not None:
                bucket["low_ocr_confidence"] = _clip01(1.0 - float(page.ocr_confidence))

        elif case.case_type == "mention":
            ocr_suspect = case.model_output.get("ocr_suspect", False)
            if ocr_suspect:
                bucket["low_quality_scan"] = 1.0
            surface = case.model_output.get("surface_form", "") or ""
            if surface:
                alpha = sum(1 for ch in surface if ch.isalpha())
                non_alpha_ratio = 1.0 - (alpha / len(surface))
                bucket["high_char_noise"] = _clip01(non_alpha_ratio)
            detection_conf = case.model_output.get("detection_confidence", 0.0)
            bucket["uncertain_mention"] = _clip01(1.0 - float(detection_conf))

        elif case.case_type == "entity_resolution":
            score = float(case.model_output.get("match_score", 0.0))
            # Distance from REFERS_TO threshold, normalized to [0,1].
            bucket["uncertain_resolution"] = _clip01(
                1.0 - 2.0 * abs(score - _REFERS_TO_THRESHOLD)
            )
            # Ambiguous band 0.65–0.85: a resolution decision near the
            # REFERS_TO threshold. A wrong call here propagates corpus-wide
            # because the entity is shared across documents — but the
            # observable signal at this point is threshold ambiguity, not
            # cross-document evidence. A separate cross_document_entity_merge
            # feature can be added later when we actually track per-entity
            # document counts.
            if _AMBIGUOUS_LOW <= score < _AMBIGUOUS_HIGH:
                bucket["ambiguous_entity_resolution"] = 1.0

        elif case.case_type == "relation":
            # A relation is only as inspectable as the claim that supports it.
            # Inherit provenance gaps from the parent claim so the relation
            # weight table's missing_source_span / missing_evidence_offsets
            # entries are not silent dead weights.
            claim_id = case.payload.get("claim_id")
            parent_claim = _find_claim(semantic, claim_id) if claim_id else None
            if parent_claim is not None:
                bucket["uncertain_parent_claim"] = _clip01(
                    1.0 - float(parent_claim.extraction_confidence)
                )
                if not parent_claim.paragraph_id:
                    bucket["missing_source_span"] = 1.0
                if parent_claim.evidence_start is None or parent_claim.evidence_end is None:
                    bucket["missing_evidence_offsets"] = 1.0


def _find_claim(semantic: SemanticBundle, claim_id: str) -> ClaimRecord | None:
    for claim in semantic.claims:
        if claim.claim_id == claim_id:
            return claim
    return None


# ----------------------------------------------------------------------
# Lightweight heuristics.
# ----------------------------------------------------------------------


def _heuristic_features(
    features: dict[str, dict[str, float]],
    cases: list[EvaluationCase],
) -> None:
    for case in cases:
        if case.case_type not in ("claim_extraction", "relation"):
            continue
        text = case.input_text or ""
        if not text:
            continue
        bucket = _ensure(features, case.case_id)
        if _CAUSAL_WORDS.search(text):
            bucket["causal_language"] = 1.0
        if _STRONG_VERBS.search(text):
            bucket["strong_verbs"] = 1.0


# ----------------------------------------------------------------------
# Doc-type bump — PNW forum docs have known broken paragraph segmentation.
# ----------------------------------------------------------------------


def _doc_type_features(
    features: dict[str, dict[str, float]],
    cases: list[EvaluationCase],
    structure: StructureBundle,
) -> None:
    doc_type = (structure.document.doc_type or "").lower()
    source_file = (structure.document.source_file or "").lower()
    is_pnw_forum = "pnw_forum" in doc_type or "pnw_forum" in source_file
    if not is_pnw_forum:
        return
    for case in cases:
        bucket = _ensure(features, case.case_id)
        bucket["broken_segmentation"] = 1.0


# ----------------------------------------------------------------------
# Detector adapter — wraps a review detector as a risk-feature contributor.
# ----------------------------------------------------------------------


def _detector_adapter(
    features: dict[str, dict[str, float]],
    object_index: dict[str, list[EvaluationCase]],
    structure: StructureBundle,
    semantic: SemanticBundle,
    detector_module,
    feature_name: str,
) -> None:
    """Invoke a review detector and harvest proposal.confidence per target.

    All four detectors expose `detect(structure, semantic, snapshot_id) ->
    list[DetectorProposal]`. We pass `snapshot_id="kettle_eval"` as a sentinel
    since we're harvesting features, not creating proposals.
    """
    try:
        proposals: list[DetectorProposal] = detector_module.detect(
            structure, semantic, "kettle_eval",
        )
    except Exception:
        _log.warning(
            "kettle_eval detector adapter for %s raised — feature skipped",
            getattr(detector_module, "__name__", "?"),
            exc_info=True,
        )
        return

    for proposal in proposals:
        for case in proposal_to_case_targets(proposal, object_index):
            bucket = _ensure(features, case.case_id)
            current = bucket.get(feature_name, 0.0)
            bucket[feature_name] = max(current, float(proposal.confidence))


def proposal_to_case_targets(
    proposal: DetectorProposal,
    object_index: dict[str, list[EvaluationCase]],
) -> list[EvaluationCase]:
    """Resolve a detector proposal's targets to the EvaluationCases they touch.

    A single detector proposal can fan out to multiple cases — e.g. an
    entity-level proposal touches every entity_resolution case for that
    entity. Funnels every adapter through this one function so they don't
    drift apart.
    """
    matches: list[EvaluationCase] = []
    seen: set[str] = set()
    for target in proposal.targets:
        for case in object_index.get(target.target_id, ()):
            if case.case_id in seen:
                continue
            seen.add(case.case_id)
            matches.append(case)
    return matches


# ----------------------------------------------------------------------
# Contributor interface — exposed for completeness and v1 extensibility.
# ----------------------------------------------------------------------

RiskContributor = Callable[
    [StructureBundle, SemanticBundle],
    dict[str, dict[str, float]],
]
