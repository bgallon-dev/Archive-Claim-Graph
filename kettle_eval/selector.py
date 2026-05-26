"""Priority scoring + top-K selection.

Formula (locked):

    priority = 0.40 * predicted_failure
             + 0.25 * uncertainty
             + 0.25 * impact
             + 0.10 * topic_novelty

`predicted_failure` is a weighted sum of risk features per case_type, squashed
through tanh. `uncertainty` is distance-to-threshold for the case type's
canonical confidence boundary. `impact` is a lookup with one content-aware
branch (sensitivity-flagged claims, ambiguous-band entity resolutions). `topic_novelty`
is the frequency-inverse proxy `1 / sqrt(1 + count_seen_for_case_type_in_batch)`
— a cheap diversity signal that prevents the queue from collapsing into
the most-numerous case type. Batch scope is per-document by default.

Replace the body of `_predicted_failure` with a learned model in v1 without
changing the signature.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import TYPE_CHECKING

from .models import EvaluationCase, ReviewLabel

if TYPE_CHECKING:
    from .store import KettleEvalStore

_log = logging.getLogger(__name__)

# Adaptive weight bounds. Wide enough to learn meaningful signal, narrow
# enough that no single label can flip the queue.
_WEIGHT_MIN = 0.05
_WEIGHT_MAX = 3.0

# Perceptron-style learning rate. Small so labels accumulate gradually.
# Multiply by signed error * fired_value (each in [-1, 1]), so per-label
# delta is bounded by lr.
_LEARNING_RATE = 0.05

# ----------------------------------------------------------------------
# Formula weights — locked. Test guards this against drift.
# ----------------------------------------------------------------------

WEIGHT_PREDICTED_FAILURE = 0.40
WEIGHT_UNCERTAINTY = 0.25
WEIGHT_IMPACT = 0.25
WEIGHT_TOPIC_NOVELTY = 0.10

PRIORITY_WEIGHTS = (
    WEIGHT_PREDICTED_FAILURE,
    WEIGHT_UNCERTAINTY,
    WEIGHT_IMPACT,
    WEIGHT_TOPIC_NOVELTY,
)

# ----------------------------------------------------------------------
# Per-case-type predicted-failure weight tables.
# Provenance gaps outweigh stylistic flags. The body of _predicted_failure
# uses tanh to keep the output in [0, 1] even if weights sum > 1 — that is
# intentional so weights can express "this matters more than 1.0 worth."
# ----------------------------------------------------------------------

_CLAIM_EXTRACTION_WEIGHTS: dict[str, float] = {
    "uncertain_extraction": 1.00,
    "missing_source_span": 1.25,
    "missing_evidence_offsets": 1.00,
    "missing_page_anchor": 0.75,
    "sensitivity_monitor": 1.00,
    "builder_repair_score": 0.80,
    "causal_language": 0.45,
    "strong_verbs": 0.35,
    "uncertain_date": 0.30,
    "invalid_evidence_offsets": 0.90,
    "overbroad_evidence_span": 0.50,
    "low_ocr_confidence": 0.50,
    "broken_segmentation": 0.40,
}

_MENTION_WEIGHTS: dict[str, float] = {
    "uncertain_mention": 0.50,
    "low_quality_scan": 0.80,
    "high_char_noise": 0.60,
    "junk_mention_score": 1.20,
    "ocr_entity_score": 0.60,
    "missing_source_span": 0.50,
    "missing_page_anchor": 0.30,
    "broken_segmentation": 0.30,
}

_ENTITY_RESOLUTION_WEIGHTS: dict[str, float] = {
    "uncertain_resolution": 1.00,
    "ambiguous_entity_resolution": 1.10,
    "ocr_entity_score": 0.80,
    "missing_source_span": 0.40,
    "missing_page_anchor": 0.30,
    "broken_segmentation": 0.30,
}

_RELATION_WEIGHTS: dict[str, float] = {
    "uncertain_parent_claim": 0.80,
    "builder_repair_score": 1.00,
    "missing_source_span": 0.80,
    "missing_evidence_offsets": 0.70,
    "causal_language": 0.35,
    "strong_verbs": 0.30,
    "broken_segmentation": 0.40,
}

_PAGE_OCR_WEIGHTS: dict[str, float] = {
    "low_ocr_confidence": 1.20,
    "broken_segmentation": 0.30,
}

_WEIGHT_TABLES: dict[str, dict[str, float]] = {
    "claim_extraction": _CLAIM_EXTRACTION_WEIGHTS,
    "mention": _MENTION_WEIGHTS,
    "entity_resolution": _ENTITY_RESOLUTION_WEIGHTS,
    "relation": _RELATION_WEIGHTS,
    "page_ocr": _PAGE_OCR_WEIGHTS,
}

# ----------------------------------------------------------------------
# Uncertainty thresholds — distance from these defines how "on the fence"
# a case is. Reviewer attention is most valuable near the boundary.
# ----------------------------------------------------------------------

_UNCERTAINTY_THRESHOLDS: dict[str, tuple[str, float]] = {
    "entity_resolution": ("match_score", 0.85),
    "claim_extraction": ("extraction_confidence", 0.90),
    "mention": ("detection_confidence", 0.70),
    "page_ocr": ("ocr_confidence", 0.70),
}

# ----------------------------------------------------------------------
# Impact lookup. Cross-document entity merges have corpus-wide blast radius;
# sensitivity-flagged claims have institutional blast radius.
# ----------------------------------------------------------------------

_IMPACT_BASE: dict[str, float] = {
    "page_ocr": 0.6,
    "entity_resolution": 0.5,
    "claim_extraction": 0.4,
    "mention": 0.3,
    "relation": 0.5,
}


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _predicted_failure(
    case: EvaluationCase,
    weights_by_type: dict[str, dict[str, float]] | None = None,
) -> tuple[float, float]:
    """Returns (mean, uncertainty) for the predicted failure estimate.

    v0 heuristic: weighted sum of risk features through tanh. Uncertainty is
    a placeholder — the magnitude of the weighted sum prior to squash gives
    a rough proxy for "how strongly we believe this is risky."

    `weights_by_type` overrides the module-level defaults when supplied (e.g.
    by `load_weights(store)` overlaying persisted adaptive weights). None
    means "use the defaults", preserving v0.2 behavior.
    """
    table = (weights_by_type or _WEIGHT_TABLES).get(case.case_type, {})
    if not table:
        return 0.0, 0.0
    raw = 0.0
    fired = 0
    for feature, value in case.risk_features.items():
        weight = table.get(feature)
        if weight is None:
            continue
        raw += weight * value
        if value > 0.0:
            fired += 1
    mean = math.tanh(raw)
    # Uncertainty proxy: 1.0 when one or zero features fire, drops as more
    # independent features agree. Cheap stand-in until a real model lands.
    uncertainty = 1.0 / (1.0 + max(0, fired - 1))
    return _clip01(mean), _clip01(uncertainty)


def load_weights(
    store: "KettleEvalStore | None" = None,
) -> dict[str, dict[str, float]]:
    """Return the effective per-case-type weight tables.

    Defaults are the prior; any persisted (case_type, feature_name) rows in
    the store overlay them as the learned posterior. When `store` is None or
    carries no weight rows, returns the module-level defaults unchanged.

    The result is a *new* dict — callers can mutate it without affecting the
    module-level defaults.
    """
    effective: dict[str, dict[str, float]] = {
        ct: dict(defaults) for ct, defaults in _WEIGHT_TABLES.items()
    }
    if store is None:
        return effective
    try:
        persisted = store.get_feature_weights()
    except Exception:
        _log.debug("kettle_eval: failed to load persisted weights", exc_info=True)
        return effective
    for case_type, feats in persisted.items():
        effective.setdefault(case_type, {})
        for name, value in feats.items():
            effective[case_type][name] = float(value)
    return effective


def _uncertainty(case: EvaluationCase) -> float:
    spec = _UNCERTAINTY_THRESHOLDS.get(case.case_type)
    if spec is None:
        return 0.0
    field, threshold = spec
    score = case.model_output.get(field)
    if score is None:
        return 0.0
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return 0.0
    return _clip01(1.0 - 2.0 * abs(score_f - threshold))


def _impact(case: EvaluationCase) -> float:
    base = _IMPACT_BASE.get(case.case_type, 0.4)
    if case.case_type == "entity_resolution":
        # Ambiguous-band resolutions get a corpus-wide impact bump because
        # a wrong merge propagates to every claim referencing the entity.
        # A future cross_document_entity_merge feature (when actual
        # cross-doc evidence is tracked) can stack additively.
        if case.risk_features.get("ambiguous_entity_resolution", 0.0) >= 0.5:
            return 0.9
    elif case.case_type == "claim_extraction":
        if case.risk_features.get("sensitivity_monitor", 0.0) > 0.0:
            return 0.7
    return base


def _topic_novelty_lookup(cases: list[EvaluationCase]) -> dict[str, float]:
    """Per-case-type frequency-inverse novelty over the current batch.

    novelty(case_type) = 1 / sqrt(1 + count_in_batch)

    Computed once per batch so the score is stable across all cases of the
    same type. Batch scope defaults to whatever list is passed in — when
    `score_cases` is invoked per-document, this is per-document novelty;
    pass a multi-document list to get corpus-level novelty.
    """
    counts = Counter(c.case_type for c in cases)
    return {
        case_type: 1.0 / math.sqrt(1.0 + n)
        for case_type, n in counts.items()
    }


def score_cases(
    cases: list[EvaluationCase],
    store: "KettleEvalStore | None" = None,
) -> None:
    """Mutate each case's priority_score, predicted_failure_mean, and
    predicted_failure_uncertainty in place.

    Call after `risk_features.compute_features` — this function reads
    `case.risk_features` and `case.model_output`.

    When `store` is supplied, persisted adaptive weights from
    `feature_weight` are overlaid on the module-level defaults. Without a
    store the function behaves identically to v0.2 (deterministic
    heuristic, no learning).
    """
    weights_by_type = load_weights(store)
    novelty_by_type = _topic_novelty_lookup(cases)

    for case in cases:
        pf_mean, pf_unc = _predicted_failure(case, weights_by_type)
        unc = _uncertainty(case)
        imp = _impact(case)
        nov = novelty_by_type.get(case.case_type, 0.0)

        priority = (
            WEIGHT_PREDICTED_FAILURE * pf_mean
            + WEIGHT_UNCERTAINTY * unc
            + WEIGHT_IMPACT * imp
            + WEIGHT_TOPIC_NOVELTY * nov
        )

        case.predicted_failure_mean = pf_mean
        case.predicted_failure_uncertainty = pf_unc
        case.priority_score = _clip01(priority)


def learn_from_label(
    store: "KettleEvalStore",
    label: ReviewLabel,
    case: EvaluationCase,
) -> dict[str, float]:
    """Nudge the persisted feature weights using one ReviewLabel.

    Perceptron-style update:

        observed = severity / 4.0
        prior    = case.predicted_failure_mean
        error    = observed - prior        # signed
        delta    = lr * error * fired_value
        new_w    = clip(current_w + delta, _WEIGHT_MIN, _WEIGHT_MAX)

    Only features that *fired* on this case (value > 0) are updated. Returns
    a dict of feature_name -> new_weight for the features that moved, so
    callers (e.g. label_cases.py) can show the reviewer what changed.

    Defensive on bad inputs: skips silently if the case is None or carries
    no fired features.
    """
    if case is None or not case.risk_features:
        return {}

    weights_by_type = load_weights(store)
    case_weights = weights_by_type.get(case.case_type, {})
    if not case_weights:
        return {}

    observed = max(0.0, min(4.0, float(label.severity))) / 4.0
    prior = float(case.predicted_failure_mean)
    error = observed - prior
    if error == 0.0:
        return {}

    moved: dict[str, float] = {}
    for feature, fired_value in case.risk_features.items():
        if fired_value <= 0:
            continue
        current = case_weights.get(feature)
        if current is None:
            # The feature fired but is not in this case_type's weight table.
            # Seed at the default-class average (modest prior) so the update
            # has somewhere to land.
            current = 0.5
        delta = _LEARNING_RATE * error * float(fired_value)
        new_w = max(_WEIGHT_MIN, min(_WEIGHT_MAX, current + delta))
        if abs(new_w - current) < 1e-9:
            continue
        store.upsert_feature_weight(case.case_type, feature, new_w)
        moved[feature] = new_w
    return moved


def top_k(cases: list[EvaluationCase], k: int) -> list[EvaluationCase]:
    """Return the K highest-priority cases. Does not mutate input order."""
    return sorted(cases, key=lambda c: c.priority_score, reverse=True)[:k]


def top_k_per_decision_class(
    cases: list[EvaluationCase],
    k: int,
) -> list[tuple[EvaluationCase, int]]:
    """Return up to K (case, cluster_size) pairs, one per decision_class_id.

    Collapses cases that encode the same underlying decision (same surface
    -> entity -> relation, or same normalized claim sentence, etc.). The
    representative for each class is the highest-priority case in that
    class. The cluster size lets the queue UI show "this decision repeats
    N times" alongside the representative.

    Cases with empty decision_class_id (pre-v0.2 rows, or future case types
    that opt out) are excluded — they would otherwise lump into one
    pseudo-class. Pair this selector with `top_k` for an audit view that
    keeps all rows visible.
    """
    if k <= 0 or not cases:
        return []

    # Group, then pick the highest-priority representative per class.
    representatives: dict[str, EvaluationCase] = {}
    counts: dict[str, int] = {}
    for case in cases:
        cls = case.decision_class_id
        if not cls:
            continue
        counts[cls] = counts.get(cls, 0) + 1
        current = representatives.get(cls)
        if current is None or case.priority_score > current.priority_score:
            representatives[cls] = case

    ordered = sorted(
        representatives.values(),
        key=lambda c: c.priority_score,
        reverse=True,
    )[:k]
    return [(c, counts[c.decision_class_id]) for c in ordered]


def top_k_balanced(
    cases: list[EvaluationCase],
    k: int,
    *,
    per_type_cap: int | None = None,
) -> list[EvaluationCase]:
    """Return up to K cases, capped per case_type to prevent queue collapse.

    The default `top_k` returns the K highest-priority cases globally — which,
    on a real corpus, can return 50 near-identical entity_resolution cases in
    a row because that case type dominates the score distribution. The
    review batch then collapses into one repeated failure pattern instead of
    mapping the failure landscape.

    This selector iterates cases in descending priority order and skips any
    case whose case_type has already hit `per_type_cap`. The result still
    favors high-priority cases, but no single case_type can monopolize the
    queue.

    Default cap: `max(1, k // n_distinct_case_types)` so the top types each
    get roughly equal shares. Pass `per_type_cap` explicitly to override —
    e.g. `per_type_cap=k` recovers the original `top_k` behavior.
    """
    if k <= 0 or not cases:
        return []

    ordered = sorted(cases, key=lambda c: c.priority_score, reverse=True)
    if per_type_cap is None:
        distinct_types = {c.case_type for c in ordered}
        per_type_cap = max(1, k // max(1, len(distinct_types)))

    counts: dict[str, int] = {}
    chosen: list[EvaluationCase] = []
    overflow: list[EvaluationCase] = []
    for case in ordered:
        if len(chosen) >= k:
            break
        if counts.get(case.case_type, 0) < per_type_cap:
            chosen.append(case)
            counts[case.case_type] = counts.get(case.case_type, 0) + 1
        else:
            overflow.append(case)

    # If under-K because some types ran dry, top up with the leftover queue.
    if len(chosen) < k:
        for case in overflow:
            if len(chosen) >= k:
                break
            chosen.append(case)

    return chosen
