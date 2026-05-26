"""Tests for the v0.3 ReviewLabel + adaptive weight layer."""
from __future__ import annotations

import pytest

import kettle_eval
from kettle_eval import FAILURE_MODES, KettleEvalStore, ReviewLabel
from kettle_eval.models import EvaluationCase
from kettle_eval.selector import (
    _WEIGHT_MAX,
    _WEIGHT_MIN,
    _WEIGHT_TABLES,
    learn_from_label,
    load_weights,
    score_cases,
)


# ----------------------------------------------------------------------
# Public API surface
# ----------------------------------------------------------------------


def test_kettle_eval_init_re_exports():
    """The package __init__ should re-export the v0.3 public surface."""
    for name in (
        "cases_from_bundle",
        "compute_features",
        "score_cases",
        "top_k",
        "top_k_balanced",
        "top_k_per_decision_class",
        "load_weights",
        "learn_from_label",
        "KettleEvalStore",
        "EvaluationCase",
        "ReviewLabel",
        "FAILURE_MODES",
    ):
        assert hasattr(kettle_eval, name), f"kettle_eval is missing public export: {name}"


def test_failure_modes_vocab_frozen():
    """The vocabulary is a frozenset (immutable) and contains the manifesto entries."""
    assert isinstance(FAILURE_MODES, frozenset)
    must_contain = {
        "ocr_misread",
        "fabricated_claim",
        "wrong_entity_merge",
        "unsupported_citation",
        "broken_provenance",
    }
    assert must_contain.issubset(FAILURE_MODES)


# ----------------------------------------------------------------------
# ReviewLabel store contract
# ----------------------------------------------------------------------


def _seed_case(store: KettleEvalStore, case_id: str = "case_x", case_type: str = "claim_extraction") -> EvaluationCase:
    case = EvaluationCase(
        case_id=case_id,
        case_family_id="fam_x",
        decision_class_id="dcls_x",
        case_type=case_type,
        corpus_id="c",
        document_id="d",
        object_id="o",
        extraction_run_id="r",
        pipeline_stage="extract.claim",
        model_name="test",
        input_text="seed",
        risk_features={"uncertain_extraction": 1.0},
        predicted_failure_mean=0.5,
        priority_score=0.5,
    )
    store.save_cases([case])
    return case


def test_review_label_roundtrip(tmp_path):
    """Add a label, read it back, and confirm the parent case flips to 'reviewed'."""
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        case = _seed_case(store)
        label = ReviewLabel(
            label_id="",
            case_id=case.case_id,
            case_family_id=case.case_family_id,
            decision_class_id=case.decision_class_id,
            severity=3,
            failure_modes=["unsupported_citation", "broken_provenance"],
            notes="cited paragraph doesn't say this",
        )
        stored = store.add_label(label)
        assert stored.label_id  # populated
        assert stored.created_at  # populated

        round_tripped = store.get_labels_for_case(case.case_id)
        assert len(round_tripped) == 1
        assert round_tripped[0].severity == 3
        assert sorted(round_tripped[0].failure_modes) == ["broken_provenance", "unsupported_citation"]
        assert round_tripped[0].notes == "cited paragraph doesn't say this"

        # Parent case moved to reviewed.
        reread = store.get_case(case.case_id)
        assert reread.review_status == "reviewed"


def test_review_label_rejects_bad_severity(tmp_path):
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        case = _seed_case(store)
        label = ReviewLabel(
            label_id="", case_id=case.case_id,
            case_family_id=case.case_family_id,
            decision_class_id=case.decision_class_id,
            severity=5,  # out of [0, 4]
        )
        with pytest.raises(ValueError):
            store.add_label(label)


def test_review_label_rejects_unknown_failure_mode(tmp_path):
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        case = _seed_case(store)
        label = ReviewLabel(
            label_id="", case_id=case.case_id,
            case_family_id=case.case_family_id,
            decision_class_id=case.decision_class_id,
            severity=2,
            failure_modes=["totally_made_up_mode"],
        )
        with pytest.raises(ValueError):
            store.add_label(label)


# ----------------------------------------------------------------------
# Adaptive feature weights
# ----------------------------------------------------------------------


def test_load_weights_falls_back_to_defaults_when_no_store():
    """No store -> exactly the module-level defaults."""
    weights = load_weights(None)
    for case_type, defaults in _WEIGHT_TABLES.items():
        for feature, weight in defaults.items():
            assert weights[case_type][feature] == weight


def test_load_weights_falls_back_to_defaults_when_no_labels(tmp_path):
    """Empty feature_weight table -> defaults preserved."""
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        weights = load_weights(store)
        assert weights["claim_extraction"]["uncertain_extraction"] == \
            _WEIGHT_TABLES["claim_extraction"]["uncertain_extraction"]


def test_load_weights_overlays_persisted_rows(tmp_path):
    """Persisted (case_type, feature_name) rows override the default."""
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        store.upsert_feature_weight("claim_extraction", "uncertain_extraction", 2.5)
        weights = load_weights(store)
        assert weights["claim_extraction"]["uncertain_extraction"] == 2.5


def test_learn_from_label_increases_weight_on_high_severity(tmp_path):
    """Severity 4 on a case that fired `uncertain_extraction=1.0` with prior
    predicted_failure_mean ~0.5 should INCREASE the feature weight."""
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        case = _seed_case(store)
        case.predicted_failure_mean = 0.5  # ensure non-zero error
        label = ReviewLabel(
            label_id="", case_id=case.case_id,
            case_family_id=case.case_family_id,
            decision_class_id=case.decision_class_id,
            severity=4,
        )
        stored = store.add_label(label)
        before = _WEIGHT_TABLES["claim_extraction"]["uncertain_extraction"]
        moved = learn_from_label(store, stored, case)
        assert "uncertain_extraction" in moved
        assert moved["uncertain_extraction"] > before


def test_learn_from_label_decreases_weight_on_severity_zero(tmp_path):
    """Severity 0 (correct) on a case that fired uncertain_extraction with
    prior predicted_failure_mean=0.5 means we over-predicted — DECREASE
    the feature weight."""
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        case = _seed_case(store)
        case.predicted_failure_mean = 0.5
        label = ReviewLabel(
            label_id="", case_id=case.case_id,
            case_family_id=case.case_family_id,
            decision_class_id=case.decision_class_id,
            severity=0,
        )
        stored = store.add_label(label)
        before = _WEIGHT_TABLES["claim_extraction"]["uncertain_extraction"]
        moved = learn_from_label(store, stored, case)
        assert "uncertain_extraction" in moved
        assert moved["uncertain_extraction"] < before


def test_learn_from_label_respects_clip_bounds(tmp_path):
    """Repeated severity-4 labels can't push a weight above _WEIGHT_MAX."""
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        case = _seed_case(store)
        case.predicted_failure_mean = 0.0  # maximum positive error
        label = ReviewLabel(
            label_id="", case_id=case.case_id,
            case_family_id=case.case_family_id,
            decision_class_id=case.decision_class_id,
            severity=4,
        )
        # Hammer the weight upward many times.
        for i in range(200):
            label.label_id = f"label_{i}"
            label.created_at = f"2026-05-12T00:00:{i:02d}"
            # add_label re-validates and inserts; we can call learn directly to skip the unique label_id concern.
            learn_from_label(store, label, case)
        final = store.get_feature_weights()["claim_extraction"]["uncertain_extraction"]
        assert final <= _WEIGHT_MAX
        assert final >= _WEIGHT_MIN


def test_score_cases_uses_persisted_weights_when_store_provided(tmp_path):
    """A clearly anomalous persisted weight must change the priority score."""
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        case = _seed_case(store)
        # Score with defaults first.
        score_cases([case], store=None)
        priority_default = case.priority_score

        # Now persist a very high weight and rescore.
        store.upsert_feature_weight("claim_extraction", "uncertain_extraction", _WEIGHT_MAX)
        # Re-fetch so we are not mutating the in-memory case from the prior call.
        case2 = store.get_case(case.case_id)
        score_cases([case2], store=store)
        assert case2.priority_score != priority_default


def test_score_cases_without_store_is_deterministic(tmp_path):
    """Without a store, score_cases is purely a function of inputs (v0.2 behavior)."""
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        _seed_case(store, case_id="case_a")
        _seed_case(store, case_id="case_b")
        # Persist a nonsense weight that would change priorities IF score
        # consulted the store. We're NOT passing the store, so it must not.
        store.upsert_feature_weight("claim_extraction", "uncertain_extraction", _WEIGHT_MAX)

    # Build two identical cases off-store (no store handle to mutate the result).
    def build():
        return EvaluationCase(
            case_id="ephemeral",
            case_family_id="f",
            decision_class_id="d",
            case_type="claim_extraction",
            corpus_id="c", document_id="d", object_id="o", extraction_run_id="r",
            pipeline_stage="extract.claim", model_name="test",
            risk_features={"uncertain_extraction": 1.0},
            predicted_failure_mean=0.0,
            priority_score=0.0,
        )
    c1, c2 = build(), build()
    score_cases([c1])
    score_cases([c2])
    assert c1.priority_score == c2.priority_score


def test_reset_feature_weights_clears_table(tmp_path):
    db = tmp_path / "eval.sqlite"
    with KettleEvalStore(db) as store:
        store.upsert_feature_weight("claim_extraction", "uncertain_extraction", 1.5)
        store.upsert_feature_weight("mention", "high_char_noise", 0.7)
        assert len(store.get_feature_weights()) == 2
        cleared = store.reset_feature_weights()
        assert cleared == 2
        assert store.get_feature_weights() == {}
