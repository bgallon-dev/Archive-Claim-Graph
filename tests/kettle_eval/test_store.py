from __future__ import annotations

import time

from kettle_eval.cases import cases_from_bundle
from kettle_eval.models import EvaluationCase
from kettle_eval.risk_features import compute_features
from kettle_eval.selector import score_cases
from kettle_eval.store import KettleEvalStore


def _make_case(
    case_id: str = "case_test",
    *,
    decision_class_id: str = "dcls_test",
    priority_score: float = 0.5,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        case_family_id="fam_test",
        decision_class_id=decision_class_id,
        case_type="claim_extraction",
        corpus_id="c",
        document_id="d",
        object_id="o",
        extraction_run_id="r",
        pipeline_stage="extract.claim",
        model_name="test",
        input_text="hello",
        risk_features={"uncertain_extraction": 0.3},
        predicted_failure_mean=0.4,
        priority_score=priority_score,
        payload={"x": 1},
    )


def test_store_roundtrip(tmp_path):
    db = tmp_path / "eval.sqlite"
    case = _make_case()
    with KettleEvalStore(db) as store:
        store.save_cases([case])
        round_tripped = store.get_case(case.case_id)
    assert round_tripped is not None
    assert round_tripped.case_id == case.case_id
    assert round_tripped.risk_features == case.risk_features
    assert round_tripped.payload == case.payload
    assert round_tripped.input_text == case.input_text
    assert round_tripped.predicted_failure_mean == case.predicted_failure_mean


def test_store_upsert_idempotent(tmp_path):
    db = tmp_path / "eval.sqlite"
    case = _make_case()
    with KettleEvalStore(db) as store:
        store.save_cases([case])
        first_count = store.count()
        first_row = store.get_case(case.case_id)
        # Sleep briefly so updated_at moves.
        time.sleep(0.01)
        # Mutate the priority and re-save the same case_id.
        case.priority_score = 0.95
        store.save_cases([case])
        second_count = store.count()
        second_row = store.get_case(case.case_id)
    assert first_count == 1
    assert second_count == 1
    assert second_row.priority_score == 0.95
    assert second_row.updated_at >= first_row.updated_at


def test_store_top_k_orders_by_priority(tmp_path):
    db = tmp_path / "eval.sqlite"
    cases = [_make_case(case_id=f"case_{i}") for i in range(5)]
    for i, c in enumerate(cases):
        c.priority_score = 0.1 * (i + 1)
    with KettleEvalStore(db) as store:
        store.save_cases(cases)
        top = store.top_k(3)
    assert len(top) == 3
    assert top[0].priority_score >= top[1].priority_score >= top[2].priority_score


def test_store_top_k_per_decision_class_collapses_duplicates(tmp_path):
    """Multiple cases sharing decision_class_id collapse into one row,
    with cluster_size reporting the dedup count."""
    db = tmp_path / "eval.sqlite"
    dup = "dcls_repeat"
    cases = [_make_case(case_id=f"d{i}", decision_class_id=dup, priority_score=0.90 - i * 0.01)
             for i in range(5)]
    cases.append(_make_case(case_id="unique", decision_class_id="dcls_unique", priority_score=0.50))
    with KettleEvalStore(db) as store:
        store.save_cases(cases)
        dedup = store.top_k_per_decision_class(10)
    # Two classes -> two rows.
    assert len(dedup) == 2
    by_class = {case.decision_class_id: (case, size) for case, size in dedup}
    rep_dup, size_dup = by_class[dup]
    assert size_dup == 5
    assert rep_dup.case_id == "d0"  # highest priority in the cluster
    rep_uniq, size_uniq = by_class["dcls_unique"]
    assert size_uniq == 1
    # Ordering: dup class first (priority 0.90 > 0.50).
    assert dedup[0][0].decision_class_id == dup


def test_store_top_k_per_decision_class_excludes_empty_class(tmp_path):
    """Rows with empty decision_class_id (pre-v0.2 migrated rows) must not
    appear in the dedup view as one fake giant cluster."""
    db = tmp_path / "eval.sqlite"
    cases = [
        _make_case(case_id="empty1", decision_class_id="", priority_score=0.99),
        _make_case(case_id="empty2", decision_class_id="", priority_score=0.98),
        _make_case(case_id="real", decision_class_id="dcls_real", priority_score=0.50),
    ]
    with KettleEvalStore(db) as store:
        store.save_cases(cases)
        dedup = store.top_k_per_decision_class(10)
    assert len(dedup) == 1
    case, size = dedup[0]
    assert case.case_id == "real"
    assert size == 1


def test_store_schema_migration_adds_decision_class_column(tmp_path):
    """An old-shape database (no decision_class_id column) must be migrated
    transparently when KettleEvalStore opens it. Simulates v0.1 -> v0.2."""
    import sqlite3 as _sqlite3
    db = tmp_path / "old_eval.sqlite"
    # Build the pre-v0.2 schema explicitly — no decision_class_id column.
    conn = _sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE evaluation_case (
            case_id TEXT PRIMARY KEY,
            case_family_id TEXT NOT NULL,
            case_type TEXT NOT NULL,
            corpus_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            page_id TEXT,
            source_span_id TEXT,
            object_id TEXT NOT NULL,
            extraction_run_id TEXT NOT NULL,
            input_text TEXT,
            pipeline_stage TEXT,
            model_name TEXT,
            prompt_hash TEXT,
            config_hash TEXT,
            predicted_failure_mean REAL NOT NULL DEFAULT 0.0,
            predicted_failure_uncertainty REAL NOT NULL DEFAULT 0.0,
            priority_score REAL NOT NULL DEFAULT 0.0,
            review_status TEXT NOT NULL DEFAULT 'unreviewed',
            risk_features_json TEXT NOT NULL DEFAULT '{}',
            model_output_json TEXT NOT NULL DEFAULT '{}',
            source_image_region_json TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.execute(
        "INSERT INTO evaluation_case (case_id, case_family_id, case_type, corpus_id, "
        "document_id, object_id, extraction_run_id, created_at, updated_at) "
        "VALUES ('legacy', 'fam', 'mention', 'c', 'd', 'o', 'r', 't', 't')"
    )
    conn.commit()
    conn.close()

    # Opening with KettleEvalStore triggers _migrate_schema().
    with KettleEvalStore(db) as store:
        case = store.get_case("legacy")
        assert case is not None
        assert case.decision_class_id == ""  # legacy rows carry the default
        # New cases save and retrieve normally.
        new = _make_case(case_id="post_migration", decision_class_id="dcls_x")
        store.save_cases([new])
        recovered = store.get_case("post_migration")
        assert recovered.decision_class_id == "dcls_x"


def test_store_full_pipeline_smoke(tmp_path, minimal_bundle):
    """End-to-end through the public API: bundle -> cases -> features -> score -> save -> read."""
    structure, semantic = minimal_bundle
    db = tmp_path / "eval.sqlite"
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    compute_features(cases, structure, semantic)
    score_cases(cases)
    with KettleEvalStore(db) as store:
        written = store.save_cases(cases)
        assert written == len(cases)
        recovered = store.top_k(len(cases))
    assert len(recovered) == len(cases)
    # Distribution should not be uniform — at least one case has > 0 risk features.
    distinct_scores = {round(c.priority_score, 4) for c in recovered}
    assert len(distinct_scores) > 1
