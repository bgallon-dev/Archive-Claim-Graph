from __future__ import annotations

from kettle_eval.cases import cases_from_bundle
from kettle_eval.models import EvaluationCase
from kettle_eval.risk_features import compute_features
from kettle_eval.selector import (
    PRIORITY_WEIGHTS,
    score_cases,
    top_k,
    top_k_balanced,
    top_k_per_decision_class,
)

from gemynd.core.models import EntityResolutionRecord


def test_priority_formula_weights_sum_to_one():
    total = sum(PRIORITY_WEIGHTS)
    assert abs(total - 1.0) < 1e-9


def _stub_case(
    case_id: str,
    case_type: str,
    priority: float,
    *,
    decision_class_id: str | None = None,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        case_family_id=f"fam_{case_id}",
        decision_class_id=decision_class_id if decision_class_id is not None else f"dcls_{case_id}",
        case_type=case_type,
        corpus_id="c",
        document_id="d",
        object_id=case_id,
        extraction_run_id="r",
        pipeline_stage="stub",
        model_name="stub",
        priority_score=priority,
    )


def test_top_k_orders_descending():
    cases = [
        _stub_case("a", "claim_extraction", 0.20),
        _stub_case("b", "claim_extraction", 0.90),
        _stub_case("c", "claim_extraction", 0.50),
    ]
    result = top_k(cases, 2)
    assert [c.case_id for c in result] == ["b", "c"]


def test_score_cases_populates_priority(minimal_bundle):
    structure, semantic = minimal_bundle
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    compute_features(cases, structure, semantic)
    for c in cases:
        c.priority_score = 0.0
    score_cases(cases)
    assert any(c.priority_score > 0.0 for c in cases)
    for c in cases:
        assert 0.0 <= c.priority_score <= 1.0


def test_missing_source_span_raises_claim_priority(bundle_factory):
    """A claim without source anchoring must outrank an otherwise-similar
    claim with clean anchoring. This guards the provenance-gap-as-risk
    philosophy."""
    structure_clean, semantic_clean = bundle_factory(
        n_claims=1, n_mentions=0, n_resolutions=0, n_links=0,
        include_source_span=True, include_evidence_offsets=True,
    )
    structure_gap, semantic_gap = bundle_factory(
        n_claims=1, n_mentions=0, n_resolutions=0, n_links=0,
        include_source_span=False, include_evidence_offsets=False,
    )

    clean_cases = cases_from_bundle(structure_clean, semantic_clean, corpus_id="c")
    gap_cases = cases_from_bundle(structure_gap, semantic_gap, corpus_id="c")
    compute_features(clean_cases, structure_clean, semantic_clean)
    compute_features(gap_cases, structure_gap, semantic_gap)
    score_cases(clean_cases)
    score_cases(gap_cases)

    clean_claim = next(c for c in clean_cases if c.case_type == "claim_extraction")
    gap_claim = next(c for c in gap_cases if c.case_type == "claim_extraction")
    assert gap_claim.priority_score > clean_claim.priority_score


def test_ambiguous_entity_resolution_has_high_impact(bundle_factory):
    """An ambiguous-band entity resolution must outrank a routine mention
    case. The impact bump reflects corpus-wide blast radius — wrong merges
    propagate to every claim that references the entity."""
    structure, semantic = bundle_factory(n_mentions=1, n_resolutions=0)
    semantic.entity_resolutions = [
        EntityResolutionRecord(
            mention_id=semantic.mentions[0].mention_id,
            entity_id="merge_target",
            relation_type="POSSIBLY_REFERS_TO",
            match_score=0.75,
        ),
    ]
    cases = cases_from_bundle(structure, semantic, corpus_id="c")
    compute_features(cases, structure, semantic)
    score_cases(cases)

    merge_case = next(c for c in cases if c.case_type == "entity_resolution")
    mention_case = next(c for c in cases if c.case_type == "mention")
    assert merge_case.priority_score > mention_case.priority_score


def test_topic_novelty_down_weights_overrepresented_type(bundle_factory):
    """A batch with many mention cases plus a single relation case should
    have the relation's novelty contribution exceed any mention's."""
    structure, semantic = bundle_factory(
        n_claims=1, n_mentions=20, n_resolutions=0, n_links=1,
    )
    cases = cases_from_bundle(structure, semantic, corpus_id="c")
    compute_features(cases, structure, semantic)

    from kettle_eval.selector import _topic_novelty_lookup

    lookup = _topic_novelty_lookup(cases)
    assert lookup["relation"] > lookup["mention"]


def test_top_k_balanced_caps_per_type():
    """Without a cap, a single case_type can monopolize the top-K. With a
    cap, the queue mixes types — preventing within-type collapse."""
    cases = (
        [_stub_case(f"r{i}", "entity_resolution", 0.90 + i * 1e-4) for i in range(30)]
        + [_stub_case(f"c{i}", "claim_extraction", 0.80 + i * 1e-4) for i in range(10)]
        + [_stub_case(f"m{i}", "mention", 0.50 + i * 1e-4) for i in range(5)]
    )
    # Naive top_k returns 10 entity_resolutions in a row.
    naive = top_k(cases, 10)
    assert all(c.case_type == "entity_resolution" for c in naive)

    # Balanced top_k mixes types — at most k // 3 = 3 of any single type.
    balanced = top_k_balanced(cases, 10)
    counts: dict[str, int] = {}
    for c in balanced:
        counts[c.case_type] = counts.get(c.case_type, 0) + 1
    assert max(counts.values()) <= max(1, 10 // 3) + 1  # +1 for overflow top-up
    assert "entity_resolution" in counts
    assert "claim_extraction" in counts


def test_top_k_balanced_returns_top_within_type():
    """Within each type, the balanced selector still picks the highest-priority
    cases — only across-type domination is prevented."""
    cases = [
        _stub_case("r_lo", "entity_resolution", 0.80),
        _stub_case("r_hi", "entity_resolution", 0.95),
        _stub_case("c_lo", "claim_extraction", 0.40),
        _stub_case("c_hi", "claim_extraction", 0.85),
    ]
    result = top_k_balanced(cases, k=2, per_type_cap=1)
    ids = {c.case_id for c in result}
    assert ids == {"r_hi", "c_hi"}


def test_top_k_balanced_recovers_top_k_with_high_cap():
    """A per_type_cap of k recovers original top_k behavior."""
    cases = [
        _stub_case(f"r{i}", "entity_resolution", 0.9 - i * 0.01)
        for i in range(5)
    ]
    naive = top_k(cases, 3)
    balanced = top_k_balanced(cases, 3, per_type_cap=3)
    assert [c.case_id for c in balanced] == [c.case_id for c in naive]


def test_top_k_balanced_handles_empty():
    assert top_k_balanced([], 5) == []
    assert top_k_balanced([_stub_case("x", "mention", 0.5)], 0) == []


def test_top_k_per_decision_class_collapses_duplicates():
    """14 cases sharing the same decision_class_id should collapse into one
    representative with cluster_size=14."""
    dup_class = "dcls_washington_water_power"
    cases = [
        _stub_case(f"d{i}", "entity_resolution", 0.92 - i * 1e-4, decision_class_id=dup_class)
        for i in range(14)
    ]
    # Plus one unique decision at slightly lower priority.
    cases.append(_stub_case("unique", "entity_resolution", 0.85, decision_class_id="dcls_unique"))

    naive = top_k(cases, 5)
    # Naive view: 5 entries, all from the dup class.
    assert all(c.decision_class_id == dup_class for c in naive)

    dedup = top_k_per_decision_class(cases, 5)
    # Dedup view: 2 entries (the dup class once, the unique once).
    assert len(dedup) == 2
    rep_classes = {case.decision_class_id for case, _ in dedup}
    assert rep_classes == {dup_class, "dcls_unique"}
    sizes = {case.decision_class_id: size for case, size in dedup}
    assert sizes[dup_class] == 14
    assert sizes["dcls_unique"] == 1
    # The representative for the dup class is the highest-priority member.
    rep = next(case for case, _ in dedup if case.decision_class_id == dup_class)
    assert rep.case_id == "d0"  # priority 0.92 (highest in the cluster)


def test_top_k_per_decision_class_excludes_empty_class_id():
    """Cases with empty decision_class_id (pre-v0.2) must not lump into a
    single pseudo-class — they are excluded entirely."""
    cases = [
        _stub_case("empty1", "mention", 0.99, decision_class_id=""),
        _stub_case("empty2", "mention", 0.98, decision_class_id=""),
        _stub_case("real", "mention", 0.50, decision_class_id="dcls_real"),
    ]
    result = top_k_per_decision_class(cases, 10)
    assert len(result) == 1
    assert result[0][0].case_id == "real"


def test_top_k_per_decision_class_orders_by_representative_priority():
    """Returned ordering is by the representative's priority, descending."""
    cases = [
        _stub_case("a_lo", "mention", 0.30, decision_class_id="dcls_a"),
        _stub_case("a_hi", "mention", 0.70, decision_class_id="dcls_a"),
        _stub_case("b", "mention", 0.50, decision_class_id="dcls_b"),
        _stub_case("c", "mention", 0.90, decision_class_id="dcls_c"),
    ]
    result = top_k_per_decision_class(cases, 3)
    class_ids = [case.decision_class_id for case, _ in result]
    assert class_ids == ["dcls_c", "dcls_a", "dcls_b"]


def test_top_k_per_decision_class_handles_empty():
    assert top_k_per_decision_class([], 5) == []
    assert top_k_per_decision_class([_stub_case("x", "mention", 0.5)], 0) == []
