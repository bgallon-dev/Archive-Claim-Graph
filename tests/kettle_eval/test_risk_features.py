from __future__ import annotations

from kettle_eval.cases import cases_from_bundle
from kettle_eval.risk_features import compute_features

from gemynd.core.models import EntityResolutionRecord, MentionRecord


def test_risk_features_clipped(minimal_bundle):
    structure, semantic = minimal_bundle
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    compute_features(cases, structure, semantic)
    for case in cases:
        for name, value in case.risk_features.items():
            assert 0.0 <= value <= 1.0, f"{case.case_type}.{name}={value} out of range"


def test_missing_source_span_fires(bundle_factory):
    structure, semantic = bundle_factory(include_source_span=False)
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    compute_features(cases, structure, semantic)
    claim_cases = [c for c in cases if c.case_type == "claim_extraction"]
    assert claim_cases
    for c in claim_cases:
        assert c.risk_features.get("missing_source_span") == 1.0


def test_missing_evidence_offsets_fires(bundle_factory):
    structure, semantic = bundle_factory(include_evidence_offsets=False)
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    compute_features(cases, structure, semantic)
    claim_cases = [c for c in cases if c.case_type == "claim_extraction"]
    assert claim_cases
    for c in claim_cases:
        assert c.risk_features.get("missing_evidence_offsets") == 1.0


def test_junk_mention_contributor_smoke(bundle_factory):
    """Adding an obvious OCR-garbage mention should produce a non-zero
    junk_mention_score on the corresponding case."""
    structure, semantic = bundle_factory(n_mentions=0)
    paragraph = structure.paragraphs[0]
    garbage = MentionRecord(
        mention_id="run_test_001:mention:garbage",
        run_id="run_test_001",
        paragraph_id=paragraph.paragraph_id,
        surface_form="$%^&*",
        normalized_form="$%^&*",
        start_offset=0,
        end_offset=5,
        detection_confidence=0.6,
    )
    semantic.mentions.append(garbage)
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    compute_features(cases, structure, semantic)
    garbage_case = next(
        c for c in cases
        if c.case_type == "mention" and c.object_id == garbage.mention_id
    )
    assert garbage_case.risk_features.get("junk_mention_score", 0.0) > 0.0


def test_pnw_forum_doc_type_bump(bundle_factory):
    structure, semantic = bundle_factory(doc_type="pnw_forum_article")
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    compute_features(cases, structure, semantic)
    assert cases
    for c in cases:
        assert c.risk_features.get("broken_segmentation") == 1.0


def test_citation_span_distance_removed(bundle_factory):
    """The old `citation_span_distance` feature must not be set anywhere —
    it has been replaced by `invalid_evidence_offsets` and
    `overbroad_evidence_span`."""
    structure, semantic = bundle_factory()
    cases = cases_from_bundle(structure, semantic, corpus_id="c")
    compute_features(cases, structure, semantic)
    for case in cases:
        assert "citation_span_distance" not in case.risk_features


def test_precise_evidence_span_does_not_fire_overbroad(bundle_factory):
    """A short, precise evidence span inside a long paragraph is GOOD
    citation — neither evidence-span feature should fire."""
    structure, semantic = bundle_factory(n_claims=1, n_mentions=0, n_resolutions=0, n_links=0)
    paragraph = structure.paragraphs[0]
    paragraph.clean_text = "X" * 500
    paragraph.char_count = 500
    cases = cases_from_bundle(structure, semantic, corpus_id="c")
    compute_features(cases, structure, semantic)
    claim_case = next(c for c in cases if c.case_type == "claim_extraction")
    assert "overbroad_evidence_span" not in claim_case.risk_features
    assert "invalid_evidence_offsets" not in claim_case.risk_features


def test_overbroad_evidence_span_fires_for_long_spans(bundle_factory):
    """A span covering >80% of the paragraph fires `overbroad_evidence_span`."""
    structure, semantic = bundle_factory(n_claims=1, n_mentions=0, n_resolutions=0, n_links=0)
    paragraph = structure.paragraphs[0]
    paragraph.clean_text = "X" * 100
    paragraph.char_count = 100
    semantic.claims[0].evidence_start = 0
    semantic.claims[0].evidence_end = 90
    cases = cases_from_bundle(structure, semantic, corpus_id="c")
    compute_features(cases, structure, semantic)
    claim_case = next(c for c in cases if c.case_type == "claim_extraction")
    assert claim_case.risk_features.get("overbroad_evidence_span") == 1.0
    assert "invalid_evidence_offsets" not in claim_case.risk_features


def test_invalid_evidence_offsets_fires_for_bad_ranges(bundle_factory):
    """Negative starts, end<=start, or end>paragraph_length all fire as
    invalid_evidence_offsets and skip the overbroad check entirely."""
    structure_a, semantic_a = bundle_factory(n_claims=1, n_mentions=0, n_resolutions=0, n_links=0)
    semantic_a.claims[0].evidence_start = 50
    semantic_a.claims[0].evidence_end = 40
    cases_a = cases_from_bundle(structure_a, semantic_a, corpus_id="c")
    compute_features(cases_a, structure_a, semantic_a)
    claim_a = next(c for c in cases_a if c.case_type == "claim_extraction")
    assert claim_a.risk_features.get("invalid_evidence_offsets") == 1.0
    assert "overbroad_evidence_span" not in claim_a.risk_features

    structure_b, semantic_b = bundle_factory(n_claims=1, n_mentions=0, n_resolutions=0, n_links=0)
    structure_b.paragraphs[0].char_count = 50
    semantic_b.claims[0].evidence_start = 0
    semantic_b.claims[0].evidence_end = 9999
    cases_b = cases_from_bundle(structure_b, semantic_b, corpus_id="c")
    compute_features(cases_b, structure_b, semantic_b)
    claim_b = next(c for c in cases_b if c.case_type == "claim_extraction")
    assert claim_b.risk_features.get("invalid_evidence_offsets") == 1.0

    structure_c, semantic_c = bundle_factory(n_claims=1, n_mentions=0, n_resolutions=0, n_links=0)
    semantic_c.claims[0].evidence_start = -1
    semantic_c.claims[0].evidence_end = 10
    cases_c = cases_from_bundle(structure_c, semantic_c, corpus_id="c")
    compute_features(cases_c, structure_c, semantic_c)
    claim_c = next(c for c in cases_c if c.case_type == "claim_extraction")
    assert claim_c.risk_features.get("invalid_evidence_offsets") == 1.0


def test_relation_case_inherits_parent_missing_source_span(bundle_factory):
    """A relation whose parent claim has no paragraph_id must inherit
    `missing_source_span=1.0`."""
    structure, semantic = bundle_factory(
        n_claims=1, n_mentions=0, n_resolutions=0, n_links=1,
        include_source_span=False,
    )
    cases = cases_from_bundle(structure, semantic, corpus_id="c")
    compute_features(cases, structure, semantic)
    relation_case = next(c for c in cases if c.case_type == "relation")
    assert relation_case.risk_features.get("missing_source_span") == 1.0


def test_relation_case_inherits_parent_missing_evidence_offsets(bundle_factory):
    """A relation whose parent claim has no evidence offsets must inherit
    `missing_evidence_offsets=1.0`."""
    structure, semantic = bundle_factory(
        n_claims=1, n_mentions=0, n_resolutions=0, n_links=1,
        include_evidence_offsets=False,
    )
    cases = cases_from_bundle(structure, semantic, corpus_id="c")
    compute_features(cases, structure, semantic)
    relation_case = next(c for c in cases if c.case_type == "relation")
    assert relation_case.risk_features.get("missing_evidence_offsets") == 1.0


def test_ambiguous_entity_resolution_feature(bundle_factory):
    """Match scores in the ambiguous 0.65-0.85 band should set the
    ambiguous_entity_resolution feature. The name reflects what is
    actually observable at this layer (threshold ambiguity) — a real
    cross_document_entity_merge feature requires per-entity document
    tracking that lives in a later phase."""
    structure, semantic = bundle_factory(
        n_resolutions=0, n_mentions=1,
    )
    semantic.entity_resolutions = [
        EntityResolutionRecord(
            mention_id=semantic.mentions[0].mention_id,
            entity_id="e1",
            relation_type="POSSIBLY_REFERS_TO",
            match_score=0.75,
        ),
    ]
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    compute_features(cases, structure, semantic)
    res_case = next(c for c in cases if c.case_type == "entity_resolution")
    assert res_case.risk_features.get("ambiguous_entity_resolution") == 1.0
    # Negative: the old misleading name must NOT be set — guards against
    # accidental re-introduction.
    assert "cross_document_entity_merge" not in res_case.risk_features
