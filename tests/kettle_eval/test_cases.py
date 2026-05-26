from __future__ import annotations

from kettle_eval.cases import cases_from_bundle
from kettle_eval.ids import make_case_family_id, make_resolution_object_id

from gemynd.core.models import EntityRecord, EntityResolutionRecord


def test_cases_from_bundle_emits_one_per_claim(minimal_bundle):
    structure, semantic = minimal_bundle
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    claim_cases = [c for c in cases if c.case_type == "claim_extraction"]
    assert len(claim_cases) == len(semantic.claims)


def test_cases_from_bundle_emits_one_per_mention(minimal_bundle):
    structure, semantic = minimal_bundle
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    mention_cases = [c for c in cases if c.case_type == "mention"]
    assert len(mention_cases) == len(semantic.mentions)


def test_cases_from_bundle_emits_one_per_page(minimal_bundle):
    structure, semantic = minimal_bundle
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    page_cases = [c for c in cases if c.case_type == "page_ocr"]
    assert len(page_cases) == len(structure.pages)


def test_case_id_stable(bundle_factory):
    structure1, semantic1 = bundle_factory(run_id="run_test_001")
    structure2, semantic2 = bundle_factory(run_id="run_test_001")
    cases1 = cases_from_bundle(structure1, semantic1, corpus_id="test_corpus")
    cases2 = cases_from_bundle(structure2, semantic2, corpus_id="test_corpus")
    ids1 = sorted(c.case_id for c in cases1)
    ids2 = sorted(c.case_id for c in cases2)
    assert ids1 == ids2


def test_case_family_id_stable_across_runs(bundle_factory):
    structure1, semantic1 = bundle_factory(run_id="run_A")
    structure2, semantic2 = bundle_factory(run_id="run_B")
    cases1 = cases_from_bundle(structure1, semantic1, corpus_id="test_corpus")
    cases2 = cases_from_bundle(structure2, semantic2, corpus_id="test_corpus")

    # Same evaluable objects across runs: page_ocr cases share family_id;
    # claim/mention/relation also share family_id because object_id is run-independent.
    page_families1 = sorted(c.case_family_id for c in cases1 if c.case_type == "page_ocr")
    page_families2 = sorted(c.case_family_id for c in cases2 if c.case_type == "page_ocr")
    assert page_families1 == page_families2

    # case_ids differ because they include the run_id.
    page_ids1 = sorted(c.case_id for c in cases1 if c.case_type == "page_ocr")
    page_ids2 = sorted(c.case_id for c in cases2 if c.case_type == "page_ocr")
    assert page_ids1 != page_ids2


def test_entity_resolution_decisions_not_collapsed(bundle_factory):
    """Two resolutions of different mentions to the same entity must produce
    two distinct cases. The evaluated unit is the decision, not the entity.
    """
    structure, semantic = bundle_factory(n_mentions=2, n_resolutions=0)
    # Two different mentions resolve to the same entity_id.
    semantic.entity_resolutions = [
        EntityResolutionRecord(
            mention_id=semantic.mentions[0].mention_id,
            entity_id="shared_entity_001",
            relation_type="REFERS_TO",
            match_score=0.91,
        ),
        EntityResolutionRecord(
            mention_id=semantic.mentions[1].mention_id,
            entity_id="shared_entity_001",
            relation_type="REFERS_TO",
            match_score=0.87,
        ),
    ]
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    res_cases = [c for c in cases if c.case_type == "entity_resolution"]
    assert len(res_cases) == 2
    # Distinct object_ids and case_ids.
    assert len({c.object_id for c in res_cases}) == 2
    assert len({c.case_id for c in res_cases}) == 2


def test_resolution_object_id_is_decision_level(bundle_factory):
    structure, semantic = bundle_factory(n_mentions=1, n_resolutions=0)
    semantic.entity_resolutions = [
        EntityResolutionRecord(
            mention_id=semantic.mentions[0].mention_id,
            entity_id="e1",
            relation_type="REFERS_TO",
            match_score=0.9,
        ),
    ]
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    res_case = next(c for c in cases if c.case_type == "entity_resolution")
    expected = make_resolution_object_id(
        semantic.mentions[0].mention_id, "e1", "REFERS_TO", semantic.extraction_run.run_id,
    )
    assert res_case.object_id == expected


def test_resolution_case_input_text_populated(bundle_factory):
    """The reviewer needs surface_form + entity name + paragraph snippet
    on an entity_resolution case, not an empty string."""
    structure, semantic = bundle_factory(n_mentions=1, n_resolutions=0)
    mention = semantic.mentions[0]
    semantic.entities.append(EntityRecord(
        entity_id="place_spokane",
        entity_type="Place",
        name="Spokane",
        normalized_form="spokane",
    ))
    semantic.entity_resolutions = [
        EntityResolutionRecord(
            mention_id=mention.mention_id,
            entity_id="place_spokane",
            relation_type="POSSIBLY_REFERS_TO",
            match_score=0.78,
        ),
    ]
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    res_case = next(c for c in cases if c.case_type == "entity_resolution")
    assert res_case.input_text  # not empty
    assert mention.surface_form in res_case.input_text
    assert "Spokane" in res_case.input_text
    assert "0.78" in res_case.input_text
    assert "POSSIBLY_REFERS_TO" in res_case.input_text
    # Payload carries the structured fields too.
    assert res_case.payload["surface_form"] == mention.surface_form
    assert res_case.payload["entity_name"] == "Spokane"


def test_decision_class_id_collapses_same_decision_across_runs(bundle_factory):
    """Two resolutions encoding the same surface + entity + relation across
    different runs/documents must share decision_class_id while keeping
    distinct case_id and case_family_id. This is the dedup contract."""
    # Two different runs, same logical decision: "Spokane" -> place_spokane
    structure_a, semantic_a = bundle_factory(
        run_id="run_A", doc_id="doc_A", n_mentions=1, n_resolutions=0,
    )
    structure_b, semantic_b = bundle_factory(
        run_id="run_B", doc_id="doc_B", n_mentions=1, n_resolutions=0,
    )

    def _attach(structure, semantic):
        mention = semantic.mentions[0]
        # Force the same normalized form across both bundles.
        mention.normalized_form = "spokane"
        mention.surface_form = "Spokane"
        semantic.entities.append(EntityRecord(
            entity_id="place_spokane",
            entity_type="Place",
            name="Spokane",
            normalized_form="spokane",
        ))
        semantic.entity_resolutions.append(EntityResolutionRecord(
            mention_id=mention.mention_id,
            entity_id="place_spokane",
            relation_type="REFERS_TO",
            match_score=0.92,
        ))

    _attach(structure_a, semantic_a)
    _attach(structure_b, semantic_b)

    cases_a = cases_from_bundle(structure_a, semantic_a, corpus_id="c")
    cases_b = cases_from_bundle(structure_b, semantic_b, corpus_id="c")
    res_a = next(c for c in cases_a if c.case_type == "entity_resolution")
    res_b = next(c for c in cases_b if c.case_type == "entity_resolution")

    # Same decision class — that's the whole point of decision_class_id.
    assert res_a.decision_class_id == res_b.decision_class_id
    assert res_a.decision_class_id.startswith("dcls_")
    # But case_id and case_family_id differ (different runs and docs).
    assert res_a.case_id != res_b.case_id
    assert res_a.case_family_id != res_b.case_family_id


def test_decision_class_id_distinguishes_different_decisions(bundle_factory):
    """Different surface forms or different target entities must yield
    different decision_class_ids."""
    structure, semantic = bundle_factory(n_mentions=2, n_resolutions=0)
    # Mention 0: Spokane -> place_spokane
    semantic.mentions[0].normalized_form = "spokane"
    semantic.mentions[0].surface_form = "Spokane"
    # Mention 1: Spokane -> place_spokane_county (different entity)
    semantic.mentions[1].normalized_form = "spokane"
    semantic.mentions[1].surface_form = "Spokane"
    semantic.entity_resolutions = [
        EntityResolutionRecord(
            mention_id=semantic.mentions[0].mention_id,
            entity_id="place_spokane",
            relation_type="REFERS_TO",
            match_score=0.90,
        ),
        EntityResolutionRecord(
            mention_id=semantic.mentions[1].mention_id,
            entity_id="place_spokane_county",
            relation_type="REFERS_TO",
            match_score=0.88,
        ),
    ]
    cases = cases_from_bundle(structure, semantic, corpus_id="c")
    res_cases = [c for c in cases if c.case_type == "entity_resolution"]
    assert len(res_cases) == 2
    # Same surface, different entity -> different classes.
    assert res_cases[0].decision_class_id != res_cases[1].decision_class_id


def test_resolution_case_input_text_handles_missing_entity(bundle_factory):
    """Even if the entity isn't in the bundle, input_text should still render."""
    structure, semantic = bundle_factory(n_mentions=1, n_resolutions=0)
    mention = semantic.mentions[0]
    semantic.entity_resolutions = [
        EntityResolutionRecord(
            mention_id=mention.mention_id,
            entity_id="orphan_entity_id",
            relation_type="REFERS_TO",
            match_score=0.91,
        ),
    ]
    cases = cases_from_bundle(structure, semantic, corpus_id="test_corpus")
    res_case = next(c for c in cases if c.case_type == "entity_resolution")
    assert "orphan_entity_id" in res_case.input_text
    assert mention.surface_form in res_case.input_text
