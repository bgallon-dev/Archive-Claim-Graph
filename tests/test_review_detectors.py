"""Tests for review detector families."""
from __future__ import annotations

import pytest

from graphrag_pipeline.models import (
    ClaimEntityLinkRecord,
    ClaimLinkDiagnosticRecord,
    ClaimLocationLinkRecord,
    ClaimRecord,
    DocumentRecord,
    EntityRecord,
    EntityResolutionRecord,
    ExtractionRunRecord,
    MentionRecord,
    PageRecord,
    ParagraphRecord,
    SemanticBundle,
    StructureBundle,
)
from graphrag_pipeline.review.detectors import ocr_entity, junk_mention, builder_repair


def _make_structure(
    doc_id: str = "doc_1",
    paragraphs: list[ParagraphRecord] | None = None,
    pages: list[PageRecord] | None = None,
) -> StructureBundle:
    doc = DocumentRecord(doc_id=doc_id, title="Test Report", source_file="test.json")
    if pages is None:
        pages = [PageRecord(page_id="p1", doc_id=doc_id, page_number=1, raw_ocr_text="", clean_text="")]
    if paragraphs is None:
        paragraphs = [ParagraphRecord(
            paragraph_id="para1", doc_id=doc_id, page_id="p1",
            section_id=None, paragraph_index=0, page_number=1,
            raw_ocr_text="Test paragraph", clean_text="Test paragraph", char_count=14,
        )]
    return StructureBundle(document=doc, pages=pages, sections=[], paragraphs=paragraphs, annotations=[])


def _make_semantic(
    entities: list[EntityRecord] | None = None,
    mentions: list[MentionRecord] | None = None,
    claims: list[ClaimRecord] | None = None,
    entity_resolutions: list[EntityResolutionRecord] | None = None,
    claim_entity_links: list[ClaimEntityLinkRecord] | None = None,
    claim_location_links: list[ClaimLocationLinkRecord] | None = None,
    claim_link_diagnostics: list[ClaimLinkDiagnosticRecord] | None = None,
) -> SemanticBundle:
    return SemanticBundle(
        extraction_run=ExtractionRunRecord(run_id="run_1"),
        claims=claims or [],
        measurements=[],
        mentions=mentions or [],
        entities=entities or [],
        entity_resolutions=entity_resolutions or [],
        claim_entity_links=claim_entity_links or [],
        claim_link_diagnostics=claim_link_diagnostics or [],
        claim_location_links=claim_location_links or [],
        claim_period_links=[],
        document_refuge_links=[],
        document_period_links=[],
        document_signed_by_links=[],
        person_affiliation_links=[],
    )


class TestOcrEntityDetector:
    def test_detects_similar_entities(self):
        entities = [
            EntityRecord(entity_id="e1", entity_type="Species", name="Mallard", normalized_form="mallard"),
            EntityRecord(entity_id="e2", entity_type="Species", name="Mallerd", normalized_form="mallerd"),
        ]
        resolutions = [
            EntityResolutionRecord(mention_id="m1", entity_id="e1", relation_type="REFERS_TO", match_score=0.9),
            EntityResolutionRecord(mention_id="m2", entity_id="e2", relation_type="REFERS_TO", match_score=0.9),
        ]
        semantic = _make_semantic(entities=entities, entity_resolutions=resolutions)
        structure = _make_structure()
        proposals = ocr_entity.detect(structure, semantic, "snap1")
        assert len(proposals) >= 1
        assert proposals[0].proposal_type in ("merge_entities", "create_alias")

    def test_no_proposals_for_different_entities(self):
        entities = [
            EntityRecord(entity_id="e1", entity_type="Species", name="Mallard", normalized_form="mallard"),
            EntityRecord(entity_id="e2", entity_type="Species", name="Coot", normalized_form="coot"),
        ]
        semantic = _make_semantic(entities=entities)
        structure = _make_structure()
        proposals = ocr_entity.detect(structure, semantic, "snap1")
        assert len(proposals) == 0

    def test_no_cross_type_merge(self):
        entities = [
            EntityRecord(entity_id="e1", entity_type="Species", name="Mallard", normalized_form="mallard"),
            EntityRecord(entity_id="e2", entity_type="Place", name="Mallard", normalized_form="mallard"),
        ]
        semantic = _make_semantic(entities=entities)
        structure = _make_structure()
        proposals = ocr_entity.detect(structure, semantic, "snap1")
        # Should not propose merging across entity types
        for p in proposals:
            patch = p.patch_spec
            if patch.get("proposal_type") == "merge_entities":
                # All targets should be same type
                pass
        # No merge between Species and Place
        assert len(proposals) == 0


class TestJunkMentionDetector:
    def test_detects_short_generic_tokens(self):
        mentions = [
            MentionRecord(
                mention_id="m1", run_id="run_1", paragraph_id="para1",
                surface_form="the", normalized_form="the",
                start_offset=0, end_offset=3, detection_confidence=0.5,
            ),
        ]
        semantic = _make_semantic(mentions=mentions)
        structure = _make_structure()
        proposals = junk_mention.detect(structure, semantic, "snap1")
        # Should detect "the" as a short generic token
        assert any(p.issue_class == "short_generic_token" for p in proposals)

    def test_detects_ocr_garbage(self):
        mentions = [
            MentionRecord(
                mention_id="m1", run_id="run_1", paragraph_id="para1",
                surface_form="3847x", normalized_form="3847x",
                start_offset=0, end_offset=5, detection_confidence=0.5,
            ),
        ]
        semantic = _make_semantic(mentions=mentions)
        structure = _make_structure()
        proposals = junk_mention.detect(structure, semantic, "snap1")
        assert any(p.issue_class == "ocr_garbage_mention" for p in proposals)


class TestBuilderRepairDetector:
    def test_detects_missing_species_focus(self):
        claims = [
            ClaimRecord(
                claim_id="c1", run_id="run_1", paragraph_id="para1",
                claim_type="population_estimate",
                source_sentence="100 mallards counted",
                normalized_sentence="100 mallards counted",
                certainty="certain", extraction_confidence=0.9,
            ),
        ]
        entities = [
            EntityRecord(entity_id="e1", entity_type="Species", name="Mallard", normalized_form="mallard"),
        ]
        diagnostics = [
            ClaimLinkDiagnosticRecord(
                claim_id="c1", relation_type="SPECIES_FOCUS",
                surface_form="mallards", normalized_form="mallards",
                diagnostic_code="NO_RESOLVED_MENTION",
            ),
        ]
        semantic = _make_semantic(
            claims=claims, entities=entities,
            claim_link_diagnostics=diagnostics,
        )
        structure = _make_structure()
        proposals = builder_repair.detect(structure, semantic, "snap1")
        species_proposals = [p for p in proposals if p.issue_class == "missing_species_focus"]
        assert len(species_proposals) >= 1
        assert species_proposals[0].proposal_type == "add_claim_entity_link"

    def test_detects_missing_location(self):
        claims = [
            ClaimRecord(
                claim_id="c1", run_id="run_1", paragraph_id="para1",
                claim_type="population_estimate",
                source_sentence="100 mallards at Main Lake",
                normalized_sentence="100 mallards at main lake",
                certainty="certain", extraction_confidence=0.9,
            ),
        ]
        entities = [
            EntityRecord(entity_id="e1", entity_type="Place", name="Main Lake", normalized_form="main lake"),
        ]
        semantic = _make_semantic(claims=claims, entities=entities)
        structure = _make_structure()
        proposals = builder_repair.detect(structure, semantic, "snap1")
        location_proposals = [p for p in proposals if p.issue_class == "missing_event_location"]
        assert len(location_proposals) >= 1
        assert location_proposals[0].proposal_type == "add_claim_location_link"

    def test_detects_method_overtrigger(self):
        claims = [
            ClaimRecord(
                claim_id="c1", run_id="run_1", paragraph_id="para1",
                claim_type="fire_incident",
                source_sentence="Fire burned 500 acres",
                normalized_sentence="fire burned 500 acres",
                certainty="certain", extraction_confidence=0.9,
            ),
        ]
        entities = [
            EntityRecord(entity_id="e1", entity_type="SurveyMethod", name="banding", normalized_form="banding"),
        ]
        claim_entity_links = [
            ClaimEntityLinkRecord(claim_id="c1", entity_id="e1", relation_type="METHOD_FOCUS"),
        ]
        semantic = _make_semantic(
            claims=claims, entities=entities,
            claim_entity_links=claim_entity_links,
        )
        structure = _make_structure()
        proposals = builder_repair.detect(structure, semantic, "snap1")
        method_proposals = [p for p in proposals if p.issue_class == "method_overtrigger"]
        assert len(method_proposals) >= 1
        assert method_proposals[0].proposal_type == "exclude_claim_from_derivation"
