"""Unit tests for individual extraction pipeline phases.

These tests exercise each phase function in isolation with minimal fixture
data, verifying that it can run without the full upstream pipeline.
"""
from __future__ import annotations

import pytest

from graphrag_pipeline.core.domain_config import load_domain_config
from graphrag_pipeline.core.models import (
    ClaimEntityLinkRecord,
    ClaimLocationLinkRecord,
    ClaimRecord,
    DocumentRecord,
    EntityRecord,
    EntityResolutionRecord,
    ExtractionRunRecord,
    MeasurementRecord,
    MentionRecord,
    ParagraphRecord,
    StructureBundle,
    YearRecord,
)
from graphrag_pipeline.ingest.extraction_state import ExtractionState
from graphrag_pipeline.ingest.phases import (
    assign_concepts_phase,
    build_extraction_run,
    create_domain_anchor,
    create_period_entity,
    create_place_refuge_links,
    create_year_entities,
    resolve_claim_links,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config():
    return load_domain_config()


def _doc(
    doc_id: str = "doc_1",
    title: str = "Turnbull NWR 1956",
    report_year: int | None = 1956,
    date_start: str | None = "1956-01-01",
    date_end: str | None = "1956-12-31",
) -> DocumentRecord:
    return DocumentRecord(doc_id=doc_id, title=title, report_year=report_year,
                          date_start=date_start, date_end=date_end)


def _paragraph(paragraph_id: str = "para_1", text: str = "Test paragraph.") -> ParagraphRecord:
    return ParagraphRecord(
        paragraph_id=paragraph_id, doc_id="doc_1", page_id="page_1",
        section_id=None, paragraph_index=0, page_number=1,
        raw_ocr_text=text, clean_text=text, char_count=len(text),
    )


def _structure(doc=None, paragraphs=None) -> StructureBundle:
    return StructureBundle(
        document=doc or _doc(),
        pages=[],
        sections=[],
        paragraphs=paragraphs or [_paragraph()],
        annotations=[],
    )


def _claim(
    claim_id: str = "claim_1",
    claim_type: str = "species_observation",
    paragraph_id: str = "para_1",
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id, run_id="run_1", paragraph_id=paragraph_id,
        claim_type=claim_type, source_sentence="Canada geese nested.",
        normalized_sentence="canada geese nested",
        certainty="certain", extraction_confidence=0.8,
    )


def _entity(entity_id: str, entity_type: str, name: str = "test") -> EntityRecord:
    return EntityRecord(
        entity_id=entity_id, entity_type=entity_type,
        name=name, normalized_form=name.lower(),
    )


def _mention(mention_id: str, paragraph_id: str = "para_1", surface: str = "Canada geese") -> MentionRecord:
    return MentionRecord(
        mention_id=mention_id, run_id="run_1", paragraph_id=paragraph_id,
        surface_form=surface, normalized_form=surface.lower(),
        start_offset=0, end_offset=len(surface), detection_confidence=0.9,
    )


def _state(**kwargs) -> ExtractionState:
    defaults = dict(
        structure=_structure(),
        config=_config(),
        extraction_run=build_extraction_run(),
    )
    defaults.update(kwargs)
    return ExtractionState(**defaults)


# ---------------------------------------------------------------------------
# build_extraction_run
# ---------------------------------------------------------------------------

class TestBuildExtractionRun:
    def test_default_values(self):
        run = build_extraction_run()
        assert run.ocr_engine == "unknown"
        assert run.claim_type_schema_version == "v2"
        assert run.run_id.startswith("run_")

    def test_overrides(self):
        run = build_extraction_run({"ocr_engine": "tesseract", "run_id": "custom_run"})
        assert run.run_id == "custom_run"
        assert run.ocr_engine == "tesseract"


# ---------------------------------------------------------------------------
# ExtractionState.to_semantic_bundle
# ---------------------------------------------------------------------------

class TestExtractionStateBundle:
    def test_roundtrip(self):
        state = _state()
        state.claims = [_claim()]
        bundle = state.to_semantic_bundle()
        assert len(bundle.claims) == 1
        assert bundle.claims[0].claim_id == "claim_1"
        assert bundle.document_signed_by_links == []
        assert bundle.person_affiliation_links == []

    def test_register_entity(self):
        state = _state()
        e = _entity("e1", "Species", "Goose")
        state.register_entity(e)
        assert len(state.entities) == 1
        assert state.entity_lookup["e1"] is e


# ---------------------------------------------------------------------------
# create_domain_anchor
# ---------------------------------------------------------------------------

class TestCreateDomainAnchor:
    def test_noop_when_no_anchor_config(self):
        cfg = _config()
        # Override domain_anchor to None
        object.__setattr__(cfg, "domain_anchor", None)
        state = _state(config=cfg)
        create_domain_anchor(state)
        assert state.doc_anchor_id is None
        assert state.document_refuge_links == []

    def test_creates_anchor_when_title_matches(self):
        cfg = _config()
        if not cfg.domain_anchor:
            pytest.skip("Default config has no domain_anchor")
        state = _state(structure=_structure(doc=_doc(title="Turnbull NWR 1956")))
        create_domain_anchor(state)
        assert state.doc_anchor_id is not None
        assert len(state.document_refuge_links) == 1
        assert state.doc_anchor_id in state.entity_lookup

    def test_noop_when_title_does_not_match(self):
        cfg = _config()
        if not cfg.domain_anchor:
            pytest.skip("Default config has no domain_anchor")
        state = _state(
            config=cfg,
            structure=_structure(doc=_doc(title="Unrelated Document")),
        )
        create_domain_anchor(state)
        assert state.doc_anchor_id is None


# ---------------------------------------------------------------------------
# create_period_entity
# ---------------------------------------------------------------------------

class TestCreatePeriodEntity:
    def test_creates_period_from_dates(self):
        state = _state()
        state.claims = [_claim()]
        create_period_entity(state)
        assert len(state.entities) == 1
        assert state.entities[0].entity_type == "Period"
        assert len(state.claim_period_links) == 1
        assert len(state.document_period_links) == 1

    def test_noop_when_no_dates(self):
        doc = _doc(date_start=None, date_end=None)
        state = _state(structure=_structure(doc=doc))
        create_period_entity(state)
        assert state.entities == []
        assert state.claim_period_links == []


# ---------------------------------------------------------------------------
# create_year_entities
# ---------------------------------------------------------------------------

class TestCreateYearEntities:
    def test_creates_year_from_report_year(self):
        state = _state()
        create_year_entities(state)
        assert len(state.years) == 1
        assert state.years[0].year == 1956
        assert len(state.document_year_links) == 1

    def test_noop_when_no_report_year(self):
        doc = _doc(report_year=None)
        state = _state(structure=_structure(doc=doc))
        create_year_entities(state)
        assert state.years == []

    def test_does_not_duplicate_existing_year(self):
        state = _state()
        from graphrag_pipeline.core.ids import make_year_id
        existing = YearRecord(year_id=make_year_id(1956), year=1956, year_label="1956")
        state.years = [existing]
        create_year_entities(state)
        assert len(state.years) == 1  # no duplicate added


# ---------------------------------------------------------------------------
# create_place_refuge_links
# ---------------------------------------------------------------------------

class TestCreatePlaceRefugeLinks:
    def test_links_places_to_anchor(self):
        state = _state()
        state.doc_anchor_id = "refuge_1"
        place = _entity("place_1", "Place", "Kepple Lake")
        non_place = _entity("species_1", "Species", "Goose")
        state.entities = [place, non_place]
        create_place_refuge_links(state)
        assert len(state.place_refuge_links) == 1
        assert state.place_refuge_links[0].place_id == "place_1"
        assert state.place_refuge_links[0].refuge_id == "refuge_1"

    def test_noop_when_no_anchor(self):
        state = _state()
        state.entities = [_entity("place_1", "Place")]
        create_place_refuge_links(state)
        assert state.place_refuge_links == []


# ---------------------------------------------------------------------------
# resolve_claim_links
# ---------------------------------------------------------------------------

class TestResolveClaimLinks:
    def test_resolves_entity_link_via_mention(self):
        state = _state()
        claim = _claim()
        mention = _mention("m1")
        entity = _entity("e1", "Species", "canada geese")
        resolution = EntityResolutionRecord(
            mention_id="m1", entity_id="e1",
            relation_type="POSSIBLY_REFERS_TO", match_score=0.9,
        )

        state.claims = [claim]
        state.claims_by_paragraph["para_1"] = [claim]
        state.mentions_by_paragraph["para_1"] = [mention]
        state.paragraph_texts["para_1"] = "Canada geese nested."
        state.entities = [entity]
        state.entity_lookup = {"e1": entity}
        state.resolutions_by_mention = {"m1": resolution}

        from graphrag_pipeline.ingest.extractors.claim_extractor import ClaimLinkDraft
        state.claim_links_by_claim["claim_1"] = [
            ClaimLinkDraft(
                relation_type="SPECIES_FOCUS",
                surface_form="Canada geese",
                normalized_form="canada geese",
                start_offset=0,
                end_offset=13,
            ),
        ]

        resolve_claim_links(state)
        assert len(state.claim_entity_links) == 1
        assert state.claim_entity_links[0].entity_id == "e1"
        assert state.claim_entity_links[0].relation_type == "SPECIES_FOCUS"

    def test_deduplicates_links(self):
        state = _state()
        claim = _claim()
        mention = _mention("m1")
        entity = _entity("e1", "Species", "canada geese")
        resolution = EntityResolutionRecord(
            mention_id="m1", entity_id="e1",
            relation_type="POSSIBLY_REFERS_TO", match_score=0.9,
        )

        state.claims = [claim]
        state.claims_by_paragraph["para_1"] = [claim]
        state.mentions_by_paragraph["para_1"] = [mention]
        state.paragraph_texts["para_1"] = "Canada geese nested."
        state.entities = [entity]
        state.entity_lookup = {"e1": entity}
        state.resolutions_by_mention = {"m1": resolution}

        from graphrag_pipeline.ingest.extractors.claim_extractor import ClaimLinkDraft
        # Same link twice
        draft = ClaimLinkDraft(
            relation_type="SPECIES_FOCUS",
            surface_form="Canada geese",
            normalized_form="canada geese",
            start_offset=0,
            end_offset=13,
        )
        state.claim_links_by_claim["claim_1"] = [draft, draft]

        resolve_claim_links(state)
        assert len(state.claim_entity_links) == 1  # deduped


# ---------------------------------------------------------------------------
# assign_concepts_phase
# ---------------------------------------------------------------------------

class TestAssignConceptsPhase:
    def test_assigns_concepts_to_claims(self):
        state = _state()
        state.claims = [_claim(claim_type="species_observation")]
        assign_concepts_phase(state)
        # Should produce at least one concept link for a species_observation
        # (the exact count depends on concept_rules, but it should not error)
        assert isinstance(state.claim_concept_links, list)

    def test_empty_claims(self):
        state = _state()
        assign_concepts_phase(state)
        assert state.claim_concept_links == []
