"""Synthetic bundle fixtures for kettle_eval tests.

Keeps tests independent of report JSON fixtures and the full pipeline. Each
fixture constructs the minimum StructureBundle + SemanticBundle needed to
exercise the case factory, risk features, and selector.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gemynd.core.models import (
    ClaimEntityLinkRecord,
    ClaimLocationLinkRecord,
    ClaimPeriodLinkRecord,
    ClaimRecord,
    DocumentRecord,
    DocumentPeriodLinkRecord,
    DocumentSignedByLinkRecord,
    EntityRecord,
    EntityResolutionRecord,
    ExtractionRunRecord,
    MentionRecord,
    PageRecord,
    ParagraphRecord,
    PersonAffiliationLinkRecord,
    SectionRecord,
    SemanticBundle,
    StructureBundle,
)


def _make_paragraph(doc_id: str, idx: int, page_id: str, text: str) -> ParagraphRecord:
    return ParagraphRecord(
        paragraph_id=f"{doc_id}:para:{idx:03d}",
        doc_id=doc_id,
        page_id=page_id,
        section_id=None,
        paragraph_index=idx,
        page_number=1,
        raw_ocr_text=text,
        clean_text=text,
        char_count=len(text),
    )


def _make_page(doc_id: str, number: int, ocr_confidence: float | None) -> PageRecord:
    return PageRecord(
        page_id=f"{doc_id}:page:{number:04d}",
        doc_id=doc_id,
        page_number=number,
        raw_ocr_text="page text",
        clean_text="page text",
        ocr_confidence=ocr_confidence,
        image_ref=None,
    )


def _make_document(doc_id: str, title: str, doc_type: str = "narrative_report") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        title=title,
        doc_type=doc_type,
        institution_id="test_corpus",
    )


def _make_run(run_id: str = "run_test_001") -> ExtractionRunRecord:
    return ExtractionRunRecord(
        run_id=run_id,
        ocr_engine="tesseract",
        ocr_version="5.0",
        ner_model="hybrid-rules-llm",
        relation_model="hybrid-rules-llm",
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        config_fingerprint="cfg_test",
    )


def _empty_semantic(run: ExtractionRunRecord) -> SemanticBundle:
    return SemanticBundle(
        extraction_run=run,
        claims=[],
        measurements=[],
        mentions=[],
        entities=[],
        entity_resolutions=[],
        claim_entity_links=[],
        claim_link_diagnostics=[],
        claim_location_links=[],
        claim_period_links=[],
        document_anchor_links=[],
        document_period_links=[],
        document_signed_by_links=[],
        person_affiliation_links=[],
    )


def build_minimal_bundle(
    *,
    run_id: str = "run_test_001",
    doc_id: str = "doc_test_001",
    doc_type: str = "narrative_report",
    n_claims: int = 2,
    n_mentions: int = 3,
    n_resolutions: int = 2,
    n_links: int = 1,
    page_ocr_confidence: float = 0.85,
    claim_extraction_confidence: float = 0.80,
    mention_detection_confidence: float = 0.75,
    resolution_match_score: float = 0.88,
    include_evidence_offsets: bool = True,
    include_source_span: bool = True,
) -> tuple[StructureBundle, SemanticBundle]:
    """Construct a synthetic structure+semantic bundle pair for testing."""
    document = _make_document(doc_id, "Test Report", doc_type=doc_type)
    page = _make_page(doc_id, 1, page_ocr_confidence)
    paragraph = _make_paragraph(doc_id, 0, page.page_id, "Test paragraph content.")
    structure = StructureBundle(
        document=document,
        pages=[page],
        sections=[],
        paragraphs=[paragraph],
        annotations=[],
    )

    run = _make_run(run_id)
    semantic = _empty_semantic(run)

    for i in range(n_claims):
        semantic.claims.append(ClaimRecord(
            claim_id=f"{run_id}:claim:{i:03d}",
            run_id=run_id,
            paragraph_id=paragraph.paragraph_id if include_source_span else None,
            claim_type="observation_count",
            source_sentence=f"Claim sentence {i}.",
            normalized_sentence=f"claim sentence {i}",
            certainty="certain",
            extraction_confidence=claim_extraction_confidence,
            evidence_start=0 if include_evidence_offsets else None,
            evidence_end=20 if include_evidence_offsets else None,
        ))

    for i in range(n_mentions):
        semantic.mentions.append(MentionRecord(
            mention_id=f"{run_id}:mention:{i:03d}",
            run_id=run_id,
            paragraph_id=paragraph.paragraph_id,
            surface_form=f"Species {i}",
            normalized_form=f"species {i}",
            start_offset=i * 10,
            end_offset=i * 10 + 8,
            detection_confidence=mention_detection_confidence,
        ))

    for i in range(n_resolutions):
        semantic.entity_resolutions.append(EntityResolutionRecord(
            mention_id=f"{run_id}:mention:{i:03d}",
            entity_id=f"species_{i:03d}",
            relation_type="REFERS_TO",
            match_score=resolution_match_score,
        ))

    for i in range(n_links):
        if i < len(semantic.claims):
            semantic.claim_entity_links.append(ClaimEntityLinkRecord(
                claim_id=semantic.claims[i].claim_id,
                entity_id=f"species_{i:03d}",
                relation_type="SPECIES_FOCUS",
            ))

    return structure, semantic


@pytest.fixture
def minimal_bundle():
    return build_minimal_bundle()


@pytest.fixture
def bundle_factory():
    return build_minimal_bundle
