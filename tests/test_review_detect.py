"""Tests for review detection orchestration and export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemynd.shared.io_utils import save_semantic_bundle, save_structure_bundle
from gemynd.core.models import (
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
from gemynd.review.detect import run_detection
from gemynd.review.export import (
    export_accepted_patches_json,
    export_proposals_csv,
    export_proposals_json,
)
from gemynd.review.store import ReviewStore
from gemynd.review.actions import accept_proposal, reject_proposal, defer_proposal


def _fixture_bundles(tmp_path: Path) -> tuple[StructureBundle, SemanticBundle, str, str]:
    doc = DocumentRecord(doc_id="doc_test", title="Test Report", source_file="test.json")
    pages = [PageRecord(page_id="p1", doc_id="doc_test", page_number=1, raw_ocr_text="text", clean_text="text")]
    paragraphs = [ParagraphRecord(
        paragraph_id="para1", doc_id="doc_test", page_id="p1",
        section_id=None, paragraph_index=0, page_number=1,
        raw_ocr_text="100 mallards counted at Main Lake",
        clean_text="100 mallards counted at Main Lake",
        char_count=35,
    )]
    structure = StructureBundle(document=doc, pages=pages, sections=[], paragraphs=paragraphs, annotations=[])

    entities = [
        EntityRecord(entity_id="e1", entity_type="Species", name="Mallard", normalized_form="mallard"),
        EntityRecord(entity_id="e2", entity_type="Species", name="Mallerd", normalized_form="mallerd"),
        EntityRecord(entity_id="e3", entity_type="Place", name="Main Lake", normalized_form="main lake"),
    ]
    mentions = [
        MentionRecord(mention_id="m1", run_id="run_1", paragraph_id="para1",
                      surface_form="mallards", normalized_form="mallards",
                      start_offset=4, end_offset=12, detection_confidence=0.9),
        MentionRecord(mention_id="m2", run_id="run_1", paragraph_id="para1",
                      surface_form="the", normalized_form="the",
                      start_offset=0, end_offset=3, detection_confidence=0.5),
    ]
    resolutions = [
        EntityResolutionRecord(mention_id="m1", entity_id="e1", relation_type="REFERS_TO", match_score=0.95),
    ]
    claims = [
        ClaimRecord(
            claim_id="c1", run_id="run_1", paragraph_id="para1",
            claim_type="population_estimate",
            source_sentence="100 mallards counted at Main Lake",
            normalized_sentence="100 mallards counted at main lake",
            certainty="certain", extraction_confidence=0.9,
        ),
    ]
    semantic = SemanticBundle(
        extraction_run=ExtractionRunRecord(run_id="run_1"),
        claims=claims,
        measurements=[],
        mentions=mentions,
        entities=entities,
        entity_resolutions=resolutions,
        claim_entity_links=[],
        claim_link_diagnostics=[],
        claim_location_links=[],
        claim_period_links=[],
        document_anchor_links=[],
        document_period_links=[],
        document_signed_by_links=[],
        person_affiliation_links=[],
    )

    struct_path = str(tmp_path / "test.structure.json")
    sem_path = str(tmp_path / "test.semantic.json")
    save_structure_bundle(struct_path, structure)
    save_semantic_bundle(sem_path, semantic)
    return structure, semantic, struct_path, sem_path


class TestRunDetection:
    def test_produces_proposals(self, tmp_path: Path):
        structure, semantic, struct_path, sem_path = _fixture_bundles(tmp_path)
        store = ReviewStore(tmp_path / "review.db")
        try:
            result = run_detection(structure, semantic, store, struct_path, sem_path)
            assert result["proposals_generated"] > 0
            assert result["proposals_upserted"] > 0
            assert result["snapshot_id"]
            assert result["review_run_id"]
        finally:
            store.close()

    def test_idempotent_detection(self, tmp_path: Path):
        structure, semantic, struct_path, sem_path = _fixture_bundles(tmp_path)
        store = ReviewStore(tmp_path / "review.db")
        try:
            result1 = run_detection(structure, semantic, store, struct_path, sem_path)
            result2 = run_detection(structure, semantic, store, struct_path, sem_path)
            # Same proposals should be upserted (not duplicated)
            proposals = store.list_proposals(limit=1000)
            proposal_ids = [p.proposal_id for p in proposals]
            # No duplicate proposal IDs
            assert len(proposal_ids) == len(set(proposal_ids))
        finally:
            store.close()

    def test_stable_proposal_ids(self, tmp_path: Path):
        structure, semantic, struct_path, sem_path = _fixture_bundles(tmp_path)
        store1 = ReviewStore(tmp_path / "review1.db")
        store2 = ReviewStore(tmp_path / "review2.db")
        try:
            result1 = run_detection(structure, semantic, store1, struct_path, sem_path)
            result2 = run_detection(structure, semantic, store2, struct_path, sem_path)
            # Same snapshot should produce same proposal IDs
            assert result1["snapshot_id"] == result2["snapshot_id"]
            proposals1 = {p.proposal_id for p in store1.list_proposals(limit=1000)}
            proposals2 = {p.proposal_id for p in store2.list_proposals(limit=1000)}
            assert proposals1 == proposals2
        finally:
            store1.close()
            store2.close()


class TestExport:
    def test_export_proposals_json(self, tmp_path: Path):
        structure, semantic, struct_path, sem_path = _fixture_bundles(tmp_path)
        store = ReviewStore(tmp_path / "review.db")
        try:
            run_detection(structure, semantic, store, struct_path, sem_path)
            output = tmp_path / "export.json"
            count = export_proposals_json(store, output)
            assert count > 0
            data = json.loads(output.read_text(encoding="utf-8"))
            assert len(data) == count
        finally:
            store.close()

    def test_export_proposals_csv(self, tmp_path: Path):
        structure, semantic, struct_path, sem_path = _fixture_bundles(tmp_path)
        store = ReviewStore(tmp_path / "review.db")
        try:
            run_detection(structure, semantic, store, struct_path, sem_path)
            output = tmp_path / "export.csv"
            count = export_proposals_csv(store, output)
            assert count > 0
            content = output.read_text(encoding="utf-8")
            assert "proposal_id" in content
        finally:
            store.close()

    def test_export_accepted_patches(self, tmp_path: Path):
        structure, semantic, struct_path, sem_path = _fixture_bundles(tmp_path)
        store = ReviewStore(tmp_path / "review.db")
        try:
            run_detection(structure, semantic, store, struct_path, sem_path)
            # Accept the first proposal
            proposals = store.list_proposals(limit=1)
            if proposals:
                accept_proposal(store, proposals[0].proposal_id, "tester")
            output = tmp_path / "patches.json"
            count = export_accepted_patches_json(store, output)
            assert count >= 1
            data = json.loads(output.read_text(encoding="utf-8"))
            assert len(data) == count
            assert "patch_spec" in data[0]
        finally:
            store.close()


class TestReviewActions:
    def test_accept_reject_defer(self, tmp_path: Path):
        structure, semantic, struct_path, sem_path = _fixture_bundles(tmp_path)
        store = ReviewStore(tmp_path / "review.db")
        try:
            run_detection(structure, semantic, store, struct_path, sem_path)
            proposals = store.list_proposals(limit=10)
            assert len(proposals) >= 1

            # Accept first
            pid = proposals[0].proposal_id
            accept_proposal(store, pid, "reviewer1", "good proposal")
            p = store.get_proposal(pid)
            assert p is not None
            assert p.status == "accepted_pending_apply"

            # Check correction event
            events = store.get_correction_events(pid)
            assert len(events) == 1
            assert events[0].action == "accept"
            assert events[0].reviewer == "reviewer1"

            # Reject another if available
            if len(proposals) >= 2:
                pid2 = proposals[1].proposal_id
                reject_proposal(store, pid2, "reviewer2", "not valid")
                p2 = store.get_proposal(pid2)
                assert p2 is not None
                assert p2.status == "rejected"

            # Defer another if available
            if len(proposals) >= 3:
                pid3 = proposals[2].proposal_id
                defer_proposal(store, pid3, "reviewer3")
                p3 = store.get_proposal(pid3)
                assert p3 is not None
                assert p3.status == "deferred"
        finally:
            store.close()
