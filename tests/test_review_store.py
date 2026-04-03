"""Tests for SQLite review store."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from gemynd.review.models import (
    CorrectionEvent,
    Proposal,
    ProposalRevision,
    ProposalTarget,
    ReviewRun,
)
from gemynd.review.store import ReviewStore


@pytest.fixture
def store(tmp_path: Path) -> ReviewStore:
    db = tmp_path / "review.db"
    s = ReviewStore(db)
    yield s
    s.close()


@pytest.fixture
def sample_run() -> ReviewRun:
    return ReviewRun(
        review_run_id="rr_test1",
        snapshot_id="snap_test1",
        doc_id="doc_1",
        structure_bundle_path="/path/to/struct.json",
        semantic_bundle_path="/path/to/sem.json",
        structure_bundle_sha256="sha_struct",
        semantic_bundle_sha256="sha_sem",
        extraction_run_id="run_1",
        created_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_proposal() -> Proposal:
    return Proposal(
        proposal_id="prop_test1",
        review_run_id="rr_test1",
        snapshot_id="snap_test1",
        anti_pattern_id="ap_ocr_spelling",
        issue_class="ocr_spelling_variant",
        proposal_type="merge_entities",
        status="queued",
        confidence=0.9,
        priority_score=0.72,
        impact_size=3,
        current_revision_id="rev_test1",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_targets() -> list[ProposalTarget]:
    return [
        ProposalTarget(proposal_id="prop_test1", target_kind="entity", target_id="e1", target_role="canonical"),
        ProposalTarget(proposal_id="prop_test1", target_kind="entity", target_id="e2", target_role="merge_source"),
    ]


class TestSchemaCreation:
    def test_creates_tables(self, store: ReviewStore):
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "review_run" in table_names
        assert "proposal" in table_names
        assert "proposal_target" in table_names
        assert "proposal_revision" in table_names
        assert "correction_event" in table_names
        assert "anti_pattern_class" in table_names

    def test_default_anti_patterns_loaded(self, store: ReviewStore):
        aps = store.get_anti_pattern_classes()
        assert len(aps) >= 9
        ids = {ap.anti_pattern_id for ap in aps}
        assert "ap_ocr_spelling" in ids
        assert "ap_missing_species" in ids


class TestReviewRun:
    def test_save_and_get(self, store: ReviewStore, sample_run: ReviewRun):
        store.save_review_run(sample_run)
        loaded = store.get_review_run("rr_test1")
        assert loaded is not None
        assert loaded.snapshot_id == "snap_test1"
        assert loaded.doc_id == "doc_1"

    def test_get_nonexistent(self, store: ReviewStore):
        assert store.get_review_run("does_not_exist") is None


class TestProposal:
    def test_upsert_and_get(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        loaded = store.get_proposal("prop_test1")
        assert loaded is not None
        assert loaded.issue_class == "ocr_spelling_variant"
        assert loaded.status == "queued"

    def test_get_targets(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        targets = store.get_proposal_targets("prop_test1")
        assert len(targets) == 2
        roles = {t.target_role for t in targets}
        assert "canonical" in roles
        assert "merge_source" in roles

    def test_upsert_idempotent(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        sample_proposal.confidence = 0.95
        store.upsert_proposal(sample_proposal, sample_targets)
        loaded = store.get_proposal("prop_test1")
        assert loaded is not None
        assert loaded.confidence == 0.95

    def test_list_proposals_default_sort(self, store: ReviewStore, sample_run: ReviewRun):
        store.save_review_run(sample_run)
        # Insert 3 proposals with different priorities
        for i, (priority, confidence) in enumerate([(0.9, 0.8), (0.5, 0.4), (0.7, 0.6)]):
            p = Proposal(
                proposal_id=f"prop_{i}",
                review_run_id="rr_test1",
                snapshot_id="snap_test1",
                anti_pattern_id="ap_ocr_spelling",
                issue_class="ocr_spelling_variant",
                proposal_type="merge_entities",
                priority_score=priority,
                confidence=confidence,
                impact_size=2,
            )
            store.upsert_proposal(p, [])

        listed = store.list_proposals()
        assert len(listed) == 3
        # Should be sorted by priority_score DESC
        assert listed[0].proposal_id == "prop_0"
        assert listed[1].proposal_id == "prop_2"
        assert listed[2].proposal_id == "prop_1"

    def test_list_filter_by_status(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        queued = store.list_proposals(status="queued")
        assert len(queued) == 1
        rejected = store.list_proposals(status="rejected")
        assert len(rejected) == 0

    def test_update_status(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        store.update_proposal_status("prop_test1", "accepted_pending_apply")
        loaded = store.get_proposal("prop_test1")
        assert loaded is not None
        assert loaded.status == "accepted_pending_apply"


class TestRevision:
    def test_save_and_get(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        rev = ProposalRevision(
            revision_id="rev_test1",
            proposal_id="prop_test1",
            revision_number=1,
            revision_kind="generated",
            patch_spec_json='{"schema_version":"v1"}',
            patch_spec_fingerprint="fp1",
            detector_name="test_detector",
            detector_version="v1",
            created_at="2024-01-01T00:00:00Z",
        )
        store.save_revision(rev)
        revisions = store.get_revisions("prop_test1")
        assert len(revisions) == 1
        assert revisions[0].detector_name == "test_detector"

    def test_latest_revision(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        for i in range(1, 4):
            rev = ProposalRevision(
                revision_id=f"rev_{i}",
                proposal_id="prop_test1",
                revision_number=i,
                revision_kind="generated" if i == 1 else "edited",
                patch_spec_json='{}',
                patch_spec_fingerprint=f"fp{i}",
            )
            store.save_revision(rev)
        latest = store.get_latest_revision("prop_test1")
        assert latest is not None
        assert latest.revision_number == 3

    def test_next_revision_number(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        assert store.next_revision_number("prop_test1") == 1
        store.save_revision(ProposalRevision(
            revision_id="rev_1", proposal_id="prop_test1", revision_number=1,
            revision_kind="generated", patch_spec_json='{}', patch_spec_fingerprint="fp",
        ))
        assert store.next_revision_number("prop_test1") == 2


class TestCorrectionEvent:
    def test_save_and_get(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        store.save_revision(ProposalRevision(
            revision_id="rev_test1", proposal_id="prop_test1", revision_number=1,
            revision_kind="generated", patch_spec_json='{}', patch_spec_fingerprint="fp",
        ))
        event = CorrectionEvent(
            event_id="ce_1",
            proposal_id="prop_test1",
            revision_id="rev_test1",
            action="accept",
            reviewer="tester",
            reviewer_note="looks good",
            created_at="2024-01-01T00:00:00Z",
        )
        store.save_correction_event(event)
        events = store.get_correction_events("prop_test1")
        assert len(events) == 1
        assert events[0].reviewer == "tester"
        assert events[0].reviewer_note == "looks good"


class TestCounts:
    def test_counts_by_status(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        counts = store.proposal_counts_by_status()
        assert counts.get("queued") == 1

    def test_counts_by_queue(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        counts = store.proposal_counts_by_queue()
        assert counts.get("ocr_entity") == 1


class TestExport:
    def test_export_proposals_json(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        data = store.export_proposals_json()
        assert len(data) == 1
        assert "targets" in data[0]
        assert len(data[0]["targets"]) == 2

    def test_export_accepted_patches_empty(self, store: ReviewStore, sample_run: ReviewRun, sample_proposal: Proposal, sample_targets: list[ProposalTarget]):
        store.save_review_run(sample_run)
        store.upsert_proposal(sample_proposal, sample_targets)
        patches = store.export_accepted_patches()
        assert patches == []
