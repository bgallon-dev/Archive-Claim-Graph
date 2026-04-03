"""Tests for review subsystem ID generation."""
from __future__ import annotations

import pytest

from gemynd.review.ids import (
    make_claim_entity_link_target_id,
    make_claim_location_link_target_id,
    make_correction_event_id,
    make_proposal_id,
    make_review_run_id,
    make_revision_id,
    make_snapshot_id,
    patch_spec_fingerprint,
    _canonicalize_patch_spec,
)


class TestSnapshotId:
    def test_deterministic(self):
        id1 = make_snapshot_id("doc1", "sha_struct", "sha_sem", "v1", "run1")
        id2 = make_snapshot_id("doc1", "sha_struct", "sha_sem", "v1", "run1")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = make_snapshot_id("doc1", "sha_a", "sha_b", "v1", "")
        id2 = make_snapshot_id("doc2", "sha_a", "sha_b", "v1", "")
        assert id1 != id2

    def test_length(self):
        sid = make_snapshot_id("doc1", "sha_struct", "sha_sem")
        assert len(sid) == 32


class TestLinkTargetIds:
    def test_claim_entity_link(self):
        tid = make_claim_entity_link_target_id("c1", "SPECIES_FOCUS", "e1")
        assert tid == "link::claim_entity::c1::SPECIES_FOCUS::e1"

    def test_claim_location_link(self):
        tid = make_claim_location_link_target_id("c1", "e1")
        assert tid == "link::claim_location::c1::OCCURRED_AT::e1"


class TestProposalId:
    def test_deterministic(self):
        targets = [("entity", "e1", "canonical"), ("entity", "e2", "merge_source")]
        patch = {"schema_version": "v1", "proposal_type": "merge_entities", "canonical_entity_id": "e1", "merge_entity_ids": ["e2"], "canonical_name": "Foo"}
        id1 = make_proposal_id("snap1", "ocr_spelling_variant", "merge_entities", targets, patch)
        id2 = make_proposal_id("snap1", "ocr_spelling_variant", "merge_entities", targets, patch)
        assert id1 == id2

    def test_target_order_independent(self):
        targets_a = [("entity", "e1", "canonical"), ("entity", "e2", "merge_source")]
        targets_b = [("entity", "e2", "merge_source"), ("entity", "e1", "canonical")]
        patch = {"schema_version": "v1", "proposal_type": "merge_entities", "canonical_entity_id": "e1", "merge_entity_ids": ["e2"], "canonical_name": "Foo"}
        id_a = make_proposal_id("snap1", "ocr_spelling_variant", "merge_entities", targets_a, patch)
        id_b = make_proposal_id("snap1", "ocr_spelling_variant", "merge_entities", targets_b, patch)
        assert id_a == id_b

    def test_different_patch_different_id(self):
        targets = [("entity", "e1", "canonical")]
        patch_a = {"schema_version": "v1", "proposal_type": "merge_entities", "canonical_entity_id": "e1", "merge_entity_ids": ["e2"], "canonical_name": "Foo"}
        patch_b = {"schema_version": "v1", "proposal_type": "merge_entities", "canonical_entity_id": "e1", "merge_entity_ids": ["e3"], "canonical_name": "Foo"}
        id_a = make_proposal_id("snap1", "ocr_spelling_variant", "merge_entities", targets, patch_a)
        id_b = make_proposal_id("snap1", "ocr_spelling_variant", "merge_entities", targets, patch_b)
        assert id_a != id_b


class TestPatchSpecFingerprint:
    def test_deterministic(self):
        spec = {"schema_version": "v1", "proposal_type": "suppress_mention", "mention_ids": ["m2", "m1"], "suppression_reason": "ocr_garbage", "scope": "semantic_only"}
        fp1 = patch_spec_fingerprint(spec)
        fp2 = patch_spec_fingerprint(spec)
        assert fp1 == fp2

    def test_key_order_independent(self):
        spec_a = {"schema_version": "v1", "proposal_type": "suppress_mention", "mention_ids": ["m1"], "suppression_reason": "ocr_garbage", "scope": "semantic_only"}
        spec_b = {"scope": "semantic_only", "suppression_reason": "ocr_garbage", "mention_ids": ["m1"], "proposal_type": "suppress_mention", "schema_version": "v1"}
        assert patch_spec_fingerprint(spec_a) == patch_spec_fingerprint(spec_b)


class TestCanonicalizeSpec:
    def test_sorts_string_lists(self):
        spec = {"mention_ids": ["m3", "m1", "m2"]}
        canonical = _canonicalize_patch_spec(spec)
        assert '"m1","m2","m3"' in canonical.replace(" ", "")


class TestHelperIds:
    def test_review_run_id(self):
        rid = make_review_run_id("snap1", "2024-01-01T00:00:00Z")
        assert rid.startswith("rr_")

    def test_revision_id(self):
        rid = make_revision_id("prop1", 1)
        assert rid.startswith("rev_")

    def test_correction_event_id(self):
        eid = make_correction_event_id("prop1", "accept", "2024-01-01T00:00:00Z")
        assert eid.startswith("ce_")
