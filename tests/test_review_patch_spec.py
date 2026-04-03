"""Tests for typed patch spec validation."""
from __future__ import annotations

import pytest

from gemynd.review.patch_spec import (
    PatchSpecValidationError,
    make_patch_spec,
    validate_patch_spec,
)


class TestValidateMergeEntities:
    def test_valid(self):
        spec = make_patch_spec(
            "merge_entities",
            canonical_entity_id="e1",
            merge_entity_ids=["e2", "e3"],
            canonical_name="Foo",
        )
        validate_patch_spec(spec)  # should not raise

    def test_missing_required_key(self):
        spec = {"schema_version": "v1", "proposal_type": "merge_entities", "canonical_entity_id": "e1"}
        with pytest.raises(PatchSpecValidationError, match="missing required"):
            validate_patch_spec(spec)

    def test_unknown_key(self):
        spec = make_patch_spec(
            "merge_entities",
            canonical_entity_id="e1",
            merge_entity_ids=["e2"],
            canonical_name="Foo",
            extra_key="bad",
        )
        with pytest.raises(PatchSpecValidationError, match="unknown keys"):
            validate_patch_spec(spec)

    def test_canonical_in_merge_ids(self):
        spec = make_patch_spec(
            "merge_entities",
            canonical_entity_id="e1",
            merge_entity_ids=["e1", "e2"],
            canonical_name="Foo",
        )
        with pytest.raises(PatchSpecValidationError, match="must not be in merge_entity_ids"):
            validate_patch_spec(spec)

    def test_unsorted_merge_ids(self):
        spec = make_patch_spec(
            "merge_entities",
            canonical_entity_id="e1",
            merge_entity_ids=["e3", "e2"],
            canonical_name="Foo",
        )
        with pytest.raises(PatchSpecValidationError, match="must be sorted"):
            validate_patch_spec(spec)

    def test_duplicate_merge_ids(self):
        spec = make_patch_spec(
            "merge_entities",
            canonical_entity_id="e1",
            merge_entity_ids=["e2", "e2"],
            canonical_name="Foo",
        )
        with pytest.raises(PatchSpecValidationError, match="must contain unique"):
            validate_patch_spec(spec)

    def test_valid_with_alias_mode(self):
        spec = make_patch_spec(
            "merge_entities",
            canonical_entity_id="e1",
            merge_entity_ids=["e2"],
            canonical_name="Foo",
            alias_mode="preserve_aliases",
        )
        validate_patch_spec(spec)

    def test_invalid_alias_mode(self):
        spec = make_patch_spec(
            "merge_entities",
            canonical_entity_id="e1",
            merge_entity_ids=["e2"],
            canonical_name="Foo",
            alias_mode="invalid_mode",
        )
        with pytest.raises(PatchSpecValidationError, match="alias_mode"):
            validate_patch_spec(spec)


class TestValidateCreateAlias:
    def test_valid(self):
        spec = make_patch_spec(
            "create_alias",
            canonical_entity_id="e1",
            alias_entity_id="e2",
            canonical_name="Foo",
        )
        validate_patch_spec(spec)

    def test_same_ids(self):
        spec = make_patch_spec(
            "create_alias",
            canonical_entity_id="e1",
            alias_entity_id="e1",
            canonical_name="Foo",
        )
        with pytest.raises(PatchSpecValidationError, match="must differ"):
            validate_patch_spec(spec)


class TestValidateSuppressMention:
    def test_valid(self):
        spec = make_patch_spec(
            "suppress_mention",
            mention_ids=["m1", "m2"],
            suppression_reason="ocr_garbage",
            scope="semantic_only",
        )
        validate_patch_spec(spec)

    def test_empty_mention_ids(self):
        spec = make_patch_spec(
            "suppress_mention",
            mention_ids=[],
            suppression_reason="ocr_garbage",
            scope="semantic_only",
        )
        with pytest.raises(PatchSpecValidationError, match="must not be empty"):
            validate_patch_spec(spec)

    def test_invalid_reason(self):
        spec = make_patch_spec(
            "suppress_mention",
            mention_ids=["m1"],
            suppression_reason="invalid_reason",
            scope="semantic_only",
        )
        with pytest.raises(PatchSpecValidationError, match="suppression_reason"):
            validate_patch_spec(spec)

    def test_invalid_scope(self):
        spec = make_patch_spec(
            "suppress_mention",
            mention_ids=["m1"],
            suppression_reason="ocr_garbage",
            scope="invalid_scope",
        )
        with pytest.raises(PatchSpecValidationError, match="scope"):
            validate_patch_spec(spec)


class TestValidateRelabelClaimLink:
    def test_valid(self):
        spec = make_patch_spec(
            "relabel_claim_link",
            claim_id="c1",
            entity_id="e1",
            old_relation_type="METHOD_FOCUS",
            new_relation_type="TOPIC_OF_CLAIM",
            evidence_basis="claim_link_diagnostic",
        )
        validate_patch_spec(spec)

    def test_same_relation_types(self):
        spec = make_patch_spec(
            "relabel_claim_link",
            claim_id="c1",
            entity_id="e1",
            old_relation_type="METHOD_FOCUS",
            new_relation_type="METHOD_FOCUS",
            evidence_basis="claim_link_diagnostic",
        )
        with pytest.raises(PatchSpecValidationError, match="must differ"):
            validate_patch_spec(spec)


class TestValidateAddClaimEntityLink:
    def test_valid(self):
        spec = make_patch_spec(
            "add_claim_entity_link",
            claim_id="c1",
            entity_id="e1",
            relation_type="SPECIES_FOCUS",
            evidence_basis="claim_link_diagnostic",
        )
        validate_patch_spec(spec)

    def test_invalid_relation_type(self):
        spec = make_patch_spec(
            "add_claim_entity_link",
            claim_id="c1",
            entity_id="e1",
            relation_type="INVALID_RELATION",
            evidence_basis="claim_link_diagnostic",
        )
        with pytest.raises(PatchSpecValidationError, match="relation_type"):
            validate_patch_spec(spec)


class TestValidateAddClaimLocationLink:
    def test_valid(self):
        spec = make_patch_spec(
            "add_claim_location_link",
            claim_id="c1",
            entity_id="e1",
            relation_type="OCCURRED_AT",
            evidence_basis="document_context",
        )
        validate_patch_spec(spec)

    def test_wrong_relation_type(self):
        spec = make_patch_spec(
            "add_claim_location_link",
            claim_id="c1",
            entity_id="e1",
            relation_type="SPECIES_FOCUS",
            evidence_basis="document_context",
        )
        with pytest.raises(PatchSpecValidationError, match="must be 'OCCURRED_AT'"):
            validate_patch_spec(spec)


class TestValidateExcludeClaimFromDerivation:
    def test_valid(self):
        spec = make_patch_spec(
            "exclude_claim_from_derivation",
            claim_id="c1",
            derivation_kind="observation",
            reason="method_overtrigger",
        )
        validate_patch_spec(spec)

    def test_invalid_derivation_kind(self):
        spec = make_patch_spec(
            "exclude_claim_from_derivation",
            claim_id="c1",
            derivation_kind="invalid",
            reason="method_overtrigger",
        )
        with pytest.raises(PatchSpecValidationError, match="derivation_kind"):
            validate_patch_spec(spec)

    def test_invalid_reason(self):
        spec = make_patch_spec(
            "exclude_claim_from_derivation",
            claim_id="c1",
            derivation_kind="observation",
            reason="arbitrary_text",
        )
        with pytest.raises(PatchSpecValidationError, match="reason"):
            validate_patch_spec(spec)


class TestTopLevelValidation:
    def test_not_a_dict(self):
        with pytest.raises(PatchSpecValidationError, match="must be a dict"):
            validate_patch_spec("not a dict")  # type: ignore[arg-type]

    def test_missing_schema_version(self):
        with pytest.raises(PatchSpecValidationError, match="schema_version"):
            validate_patch_spec({"proposal_type": "merge_entities"})

    def test_unknown_schema_version(self):
        with pytest.raises(PatchSpecValidationError, match="Unsupported schema_version"):
            validate_patch_spec({"schema_version": "v999", "proposal_type": "merge_entities"})

    def test_unknown_proposal_type(self):
        with pytest.raises(PatchSpecValidationError, match="Unknown proposal_type"):
            validate_patch_spec({"schema_version": "v1", "proposal_type": "unknown_type"})
