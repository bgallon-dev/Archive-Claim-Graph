"""Adversarial tests for domain-agnosticism of the review detectors.

Covers the wiring of the ``config=`` seam into ``builder_repair`` and
``junk_mention`` (previously inert / Turnbull-hardcoded):

  * PARITY   — Turnbull's own config reproduces the legacy hardcoded behaviour
               byte-for-byte, both on the ``config is None`` fallback path and
               on the real ``config=<turnbull>`` path.
  * AGNOSTIC — a domain that declares no (or a different) policy does NOT
               inherit Turnbull's wildlife/institutional rules.
  * HOSTILE  — malformed / partial config must not crash the detector.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gemynd.core.domain_config import load_domain_config
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR
from gemynd.core.models import (
    ClaimEntityLinkRecord,
    ClaimLinkDiagnosticRecord,
    ClaimRecord,
    DocumentRecord,
    EntityRecord,
    ExtractionRunRecord,
    MentionRecord,
    PageRecord,
    ParagraphRecord,
    SemanticBundle,
    StructureBundle,
)
from gemynd.review.detectors import builder_repair, junk_mention
from gemynd.review import detect as review_detect
from gemynd.review.patch_spec import (
    ALLOWED_CLAIM_ENTITY_RELATIONS,
    PatchSpecValidationError,
    make_patch_spec,
    validate_patch_spec,
)


@pytest.fixture(scope="module")
def turnbull_cfg():
    return load_domain_config(TURNBULL_RESOURCES_DIR)


# --------------------------------------------------------------------------- #
# bundle helpers
# --------------------------------------------------------------------------- #
def _structure(paragraphs: list[ParagraphRecord], doc_id: str = "doc_1") -> StructureBundle:
    doc = DocumentRecord(doc_id=doc_id, title="T", source_file="t.json")
    pages = [PageRecord(page_id="p1", doc_id=doc_id, page_number=1, raw_ocr_text="", clean_text="")]
    return StructureBundle(document=doc, pages=pages, sections=[], paragraphs=paragraphs, annotations=[])


def _para(pid: str, idx: int, text: str) -> ParagraphRecord:
    return ParagraphRecord(
        paragraph_id=pid, doc_id="doc_1", page_id="p1", section_id=None,
        paragraph_index=idx, page_number=1, raw_ocr_text=text, clean_text=text,
        char_count=len(text),
    )


def _semantic(**kw) -> SemanticBundle:
    return SemanticBundle(
        extraction_run=ExtractionRunRecord(run_id="run_1"),
        claims=kw.get("claims", []),
        measurements=[],
        mentions=kw.get("mentions", []),
        entities=kw.get("entities", []),
        entity_resolutions=[],
        claim_entity_links=kw.get("claim_entity_links", []),
        claim_link_diagnostics=kw.get("claim_link_diagnostics", []),
        claim_location_links=kw.get("claim_location_links", []),
        claim_period_links=[],
        document_anchor_links=[],
        document_period_links=[],
        document_signed_by_links=[],
        person_affiliation_links=[],
    )


def _claim(cid: str, ctype: str) -> ClaimRecord:
    return ClaimRecord(
        claim_id=cid, run_id="run_1", paragraph_id="para1", claim_type=ctype,
        source_sentence="s", normalized_sentence="s",
        certainty="certain", extraction_confidence=0.9,
    )


def _proj(p) -> tuple:
    """Behavioural fingerprint of a proposal, independent of evidence blobs."""
    return (
        p.issue_class, p.proposal_type, p.anti_pattern_id,
        p.patch_spec.get("relation_type"),
        tuple(sorted(t.target_id for t in p.targets)),
    )


# --------------------------------------------------------------------------- #
# builder_repair — PARITY
# --------------------------------------------------------------------------- #
class TestBuilderRepairParity:
    def test_turnbull_config_policy_equals_module_default(self, turnbull_cfg):
        assert builder_repair._policy_from_config(turnbull_cfg) == builder_repair._DEFAULT_POLICY

    def test_none_vs_turnbull_config_behavioural_parity(self, turnbull_cfg):
        # A bundle that exercises all three checks at once.
        structure = _structure([_para("para1", 0, "body text about the marsh")])
        semantic = _semantic(
            claims=[_claim("c1", "population_estimate"), _claim("c2", "fire_incident")],
            entities=[
                EntityRecord(entity_id="e_sp", entity_type="Species", name="Mallard", normalized_form="mallard"),
                EntityRecord(entity_id="e_pl", entity_type="Place", name="Pool 2", normalized_form="pool 2"),
                EntityRecord(entity_id="e_sm", entity_type="SurveyMethod", name="banding", normalized_form="banding"),
            ],
            claim_entity_links=[ClaimEntityLinkRecord(claim_id="c2", entity_id="e_sm", relation_type="METHOD_FOCUS")],
            claim_link_diagnostics=[ClaimLinkDiagnosticRecord(
                claim_id="c1", relation_type="SPECIES_FOCUS", surface_form="mallards",
                normalized_form="mallards", diagnostic_code="NO_RESOLVED_MENTION",
            )],
        )
        none_props = builder_repair.detect(structure, semantic, "snap", config=None)
        cfg_props = builder_repair.detect(structure, semantic, "snap", config=turnbull_cfg)

        assert none_props, "test bundle should trigger proposals"
        assert sorted(map(_proj, none_props)) == sorted(map(_proj, cfg_props))
        # all three families present, proving the bundle is a real exercise
        classes = {p.issue_class for p in none_props}
        assert classes == {"missing_species_focus", "missing_event_location", "method_overtrigger"}


# --------------------------------------------------------------------------- #
# builder_repair — AGNOSTIC
# --------------------------------------------------------------------------- #
class TestBuilderRepairAgnostic:
    def test_empty_block_emits_nothing_for_turnbull_typed_claims(self):
        """config present but no builder_repair block → no Turnbull leak."""
        cfg = SimpleNamespace(builder_repair={}, claim_location_relation="OCCURRED_AT")
        structure = _structure([_para("para1", 0, "body")])
        semantic = _semantic(
            claims=[_claim("c1", "population_estimate")],
            entities=[
                EntityRecord(entity_id="e_sp", entity_type="Species", name="Mallard", normalized_form="mallard"),
                EntityRecord(entity_id="e_pl", entity_type="Place", name="Pool 2", normalized_form="pool 2"),
            ],
            claim_link_diagnostics=[ClaimLinkDiagnosticRecord(
                claim_id="c1", relation_type="SPECIES_FOCUS", surface_form="mallards",
                normalized_form="mallards", diagnostic_code="NO_RESOLVED_MENTION",
            )],
        )
        assert builder_repair.detect(structure, semantic, "snap", config=cfg) == []

    def test_custom_domain_drives_focus_relation(self):
        """A newspaper-style domain repairs SUBJECT_OF_CLAIM/Person, not SPECIES_FOCUS."""
        cfg = SimpleNamespace(
            builder_repair={
                "focus_relation": "SUBJECT_OF_CLAIM",
                "focus_candidate_entity_types": ["Person"],
                "focus_expecting_types": ["person_profile"],
            },
            claim_location_relation="OCCURRED_AT",
        )
        structure = _structure([_para("para1", 0, "John Doe, a labor organizer")])
        semantic = _semantic(
            claims=[_claim("c1", "person_profile")],
            entities=[EntityRecord(entity_id="e_pn", entity_type="Person", name="John Doe", normalized_form="john doe")],
            claim_link_diagnostics=[ClaimLinkDiagnosticRecord(
                claim_id="c1", relation_type="SUBJECT_OF_CLAIM", surface_form="John Doe",
                normalized_form="john doe", diagnostic_code="NO_RESOLVED_MENTION",
            )],
        )
        props = builder_repair.detect(structure, semantic, "snap", config=cfg)
        focus = [p for p in props if p.proposal_type == "add_claim_entity_link"]
        assert len(focus) == 1
        assert focus[0].patch_spec.get("relation_type") == "SUBJECT_OF_CLAIM"
        assert focus[0].patch_spec.get("entity_id") == "e_pn"

    def test_custom_domain_ignores_turnbull_species_types(self):
        """population_estimate is NOT in the custom domain's expecting set → skipped."""
        cfg = SimpleNamespace(
            builder_repair={
                "focus_relation": "SUBJECT_OF_CLAIM",
                "focus_candidate_entity_types": ["Person"],
                "focus_expecting_types": ["person_profile"],
            },
            claim_location_relation="OCCURRED_AT",
        )
        structure = _structure([_para("para1", 0, "100 mallards")])
        semantic = _semantic(
            claims=[_claim("c1", "population_estimate")],
            entities=[EntityRecord(entity_id="e_sp", entity_type="Species", name="Mallard", normalized_form="mallard")],
            claim_link_diagnostics=[ClaimLinkDiagnosticRecord(
                claim_id="c1", relation_type="SPECIES_FOCUS", surface_form="mallards",
                normalized_form="mallards", diagnostic_code="NO_RESOLVED_MENTION",
            )],
        )
        assert builder_repair.detect(structure, semantic, "snap", config=cfg) == []


# --------------------------------------------------------------------------- #
# builder_repair — HOSTILE
# --------------------------------------------------------------------------- #
class TestBuilderRepairHostile:
    def test_expecting_types_without_relation_does_not_crash(self):
        cfg = SimpleNamespace(
            builder_repair={"focus_expecting_types": ["population_estimate"]},  # no focus_relation
            claim_location_relation="OCCURRED_AT",
        )
        structure = _structure([_para("para1", 0, "body")])
        semantic = _semantic(
            claims=[_claim("c1", "population_estimate")],
            entities=[EntityRecord(entity_id="e_sp", entity_type="Species", name="M", normalized_form="m")],
        )
        assert builder_repair.detect(structure, semantic, "snap", config=cfg) == []

    def test_method_types_without_relation_does_not_crash(self):
        cfg = SimpleNamespace(
            builder_repair={"method_overtrigger_types": ["fire_incident"]},  # no method_relation
            claim_location_relation="OCCURRED_AT",
        )
        structure = _structure([_para("para1", 0, "body")])
        semantic = _semantic(
            claims=[_claim("c1", "fire_incident")],
            entities=[EntityRecord(entity_id="e_sm", entity_type="SurveyMethod", name="b", normalized_form="b")],
            claim_entity_links=[ClaimEntityLinkRecord(claim_id="c1", entity_id="e_sm", relation_type="METHOD_FOCUS")],
        )
        assert builder_repair.detect(structure, semantic, "snap", config=cfg) == []


# --------------------------------------------------------------------------- #
# junk_mention
# --------------------------------------------------------------------------- #
def _jm_cfg(header_patterns=()):
    return SimpleNamespace(
        junk_mention_filters={"header_patterns": list(header_patterns)},
        ocr_corrections=frozenset(),
        seed_entities=[],
    )


def _header_bundle(header_text: str, surface: str = "Smith"):
    structure = _structure([_para("para1", 0, header_text)])
    semantic = _semantic(mentions=[MentionRecord(
        mention_id="m1", run_id="run_1", paragraph_id="para1",
        surface_form=surface, normalized_form=surface.lower(),
        start_offset=0, end_offset=len(surface), detection_confidence=0.5,
    )])
    return structure, semantic


class TestJunkMentionParity:
    def test_turnbull_effective_headers_match_legacy(self, turnbull_cfg):
        extra = junk_mention._compile_domain_header_patterns(
            tuple(turnbull_cfg.junk_mention_filters.get("header_patterns") or ())
        )
        eff = tuple(junk_mention._STRUCTURAL_HEADER_PATTERNS) + extra
        legacy = tuple(junk_mention._HEADER_PATTERNS)
        assert [(p.pattern, p.flags) for p in eff] == [(p.pattern, p.flags) for p in legacy]

    def test_institutional_header_suppressed_none_and_turnbull(self, turnbull_cfg):
        structure, semantic = _header_bundle("Refuge Manager J. Smith reporting")
        for cfg in (None, turnbull_cfg):
            props = junk_mention.detect(structure, semantic, "snap", config=cfg)
            assert any(p.issue_class == "header_contamination" for p in props), cfg


class TestJunkMentionAgnostic:
    def test_bare_domain_does_not_suppress_institutional_header(self):
        structure, semantic = _header_bundle("Refuge Manager J. Smith reporting")
        props = junk_mention.detect(structure, semantic, "snap", config=_jm_cfg())
        assert not any(p.issue_class == "header_contamination" for p in props)

    def test_custom_domain_header_pattern_applies(self):
        structure, semantic = _header_bundle("Weekly Gazette Established 1901", surface="Gazette")
        cfg = _jm_cfg(header_patterns=[r"weekly\s+gazette"])
        props = junk_mention.detect(structure, semantic, "snap", config=cfg)
        assert any(p.issue_class == "header_contamination" for p in props)
        # and the same custom domain does NOT know Turnbull's institutional header
        s2, sem2 = _header_bundle("Refuge Manager J. Smith reporting")
        assert not any(
            p.issue_class == "header_contamination"
            for p in junk_mention.detect(s2, sem2, "snap", config=cfg)
        )


class TestJunkMentionHostile:
    def test_invalid_regex_in_config_is_skipped_not_raised(self):
        structure, semantic = _header_bundle("Weekly Gazette Established 1901", surface="Gazette")
        cfg = _jm_cfg(header_patterns=["[", r"weekly\s+gazette"])  # "[" is invalid
        # must not raise, and the valid pattern still fires
        props = junk_mention.detect(structure, semantic, "snap", config=cfg)
        assert any(p.issue_class == "header_contamination" for p in props)


# --------------------------------------------------------------------------- #
# patch_spec validation — Step 2 (domain relation vocabulary)
# --------------------------------------------------------------------------- #
def _entity_link_spec(relation: str) -> dict:
    return make_patch_spec(
        "add_claim_entity_link", claim_id="c1", entity_id="e1",
        relation_type=relation, evidence_basis="claim_link_diagnostic",
    )


def _location_link_spec(relation: str) -> dict:
    return make_patch_spec(
        "add_claim_location_link", claim_id="c1", entity_id="e1",
        relation_type=relation, evidence_basis="document_context",
    )


class TestPatchSpecParity:
    def test_default_path_unchanged(self):
        # Turnbull relation validates; a bogus one is rejected — no kwargs.
        validate_patch_spec(_entity_link_spec("SPECIES_FOCUS"))
        with pytest.raises(PatchSpecValidationError):
            validate_patch_spec(_entity_link_spec("MENTIONS_PERSON"))

    def test_turnbull_config_vocab_matches_default(self, turnbull_cfg):
        # Passing Turnbull's own vocab is a no-op vs the module default.
        for rel in ALLOWED_CLAIM_ENTITY_RELATIONS:
            validate_patch_spec(_entity_link_spec(rel), allowed_relations=turnbull_cfg.claim_entity_relations)
        validate_patch_spec(_location_link_spec("OCCURRED_AT"), location_relation=turnbull_cfg.claim_location_relation)


class TestPatchSpecDomainVocab:
    def test_novel_entity_relation_accepted_with_domain_vocab(self):
        spec = _entity_link_spec("MENTIONS_PERSON")
        # rejected under the Turnbull default...
        with pytest.raises(PatchSpecValidationError):
            validate_patch_spec(spec)
        # ...accepted when the domain declares it.
        validate_patch_spec(spec, allowed_relations=frozenset({"MENTIONS_PERSON"}))

    def test_novel_location_relation_accepted(self):
        spec = _location_link_spec("RECORDED_IN")
        with pytest.raises(PatchSpecValidationError):
            validate_patch_spec(spec)  # default OCCURRED_AT
        validate_patch_spec(spec, location_relation="RECORDED_IN")

    def test_gate_not_disabled_relation_outside_domain_set_still_rejected(self):
        # A relation not in the *domain's* set is still rejected — the gate is
        # parameterized, not bypassed.
        with pytest.raises(PatchSpecValidationError):
            validate_patch_spec(_entity_link_spec("SPECIES_FOCUS"), allowed_relations=frozenset({"MENTIONS_PERSON"}))
        with pytest.raises(PatchSpecValidationError):
            validate_patch_spec(_entity_link_spec("X"), allowed_relations=frozenset())


class TestValidateAndFilterIntegration:
    """End-to-end: a custom-domain detector emits a novel relation (Step 1) and
    detect-time validation accepts it (Step 2)."""

    def _custom_bundle_and_cfg(self):
        cfg = SimpleNamespace(
            builder_repair={
                "focus_relation": "MENTIONS_PERSON",
                "focus_candidate_entity_types": ["Person"],
                "focus_expecting_types": ["person_profile"],
            },
            claim_location_relation="OCCURRED_AT",
            claim_entity_relations=frozenset({"MENTIONS_PERSON"}),
        )
        structure = _structure([_para("para1", 0, "John Doe, organizer")])
        semantic = _semantic(
            claims=[_claim("c1", "person_profile")],
            entities=[EntityRecord(entity_id="e_pn", entity_type="Person", name="John Doe", normalized_form="john doe")],
            claim_link_diagnostics=[ClaimLinkDiagnosticRecord(
                claim_id="c1", relation_type="MENTIONS_PERSON", surface_form="John Doe",
                normalized_form="john doe", diagnostic_code="NO_RESOLVED_MENTION",
            )],
        )
        return structure, semantic, cfg

    def test_novel_relation_survives_validation_with_config(self):
        structure, semantic, cfg = self._custom_bundle_and_cfg()
        props = builder_repair.detect(structure, semantic, "snap", config=cfg)
        assert props and props[0].patch_spec.get("relation_type") == "MENTIONS_PERSON"
        valid, rejected = review_detect._validate_and_filter(props, "snap", cfg)
        assert len(valid) == len(props)
        assert rejected == []

    def test_same_proposal_rejected_without_config(self):
        structure, semantic, cfg = self._custom_bundle_and_cfg()
        props = builder_repair.detect(structure, semantic, "snap", config=cfg)
        valid, rejected = review_detect._validate_and_filter(props, "snap", config=None)
        assert valid == []
        assert rejected and "MENTIONS_PERSON" in rejected[0]["error"]
