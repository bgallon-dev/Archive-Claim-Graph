from graphrag_pipeline.extractors.claim_extractor import ClaimDraft, ClaimLinkDraft
from graphrag_pipeline.extractors.mention_extractor import MentionDraft
from graphrag_pipeline.models import EntityRecord, EntityResolutionRecord
from graphrag_pipeline.pipeline import extract_semantic, parse_source, quality_report


class StaticClaimExtractor:
    def __init__(self, drafts: list[ClaimDraft]) -> None:
        self._drafts = drafts

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        return list(self._drafts)


class StaticMentionExtractor:
    def __init__(self, drafts: list[MentionDraft]) -> None:
        self._drafts = drafts

    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        return list(self._drafts)


class MappingResolver:
    def __init__(self, mapping: dict[tuple[int, int], tuple[EntityRecord, str]]) -> None:
        self._mapping = mapping

    def resolve(self, mentions: list) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        entities: dict[str, EntityRecord] = {}
        resolutions: list[EntityResolutionRecord] = []
        for mention in mentions:
            resolved = self._mapping.get((mention.start_offset, mention.end_offset))
            if not resolved:
                continue
            entity, relation_type = resolved
            entities[entity.entity_id] = entity
            resolutions.append(
                EntityResolutionRecord(
                    mention_id=mention.mention_id,
                    entity_id=entity.entity_id,
                    relation_type=relation_type,
                    match_score=0.99 if relation_type == "REFERS_TO" else 0.72,
                )
            )
        return list(entities.values()), resolutions


def _single_paragraph_structure(text: str):
    return parse_source(
        {
            "metadata": {"title": "Claim Link Test", "report_year": 1938},
            "pages": [{"page_number": 1, "raw_text": text}],
        }
    )


def test_exact_span_claim_link_match_emits_typed_relation() -> None:
    text = "Mallards were observed."
    structure = _single_paragraph_structure(text)
    semantic = extract_semantic(
        structure,
        claim_extractor=StaticClaimExtractor(
            [
                ClaimDraft(
                    claim_type="species_presence",
                    source_sentence=text,
                    normalized_sentence=text.lower(),
                    epistemic_status="certain",
                    extraction_confidence=0.9,
                    evidence_start=0,
                    evidence_end=len(text),
                    claim_links=[
                        ClaimLinkDraft(
                            surface_form="Mallards",
                            normalized_form="mallards",
                            relation_type="SPECIES_FOCUS",
                            start_offset=0,
                            end_offset=8,
                            entity_type_hint="Species",
                        )
                    ],
                )
            ]
        ),
        mention_extractor=StaticMentionExtractor(
            [MentionDraft("Mallards", "mallards", 0, 8, 0.95, [])]
        ),
        resolver=MappingResolver(
            {
                (0, 8): (
                    EntityRecord("species_mallard", "Species", "mallard", "mallard"),
                    "REFERS_TO",
                )
            }
        ),
        run_overrides={"run_id": "run_exact_link", "run_timestamp": "2026-03-13T00:00:00+00:00"},
    )

    assert [(link.entity_id, link.relation_type) for link in semantic.claim_entity_links] == [
        ("species_mallard", "SPECIES_FOCUS")
    ]
    assert semantic.claim_link_diagnostics == []


def test_normalized_form_fallback_links_unique_candidate() -> None:
    text = "Mallards were observed."
    structure = _single_paragraph_structure(text)
    semantic = extract_semantic(
        structure,
        claim_extractor=StaticClaimExtractor(
            [
                ClaimDraft(
                    claim_type="species_presence",
                    source_sentence=text,
                    normalized_sentence=text.lower(),
                    epistemic_status="certain",
                    extraction_confidence=0.9,
                    evidence_start=0,
                    evidence_end=len(text),
                    claim_links=[
                        ClaimLinkDraft(
                            surface_form="Mallards",
                            normalized_form="mallards",
                            relation_type="SPECIES_FOCUS",
                            entity_type_hint="Species",
                        )
                    ],
                )
            ]
        ),
        mention_extractor=StaticMentionExtractor(
            [MentionDraft("Mallards", "mallards", 0, 8, 0.95, [])]
        ),
        resolver=MappingResolver(
            {
                (0, 8): (
                    EntityRecord("species_mallard", "Species", "mallard", "mallard"),
                    "REFERS_TO",
                )
            }
        ),
        run_overrides={"run_id": "run_normalized_link", "run_timestamp": "2026-03-13T00:00:00+00:00"},
    )

    assert len(semantic.claim_entity_links) == 1
    assert semantic.claim_entity_links[0].relation_type == "SPECIES_FOCUS"
    assert semantic.claim_link_diagnostics == []


def test_ambiguous_normalized_fallback_emits_diagnostic() -> None:
    text = "Marsh near marsh dried."
    structure = _single_paragraph_structure(text)
    semantic = extract_semantic(
        structure,
        claim_extractor=StaticClaimExtractor(
            [
                ClaimDraft(
                    claim_type="habitat_condition",
                    source_sentence=text,
                    normalized_sentence=text.lower(),
                    epistemic_status="certain",
                    extraction_confidence=0.85,
                    evidence_start=0,
                    evidence_end=len(text),
                    claim_links=[
                        ClaimLinkDraft(
                            surface_form="marsh",
                            normalized_form="marsh",
                            relation_type="HABITAT_FOCUS",
                            entity_type_hint="Habitat",
                        )
                    ],
                )
            ]
        ),
        mention_extractor=StaticMentionExtractor(
            [
                MentionDraft("Marsh", "marsh", 0, 5, 0.95, []),
                MentionDraft("marsh", "marsh", 11, 16, 0.95, []),
            ]
        ),
        resolver=MappingResolver(
            {
                (0, 5): (EntityRecord("habitat_marsh", "Habitat", "marsh", "marsh"), "REFERS_TO"),
                (11, 16): (EntityRecord("habitat_marsh", "Habitat", "marsh", "marsh"), "REFERS_TO"),
            }
        ),
        run_overrides={"run_id": "run_ambiguous_link", "run_timestamp": "2026-03-13T00:00:00+00:00"},
    )

    assert semantic.claim_entity_links == []
    assert [diagnostic.diagnostic_code for diagnostic in semantic.claim_link_diagnostics] == ["AMBIGUOUS_FALLBACK"]

    report = quality_report(structure, semantic)
    assert report["claim_link_diagnostic_counts"]["AMBIGUOUS_FALLBACK"] == 1


def test_entity_type_hint_disambiguates_normalized_fallback() -> None:
    text = "Turnbull and Turnbull were discussed."
    structure = _single_paragraph_structure(text)
    semantic = extract_semantic(
        structure,
        claim_extractor=StaticClaimExtractor(
            [
                ClaimDraft(
                    claim_type="public_contact",
                    source_sentence=text,
                    normalized_sentence=text.lower(),
                    epistemic_status="certain",
                    extraction_confidence=0.83,
                    evidence_start=0,
                    evidence_end=len(text),
                    claim_links=[
                        ClaimLinkDraft(
                            surface_form="Turnbull",
                            normalized_form="turnbull",
                            relation_type="LOCATION_FOCUS",
                            entity_type_hint="Refuge",
                        )
                    ],
                )
            ]
        ),
        mention_extractor=StaticMentionExtractor(
            [
                MentionDraft("Turnbull", "turnbull", 0, 8, 0.9, []),
                MentionDraft("Turnbull", "turnbull", 13, 21, 0.9, []),
            ]
        ),
        resolver=MappingResolver(
            {
                (0, 8): (EntityRecord("place_turnbull", "Place", "Turnbull", "turnbull"), "REFERS_TO"),
                (13, 21): (EntityRecord("refuge_turnbull", "Refuge", "Turnbull Refuge", "turnbull refuge"), "REFERS_TO"),
            }
        ),
        run_overrides={"run_id": "run_type_hint", "run_timestamp": "2026-03-13T00:00:00+00:00"},
    )

    assert [(link.entity_id, link.relation_type) for link in semantic.claim_entity_links] == [
        ("refuge_turnbull", "LOCATION_FOCUS")
    ]
    assert [diagnostic.diagnostic_code for diagnostic in semantic.claim_link_diagnostics] == [
        "RELATION_COMPATIBILITY_WEAK"
    ]
