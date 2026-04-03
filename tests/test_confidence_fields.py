from gemynd.ingest.extractors.claim_extractor import LLMClaimExtractor
from gemynd.core.models import ClaimRecord, EntityResolutionRecord


def test_claim_record_serializes_epistemic_status_instead_of_certainty() -> None:
    claim = ClaimRecord(
        claim_id="c1",
        run_id="run_1",
        paragraph_id="p1",
        claim_type="population_estimate",
        source_sentence="Approximately 50 mallards were observed.",
        normalized_sentence="approximately 50 mallards were observed.",
        certainty="uncertain",
        extraction_confidence=0.72,
    )

    payload = claim.to_dict()
    node_props = claim.node_props()

    assert payload["epistemic_status"] == "uncertain"
    assert "certainty" not in payload
    assert node_props["epistemic_status"] == "uncertain"
    assert "certainty" not in node_props


def test_claim_record_from_dict_accepts_legacy_certainty_key() -> None:
    claim = ClaimRecord.from_dict(
        {
            "claim_id": "c1",
            "run_id": "run_1",
            "paragraph_id": "p1",
            "claim_type": "population_estimate",
            "source_sentence": "Mallards were observed.",
            "normalized_sentence": "mallards were observed.",
            "certainty": "certain",
            "extraction_confidence": 0.81,
        }
    )

    assert claim.certainty == "certain"
    assert claim.epistemic_status == "certain"


def test_llm_claim_extractor_accepts_new_and_legacy_epistemic_keys() -> None:
    class Adapter:
        def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
            return [
                {
                    "claim_type": "population_estimate",
                    "source_sentence": "Mallards were observed.",
                    "epistemic_status": "certain",
                    "extraction_confidence": 0.9,
                },
                {
                    "claim_type": "species_presence",
                    "source_sentence": "Pintails were possibly present.",
                    "certainty": "uncertain",
                    "extraction_confidence": 0.6,
                    "claim_links": [
                        {
                            "surface_form": "Pintails",
                            "normalized_form": "pintails",
                            "relation_type": "SPECIES_FOCUS",
                            "entity_type_hint": "Species",
                        }
                    ],
                },
            ]

    drafts = LLMClaimExtractor(Adapter()).extract("ignored")

    assert [draft.epistemic_status for draft in drafts] == ["certain", "uncertain"]
    assert drafts[1].claim_links[0].relation_type == "SPECIES_FOCUS"
    assert drafts[1].claim_links[0].entity_type_hint == "Species"


def test_entity_resolution_record_keeps_match_score_specific() -> None:
    payload = EntityResolutionRecord(
        mention_id="m1",
        entity_id="e1",
        relation_type="REFERS_TO",
        match_score=0.93,
    ).to_dict()

    assert payload["match_score"] == 0.93
    assert "score" not in payload
