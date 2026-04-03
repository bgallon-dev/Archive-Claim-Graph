from gemynd.core.models import ClaimRecord


def test_claim_record_serializes_source_sentence_not_raw_sentence() -> None:
    claim = ClaimRecord(
        claim_id="c1",
        run_id="run_1",
        paragraph_id="p1",
        claim_type="population_estimate",
        source_sentence="Mallards were observed.",
        normalized_sentence="mallards were observed.",
        certainty="certain",
        extraction_confidence=0.8,
    )

    payload = claim.to_dict()

    assert payload["source_sentence"] == "Mallards were observed."
    assert "raw_sentence" not in payload


