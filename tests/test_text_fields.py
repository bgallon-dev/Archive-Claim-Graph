from graphrag_pipeline.models import ClaimRecord, DocumentRecord, PageRecord, ParagraphRecord


def test_structure_records_load_legacy_raw_text_as_raw_ocr_text() -> None:
    document = DocumentRecord.from_dict(
        {
            "doc_id": "d1",
            "title": "Legacy",
            "raw_text": "legacy document text",
            "clean_text": "legacy document text",
        }
    )
    page = PageRecord.from_dict(
        {
            "page_id": "p1",
            "doc_id": "d1",
            "page_number": 1,
            "raw_text": "legacy page text",
            "clean_text": "legacy page text",
        }
    )
    paragraph = ParagraphRecord.from_dict(
        {
            "paragraph_id": "para1",
            "doc_id": "d1",
            "page_id": "p1",
            "section_id": None,
            "paragraph_index": 1,
            "page_number": 1,
            "raw_text": "legacy paragraph text",
            "clean_text": "legacy paragraph text",
            "char_count": 21,
        }
    )

    assert document.raw_ocr_text == "legacy document text"
    assert page.raw_ocr_text == "legacy page text"
    assert paragraph.raw_ocr_text == "legacy paragraph text"


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


def test_claim_record_loads_legacy_raw_sentence_alias() -> None:
    claim = ClaimRecord.from_dict(
        {
            "claim_id": "c1",
            "run_id": "run_1",
            "paragraph_id": "p1",
            "claim_type": "population_estimate",
            "raw_sentence": "Mallards were observed.",
            "normalized_sentence": "mallards were observed.",
            "certainty": "certain",
            "extraction_confidence": 0.8,
        }
    )

    assert claim.source_sentence == "Mallards were observed."
