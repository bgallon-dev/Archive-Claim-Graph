from pathlib import Path

from graphrag_pipeline.source_parser import _infer_year_from_filename, parse_source_file, parse_source_payload


def test_parse_source_sections_and_paragraphs(fixtures_dir: Path) -> None:
    bundle = parse_source_file(fixtures_dir / "report1.json")

    assert bundle.document.doc_type == "narrative_report"
    assert bundle.document.report_year == 1938
    assert bundle.document.page_count == 2

    headings = [section.heading.lower() for section in bundle.sections]
    assert any("general" in heading for heading in headings)
    assert any("wildlife" in heading for heading in headings)
    assert any("economic uses" in heading for heading in headings)

    assert len(bundle.paragraphs) >= 4
    assert all(paragraph.section_id for paragraph in bundle.paragraphs)


def test_infer_year_bare() -> None:
    assert _infer_year_from_filename("TBL_1938a_NarrativeReport") == 1938


def test_infer_year_range_returns_earliest() -> None:
    assert _infer_year_from_filename("TBL_1940-1941_AnnualReport") == 1940


def test_infer_year_yyyymm() -> None:
    assert _infer_year_from_filename("TBL_200601_Report") == 2006


def test_infer_year_no_year_returns_none() -> None:
    assert _infer_year_from_filename("no_year_here") is None


def test_filename_year_fallback_when_no_metadata_year() -> None:
    payload = {
        "metadata": {
            "title": "Test Report",
        },
        "pages": [{"page_number": 1, "raw_ocr_text": "Some text."}],
    }
    bundle = parse_source_payload(payload, source_file="TBL_1952_report.json")
    assert bundle.document.report_year == 1952


def test_metadata_year_takes_priority_over_filename() -> None:
    payload = {
        "metadata": {
            "title": "Test Report",
            "report_year": 1999,
        },
        "pages": [{"page_number": 1, "raw_ocr_text": "Some text."}],
    }
    bundle = parse_source_payload(payload, source_file="TBL_1938a_report.json")
    assert bundle.document.report_year == 1999


def test_structure_uses_raw_ocr_text_in_runtime_schema() -> None:
    payload = {
        "metadata": {
            "title": "Test Report",
            "report_year": 1999,
        },
        "pages": [{"page_number": 1, "raw_ocr_text": "Some text."}],
    }
    bundle = parse_source_payload(payload, source_file="TBL_1999_report.json")

    document_payload = bundle.document.to_dict()
    page_payload = bundle.pages[0].to_dict()
    paragraph_payload = bundle.paragraphs[0].to_dict()

    assert "raw_ocr_text" in document_payload
    assert "raw_text" not in document_payload
    assert "raw_ocr_text" in page_payload
    assert "raw_text" not in page_payload
    assert "raw_ocr_text" in paragraph_payload
    assert "raw_text" not in paragraph_payload
