from pathlib import Path

from gemynd.core.domain_config import load_domain_config
from gemynd.ingest.source_parser import (
    _extract_loc_sequence,
    _infer_year_from_filename,
    parse_source_file,
    parse_source_payload,
)
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR

# Turnbull source profile (default_doc_type=narrative_report, loc_sequence on).
_TURNBULL_PROFILE = load_domain_config(TURNBULL_RESOURCES_DIR).source_profile
# Minimal profile that enables the LoC filename heuristics (edition zero-strip,
# trailing-sequence) for the newspaper-style metadata test.
_LOC_PROFILE = {"filename_inference": ["loc_sequence"]}


def test_parse_source_sections_and_paragraphs(fixtures_dir: Path) -> None:
    bundle = parse_source_file(fixtures_dir / "report1.json", source_profile=_TURNBULL_PROFILE)

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


def test_extract_loc_sequence_trailing_digits() -> None:
    # LoC canonical URL form is ``?sp=42`` not ``?sp=0042`` — the helper
    # zero-strips so the URL template can use ``{loc_sequence}`` directly.
    fname = "1_1909-11-10_p1_sn88085947_00211108617_1909111001_0042"
    assert _extract_loc_sequence(fname) == "42"


def test_extract_loc_sequence_with_extension() -> None:
    fname = "1_1909-11-10_p1_sn88085947_00211108617_1909111001_0042.json"
    assert _extract_loc_sequence(fname) == "42"


def test_extract_loc_sequence_preserves_zero() -> None:
    # Edge case: a sequence of all zeros should normalize to "0", not "".
    fname = "anything_0000"
    assert _extract_loc_sequence(fname) == "0"


def test_extract_loc_sequence_non_conforming_returns_none() -> None:
    assert _extract_loc_sequence("TBL_1938a_NarrativeReport") is None
    assert _extract_loc_sequence("trailing_three_999") is None
    assert _extract_loc_sequence("trailing_five_99999") is None


def test_extract_loc_sequence_empty_input() -> None:
    assert _extract_loc_sequence(None) is None
    assert _extract_loc_sequence("") is None


def test_newspaper_metadata_populates_lccn_edition_sequence() -> None:
    payload = {
        "metadata": {
            "title": "Allow Free Speech Under Proper Regulations.",
            "doc_type": "newspaper_article",
            "date_start": "1909-11-10",
            "lccn": "sn88085947",
            "edition": "01",
        },
        "pages": [{"page_number": 1, "raw_ocr_text": "Body."}],
    }
    bundle = parse_source_payload(
        payload,
        source_file="1_1909-11-10_p1_sn88085947_00211108617_1909111001_0042.json",
        source_profile=_LOC_PROFILE,
    )
    assert bundle.document.lccn == "sn88085947"
    # edition and loc_sequence are stored zero-stripped for direct use in
    # the canonical loc.gov URL form (ed-1, ?sp=42 — not ed-01, ?sp=0042).
    assert bundle.document.edition == "1"
    assert bundle.document.loc_sequence == "42"


def test_lccn_edition_default_to_none_when_metadata_missing() -> None:
    payload = {
        "metadata": {"title": "Generic Report", "report_year": 1938},
        "pages": [{"page_number": 1, "raw_ocr_text": "Body."}],
    }
    bundle = parse_source_payload(payload, source_file="TBL_1938_report.json")
    assert bundle.document.lccn is None
    assert bundle.document.edition is None
    assert bundle.document.loc_sequence is None


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
