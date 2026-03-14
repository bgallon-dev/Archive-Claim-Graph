from pathlib import Path

from graphrag_pipeline.graph.writer import InMemoryGraphWriter
from graphrag_pipeline.models import DocumentRecord, YearRecord
from graphrag_pipeline.pipeline import extract_semantic
from graphrag_pipeline.source_parser import parse_source_file, parse_source_payload


def test_document_record_serializes_report_year_not_year() -> None:
    document = DocumentRecord.from_dict(
        {
            "doc_id": "d1",
            "title": "Legacy report",
            "year": "1938",
            "raw_ocr_text": "legacy text",
            "clean_text": "legacy text",
        }
    )

    payload = document.to_dict()

    assert document.report_year == 1938
    assert payload["report_year"] == 1938
    assert "year" not in payload


def test_source_parser_accepts_legacy_metadata_year_alias() -> None:
    bundle = parse_source_payload(
        {
            "metadata": {
                "title": "Legacy report",
                "year": "1942",
            },
            "pages": [{"page_number": 1, "raw_text": "Some text."}],
        }
    )

    assert bundle.document.report_year == 1942


def test_year_record_serializes_year_label_not_label() -> None:
    year = YearRecord.from_dict(
        {
            "year_id": "year_1938",
            "year": 1938,
            "label": "1938",
        }
    )

    payload = year.to_dict()

    assert year.year_label == "1938"
    assert payload["year_label"] == "1938"
    assert "label" not in payload


def test_year_fields_are_scoped_to_their_canonical_node_types(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_year_fields", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    writer = InMemoryGraphWriter()
    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)

    document_nodes = list(writer.node_store.get("Document", {}).values())
    assert document_nodes, "Expected a Document node."
    for props in document_nodes:
        assert "report_year" in props
        assert "year" not in props
        assert "year_label" not in props

    year_nodes = list(writer.node_store.get("Year", {}).values())
    assert year_nodes, "Expected Year nodes."
    for props in year_nodes:
        assert "year" in props
        assert "year_label" in props
        assert "report_year" not in props

    observation_nodes = list(writer.node_store.get("Observation", {}).values())
    assert observation_nodes, "Expected Observation nodes."
    for props in observation_nodes:
        assert "year" in props
        assert "year_source" in props
        assert "year_id" not in props
        assert "year_label" not in props
        assert "report_year" not in props

    event_nodes = list(writer.node_store.get("Event", {}).values())
    assert event_nodes, "Expected Event nodes."
    for props in event_nodes:
        assert "year" in props
        assert "year_source" in props
        assert "year_id" not in props
        assert "year_label" not in props
        assert "report_year" not in props
