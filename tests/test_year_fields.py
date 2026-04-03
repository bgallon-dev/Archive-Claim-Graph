from pathlib import Path

from gemynd.ingest.graph.writer import InMemoryGraphWriter
from gemynd.ingest.pipeline import extract_semantic
from gemynd.ingest.source_parser import parse_source_file


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
