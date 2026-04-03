import json
from pathlib import Path

from gemynd.cli import main
from gemynd.ingest.graph.writer import InMemoryGraphWriter
from gemynd.shared.io_utils import save_semantic_bundle, save_structure_bundle
from gemynd.core.models import EntityRecord
from gemynd.ingest.pipeline import extract_semantic
from gemynd.ingest.source_parser import parse_source_file


def test_entity_record_keeps_entity_type_in_bundle_but_not_node_props() -> None:
    entity = EntityRecord(
        entity_id="species_1",
        entity_type="Species",
        name="mallard",
        normalized_form="mallard",
        properties={"taxon_group": "bird"},
    )

    payload = entity.to_dict()
    node_props = entity.node_props()

    assert payload["entity_type"] == "Species"
    assert node_props["entity_id"] == "species_1"
    assert "entity_type" not in node_props
    assert node_props["taxon_group"] == "bird"


def test_entity_type_not_persisted_on_graph_nodes(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_group_f", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    writer = InMemoryGraphWriter()
    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)

    for label in ("Entity", "Refuge", "Place", "Person", "Organization", "Species", "Activity", "Period", "Habitat", "SurveyMethod"):
        for node_props in writer.node_store.get(label, {}).values():
            assert "entity_type" not in node_props, f"{label} nodes must not persist entity_type as a property."


def test_cli_memory_load_graph_reports_counts_not_nodes_or_relationships(fixtures_dir: Path, tmp_path: Path, capsys) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_cli_group_f", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    structure_path = tmp_path / "report1.structure.json"
    semantic_path = tmp_path / "report1.semantic.json"
    save_structure_bundle(structure_path, structure)
    save_semantic_bundle(semantic_path, semantic)

    exit_code = main(
        [
            "load-graph",
            "--structure",
            str(structure_path),
            "--semantic",
            str(semantic_path),
            "--backend",
            "memory",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["backend"] == "memory"
    assert payload["node_count"] > 0
    assert payload["relationship_count"] > 0
    assert "nodes" not in payload
    assert "relationships" not in payload
