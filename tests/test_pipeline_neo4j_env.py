from __future__ import annotations

from pathlib import Path

from graphrag_pipeline.ingest.pipeline import extract_semantic, load_graph
from graphrag_pipeline.ingest.source_parser import parse_source_file


class _DummyNeo4jWriter:
    captured: dict[str, object] = {}

    def __init__(self, **kwargs):
        _DummyNeo4jWriter.captured = dict(kwargs)

    def create_schema(self) -> None:
        return None

    def load_structure(self, structure) -> None:
        return None

    def load_semantic(self, structure, semantic) -> None:
        return None


def test_load_graph_reads_neo4j_trust_env(monkeypatch, fixtures_dir: Path) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setenv("NEO4J_TRUST", "custom")
    monkeypatch.setenv("NEO4J_CA_CERT", "C:/tmp/ca.crt")

    monkeypatch.setattr("graphrag_pipeline.ingest.pipeline.Neo4jGraphWriter", _DummyNeo4jWriter)

    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(structure, run_overrides={"run_id": "run_env", "run_timestamp": "2026-03-10T00:00:00+00:00"})

    load_graph(structure, semantic, backend="neo4j")
    assert _DummyNeo4jWriter.captured["uri"] == "bolt://localhost:7687"
    assert _DummyNeo4jWriter.captured["trust_mode"] == "custom"
    assert _DummyNeo4jWriter.captured["ca_cert_path"] == "C:/tmp/ca.crt"
