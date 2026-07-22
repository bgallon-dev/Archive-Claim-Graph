"""`retain_bundles=False` — stream each doc into Neo4j, then delete its bundles.

Keeps disk flat (in-flight docs only) instead of accumulating the whole corpus.
Neo4j is stubbed so these run without a live database.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import gemynd.ingest.pipeline as pipeline_module
from gemynd.ingest.checkpoint import CHECKPOINT_FILENAME
from gemynd.ingest.pipeline import run_e2e
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR


class _DummyNeo4jWriter:
    def __init__(self, **kwargs) -> None:
        pass

    def create_schema(self) -> None:
        pass

    def load_structure(self, _structure) -> None:
        pass

    def load_semantic(self, _structure, _semantic) -> None:
        pass


class _DummyExecutor:
    def __init__(self, **kwargs) -> None:
        pass

    def run(self, *_a, **_k):
        return []  # no already-ingested duplicates

    def close(self) -> None:
        pass


@pytest.fixture()
def _stub_neo4j(monkeypatch):
    monkeypatch.setattr(pipeline_module, "Neo4jGraphWriter", _DummyNeo4jWriter)
    import gemynd.retrieval.executor as executor_module
    monkeypatch.setattr(executor_module, "Neo4jQueryExecutor", _DummyExecutor)


_NEO4J = dict(neo4j_uri="bolt://stub", neo4j_user="u", neo4j_password="p")


def _inputs(fixtures_dir: Path, n: int) -> list[str]:
    return [str(fixtures_dir / f"report{i}.json") for i in range(1, n + 1)]


def test_stream_deletes_bundles_after_graph_write(_stub_neo4j, fixtures_dir, tmp_path) -> None:
    summary = run_e2e(
        _inputs(fixtures_dir, 1), tmp_path, backend="neo4j",
        resources_dir=TURNBULL_RESOURCES_DIR, retain_bundles=False, **_NEO4J,
    )
    assert summary["documents_processed"] == 1
    # bundles gone -> disk stays flat
    assert list(tmp_path.glob("*.semantic.json")) == []
    assert list(tmp_path.glob("*.structure.json")) == []
    # ...but the doc is checkpointed, so a resume still skips it
    assert (tmp_path / CHECKPOINT_FILENAME).exists()
    assert (tmp_path / CHECKPOINT_FILENAME).read_text(encoding="utf-8").strip()


def test_stream_deletes_bundles_multiworker(_stub_neo4j, fixtures_dir, tmp_path) -> None:
    # Multi-worker: pool workers (Phase 1) write bundles; the parent (Phase 2)
    # writes to the graph and deletes them. Peak disk = one batch, not the corpus.
    summary = run_e2e(
        _inputs(fixtures_dir, 3), tmp_path, backend="neo4j", workers=2,
        resources_dir=TURNBULL_RESOURCES_DIR, retain_bundles=False, **_NEO4J,
    )
    assert summary["documents_processed"] == 3
    assert list(tmp_path.glob("*.semantic.json")) == []
    assert list(tmp_path.glob("*.structure.json")) == []


def test_default_retains_bundles(_stub_neo4j, fixtures_dir, tmp_path) -> None:
    run_e2e(
        _inputs(fixtures_dir, 1), tmp_path, backend="neo4j",
        resources_dir=TURNBULL_RESOURCES_DIR, **_NEO4J,  # retain_bundles defaults True
    )
    assert list(tmp_path.glob("*.semantic.json"))
    assert list(tmp_path.glob("*.structure.json"))


def test_memory_backend_ignores_retain_false(fixtures_dir, tmp_path) -> None:
    # Safety guard: with the memory backend the bundles ARE the only output, so
    # retain_bundles=False must be ignored rather than deleting everything.
    run_e2e(
        _inputs(fixtures_dir, 1), tmp_path, backend="memory",
        resources_dir=TURNBULL_RESOURCES_DIR, retain_bundles=False,
    )
    assert list(tmp_path.glob("*.semantic.json"))
    assert list(tmp_path.glob("*.structure.json"))
