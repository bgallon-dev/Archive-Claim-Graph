from __future__ import annotations

import os
from pathlib import Path

import pytest


def _load_dotenv(env_path: Path) -> None:
    """Parse a .env file and set missing variables into os.environ."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).parent.parent / ".env")


# Turnbull-corpus entity labels used by tests that build a writer directly.
# Any test that constructs InMemoryGraphWriter / Neo4jGraphWriter without going
# through load_graph() should pass entity_labels=TEST_ENTITY_LABELS.
TEST_ENTITY_LABELS: frozenset[str] = frozenset({
    "Refuge", "Place", "Person", "Organization", "Species",
    "Activity", "Period", "Habitat", "SurveyMethod",
})


@pytest.fixture()
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def populated_writer(tmp_path_factory):
    """Run the full pipeline on all 3 fixture reports into an InMemoryGraphWriter."""
    from gemynd.ingest.pipeline import run_e2e

    _fixtures = Path(__file__).parent / "fixtures"
    tmp = tmp_path_factory.mktemp("graph_data")
    result = run_e2e(
        [str(_fixtures / f) for f in ("report1.json", "report2.json", "report3.json")],
        tmp,
        backend="memory",
    )
    return result["writer"]


@pytest.fixture(scope="session")
def populated_executor(populated_writer):
    """InMemoryQueryExecutor backed by the fully-populated writer."""
    from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor

    return InMemoryQueryExecutor(
        populated_writer,
        entity_labels=TEST_ENTITY_LABELS,
        anchor_entity_type="Refuge",
        anchor_relation="ABOUT_REFUGE",
    )
