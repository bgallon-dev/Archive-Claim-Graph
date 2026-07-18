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


# ── Hermetic claim-extraction: never call a live model from the test suite ─────
# `.env` legitimately carries production claim-LLM routing (CLAIM_LLM_BASE_URL →
# a local LM Studio/vLLM server) and a real ANTHROPIC_API_KEY. Loading it above
# would otherwise make `extract_semantic` issue real model calls — slow,
# non-deterministic, and (for Anthropic) billable. Strip that routing so the
# claim LLM path degrades to NullLLMAdapter. Tests that exercise an adapter set
# these explicitly themselves (monkeypatch / patch.dict), which still works.
#
# This previously "worked" only by accident: .env pointed at a dead server, so
# calls failed and returned no claims. Pointing it at a running server surfaced
# the coupling as a spurious test failure.
for _var in (
    "CLAIM_LLM_BASE_URL",
    "CLAIM_LLM_PROVIDER",
    "CLAIM_LLM_MODEL",
    "CLAIM_LLM_API_KEY",
    "CLAIM_LLM_EXTRA_BODY",
    "CLAIM_LLM_SCHEMA_PROFILE",
    "CLAIM_LLM_JSON_SCHEMA",
    "CLAIM_LLM_GATE",
    "CLAIM_LLM_ROLE",
    "CLAIM_RULES_FALLBACK",
    "CLAIM_VALIDATOR_BATCH",
    "CLAIM_EXTRACTION_MODEL",
    "ANTHROPIC_API_KEY",
    "Anthropic_API_Key",
):
    os.environ.pop(_var, None)


# Turnbull-corpus entity labels used by tests that build a writer directly.
# Any test that constructs InMemoryGraphWriter / Neo4jGraphWriter without going
# through load_graph() should pass entity_labels=TEST_ENTITY_LABELS.
TEST_ENTITY_LABELS: frozenset[str] = frozenset({
    "Refuge", "Place", "Person", "Organization", "Species",
    "Activity", "Period", "Habitat", "SurveyMethod",
})

# Role → (target label, edge type) tables matching the wildlife domain schema.
# Tests that build a writer directly pass these to keep the role-driven edge
# emission producing the same edges as the legacy hardcoded path.
TEST_OBSERVATION_ROLE_EDGES: dict[str, tuple[str, str]] = {
    "species":       ("Species",      "OF_SPECIES"),
    "refuge":        ("Refuge",       "AT_REFUGE"),
    "habitat":       ("Habitat",      "IN_HABITAT"),
    "survey_method": ("SurveyMethod", "USED_METHOD"),
}
TEST_EVENT_ROLE_EDGES: dict[str, tuple[str, str]] = {
    "species":       ("Species",      "INVOLVED_SPECIES"),
    "refuge":        ("Refuge",       "OCCURRED_AT"),
    "habitat":       ("Habitat",      "IN_HABITAT"),
    "survey_method": ("SurveyMethod", "USED_METHOD"),
}

# Categorial label projection matching the Turnbull domain schema. Tests that
# build a writer directly pass this so Refuge entities also get the :Place label.
TEST_CATEGORIAL_LABEL_MAP: dict[str, str] = {"Refuge": "Place"}


@pytest.fixture()
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def populated_writer(tmp_path_factory):
    """Run the full pipeline on all 3 fixture reports into an InMemoryGraphWriter."""
    from gemynd.ingest.pipeline import run_e2e
    from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR

    _fixtures = Path(__file__).parent / "fixtures"
    tmp = tmp_path_factory.mktemp("graph_data")
    result = run_e2e(
        [str(_fixtures / f) for f in ("report1.json", "report2.json", "report3.json")],
        tmp,
        backend="memory",
        resources_dir=TURNBULL_RESOURCES_DIR,
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
