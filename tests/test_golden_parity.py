"""Byte-for-byte Turnbull parity guard.

Re-extracts the SemanticBundle for the Turnbull fixtures with pinned run
metadata and diffs against frozen golden snapshots. This is the hard guarantee
behind the domain-agnosticism refactor: relocating Turnbull's hardcoded ontology
into config must not change Turnbull's extraction output by a single byte.

If extraction legitimately changes, regenerate the goldens by writing the same
``json.dumps(semantic.to_dict(), indent=2, sort_keys=True)`` payload to
``tests/fixtures/golden/<stem>.semantic.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemynd.core.domain_config import load_domain_config
from gemynd.ingest.pipeline import extract_semantic, parse_source
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR

_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"

# Every volatile run field is pinned so the bundle is fully deterministic.
_PINNED = {
    "run_id": "run_golden",
    "run_timestamp": "2020-01-01T00:00:00+00:00",
    "ocr_engine": "golden",
    "ocr_version": "golden",
    "normalizer_version": "v1",
    "ner_model": "rules",
    "relation_model": "rules",
}


@pytest.fixture(scope="module")
def _turnbull_config():
    return load_domain_config(TURNBULL_RESOURCES_DIR)


@pytest.mark.parametrize("stem", ["report1", "report2", "report3"])
def test_turnbull_semantic_bundle_matches_golden(stem: str, fixtures_dir: Path, _turnbull_config) -> None:
    structure = parse_source(
        fixtures_dir / f"{stem}.json", source_profile=_turnbull_config.source_profile
    )
    semantic = extract_semantic(
        structure, config=_turnbull_config, no_llm=True, run_overrides=dict(_PINNED)
    )
    produced = json.dumps(semantic.to_dict(), indent=2, sort_keys=True, ensure_ascii=True)
    golden = (_GOLDEN_DIR / f"{stem}.semantic.json").read_text(encoding="utf-8")
    assert produced == golden, (
        f"Turnbull extraction for {stem} diverged from the frozen golden — "
        f"the refactor changed Turnbull output. Inspect the diff before regenerating."
    )
