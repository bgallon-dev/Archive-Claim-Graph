"""Tests for scripts/generate_ocr_variants.py (Phase 4B)."""
from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

import pytest

# Import the script module directly (no package install required).
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_ocr_variants.py"
_RESOURCES_DIR = Path(__file__).parent.parent / "graphrag_pipeline" / "resources"

import importlib.util

_spec = importlib.util.spec_from_file_location("generate_ocr_variants", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["generate_ocr_variants"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

BUILTIN_CONFUSIONS = _mod.BUILTIN_CONFUSIONS
OCRConfusionMatrix = _mod.OCRConfusionMatrix
generate_variants = _mod.generate_variants
_write_seed_patch = _mod._write_seed_patch


# ---------------------------------------------------------------------------
# 4B-1: from_sources() with no args includes all BUILTIN_CONFUSIONS
# ---------------------------------------------------------------------------

def test_from_sources_no_args_includes_all_builtins() -> None:
    matrix = OCRConfusionMatrix.from_sources()
    for key in BUILTIN_CONFUSIONS:
        assert key in matrix.pairs, f"Built-in pair {key!r} missing from matrix"


# ---------------------------------------------------------------------------
# 4B-2: known_errors dict adds char-pair entries at weight >= 0.20
# ---------------------------------------------------------------------------

def test_from_sources_known_errors_adds_pairs_at_high_weight() -> None:
    # "tumbull" → "turnbull" should give ("rn", "m") pair at >= 0.20
    matrix = OCRConfusionMatrix.from_sources(known_errors={"tumbull": "turnbull"})
    # ("rn", "m") means "rn" is misread as "m"
    pair = ("rn", "m")
    assert pair in matrix.pairs
    assert matrix.pairs[pair] >= 0.20


# ---------------------------------------------------------------------------
# 4B-3: generate_variants produces the rn→m substitution for "mallard"
# (mallard doesn't have rn, but "turnbull" → "tumbull" is the classic case)
# ---------------------------------------------------------------------------

def test_generate_variants_produces_rn_m_substitution() -> None:
    matrix = OCRConfusionMatrix.from_sources()
    # "Turnbull" contains "rn" which should be substituted by "m"
    results = dict(generate_variants("Turnbull", matrix))
    assert "Tumbull" in results, "Expected 'Tumbull' variant for 'Turnbull'"


# ---------------------------------------------------------------------------
# 4B-4: min_prob filter excludes low-probability variants
# ---------------------------------------------------------------------------

def test_generate_variants_min_prob_filter() -> None:
    # Build a matrix with only a very low-probability pair
    matrix = OCRConfusionMatrix(pairs={("u", "v"): 0.01})
    # min_prob=0.05 should exclude it
    results = generate_variants("Turnbull", matrix, min_prob=0.05)
    assert results == [], "Expected no variants when all pairs are below min_prob"
    # min_prob=0.005 should include it
    results_low = generate_variants("Turnbull", matrix, min_prob=0.005)
    assert len(results_low) > 0, "Expected variants when min_prob is below pair probability"


# ---------------------------------------------------------------------------
# 4B-5: seed_patch format includes correct entity_type column
# ---------------------------------------------------------------------------

def test_write_seed_patch_includes_entity_type() -> None:
    seed_rows = [
        {"entity_type": "Species", "name": "Mallard", "prop_key": "species_id", "prop_value": "mallard"},
    ]
    variant_rows = [
        {"entity_name": "Mallard", "variant": "Mallord", "probability": 0.10},
    ]
    out = io.StringIO()
    _write_seed_patch(seed_rows, variant_rows, out)
    out.seek(0)
    reader = list(csv.DictReader(out))
    assert len(reader) == 1
    assert reader[0]["entity_type"] == "Species"
    assert reader[0]["name"] == "Mallord"
    assert reader[0]["prop_key"] == "species_id"


# ---------------------------------------------------------------------------
# 4B-6: CLI smoke test — runs on Turnbull resources without error
# ---------------------------------------------------------------------------

def test_cli_smoke_test_runs_without_error() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--resources-dir", str(_RESOURCES_DIR),
            "--output-format", "csv",
            "--output", "-",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Script exited with code {result.returncode}.\nstdout: {result.stdout[:500]}"
        f"\nstderr: {result.stderr[:500]}"
    )
    # Output should have a CSV header
    assert "entity_name" in result.stdout
