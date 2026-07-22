"""Bulk `load-graph --input-dir`: streaming + resumable.

The bundle directory for a large corpus can hold hundreds of thousands of
pairs, so the loader must not read them all into memory, and an interrupted
load must resume rather than restart.
"""
from __future__ import annotations

import json
from pathlib import Path

from gemynd.cli import main as cli_main
from gemynd.ingest.checkpoint import LOAD_CHECKPOINT_FILENAME
from gemynd.ingest.pipeline import run_e2e
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR


def _make_bundles(fixtures_dir: Path, out: Path) -> int:
    inputs = [str(fixtures_dir / f"report{n}.json") for n in (1, 2, 3)]
    run_e2e(inputs, out, backend="memory", resources_dir=TURNBULL_RESOURCES_DIR)
    return len(list(out.glob("*.structure.json")))


def _load(bundle_dir: Path, capsys) -> dict:
    rc = cli_main([
        "load-graph",
        "--input-dir", str(bundle_dir),
        "--backend", "memory",
        "--domain-dir", str(TURNBULL_RESOURCES_DIR),
    ])
    assert rc in (0, None)
    return json.loads(capsys.readouterr().out)


def test_bulk_load_writes_all_bundles(fixtures_dir: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "bundles"
    n = _make_bundles(fixtures_dir, out)
    assert n == 3

    result = _load(out, capsys)
    assert result["status"] == "ok"
    assert result["documents_loaded"] == 3
    assert result["already_loaded"] == 0
    assert (out / LOAD_CHECKPOINT_FILENAME).exists()


def test_bulk_load_resumes_from_checkpoint(fixtures_dir: Path, tmp_path: Path, capsys) -> None:
    """A second run must skip everything already loaded, not redo it."""
    out = tmp_path / "bundles"
    _make_bundles(fixtures_dir, out)

    first = _load(out, capsys)
    assert first["documents_loaded"] == 3

    second = _load(out, capsys)
    assert second["documents_loaded"] == 0
    assert second["already_loaded"] == 3


def test_bulk_load_checkpoint_is_separate_from_ingest_checkpoint(
    fixtures_dir: Path, tmp_path: Path, capsys
) -> None:
    """The load checkpoint must not collide with the ingest checkpoint that
    lives in the same directory (which would corrupt resume for either)."""
    from gemynd.ingest.checkpoint import CHECKPOINT_FILENAME

    out = tmp_path / "bundles"
    _make_bundles(fixtures_dir, out)
    _load(out, capsys)

    assert (out / CHECKPOINT_FILENAME).exists()       # written by run_e2e
    assert (out / LOAD_CHECKPOINT_FILENAME).exists()  # written by load-graph
    assert CHECKPOINT_FILENAME != LOAD_CHECKPOINT_FILENAME
    # ingest checkpoint still only holds the ingest entries
    ingest_ids = {
        json.loads(l)["doc_id"]
        for l in (out / CHECKPOINT_FILENAME).read_text(encoding="utf-8").splitlines() if l.strip()
    }
    load_ids = {
        json.loads(l)["doc_id"]
        for l in (out / LOAD_CHECKPOINT_FILENAME).read_text(encoding="utf-8").splitlines() if l.strip()
    }
    assert ingest_ids == load_ids  # same docs, tracked independently
