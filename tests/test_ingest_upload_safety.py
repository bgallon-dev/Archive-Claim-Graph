"""Regression tests for the ingest upload path-traversal guard.

`_safe_staged_dest` is the guard promised by ``ingest/CLAUDE.md`` ("path
traversal guards prevent directory escape in server contexts"). Before it
existed, ``create_ingest_job`` did ``dest = staging_dir / upload.filename`` with
an attacker-controlled filename — an absolute path made ``/`` discard the
staging dir and ``..`` segments escaped it, turning ``write_bytes`` into an
arbitrary-file write. These tests pin the guarantee: whatever the filename, the
destination stays a direct child of the staging directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gemynd.ingest.web.app import _safe_staged_dest


def _assert_contained(dest: Path, staging: Path, expected_name: str) -> None:
    staging_resolved = staging.resolve()
    assert dest.parent == staging_resolved, f"{dest} escaped {staging_resolved}"
    assert dest.name == expected_name
    # Belt-and-suspenders: the resolved dest must be relative to staging.
    dest.relative_to(staging_resolved)


def test_plain_filename_stays_in_staging(tmp_path: Path) -> None:
    _assert_contained(_safe_staged_dest(tmp_path, "report.json"), tmp_path, "report.json")


def test_absolute_path_is_reduced_to_basename(tmp_path: Path) -> None:
    # An absolute right-operand would make ``staging / name`` == the absolute
    # path itself under pathlib. The guard must strip it back to a basename.
    dest = _safe_staged_dest(tmp_path, "/etc/cron.d/evil.json")
    _assert_contained(dest, tmp_path, "evil.json")


@pytest.mark.parametrize(
    "hostile",
    [
        "../../../../etc/evil.json",
        "..\\..\\Windows\\System32\\evil.pdf",
        "subdir/nested/evil.json",
        "....//evil.json",
    ],
)
def test_traversal_attempts_are_neutralized(tmp_path: Path, hostile: str) -> None:
    dest = _safe_staged_dest(tmp_path, hostile)
    # Regardless of the traversal shape, the write lands directly in staging.
    assert dest.parent == tmp_path.resolve()
    dest.relative_to(tmp_path.resolve())


@pytest.mark.parametrize("bad", ["", ".", "..", None, "/", "../", "foo/.."])
def test_degenerate_names_are_rejected(tmp_path: Path, bad: str | None) -> None:
    # These reduce to no usable basename; the handler must reject, not guess.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _safe_staged_dest(tmp_path, bad)
    assert exc.value.status_code == 422
