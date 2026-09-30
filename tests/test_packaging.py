from __future__ import annotations

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = Path(__file__).resolve().parents[1]


def test_every_package_on_disk_is_declared_for_installation() -> None:
    """A missing entry installs cleanly but crashes on import (gemynd-mcp startup)."""
    declared = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
            "setuptools"
        ]["packages"]
    )
    on_disk = {
        ".".join(init.parent.relative_to(ROOT).parts)
        for init in (ROOT / "gemynd").rglob("__init__.py")
        if "__pycache__" not in init.parts
    }
    assert on_disk - declared == set()
