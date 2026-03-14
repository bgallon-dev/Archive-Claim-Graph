from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import SemanticBundle, StructureBundle


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_structure_bundle(path: str | Path, bundle: StructureBundle) -> None:
    save_json(path, bundle.to_dict())


def load_structure_bundle(path: str | Path) -> StructureBundle:
    return StructureBundle.from_dict(load_json(path))


def save_semantic_bundle(path: str | Path, bundle: SemanticBundle) -> None:
    save_json(path, bundle.to_dict())


def load_semantic_bundle(path: str | Path) -> SemanticBundle:
    return SemanticBundle.from_dict(load_json(path))
