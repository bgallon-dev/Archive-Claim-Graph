from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import SemanticBundle, StructureBundle


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                key: json.dumps(value, ensure_ascii=True)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )

    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_rows)


def save_structure_bundle(path: str | Path, bundle: StructureBundle) -> None:
    save_json(path, bundle.to_dict())


def load_structure_bundle(path: str | Path) -> StructureBundle:
    return StructureBundle.from_dict(load_json(path))


def save_semantic_bundle(path: str | Path, bundle: SemanticBundle) -> None:
    save_json(path, bundle.to_dict())


def load_semantic_bundle(path: str | Path) -> SemanticBundle:
    return SemanticBundle.from_dict(load_json(path))
