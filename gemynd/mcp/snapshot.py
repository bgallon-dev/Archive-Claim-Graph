"""Versioned, integrity-checked evidence snapshots for standalone serving."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SNAPSHOT_FORMAT = "kettle-evidence-snapshot-v1"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict) and "evidence" in payload:
                rows = payload["evidence"]
            else:
                # A one-record JSONL file is also valid JSON; preserve that
                # record instead of interpreting it as an empty envelope.
                rows = [payload]
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(f"evidence input is not valid JSON or JSONL: {exc}") from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("evidence input must contain an array of objects")
    return [dict(row) for row in rows]


def validate_rows(rows: list[dict[str, Any]]) -> None:
    for number, row in enumerate(rows, start=1):
        if not isinstance(row.get("evidence_id"), str) or not row["evidence_id"].strip():
            raise ValueError(f"evidence row {number} is missing evidence_id")
        if not isinstance(row.get("claim"), str) or not row["claim"].strip():
            raise ValueError(f"evidence row {number} is missing claim")
        if not isinstance(row.get("source"), dict):
            raise ValueError(f"evidence row {number} is missing source metadata")


def build_snapshot(input_path: str | Path, output_path: str | Path, *, snapshot_id: str) -> dict[str, Any]:
    """Write a deterministic snapshot and return its manifest."""
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rows = _read_rows(source)
    validate_rows(rows)
    evidence_path = destination / "evidence.jsonl"
    with evidence_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    manifest = {
        "format": SNAPSHOT_FORMAT,
        "snapshot_id": snapshot_id,
        "record_count": len(rows),
        "evidence_file": evidence_path.name,
        "sha256": digest,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


class EvidenceSnapshot:
    """Load a snapshot only after validating its manifest and content hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        manifest_path = self.path / "manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"snapshot manifest is unavailable or invalid: {exc}") from exc
        if self.manifest.get("format") != SNAPSHOT_FORMAT:
            raise ValueError("unsupported evidence snapshot format")
        snapshot_id = self.manifest.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("snapshot manifest is missing snapshot_id")
        filename = self.manifest.get("evidence_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("snapshot evidence_file must be a filename")
        evidence_path = self.path / filename
        if not evidence_path.is_file():
            raise ValueError("snapshot evidence file is missing")
        digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        if digest != self.manifest.get("sha256"):
            raise ValueError("snapshot evidence checksum mismatch")
        rows = _read_rows(evidence_path)
        validate_rows(rows)
        if len(rows) != int(self.manifest.get("record_count", -1)):
            raise ValueError("snapshot record_count does not match evidence file")
        self.records = rows
