"""Review export – JSON/CSV snapshots of proposals, revisions, and patches."""
from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from .store import ReviewStore


def export_proposals_json(
    store: ReviewStore,
    output_path: str | Path,
    *,
    status: str | None = None,
    snapshot_id: str | None = None,
) -> int:
    """Export proposals (with targets, revisions, events) to a JSON file."""
    data = store.export_proposals_json(status=status, snapshot_id=snapshot_id)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    return len(data)


def export_proposals_csv(
    store: ReviewStore,
    output_path: str | Path,
    *,
    status: str | None = None,
    snapshot_id: str | None = None,
) -> int:
    """Export proposals to a flat CSV file (one row per proposal)."""
    proposals = store.list_proposals(status=status, snapshot_id=snapshot_id, limit=10000)
    if not proposals:
        Path(output_path).write_text("", encoding="utf-8")
        return 0

    rows: list[dict[str, Any]] = []
    for p in proposals:
        rev = store.get_latest_revision(p.proposal_id)
        row: dict[str, Any] = p.to_dict()
        if rev:
            row["patch_spec_json"] = rev.patch_spec_json
            row["detector_name"] = rev.detector_name
            row["detector_version"] = rev.detector_version
            row["validation_state"] = rev.validation_state
        rows.append(row)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_accepted_patches_json(
    store: ReviewStore,
    output_path: str | Path,
    *,
    snapshot_id: str | None = None,
) -> int:
    """Export accepted patch_spec payloads to a JSON file."""
    patches = store.export_accepted_patches(snapshot_id=snapshot_id)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(patches, indent=2, ensure_ascii=True), encoding="utf-8")
    return len(patches)


def export_revision_history_json(
    store: ReviewStore,
    output_path: str | Path,
    proposal_id: str,
) -> int:
    """Export the full revision history for one proposal."""
    revisions = store.get_revisions(proposal_id)
    events = store.get_correction_events(proposal_id)
    data = {
        "proposal_id": proposal_id,
        "revisions": [r.to_dict() for r in revisions],
        "correction_events": [e.to_dict() for e in events],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    return len(revisions)
