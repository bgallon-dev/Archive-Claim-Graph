"""Validate an evidence snapshot before upload, activation, or rollback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemynd.mcp.snapshot import SNAPSHOT_FORMAT, EvidenceSnapshot  # noqa: E402
from gemynd.mcp.sqlite_snapshot import (  # noqa: E402
    SUPPORTED_SQLITE_SNAPSHOT_FORMATS,
    validate_sqlite_snapshot,
)


def validate(
    snapshot_path: Path,
    *,
    minimum_records: int = 1,
    expected_snapshot_id: str | None = None,
    expected_institutions: set[str] | None = None,
) -> dict:
    manifest_path = snapshot_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"snapshot manifest is unavailable or invalid: {exc}") from exc
    snapshot_format = manifest.get("format")
    if snapshot_format in SUPPORTED_SQLITE_SNAPSHOT_FORMATS:
        manifest, _database = validate_sqlite_snapshot(snapshot_path)
    elif snapshot_format == SNAPSHOT_FORMAT:
        manifest = EvidenceSnapshot(snapshot_path).manifest
    else:
        raise ValueError(f"unsupported snapshot format: {snapshot_format!r}")
    record_count = int(manifest.get("record_count", -1))
    if record_count < minimum_records:
        raise ValueError(f"snapshot has {record_count} records; minimum is {minimum_records}")
    if expected_snapshot_id and manifest.get("snapshot_id") != expected_snapshot_id:
        raise ValueError(
            f"snapshot_id is {manifest.get('snapshot_id')!r}; expected {expected_snapshot_id!r}"
        )
    if expected_institutions:
        actual = set((manifest.get("institutions") or {}).keys())
        missing = expected_institutions - actual
        if missing:
            raise ValueError(
                "snapshot is missing expected institution(s): " + ", ".join(sorted(missing))
            )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--minimum-records", type=int, default=1)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--institution-id", action="append", default=[])
    args = parser.parse_args()
    try:
        manifest = validate(
            args.snapshot.resolve(),
            minimum_records=max(0, args.minimum_records),
            expected_snapshot_id=args.snapshot_id,
            expected_institutions=set(args.institution_id),
        )
    except (OSError, ValueError) as exc:
        print(f"snapshot validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "valid",
                "snapshot_id": manifest.get("snapshot_id"),
                "format": manifest.get("format"),
                "record_count": manifest.get("record_count"),
                "size_bytes": manifest.get("size_bytes"),
                "institutions": manifest.get("institutions", {}),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
