"""Build a versioned standalone Kettle evidence snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemynd.mcp.snapshot import build_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON or JSONL evidence input")
    parser.add_argument("--output", type=Path, required=True, help="New snapshot directory")
    parser.add_argument("--snapshot-id", required=True, help="Immutable release identifier")
    args = parser.parse_args()
    manifest = build_snapshot(args.input, args.output, snapshot_id=args.snapshot_id)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
