"""Backfill lccn / edition / loc_sequence on existing newspaper Document nodes.

These three fields were added to ``DocumentRecord`` as part of the LoC URL
migration (canonical ``loc.gov/resource/`` form). Documents ingested before
that change have NULL values, so the LoC source-link buttons in the
retrieval UI do not render. The Neo4j-side hash check in
``gemynd/ingest/pipeline.py`` skips re-ingestion of unchanged inputs, so
re-running ``ingest_newspapers.py`` will not populate them.

This script reads the input JSONs and the source-file naming convention,
then SETs the three fields on the existing nodes via a batched UNWIND.
No claim re-extraction, no node re-creation, no LLM cost. Idempotent —
safe to re-run.

Usage:
    python scripts/backfill_loc_fields.py             # apply
    python scripts/backfill_loc_fields.py --dry-run   # preview without writing
    python scripts/backfill_loc_fields.py --input-dir ../OtherCorpus/json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from gemynd.ingest.source_parser import _extract_loc_sequence, _strip_leading_zeros  # noqa: E402
from gemynd.retrieval.executor import Neo4jQueryExecutor  # noqa: E402


_GRAPHRAG_ROOT = _REPO_ROOT.parent
_DEFAULT_INPUT_DIR = _GRAPHRAG_ROOT / "Newspapers" / "gemynd_input"


_BACKFILL_QUERY = """
UNWIND $rows AS row
MATCH (d:Document {archive_ref: row.archive_ref})
SET d.lccn = row.lccn,
    d.edition = row.edition,
    d.loc_sequence = row.loc_sequence
RETURN row.archive_ref AS archive_ref
"""


def _build_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  warn: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        meta = payload.get("metadata") or {}
        archive_ref = meta.get("archive_ref")
        if not archive_ref:
            continue
        rows.append({
            "archive_ref": archive_ref,
            "lccn": meta.get("lccn"),
            # Mirror source_parser.parse_source_payload normalization so the
            # backfilled values match what fresh ingest would write.
            "edition": _strip_leading_zeros(meta.get("edition")),
            "loc_sequence": _extract_loc_sequence(meta.get("source_file") or path.name),
        })
    return rows


def _chunked(seq: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _print_coverage(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    with_lccn = sum(1 for r in rows if r["lccn"])
    with_edition = sum(1 for r in rows if r["edition"])
    with_seq = sum(1 for r in rows if r["loc_sequence"])
    distinct_lccns = sorted({r["lccn"] for r in rows if r["lccn"]})
    print(f"  with lccn:         {with_lccn}/{total}")
    print(f"  with edition:      {with_edition}/{total}")
    print(f"  with loc_sequence: {with_seq}/{total}")
    print(f"  distinct lccns:    {distinct_lccns}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill lccn/edition/loc_sequence on existing newspaper Document nodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print coverage stats and a row sample without connecting to Neo4j.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help=f"Directory of newspaper input JSONs. Default: {_DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="UNWIND batch size for the SET query. Default: 500.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"input dir not found: {args.input_dir}", file=sys.stderr)
        return 2

    rows = _build_rows(args.input_dir)
    print(f"read {len(rows)} input rows from {args.input_dir}")
    _print_coverage(rows)

    if not rows:
        print("no rows to backfill — exiting")
        return 0

    if args.dry_run:
        print("\n--dry-run: not connecting to Neo4j. Sample rows:")
        for sample in rows[:3]:
            print(f"  {sample}")
        return 0

    for env_var in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        if not os.environ.get(env_var):
            print(f"required env var not set: {env_var}", file=sys.stderr)
            return 2

    executor = Neo4jQueryExecutor(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        trust_mode=os.environ.get("NEO4J_TRUST", "system"),
    )

    matched_refs: set[str] = set()
    try:
        for batch_idx, batch in enumerate(_chunked(rows, args.batch_size), start=1):
            result = executor.run(_BACKFILL_QUERY, {"rows": batch})
            matched_refs.update(r["archive_ref"] for r in result)
            print(f"  batch {batch_idx}: matched {len(result)}/{len(batch)}")
    finally:
        executor.close()

    input_refs = {r["archive_ref"] for r in rows}
    not_found = sorted(input_refs - matched_refs)

    print()
    print(f"updated:            {len(matched_refs)}")
    print(f"not_found_in_neo4j: {len(not_found)}")
    if not_found:
        print(f"  first 5 missing: {not_found[:5]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
