"""Export a scoped, source-linked Neo4j corpus into a standalone SQLite snapshot."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from gemynd.core.evidence_fingerprint import evidence_fingerprint_sha256  # noqa: E402
from gemynd.mcp.sqlite_snapshot import build_sqlite_snapshot  # noqa: E402
from gemynd.shared.neo4j_helpers import build_neo4j_executor  # noqa: E402
from gemynd.shared.settings import Settings  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# This deliberately mirrors production retrieval policy: public, unrestricted,
# non-deleted documents and only the latest extraction run for each document.
# institution_ids is fail-closed unless --all-institutions is explicitly used.
EXPORT_QUERY = """
MATCH (c:Claim)<-[:HAS_CLAIM]-(p:Paragraph)
      <-[:HAS_PARAGRAPH]-(:Section)<-[:HAS_SECTION]-(pg:Page)
      <-[:HAS_PAGE]-(d:Document)
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_id: c.run_id})
WHERE c.claim_id > $after_claim_id
  AND ($institution_ids IS NULL OR d.institution_id IN $institution_ids)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
  AND c.source_sentence IS NOT NULL
  AND trim(c.source_sentence) <> ''
  AND d.deleted_at IS NULL
  AND coalesce(d.access_level, 'public') = 'public'
  AND coalesce(d.donor_restricted, false) = false
  AND NOT EXISTS {
    MATCH (d)-[:PROCESSED_BY]->(newer:ExtractionRun)
    WHERE newer.run_timestamp > lr.run_timestamp
  }
CALL (p) {
  OPTIONAL MATCH (p)-[:CONTAINS_MENTION]->(m:Mention)-[:REFERS_TO]->(e)
  WITH m, e WHERE e IS NOT NULL
  RETURN collect(DISTINCT {
    entity_id: e.entity_id,
    entity_type: coalesce(e.entity_type, head(labels(e))),
    name: coalesce(e.name, m.surface_form)
  }) AS subjects
}
RETURN c.claim_id AS claim_id,
       c.source_sentence AS source_sentence,
       c.extraction_confidence AS confidence,
       c.claim_type AS claim_type,
       c.epistemic_status AS attestation,
       c.fingerprint_sha256 AS fingerprint_sha256,
       d.doc_id AS doc_id,
       d.title AS doc_title,
       d.institution_id AS institution_id,
       d.series AS series,
       d.date_start AS date_start,
       d.date_end AS date_end,
       d.report_year AS report_year,
       d.source_url AS source_url,
       d.archive_ref AS archive_ref,
       d.lccn AS lccn,
       d.edition AS edition,
       pg.page_number AS page_number,
       p.paragraph_id AS paragraph_id,
       subjects
ORDER BY c.claim_id, p.paragraph_id
LIMIT $batch_size
"""


def _year(row: dict[str, Any]) -> int | None:
    value = row.get("report_year") or row.get("date_start")
    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(value or ""))
    return int(match.group(1)) if match else None


def _source_url(row: dict[str, Any]) -> str | None:
    source_url = str(row.get("source_url") or "").strip()
    if source_url:
        return source_url
    archive_ref = str(row.get("archive_ref") or "").strip()
    if archive_ref.startswith(("https://", "http://")):
        return archive_ref
    lccn = str(row.get("lccn") or "").strip()
    date_start = str(row.get("date_start") or "").strip()
    edition = str(row.get("edition") or "").strip()
    if lccn and date_start and edition:
        return f"https://www.loc.gov/item/{lccn}/{date_start}/ed-{edition}/"
    return None


def _git_commit() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "status",
                "--porcelain",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _rows(
    executor: Any,
    batch_size: int,
    max_records: int | None,
    institution_ids: list[str] | None,
) -> Iterator[dict[str, Any]]:
    after = ""
    emitted = 0
    while True:
        rows = executor.run(
            EXPORT_QUERY,
            {
                "after_claim_id": after,
                "batch_size": batch_size,
                "institution_ids": institution_ids,
            },
        )
        if not rows:
            return
        last_claim = after
        for row in rows:
            claim_id = str(row.get("claim_id") or "")
            if not claim_id:
                continue
            last_claim = claim_id
            sentence = str(row.get("source_sentence") or "").strip()
            doc_id = str(row.get("doc_id") or "")
            paragraph_id = str(row.get("paragraph_id") or "")
            fingerprint = str(row.get("fingerprint_sha256") or "").strip()
            if not fingerprint:
                fingerprint = evidence_fingerprint_sha256(
                    claim_id=claim_id,
                    doc_id=doc_id,
                    paragraph_id=paragraph_id,
                    source_sentence=sentence,
                )
            yield {
                "evidence_id": claim_id,
                "provenance_id": fingerprint,
                "claim": sentence,
                "excerpt": sentence,
                "year": _year(row),
                "confidence": row.get("confidence"),
                "claim_type": row.get("claim_type"),
                "attestation": row.get("attestation") or "direct",
                "subjects": row.get("subjects") or [],
                "relationships": [],
                "place": [],
                "source": {
                    "institution": row.get("institution_id"),
                    "collection": row.get("series"),
                    "document": row.get("doc_title"),
                    "source_id": doc_id,
                    "page_or_record": row.get("page_number") or paragraph_id,
                    "page_number": row.get("page_number"),
                    "paragraph_id": paragraph_id,
                    "date_start": row.get("date_start"),
                    "date_end": row.get("date_end"),
                    "source_url": _source_url(row),
                    "evidence_fingerprint": fingerprint,
                },
            }
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return
        if last_claim == after:
            return
        after = last_claim
        print(f"exported {emitted} claims; after={after}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--snapshot-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Build a development snapshot and mark it as coming from a dirty tree.",
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--institution-id",
        action="append",
        dest="institution_ids",
        help="Institution/corpus to export; repeat for multiple corpora.",
    )
    scope.add_argument(
        "--all-institutions",
        action="store_true",
        help="Explicitly export every institution. Avoid for tenant-scoped releases.",
    )
    args = parser.parse_args()
    institution_ids = None if args.all_institutions else sorted(set(args.institution_ids))
    settings = Settings.from_env()
    built_at = datetime.now(timezone.utc).isoformat()
    code_dirty = _git_dirty()
    if code_dirty and not args.allow_dirty:
        raise SystemExit("refusing to build a production snapshot from a dirty working tree")
    executor = build_neo4j_executor(settings)
    try:
        manifest = build_sqlite_snapshot(
            _rows(executor, args.batch_size, args.max_records, institution_ids),
            args.output,
            snapshot_id=args.snapshot_id,
            scope=(
                "public latest-run source-linked claims for "
                + (", ".join(institution_ids) if institution_ids else "all institutions")
            ),
            metadata={
                "built_at": built_at,
                "code_commit": _git_commit(),
                "code_dirty": code_dirty,
                "source_database": settings.neo4j_database,
                "institution_ids": institution_ids or ["*"],
                "public_only": True,
                "latest_run_only": True,
            },
        )
    finally:
        executor.close()
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
