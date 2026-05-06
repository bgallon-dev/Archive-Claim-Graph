"""Consolidate duplicate-normalized-form Entity nodes in Neo4j.

The graph can accumulate distinct Entity nodes that share a `normalized_form`
but differ in `entity_type` (e.g. Organization "WASHINGTON STATE" alongside
seeded Place "Washington State"). This script picks one canonical variant per
form, reroutes all incoming relationships onto it, and soft-deletes the rest.

The companion fix in `gemynd/core/resolver.py` (supplementary-candidate skip
keyed on normalized_form alone) prevents new duplicates from appearing on
ingest. Run that fix FIRST, then run this script.

Usage::

    # Dry-run: emit a report of proposed merges without touching the graph.
    python scripts/consolidate_duplicate_entities.py \\
        --resources-dir gemynd/resources \\
        --resources-dir gemynd/resources_newspaper \\
        --output-report consolidate_report.csv

    # Execute after reviewing the report.
    python scripts/consolidate_duplicate_entities.py \\
        --resources-dir gemynd/resources \\
        --resources-dir gemynd/resources_newspaper \\
        --execute

Canonical-selection rule:
    1. The variant whose (entity_type, normalized_form) matches a seed CSV
       entry from any --resources-dir.
    2. If no seed match, the variant with the highest claim_count.
    3. If two seed matches across resource dirs disagree on type, or if the
       claim-count tie-break is itself a tie, the group is SKIPPED and emitted
       as `manual_review` in the report.

Environment variable fallbacks: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
NEO4J_DATABASE.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("consolidate_duplicate_entities")

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from gemynd.core.ids import make_entity_id  # noqa: E402
from gemynd.core.resolver import default_seed_entities  # noqa: E402
from gemynd.shared.env import load_dotenv  # noqa: E402


# Period entities are intentionally per-document (entity_id includes source_title
# in its hash) and they all share the date-range string as normalized_form, so
# they would otherwise appear as massive duplicate groups. Merging them would
# corrupt per-document provenance — exclude at the source.
_DUPLICATE_GROUPS_QUERY = """
MATCH (e:Entity)
WHERE coalesce(e.deleted_at, 0) = 0
  AND NOT 'Period' IN labels(e)
  AND e.normalized_form IS NOT NULL
  AND e.normalized_form <> ''
WITH e.normalized_form AS nf, collect(e) AS group
WHERE size(group) > 1
UNWIND group AS node
OPTIONAL MATCH (node)<-[r]-(src)
WITH nf, node, count(r) AS in_degree
OPTIONAL MATCH (node)<-[]-(c:Claim)
WITH nf, node, in_degree, count(DISTINCT c) AS claim_count
RETURN nf AS normalized_form,
       collect({
         entity_id: node.entity_id,
         entity_type: [l IN labels(node) WHERE l <> 'Entity'][0],
         name: node.name,
         claim_count: claim_count,
         in_degree: in_degree
       }) AS variants
ORDER BY normalized_form
"""

_INCIDENT_REL_TYPES_QUERY = """
MATCH (dupe:Entity {entity_id: $dupe_id})<-[r]-()
RETURN DISTINCT type(r) AS rel_type
"""

# Per-relation rewrite. The relation type is interpolated into the query
# string (Cypher does not allow parameterized relation types), so all callers
# MUST pass rel_type from the discovery query above (no user input).
_REROUTE_TEMPLATE = """
MATCH (dupe:Entity {{entity_id: $dupe_id}})<-[r:`{rel_type}`]-(src)
WITH r, src, properties(r) AS props
MATCH (canonical:Entity {{entity_id: $canonical_id}})
MERGE (src)-[r2:`{rel_type}`]->(canonical)
SET r2 += props
DELETE r
RETURN count(r2) AS rerouted
"""

_SOFT_DELETE_QUERY = """
MATCH (dupe:Entity {entity_id: $dupe_id})
SET dupe.deleted_at = timestamp(),
    dupe.merged_into = $canonical_id
RETURN dupe.entity_id AS marked
"""


def _load_seed_keys(resources_dirs: list[Path]) -> dict[str, set[str]]:
    """Return {normalized_form: {entity_type, ...}} across all resource dirs."""
    seed_keys: dict[str, set[str]] = defaultdict(set)
    for rd in resources_dirs:
        for entity in default_seed_entities(rd):
            seed_keys[entity.normalized_form].add(entity.entity_type)
    return dict(seed_keys)


def _classify_group(
    normalized_form: str,
    variants: list[dict],
    seed_keys: dict[str, set[str]],
) -> tuple[str | None, list[str], str]:
    """Choose canonical entity_id for a duplicate group.

    Returns (canonical_entity_id, dupe_entity_ids, decision_reason). If the
    group should be skipped, canonical_entity_id is None and reason explains.

    Selection rules, in order:
      1. Among variants whose entity_type matches a seed type for this form,
         prefer the one whose entity_id equals make_entity_id(type, form) —
         that is the entity created by the standard ingest path. Synthetic
         IDs like ``place_created_tacoma`` or stale-prefix IDs like
         ``person_fc71...`` (Organization labelled later) lose to it.
      2. If exactly one variant has the canonical hashed ID, take it.
      3. If no variant has the canonical hashed ID but there is a single
         seed-typed variant, take it (e.g. only the synthetic exists).
      4. If none matches a seed, fall back to claim count.
      5. Ties → manual review.
    """
    if not normalized_form:
        return None, [], "skip:null_normalized_form"

    seed_types = seed_keys.get(normalized_form, set())
    seed_matches = [v for v in variants if v["entity_type"] in seed_types]

    if seed_matches:
        with_canonical_id = [
            v for v in seed_matches
            if v["entity_id"] == make_entity_id(v["entity_type"], normalized_form)
        ]
        if len(with_canonical_id) == 1:
            canonical = with_canonical_id[0]
            reason = f"seed_canonical:{canonical['entity_type']}"
        elif len(with_canonical_id) > 1:
            return None, [], (
                f"seed_disagreement: multiple canonical IDs across types "
                f"{sorted({v['entity_type'] for v in with_canonical_id})}"
            )
        elif len(seed_matches) == 1:
            canonical = seed_matches[0]
            reason = f"seed_match_no_canonical_id:{canonical['entity_type']}"
        else:
            # Several synthetic variants of seed types but no canonical ID.
            ranked = sorted(
                seed_matches,
                key=lambda v: (v["claim_count"], v["in_degree"]),
                reverse=True,
            )
            canonical = ranked[0]
            reason = f"seed_match_synthetic_only:{canonical['entity_type']}"
    else:
        # No seed match — fall back to claim count.
        ranked = sorted(
            variants,
            key=lambda v: (v["claim_count"], v["in_degree"]),
            reverse=True,
        )
        if (
            len(ranked) >= 2
            and ranked[0]["claim_count"] == ranked[1]["claim_count"]
            and ranked[0]["in_degree"] == ranked[1]["in_degree"]
        ):
            return None, [], "tie_no_seed: claim_count and in_degree both tied"
        canonical = ranked[0]
        reason = f"claim_count:{canonical['claim_count']}"

    dupes = [v["entity_id"] for v in variants if v["entity_id"] != canonical["entity_id"]]
    return canonical["entity_id"], dupes, reason


def _emit_report(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "normalized_form", "decision", "canonical_entity_id", "canonical_type",
        "canonical_name", "canonical_claim_count", "dupe_entity_ids",
        "dupes_total_claims", "variant_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _log.info("Wrote report to %s (%d groups)", path, len(rows))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument(
        "--resources-dir", action="append", type=Path, default=None,
        help="Resource directory containing seed_entities.csv. Pass multiple times for multi-corpus deployments.",
    )
    parser.add_argument(
        "--output-report", type=Path, default=Path("consolidate_report.csv"),
        help="Path for the per-group decision report (CSV).",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Apply the merges. Without this flag the script is dry-run.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Number of dupe-merges per Neo4j transaction (default: 500).",
    )
    args = parser.parse_args()

    resources_dirs = args.resources_dir or [
        _project_root / "gemynd" / "resources",
        _project_root / "gemynd" / "resources_newspaper",
    ]
    seed_keys = _load_seed_keys(resources_dirs)
    _log.info("Loaded %d distinct normalized seed forms from %d resource dirs",
              len(seed_keys), len(resources_dirs))

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("neo4j package required: pip install neo4j", file=sys.stderr)
        sys.exit(1)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    driver.verify_connectivity()

    try:
        with driver.session(database=args.neo4j_database) as session:
            result = session.run(_DUPLICATE_GROUPS_QUERY)
            groups = [{"normalized_form": r["normalized_form"], "variants": r["variants"]} for r in result]
        _log.info("Found %d duplicate-form groups", len(groups))

        report_rows: list[dict] = []
        merge_plan: list[tuple[str, list[str]]] = []  # (canonical_id, dupe_ids)
        for group in groups:
            nf = group["normalized_form"]
            variants = group["variants"]
            canonical_id, dupe_ids, reason = _classify_group(nf, variants, seed_keys)
            canonical_variant = next((v for v in variants if v["entity_id"] == canonical_id), None) if canonical_id else None
            dupes_total_claims = sum(
                v["claim_count"] for v in variants if v["entity_id"] in set(dupe_ids)
            )
            report_rows.append({
                "normalized_form": nf,
                "decision": reason if canonical_id else f"SKIP:{reason}",
                "canonical_entity_id": canonical_id or "",
                "canonical_type": canonical_variant["entity_type"] if canonical_variant else "",
                "canonical_name": canonical_variant["name"] if canonical_variant else "",
                "canonical_claim_count": canonical_variant["claim_count"] if canonical_variant else "",
                "dupe_entity_ids": ";".join(dupe_ids),
                "dupes_total_claims": dupes_total_claims,
                "variant_count": len(variants),
            })
            if canonical_id:
                merge_plan.append((canonical_id, dupe_ids))

        _emit_report(report_rows, args.output_report)

        if not args.execute:
            _log.info("Dry-run mode. Re-run with --execute to apply %d merges affecting %d duplicate nodes.",
                      len(merge_plan), sum(len(d) for _, d in merge_plan))
            return

        # ── Execute mode ────────────────────────────────────────────────
        total_dupes = sum(len(dupes) for _, dupes in merge_plan)
        _log.info("Executing %d merges affecting %d duplicate nodes (batch size %d)",
                  len(merge_plan), total_dupes, args.batch_size)

        merged = 0
        for canonical_id, dupe_ids in merge_plan:
            with driver.session(database=args.neo4j_database) as session:
                for dupe_id in dupe_ids:
                    rel_types = [
                        r["rel_type"]
                        for r in session.run(_INCIDENT_REL_TYPES_QUERY, dupe_id=dupe_id)
                    ]
                    for rel_type in rel_types:
                        session.run(
                            _REROUTE_TEMPLATE.format(rel_type=rel_type),
                            dupe_id=dupe_id, canonical_id=canonical_id,
                        )
                    session.run(_SOFT_DELETE_QUERY, dupe_id=dupe_id, canonical_id=canonical_id)
                    merged += 1
                    if merged % args.batch_size == 0:
                        _log.info("Progress: %d / %d duplicates merged", merged, total_dupes)
        _log.info("Done. Merged %d duplicate nodes into %d canonicals.", merged, len(merge_plan))

    finally:
        driver.close()


if __name__ == "__main__":
    main()
