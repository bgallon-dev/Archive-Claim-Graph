"""
run_seeding_candidates.py — Run the seeding-candidate ranking against Neo4j.

Loads seeded entity names + aliases from one or more seed_entities.csv files
and passes them as `$seed_names` to tools/seeding_candidates.cypher (which
excludes them). Writes results to a CSV.

Usage:
    python tools/run_seeding_candidates.py [--output FILE] [--seeds CSV] [--limit N]

Options:
    --output FILE   Output CSV path (default: seeding_candidates.csv)
    --seeds CSV     Path to a seed_entities.csv whose names should be excluded.
                    Repeatable. If omitted, defaults to the active resources:
                      gemynd/resources/seed_entities.csv
                      gemynd/resources_newspaper/seed_entities.csv
    --limit N       Override the LIMIT in the cypher (default: 200, file's value)
    --no-filter     Skip the seed-name exclusion (pass an empty list).

Requires NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (loaded from .env).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from gemynd.shared.env import load_dotenv
from gemynd.shared.settings import Settings
from gemynd.ingest.graph.writer import _build_driver_kwargs

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("neo4j package is not installed. Install with: pip install neo4j")

CYPHER_PATH = Path(__file__).resolve().parent / "seeding_candidates.cypher"
DEFAULT_SEED_CSVS = [
    _REPO_ROOT / "gemynd" / "resources" / "seed_entities.csv",
    _REPO_ROOT / "gemynd" / "resources_newspaper" / "seed_entities.csv",
]


def load_seed_names(csv_paths: list[Path]) -> list[str]:
    """Return the union of lowercased names + aliases across the given CSVs."""
    names: set[str] = set()
    for path in csv_paths:
        if not path.exists():
            print(f"[warn] seed CSV not found: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or "").strip()
                if name:
                    names.add(name.lower())
                if (row.get("prop_key") or "").strip().lower() == "alias":
                    alias = (row.get("prop_value") or "").strip()
                    if alias:
                        names.add(alias.lower())
    return sorted(names)


def run_query(
    settings: Settings,
    seed_names: list[str],
    limit: int | None,
    min_doc_spread: int,
    min_claim_types: int,
    exclude_periods: bool,
) -> list[dict]:
    cypher = CYPHER_PATH.read_text(encoding="utf-8")
    if limit is not None:
        # Cypher LIMIT requires a literal/parameter, not arithmetic — string
        # substitution is the simplest way to parameterize it from the file.
        cypher = cypher.replace("LIMIT 200", f"LIMIT {int(limit)}")

    driver_kwargs = _build_driver_kwargs(
        uri=settings.neo4j_uri,
        trust_mode=settings.neo4j_trust,
        ca_cert_path=settings.neo4j_ca_cert,
    )
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        **driver_kwargs,
    )
    rows: list[dict] = []
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(
            cypher,
            seed_names=seed_names,
            min_doc_spread=min_doc_spread,
            min_claim_types=min_claim_types,
            exclude_periods=exclude_periods,
        )
        for record in result:
            rows.append(dict(record))
    driver.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank corpus entities that are NOT yet in seed_entities.csv.",
    )
    parser.add_argument("--output", default="seeding_candidates.csv")
    parser.add_argument(
        "--seeds",
        action="append",
        type=Path,
        help="Path to a seed_entities.csv (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument(
        "--min-docs",
        type=int,
        default=2,
        help="Minimum document_spread (default 2; lower to surface thinner candidates).",
    )
    parser.add_argument(
        "--min-types",
        type=int,
        default=2,
        help="Minimum distinct_claim_types (default 2).",
    )
    parser.add_argument(
        "--include-periods",
        action="store_true",
        help="Include Period (date-bucket) entities in the ranking.",
    )
    args = parser.parse_args()

    seed_paths = args.seeds or DEFAULT_SEED_CSVS
    if args.no_filter:
        seed_names: list[str] = []
        print("Seed-name filter disabled (--no-filter).")
    else:
        seed_names = load_seed_names(seed_paths)
        print(
            f"Loaded {len(seed_names)} seed names/aliases from "
            f"{len(seed_paths)} CSV(s): "
            + ", ".join(str(p.relative_to(_REPO_ROOT)) for p in seed_paths)
        )

    load_dotenv(_REPO_ROOT / ".env")
    settings = Settings.from_env()
    if not settings.neo4j_uri:
        sys.exit("NEO4J_URI not set. Export Neo4j connection env vars first.")

    rows = run_query(
        settings,
        seed_names,
        args.limit,
        min_doc_spread=args.min_docs,
        min_claim_types=args.min_types,
        exclude_periods=not args.include_periods,
    )
    if not rows:
        print("No results.")
        return

    out_path = Path(args.output)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} candidates to {out_path}")


if __name__ == "__main__":
    main()
