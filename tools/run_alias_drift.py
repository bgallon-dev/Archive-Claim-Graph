"""
run_alias_drift.py — Surface entities that look like un-merged variants of
seeded entities (e.g. "NORTHERN PACIFIC RAILROAD CO" vs seeded
"Northern Pacific Railway").

Loads names + aliases from seed_entities.csv files and passes them to
tools/alias_drift.cypher. The cypher returns entities whose name contains a
multi-word seed name as a substring but isn't an exact match — i.e.,
candidates for resolver-layer merging, not new-seed addition.

Usage:
    python tools/run_alias_drift.py [--output FILE] [--seeds CSV] [--limit N]

Requires NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (loaded from .env).
"""
from __future__ import annotations

import argparse
import csv
import json
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

# Reuse the seed-name loader from the seeding-candidates runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seeding_candidates import load_seed_names, DEFAULT_SEED_CSVS  # noqa: E402

CYPHER_PATH = Path(__file__).resolve().parent / "alias_drift.cypher"


def run_query(
    settings: Settings, seed_names: list[str], limit: int | None
) -> list[dict]:
    cypher = CYPHER_PATH.read_text(encoding="utf-8")
    if limit is not None:
        cypher = cypher.replace("LIMIT 100", f"LIMIT {int(limit)}")

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
        result = session.run(cypher, seed_names=seed_names)
        for record in result:
            rows.append(dict(record))
    driver.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find entities that look like un-merged variants of seeded ones.",
    )
    parser.add_argument("--output", default="alias_drift.csv")
    parser.add_argument(
        "--seeds",
        action="append",
        type=Path,
        help="Path to a seed_entities.csv (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    seed_paths = args.seeds or DEFAULT_SEED_CSVS
    seed_names = load_seed_names(seed_paths)
    print(
        f"Loaded {len(seed_names)} seed names/aliases from {len(seed_paths)} CSV(s)"
    )

    load_dotenv(_REPO_ROOT / ".env")
    settings = Settings.from_env()
    if not settings.neo4j_uri:
        sys.exit("NEO4J_URI not set. Export Neo4j connection env vars first.")

    rows = run_query(settings, seed_names, args.limit)
    if not rows:
        print("No alias-drift candidates found.")
        return

    out_path = Path(args.output)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # neo4j returns lists for `likely_canonical_seed` — JSON-encode
            # so the CSV cell is round-trippable.
            row_out = dict(row)
            if isinstance(row_out.get("likely_canonical_seed"), list):
                row_out["likely_canonical_seed"] = json.dumps(
                    row_out["likely_canonical_seed"]
                )
            writer.writerow(row_out)
    print(f"Wrote {len(rows)} candidates to {out_path}")


if __name__ == "__main__":
    main()
