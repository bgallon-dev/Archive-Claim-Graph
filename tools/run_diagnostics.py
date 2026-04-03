"""
run_diagnostics.py — Run entity/mention diagnostic queries against Neo4j.

Usage:
    python tools/run_diagnostics.py [--output FILE]
"""
from __future__ import annotations

import argparse
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

DIAGNOSTICS = [
    (
        "1. Entity counts by type",
        """\
MATCH (e:Entity)
WITH [l IN labels(e) WHERE l <> 'Entity'] AS types
UNWIND types AS type
RETURN type, count(*) AS count
ORDER BY count DESC
""",
    ),
    (
        "2. Species list — verify merges",
        """\
MATCH (e:Entity:Species)
RETURN e.name AS name, COUNT {(m)-[:REFERS_TO]->(e)} AS refers_count
ORDER BY e.name
""",
    ),
    (
        "3. Person list — verify Whitmore/West Tritt gone",
        """\
MATCH (e:Entity:Person)
RETURN e.name AS name, COUNT {(m)-[:REFERS_TO]->(e)} AS refers_count
ORDER BY refers_count DESC
""",
    ),
    (
        "4. Place list — verify Whitmore/West Tritt appear here",
        """\
MATCH (e:Entity:Place)
RETURN e.name AS name, e.place_type AS place_type,
       COUNT {(m)-[:REFERS_TO]->(e)} AS refers_count
ORDER BY refers_count DESC
""",
    ),
    (
        "5. Organization list — verify State Game Department gone",
        """\
MATCH (e:Entity:Organization)
RETURN e.name AS name, COUNT {(m)-[:REFERS_TO]->(e)} AS refers_count
ORDER BY refers_count DESC
""",
    ),
    (
        "6. EWC succession chain",
        """\
MATCH path = (e:Entity:Organization)-[:SUCCEEDED_BY*]->(successor)
WHERE e.name = 'Eastern Washington College'
RETURN [n IN nodes(path) | n.name] AS chain,
       [r IN relationships(path) | r.year] AS years
""",
    ),
    (
        "7. Total graph size — nodes",
        """\
MATCH (n) RETURN 'nodes' AS type, count(n) AS total
""",
    ),
    (
        "8. Total graph size — relationships",
        """\
MATCH ()-[r]->() RETURN 'rels' AS type, count(r) AS total
""",
    ),
]


def run_all(settings: Settings) -> list[tuple[str, list[dict]]]:
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

    results: list[tuple[str, list[dict]]] = []
    with driver.session(database=settings.neo4j_database) as session:
        for title, query in DIAGNOSTICS:
            records = [dict(r) for r in session.run(query)]
            results.append((title, records))
    driver.close()
    return results


def format_results(results: list[tuple[str, list[dict]]]) -> str:
    lines: list[str] = []
    for title, records in results:
        lines.append(f"\n{'='*70}")
        lines.append(f"  {title}")
        lines.append(f"{'='*70}")
        if not records:
            lines.append("  (no results)")
            continue
        # Print column headers from first record
        keys = list(records[0].keys())
        header = " | ".join(f"{k:>20s}" for k in keys)
        lines.append(header)
        lines.append("-" * len(header))
        for rec in records:
            row = " | ".join(f"{str(rec[k]):>20s}" for k in keys)
            lines.append(row)
        lines.append(f"  ({len(records)} rows)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run entity/mention diagnostics")
    parser.add_argument("--output", default=None, help="Write results to file (default: stdout only)")
    args = parser.parse_args()

    load_dotenv(_REPO_ROOT / ".env")
    settings = Settings.from_env()
    if not settings.neo4j_uri:
        sys.exit("NEO4J_URI not set. Export Neo4j connection env vars first.")

    results = run_all(settings)
    output = format_results(results)
    print(output)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output, encoding="utf-8")
        print(f"\nResults also written to {out_path}")


if __name__ == "__main__":
    main()
