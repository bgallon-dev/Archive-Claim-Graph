"""
dump_entity_names.py — Export paragraphs with their resolved mentions from Neo4j.

Outputs a CSV with columns: paragraph_id, text, mentions
where mentions is a JSON array of objects with entity_type, confidence,
start_offset, end_offset, and surface_form.

Usage:
    python tools/dump_entity_names.py [options]

Options:
    --output FILE   Path to output file (default: entity_mentions.csv)

Requires NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables
(loaded automatically from .env).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running from the repo root or tools/ directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from gemynd.shared.env import load_dotenv
from gemynd.shared.settings import Settings
from gemynd.ingest.graph.writer import _build_driver_kwargs

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("neo4j package is not installed. Install with: pip install neo4j")

QUERY = """\
MATCH (p:Paragraph)-[:CONTAINS_MENTION]->(m:Mention)
OPTIONAL MATCH (m)-[:REFERS_TO|POSSIBLY_REFERS_TO]->(e:Entity)
WITH p.paragraph_id AS paragraph_id,
     p.clean_text    AS text,
     m.surface_form  AS surface_form,
     m.start_offset  AS start_offset,
     m.end_offset    AS end_offset,
     m.detection_confidence AS confidence,
     labels(e)       AS entity_labels
ORDER BY paragraph_id, start_offset
WITH paragraph_id, text,
     collect({
       entity_type: COALESCE(
         HEAD([l IN entity_labels WHERE l <> 'Entity' AND l <> 'Place']),
         HEAD([l IN entity_labels WHERE l <> 'Entity']),
         'unknown'
       ),
       confidence: confidence,
       end_offset: end_offset,
       start_offset: start_offset,
       surface_form: surface_form
     }) AS mentions
RETURN paragraph_id, text, mentions
ORDER BY paragraph_id
"""


def fetch_paragraph_mentions(settings: Settings) -> list[dict]:
    """Return paragraph rows with aggregated mention data."""
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
        result = session.run(QUERY)
        for record in result:
            rows.append({
                "paragraph_id": record["paragraph_id"],
                "text": record["text"],
                "mentions": record["mentions"],
            })
    driver.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump paragraphs with entity mentions from Neo4j",
    )
    parser.add_argument("--output", default="entity_mentions.csv", help="Output file path")
    args = parser.parse_args()

    load_dotenv(_REPO_ROOT / ".env")
    settings = Settings.from_env()
    if not settings.neo4j_uri:
        sys.exit("NEO4J_URI not set. Export Neo4j connection env vars first.")

    rows = fetch_paragraph_mentions(settings)

    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph_id", "text", "mentions"])
        for row in rows:
            writer.writerow([
                row["paragraph_id"],
                row["text"],
                json.dumps(row["mentions"], indent=2),
            ])

    print(f"Wrote {len(rows)} paragraphs to {out_path}")


if __name__ == "__main__":
    main()
