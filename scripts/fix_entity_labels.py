"""One-time migration: add secondary labels to existing Neo4j entity nodes.

The Neo4jGraphWriter previously created separate :Entity and :Place nodes
instead of adding those labels to existing domain-typed nodes.  This script
adds the missing labels and removes the resulting orphan nodes.

Usage:
    python -m scripts.fix_entity_labels \
        [--neo4j-uri bolt://localhost:7687] \
        [--neo4j-user neo4j] \
        [--neo4j-password <password>] \
        [--neo4j-database neo4j] \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys

from gemynd.shared.env import load_dotenv

DOMAIN_LABELS = [
    "Refuge", "Place", "Person", "Organization",
    "Species", "Activity", "Period", "Habitat", "SurveyMethod",
]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fix entity node labels in Neo4j")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--dry-run", action="store_true", help="Print queries without executing")
    args = parser.parse_args()

    if not args.dry_run:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            print("neo4j package is required. Install with: pip install neo4j", file=sys.stderr)
            sys.exit(1)
        driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
        driver.verify_connectivity()

    def run(query: str, desc: str) -> None:
        if args.dry_run:
            print(f"[dry-run] {query}")
        else:
            with driver.session(database=args.neo4j_database) as session:
                result = session.run(query).single()
                count = result[0] if result else 0
                print(f"{desc} — {count}")

    # Step 1: Delete orphan nodes FIRST so uniqueness constraints don't block
    # label additions in steps 2-3.
    run(
        "MATCH (n:Entity) WHERE NOT (n)--() DELETE n RETURN count(n) AS deleted",
        "Deleted orphan :Entity-only nodes",
    )
    run(
        "MATCH (n:Place) WHERE NOT (n)--() AND NOT n:Refuge DELETE n RETURN count(n) AS deleted",
        "Deleted orphan :Place-only nodes",
    )

    # Step 2: Add :Entity label to all domain-typed nodes.
    for label in DOMAIN_LABELS:
        run(
            f"MATCH (n:{label}) WHERE NOT n:Entity SET n:Entity RETURN count(n) AS affected",
            f"Added :Entity to :{label} nodes",
        )

    # Step 3: Add :Place label to :Refuge nodes.
    run(
        "MATCH (n:Refuge) WHERE NOT n:Place SET n:Place RETURN count(n) AS affected",
        "Added :Place to :Refuge nodes",
    )

    if not args.dry_run:
        driver.close()

    print("Done.")


if __name__ == "__main__":
    main()
