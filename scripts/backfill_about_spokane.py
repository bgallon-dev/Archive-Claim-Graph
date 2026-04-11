"""One-shot backfill: add ABOUT_PLACE edges from every newspaper Document
to the canonical Spokane Place entity, then verify coverage.

The retrieval layer's anchor-based temporal query strategy expects every
document in a corpus to be linked to a single anchor entity. Turnbull gets
ABOUT_REFUGE via the Turnbull Refuge; the newspaper corpus is blanket-linked
to Spokane via ABOUT_PLACE. The ingest pipeline creates this edge for new
documents; this script is idempotent (MERGE) so it is safe to re-run.
"""
import os
from pathlib import Path

_here = Path(__file__).resolve().parent
for candidate in (_here / ".env", _here.parent / ".env", _here.parent.parent / ".env"):
    if candidate.exists():
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        break

from gemynd.retrieval.executor import Neo4jQueryExecutor

executor = Neo4jQueryExecutor(
    uri=os.environ["NEO4J_URI"],
    user=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "neo4j"),
    trust_mode=os.environ.get("NEO4J_TRUST", "system"),
)

RESOLVE_SPOKANE = """
MATCH (p:Place)
WHERE toLower(p.name) = 'spokane'
   OR toLower(p.normalized_form) = 'spokane'
RETURN p.entity_id AS entity_id
LIMIT 1
"""

BACKFILL = """
MATCH (d:Document {institution_id: 'spokane_newspaper'})
MATCH (p:Place {entity_id: $spokane_id})
WHERE NOT (d)-[:ABOUT_PLACE]->(p)
MERGE (d)-[:ABOUT_PLACE]->(p)
RETURN count(*) AS edges_created
"""

VERIFY = """
MATCH (d:Document {institution_id: 'spokane_newspaper'})-[:ABOUT_PLACE]->(p:Place {entity_id: $spokane_id})
RETURN count(d) AS linked_documents
"""

rows = executor.run(RESOLVE_SPOKANE)
if not rows:
    print("ERROR: no Place entity named 'Spokane' found in graph. Ingest the newspaper seed entities first.")
    executor.close()
    raise SystemExit(1)
spokane_id = rows[0]["entity_id"]
print("spokane entity_id:", spokane_id)

rows = executor.run(BACKFILL, {"spokane_id": spokane_id})
print("edges_created:", rows[0]["edges_created"] if rows else "no result")

rows = executor.run(VERIFY, {"spokane_id": spokane_id})
print("linked_documents:", rows[0]["linked_documents"] if rows else "no result")

executor.close()
