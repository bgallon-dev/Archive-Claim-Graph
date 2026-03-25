"""One-shot backfill: add ABOUT_REFUGE edges from all Tbl/Turnbull documents
to the canonical Turnbull Refuge entity, then verify coverage."""
import os
from pathlib import Path

# Load .env if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from graphrag_pipeline.retrieval.executor import Neo4jQueryExecutor

executor = Neo4jQueryExecutor(
    uri=os.environ["NEO4J_URI"],
    user=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "neo4j"),
    trust_mode=os.environ.get("NEO4J_TRUST", "system"),
)

BACKFILL = """
MATCH (d:Document), (r:Refuge {entity_id: 'refuge_83a540d8bc35b099'})
WHERE (toLower(d.title) STARTS WITH 'tbl'
    OR toLower(d.title) CONTAINS 'turnbull')
  AND NOT (d)-[:ABOUT_REFUGE]->(r)
MERGE (d)-[:ABOUT_REFUGE]->(r)
RETURN count(*) AS edges_created
"""

VERIFY = """
MATCH (d:Document)-[:ABOUT_REFUGE]->(r:Refuge {entity_id: 'refuge_83a540d8bc35b099'})
RETURN count(d) AS linked_documents
"""

rows = executor.run(BACKFILL)
print("edges_created:", rows[0]["edges_created"] if rows else "no result")

rows = executor.run(VERIFY)
print("linked_documents:", rows[0]["linked_documents"] if rows else "no result")

executor.close()
