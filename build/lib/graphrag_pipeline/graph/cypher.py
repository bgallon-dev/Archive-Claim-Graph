from __future__ import annotations

ID_CONSTRAINTS: dict[str, str] = {
    "Document": "doc_id",
    "Page": "page_id",
    "Section": "section_id",
    "Paragraph": "paragraph_id",
    "Annotation": "annotation_id",
    "ExtractionRun": "run_id",
    "Claim": "claim_id",
    "Measurement": "measurement_id",
    "Mention": "mention_id",
    "Refuge": "entity_id",
    "Place": "entity_id",
    "Person": "entity_id",
    "Organization": "entity_id",
    "Species": "entity_id",
    "Activity": "entity_id",
    "Period": "entity_id",
}

INDEX_STATEMENTS: list[str] = [
    "CREATE INDEX document_report_year IF NOT EXISTS FOR (n:Document) ON (n.report_year)",
    "CREATE INDEX claim_type IF NOT EXISTS FOR (n:Claim) ON (n.claim_type)",
    "CREATE INDEX measurement_name IF NOT EXISTS FOR (n:Measurement) ON (n.name)",
    "CREATE INDEX period_dates IF NOT EXISTS FOR (n:Period) ON (n.start_date, n.end_date)",
    "CREATE INDEX mention_ocr_suspect IF NOT EXISTS FOR (n:Mention) ON (n.ocr_suspect)",
]


def build_constraint_statements() -> list[str]:
    statements: list[str] = []
    for label, property_name in ID_CONSTRAINTS.items():
        constraint_name = f"{label.lower()}_{property_name}"
        statements.append(
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
        )
    return statements


SCHEMA_STATEMENTS: list[str] = build_constraint_statements() + INDEX_STATEMENTS


LATEST_VIEW_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (p:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: lr.run_id})
OPTIONAL MATCH (c)-[:HAS_MEASUREMENT]->(m:Measurement)
OPTIONAL MATCH (c)-[:ABOUT]->(e:Entity)
RETURN d.doc_id AS doc_id, lr.run_id AS run_id, c.claim_id AS claim_id, c.claim_type AS claim_type,
       p.paragraph_id AS paragraph_id, collect(DISTINCT m) AS measurements, collect(DISTINCT e) AS entities
"""


AUDIT_VIEW_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
MATCH (p:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: r.run_id})
RETURN d.doc_id AS doc_id, r.run_id AS run_id, r.run_timestamp AS run_timestamp,
       c.claim_id AS claim_id, c.claim_type AS claim_type, c.extraction_confidence AS extraction_confidence
ORDER BY d.doc_id, r.run_timestamp, c.claim_id
"""
