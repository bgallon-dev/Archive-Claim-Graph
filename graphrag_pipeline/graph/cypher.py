from __future__ import annotations

from ..claim_contract import CLAIM_ENTITY_RELATION_PRECEDENCE

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
    "Observation": "observation_id",
    "Year": "year_id",
    "Refuge": "entity_id",
    "Place": "entity_id",
    "Person": "entity_id",
    "Organization": "entity_id",
    "Species": "entity_id",
    "Activity": "entity_id",
    "Period": "entity_id",
    "Habitat": "entity_id",
    "SurveyMethod": "entity_id",
    "Event": "event_id",
}

INDEX_STATEMENTS: list[str] = [
    "CREATE INDEX document_report_year IF NOT EXISTS FOR (n:Document) ON (n.report_year)",
    "CREATE INDEX claim_type IF NOT EXISTS FOR (n:Claim) ON (n.claim_type)",
    "CREATE INDEX measurement_name IF NOT EXISTS FOR (n:Measurement) ON (n.name)",
    "CREATE INDEX period_dates IF NOT EXISTS FOR (n:Period) ON (n.start_date, n.end_date)",
    "CREATE INDEX mention_ocr_suspect IF NOT EXISTS FOR (n:Mention) ON (n.ocr_suspect)",
    "CREATE INDEX observation_type IF NOT EXISTS FOR (n:Observation) ON (n.observation_type)",
    "CREATE INDEX year_value IF NOT EXISTS FOR (n:Year) ON (n.year)",
    # observation_species and observation_year dropped: species_id/year_id are no longer stored
    # on Observation nodes; use OF_SPECIES / IN_YEAR edges instead.
    "CREATE INDEX observation_year_int IF NOT EXISTS FOR (n:Observation) ON (n.year)",
    "CREATE INDEX period_type IF NOT EXISTS FOR (n:Period) ON (n.period_type)",
    "CREATE INDEX measurement_date IF NOT EXISTS FOR (n:Measurement) ON (n.measurement_date)",
    "CREATE INDEX run_config IF NOT EXISTS FOR (n:ExtractionRun) ON (n.config_fingerprint)",
    "CREATE INDEX event_type IF NOT EXISTS FOR (n:Event) ON (n.event_type)",
    # event_year dropped: year_id is no longer stored on Event nodes; use IN_YEAR edge instead.
    "CREATE INDEX event_year_int IF NOT EXISTS FOR (n:Event) ON (n.year)",
    # Fulltext index for retrieval-layer keyword fallback search on claim text.
    "CREATE FULLTEXT INDEX claim_normalized_sentence IF NOT EXISTS FOR (n:Claim) ON EACH [n.normalized_sentence]",
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
CLAIM_ENTITY_RELATION_CYPHER = ", ".join(
    f"'{relation}'" for relation in CLAIM_ENTITY_RELATION_PRECEDENCE
)


LATEST_VIEW_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (p:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: lr.run_id})
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(om:Measurement)
OPTIONAL MATCH (c)-[:HAS_MEASUREMENT]->(dm:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (c)-[cer]->(e:Entity)
WHERE type(cer) IN [""" + CLAIM_ENTITY_RELATION_CYPHER + """]
WITH d, lr, c, p, obs,
     collect(DISTINCT om) + collect(DISTINCT dm) AS measurements,
     collect(DISTINCT e) AS entities,
     collect(DISTINCT CASE WHEN e IS NULL THEN NULL ELSE {name: e.normalized_form, relation_type: type(cer)} END) AS entity_links,
     collect(DISTINCT sp) AS species,
     collect(DISTINCT y) AS years
RETURN d.doc_id AS doc_id, lr.run_id AS run_id, c.claim_id AS claim_id, c.claim_type AS claim_type,
       p.paragraph_id AS paragraph_id, obs.observation_id AS observation_id, obs.observation_type AS observation_type,
       measurements, entities, entity_links, species, years
"""


AUDIT_VIEW_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
MATCH (p:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: r.run_id})
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
RETURN d.doc_id AS doc_id, r.run_id AS run_id, r.run_timestamp AS run_timestamp,
       c.claim_id AS claim_id, c.claim_type AS claim_type, c.extraction_confidence AS extraction_confidence,
       obs.observation_id AS observation_id, obs.observation_type AS observation_type
ORDER BY d.doc_id, r.run_timestamp, c.claim_id
"""

# ---------------------------------------------------------------------------
# Retrieval-layer query templates
# All queries anchor to the latest ExtractionRun per Document to guarantee
# freshness after re-extraction runs.
# ---------------------------------------------------------------------------

PROVENANCE_CHAIN_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (sec:Section)-[:HAS_PARAGRAPH]->(para:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: lr.run_id})
WHERE c.claim_id = $claim_id
MATCH (pg:Page)-[:HAS_SECTION]->(sec)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(om:Measurement)
OPTIONAL MATCH (c)-[:HAS_MEASUREMENT]->(dm:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
RETURN d, pg, sec, para, c, obs, sp, y,
       collect(DISTINCT om) + collect(DISTINCT dm) AS measurements
"""

ENTITY_ANCHORED_CLAIMS_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[]->(e:Entity {entity_id: $entity_id})
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
RETURN d, pg, sec, para, c, obs, sp, y, measurements
ORDER BY c.extraction_confidence DESC
LIMIT $limit
"""

FULLTEXT_CLAIMS_QUERY = """
CALL db.index.fulltext.queryNodes('claim_normalized_sentence', $search_text)
YIELD node AS c, score
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d:Document)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
RETURN d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements, score
ORDER BY score DESC
LIMIT $limit
"""

SPECIES_TREND_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[:SUPPORTS]->(obs:Observation)-[:OF_SPECIES]->(sp:Species {entity_id: $species_id})
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
RETURN sp.name AS species, y.year AS year,
       count(obs) AS observation_count,
       avg(c.extraction_confidence) AS avg_confidence,
       collect({name: m.name, value: m.numeric_value, unit: m.unit, approximate: m.approximate}) AS measurements
ORDER BY y.year
"""

CORPUS_STATS_QUERY = """
CALL { MATCH (p:Paragraph) RETURN count(p) AS n } WITH n AS total_paragraphs
CALL { MATCH (d:Document)  RETURN count(d) AS n } WITH total_paragraphs, n AS total_documents
RETURN total_paragraphs, total_documents
"""

HABITAT_CONDITION_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[:SUPPORTS]->(obs:Observation)-[:IN_HABITAT]->(h:Habitat {entity_id: $habitat_id})
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
RETURN h.name AS habitat, y.year AS year, sp.name AS species,
       count(obs) AS observation_count,
       avg(c.extraction_confidence) AS avg_confidence,
       collect({name: m.name, value: m.numeric_value, unit: m.unit}) AS measurements
ORDER BY y.year, sp.name
"""
