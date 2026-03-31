from __future__ import annotations

from collections.abc import Iterable

from graphrag_pipeline.core.claim_contract import CLAIM_ENTITY_RELATION_PRECEDENCE

# Structural constraints that every domain needs.
_STRUCTURAL_CONSTRAINTS: dict[str, str] = {
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
    "Event": "event_id",
}

# Default entity labels for the Turnbull domain.
_DEFAULT_ENTITY_LABELS: frozenset[str] = frozenset({
    "Refuge", "Place", "Person", "Organization", "Species",
    "Activity", "Period", "Habitat", "SurveyMethod",
})


def build_id_constraints(entity_labels: Iterable[str] | None = None) -> dict[str, str]:
    """Return combined structural + domain entity ID constraints."""
    labels = entity_labels if entity_labels is not None else _DEFAULT_ENTITY_LABELS
    result = dict(_STRUCTURAL_CONSTRAINTS)
    for label in labels:
        result[label] = "entity_id"
    return result


# Module-level default for backward compatibility.
ID_CONSTRAINTS: dict[str, str] = build_id_constraints()

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
    # Access control and institution isolation indexes.
    "CREATE INDEX document_access_level IF NOT EXISTS FOR (n:Document) ON (n.access_level)",
    "CREATE INDEX document_institution IF NOT EXISTS FOR (n:Document) ON (n.institution_id)",
    # Soft-delete index for fast IS NULL filter.
    "CREATE INDEX document_deleted_at IF NOT EXISTS FOR (n:Document) ON (n.deleted_at)",
    # Quarantine index for fast quarantine_status filter on claims and documents.
    "CREATE INDEX claim_quarantine IF NOT EXISTS FOR (n:Claim) ON (n.quarantine_status)",
    "CREATE INDEX document_quarantine IF NOT EXISTS FOR (n:Document) ON (n.quarantine_status)",
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

_DEFAULT_RELATION_CYPHER = ", ".join(
    f"'{relation}'" for relation in CLAIM_ENTITY_RELATION_PRECEDENCE
)
# Backward-compatible alias.
CLAIM_ENTITY_RELATION_CYPHER = _DEFAULT_RELATION_CYPHER


def build_latest_view_query(relation_cypher: str | None = None) -> str:
    """Build the LATEST_VIEW_QUERY with parameterized relation types."""
    rel_cypher = relation_cypher if relation_cypher is not None else _DEFAULT_RELATION_CYPHER
    return """
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
WHERE type(cer) IN [""" + rel_cypher + """]
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


LATEST_VIEW_QUERY = build_latest_view_query()


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
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (sec:Section)-[:HAS_PARAGRAPH]->(para:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: lr.run_id})
WHERE c.claim_id = $claim_id
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
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
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[rel]->(e:Entity {entity_id: $entity_id})
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
WITH d, pg, sec, para, c, obs, sp, y, type(rel) AS traversal_rel_type, collect(DISTINCT m) AS measurements
WHERE ($year_min IS NULL OR y IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y IS NULL OR y.year <= $year_max)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN d, pg, sec, para, c, obs, sp, y, measurements, traversal_rel_type
ORDER BY c.extraction_confidence DESC
LIMIT $limit
"""

FULLTEXT_CLAIMS_QUERY = """
CALL db.index.fulltext.queryNodes('claim_normalized_sentence', $search_text)
YIELD node AS c, score
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d:Document)
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
RETURN d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements, score
ORDER BY score DESC
LIMIT $limit
"""

TEMPORAL_CLAIMS_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[:SUPPORTS]->(obs:Observation)-[:IN_YEAR]->(y:Year)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
  AND ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
RETURN d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements,
       'IN_YEAR' AS traversal_rel_type
ORDER BY y.year, c.extraction_confidence DESC
LIMIT $limit
"""

TEMPORAL_CLAIMS_QUERY_WITH_REFUGE = """
MATCH (d:Document)-[:ABOUT_REFUGE]->(ref:Refuge {entity_id: $refuge_id})
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
MATCH (d)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})
WHERE ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
MATCH (c)-[:SUPPORTS]->(obs:Observation)-[:IN_YEAR]->(y:Year)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)
           <-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
RETURN d, pg, sec, para, c, obs, sp, y,
       collect(DISTINCT m) AS measurements,
       'ABOUT_REFUGE' AS traversal_rel_type
ORDER BY y.year, c.extraction_confidence DESC
LIMIT $limit
"""
# traversal_rel_type is 'ABOUT_REFUGE' (not 'IN_YEAR') because the
# distinguishing path for this template is the document→refuge anchor.
# The IN_YEAR traversal is implicit to all temporal queries; logging
# ABOUT_REFUGE correctly identifies which retrieval strategy was used.

MULTI_ENTITY_CLAIMS_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[rel]->(e:Entity)
WHERE e.entity_id IN $entity_ids
  AND ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y IS NULL OR y.year <= $year_max)
WITH d, pg, sec, para, c, obs, sp, y, type(rel) AS traversal_rel_type,
     collect(DISTINCT m) AS measurements,
     collect(DISTINCT e.entity_id) AS matched_entity_ids
RETURN d, pg, sec, para, c, obs, sp, y, measurements, traversal_rel_type,
       matched_entity_ids
ORDER BY size(matched_entity_ids) DESC, c.extraction_confidence DESC
LIMIT $limit
"""

CLAIM_TYPE_SCOPED_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})
WHERE c.claim_type IN $claim_types
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y IS NULL OR y.year <= $year_max)
  AND ($entity_ids IS NULL OR EXISTS {
    MATCH (c)-[]->(e:Entity) WHERE e.entity_id IN $entity_ids
  })
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       c.claim_type AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
LIMIT $limit
"""

SPECIES_TREND_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[:SUPPORTS]->(obs:Observation)-[:OF_SPECIES]->(sp:Species {entity_id: $species_id})
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN sp.name AS species, y.year AS year,
       count(obs) AS observation_count,
       avg(c.extraction_confidence) AS avg_confidence,
       collect({name: m.name, value: m.numeric_value, unit: m.unit, approximate: m.approximate}) AS measurements
ORDER BY y.year
"""

CORPUS_STATS_QUERY = """
CALL () { MATCH (p:Paragraph) RETURN count(p) AS n } WITH n AS total_paragraphs
CALL () { MATCH (d:Document)  RETURN count(d) AS n } WITH total_paragraphs, n AS total_documents
RETURN total_paragraphs, total_documents
"""

HABITAT_CONDITION_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[:SUPPORTS]->(obs:Observation)-[:IN_HABITAT]->(h:Habitat {entity_id: $habitat_id})
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN h.name AS habitat, y.year AS year, sp.name AS species,
       count(obs) AS observation_count,
       avg(c.extraction_confidence) AS avg_confidence,
       collect({name: m.name, value: m.numeric_value, unit: m.unit}) AS measurements
ORDER BY y.year, sp.name
"""

# ---------------------------------------------------------------------------
# Admin document management queries
# ---------------------------------------------------------------------------

SOFT_DELETE_DOCUMENT_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
WHERE d.institution_id = $institution_id
SET d.deleted_at = $deleted_at, d.deleted_by = $deleted_by
RETURN d.doc_id AS doc_id, d.title AS title
"""

RESTORE_DOCUMENT_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
WHERE d.institution_id = $institution_id
SET d.deleted_at = null, d.deleted_by = null
RETURN d.doc_id AS doc_id, d.title AS title
"""

LIST_DOCUMENTS_QUERY = """
MATCH (d:Document)
WHERE d.institution_id = $institution_id
RETURN d.doc_id AS doc_id, d.title AS title, d.access_level AS access_level,
       d.deleted_at AS deleted_at, d.deleted_by AS deleted_by
ORDER BY d.title
"""

# ---------------------------------------------------------------------------
# Integrity verification query (used by CLI verify-integrity command)
# ---------------------------------------------------------------------------

INTEGRITY_CHECK_QUERY = """
MATCH (d:Document)
WHERE d.deleted_at IS NULL
  AND ($institution_id IS NULL OR d.institution_id = $institution_id)
  AND d.file_hash IS NOT NULL
  AND d.source_file IS NOT NULL
RETURN d.doc_id AS doc_id, d.title AS title, d.file_hash AS file_hash,
       d.source_file AS source_file, d.institution_id AS institution_id
ORDER BY d.institution_id, d.title
"""

# Pre-ingest duplicate detection: check whether any of the given file_hashes
# already exist in the graph (ignoring soft-deleted documents).
DUPLICATE_HASH_CHECK_QUERY = """
MATCH (d:Document)
WHERE d.file_hash IN $file_hashes
  AND d.deleted_at IS NULL
RETURN d.file_hash AS file_hash, d.doc_id AS doc_id
"""

# Graph-backed entity resolution: fetch all active Entity nodes so the
# DictionaryFuzzyResolver can use graph-resident entities as supplementary
# candidates alongside seed_entities.csv.
GRAPH_ENTITY_FETCH_QUERY = """
MATCH (e:Entity)
WHERE e.deleted_at IS NULL OR e.deleted_at = ''
RETURN e.entity_id           AS entity_id,
       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type,
       e.name                AS name,
       e.normalized_form     AS normalized_form
ORDER BY entity_type, name
"""

# ---------------------------------------------------------------------------
# Sensitivity quarantine operation queries
# ---------------------------------------------------------------------------

QUARANTINE_CLAIM_QUERY = """
MATCH (c:Claim {claim_id: $claim_id})
SET c.quarantine_status = 'quarantined',
    c.quarantine_reason = $reason,
    c.quarantine_timestamp = $quarantine_timestamp
RETURN c.claim_id AS claim_id
"""

RESTORE_CLAIM_QUERY = """
MATCH (c:Claim {claim_id: $claim_id})
SET c.quarantine_status = 'reviewed_cleared',
    c.quarantine_reason = null,
    c.quarantine_timestamp = null
RETURN c.claim_id AS claim_id
"""

RESTRICT_CLAIM_PERMANENTLY_QUERY = """
MATCH (c:Claim {claim_id: $claim_id})
SET c.quarantine_status = 'reviewed_restricted'
RETURN c.claim_id AS claim_id
"""

QUARANTINE_DOCUMENT_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
WHERE d.institution_id = $institution_id
SET d.quarantine_status = 'quarantined',
    d.quarantine_reason = $reason,
    d.quarantine_timestamp = $quarantine_timestamp
RETURN d.doc_id AS doc_id
"""

COUNT_QUARANTINED_CLAIMS_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND c.quarantine_status = 'quarantined'
RETURN count(c) AS quarantined_count
"""

# ---------------------------------------------------------------------------
# Background sensitivity scan batch query
# Fetches active claims in pages for the SensitivityMonitor.run_full_scan().
# ---------------------------------------------------------------------------

SENSITIVITY_SCAN_BATCH_QUERY = """
MATCH (c:Claim)
WHERE (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
  AND ($institution_id IS NULL OR EXISTS {
      MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c)
      WHERE d.institution_id = $institution_id
  })
RETURN c.claim_id AS claim_id, c.source_sentence AS source_sentence,
       c.claim_type AS claim_type
SKIP $offset LIMIT $batch_size
"""

# ---------------------------------------------------------------------------
# Collection statistics queries
# Used by the GET /stats endpoint in the query API web app.
# All document-scoped queries filter by institution_id, deleted_at, and
# access_level so archivists only see their own institution's permitted data.
# ---------------------------------------------------------------------------

STATS_DOC_OVERVIEW_QUERY = """
MATCH (d:Document)
WHERE d.institution_id = $institution_id
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
RETURN count(d) AS total_docs,
       min(d.report_year) AS earliest_year,
       max(d.report_year) AS latest_year,
       sum(d.page_count) AS total_pages,
       count(CASE WHEN d.donor_restricted THEN 1 ELSE null END) AS donor_restricted_count
"""

STATS_DOC_TYPE_QUERY = """
MATCH (d:Document)
WHERE d.institution_id = $institution_id
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
RETURN d.doc_type AS doc_type, count(d) AS count
ORDER BY count DESC
"""

STATS_CLAIM_TYPE_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE d.institution_id = $institution_id
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN c.claim_type AS claim_type,
       count(c) AS count,
       avg(c.extraction_confidence) AS avg_confidence
ORDER BY count DESC
"""

STATS_ENTITY_TYPE_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)-[]->(e:Entity)
WHERE d.institution_id = $institution_id
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
RETURN [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type, count(DISTINCT e) AS count
ORDER BY count DESC
"""

STATS_TEMPORAL_COVERAGE_QUERY = """
MATCH (d:Document)
WHERE d.institution_id = $institution_id
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND d.report_year IS NOT NULL
RETURN d.report_year AS year, count(d) AS doc_count
ORDER BY year
"""

STATS_CONFIDENCE_DISTRIBUTION_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE d.institution_id = $institution_id
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN
  count(c) AS total_claims,
  avg(c.extraction_confidence) AS avg_confidence,
  count(CASE WHEN c.extraction_confidence >= 0.85 THEN 1 ELSE null END) AS high_count,
  count(CASE WHEN c.extraction_confidence >= 0.70 AND c.extraction_confidence < 0.85 THEN 1 ELSE null END) AS medium_count,
  count(CASE WHEN c.extraction_confidence < 0.70 THEN 1 ELSE null END) AS low_count,
  count(CASE WHEN c.certainty = 'uncertain' THEN 1 ELSE null END) AS uncertain_epistemic_count
"""

# ---------------------------------------------------------------------------
# Gap Detection Queries
# ---------------------------------------------------------------------------

GAP_TEMPORAL_DENSITY_QUERY = """
MATCH (d:Document)
WHERE d.institution_id = $institution_id
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND d.report_year IS NOT NULL
OPTIONAL MATCH (d)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN d.report_year AS year, count(DISTINCT d) AS doc_count, count(c) AS claim_count
ORDER BY year
"""

def build_gap_entity_depth_query(
    relation_types: Iterable[str] | None = None,
    entity_labels: Iterable[str] | None = None,
) -> str:
    """Build the GAP_ENTITY_DEPTH_QUERY with parameterized relation types and entity labels."""
    if relation_types is None:
        relation_types = ["SUBJECT_OF_CLAIM", "SPECIES_FOCUS", "HABITAT_FOCUS",
                          "LOCATION_FOCUS", "MANAGEMENT_TARGET"]
    if entity_labels is None:
        entity_labels = ["Species", "Person", "Organization", "Habitat", "Activity"]
    rel_list = ", ".join(f"'{r}'" for r in relation_types)
    label_filter = " OR ".join(f"e:{label}" for label in entity_labels)
    return f"""
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)-[:PRODUCED]->(c:Claim)-[rel]->(e:Entity)
WHERE d.institution_id = $institution_id AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
  AND type(rel) IN [{rel_list}]
  AND ({label_filter})
WITH e, count(c) AS primary_claim_count,
     [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type
WHERE primary_claim_count <= $thin_threshold
RETURN e.entity_id AS entity_id, e.name AS name, entity_type,
       primary_claim_count
ORDER BY primary_claim_count ASC, entity_type
LIMIT $limit
"""


GAP_ENTITY_DEPTH_QUERY = build_gap_entity_depth_query()

GAP_GEOGRAPHIC_COVERAGE_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)-[:PRODUCED]->(c:Claim)-[rel]->(e:Entity)
WHERE d.institution_id = $institution_id AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
  AND type(rel) IN ['OCCURRED_AT', 'LOCATION_FOCUS']
  AND (e:Place OR e:Refuge)
WITH e, count(c) AS geo_claim_count,
     [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type
WHERE geo_claim_count <= $thin_threshold
RETURN e.entity_id AS entity_id, e.name AS name, entity_type,
       geo_claim_count AS location_specific_claims
ORDER BY geo_claim_count ASC
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Relationship Mapping Queries
# ---------------------------------------------------------------------------

ENTITY_SEARCH_QUERY = """
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($query)
   OR toLower(e.normalized_form) CONTAINS toLower($query)
RETURN e.entity_id AS entity_id, e.name AS name,
       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type
ORDER BY e.name
LIMIT $limit
"""

ENTITY_DETAIL_QUERY = """
MATCH (e:Entity {entity_id: $entity_id})
OPTIONAL MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)-[rel]->(e)
WHERE d.institution_id = $institution_id AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN e.entity_id AS entity_id, e.name AS name,
       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type,
       count(DISTINCT c) AS claim_count
"""

ENTITY_NEIGHBORHOOD_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)-[r1]->(e1:Entity {entity_id: $entity_id})
WHERE d.institution_id = $institution_id AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
WITH c, e1
MATCH (c)-[r2]->(e2:Entity)
WHERE e2 <> e1
WITH e1, e2, c,
     type(r2) AS rel_type,
     c.source_sentence AS sentence,
     c.claim_id AS claim_id
WITH e1, e2,
     collect(DISTINCT rel_type) AS relationship_types,
     collect(DISTINCT claim_id)[..5] AS sample_claim_ids,
     collect(DISTINCT sentence)[..3] AS sample_sentences,
     count(DISTINCT c) AS co_occurrence_count
ORDER BY co_occurrence_count DESC
LIMIT $limit
RETURN e2.entity_id AS entity_id, e2.name AS name,
       [x IN labels(e2) WHERE x <> 'Entity'][0] AS entity_type,
       co_occurrence_count, relationship_types, sample_claim_ids, sample_sentences
"""
