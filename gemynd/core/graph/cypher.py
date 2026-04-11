from __future__ import annotations

import functools
import re
from collections.abc import Iterable

from gemynd.core.claim_contract import CLAIM_ENTITY_RELATION_PRECEDENCE

# Truly universal: every domain produces documents, paragraphs, claims, measurements.
_UNIVERSAL_CONSTRAINTS: dict[str, str] = {
    "Document": "doc_id",
    "Page": "page_id",
    "Section": "section_id",
    "Paragraph": "paragraph_id",
    "Annotation": "annotation_id",
    "ExtractionRun": "run_id",
    "Claim": "claim_id",
    "Measurement": "measurement_id",
    "Mention": "mention_id",
}

# Derivation-dependent: only created if the domain has a derivation registry
# producing Observation / Event nodes.
_DERIVATION_CONSTRAINTS: dict[str, str] = {
    "Observation": "observation_id",
    "Year": "year_id",
    "Event": "event_id",
}


def build_id_constraints(
    entity_labels: Iterable[str],
    *,
    include_derivation: bool = True,
) -> dict[str, str]:
    """Return combined structural + domain entity ID constraints.

    When ``include_derivation`` is False, Observation/Year/Event constraints
    are omitted — use this for domains whose derivation_registry is empty.
    """
    result = dict(_UNIVERSAL_CONSTRAINTS)
    if include_derivation:
        result.update(_DERIVATION_CONSTRAINTS)
    for label in entity_labels:
        result[label] = "entity_id"
    return result

_UNIVERSAL_INDEX_STATEMENTS: list[str] = [
    "CREATE INDEX document_report_year IF NOT EXISTS FOR (n:Document) ON (n.report_year)",
    "CREATE INDEX claim_type IF NOT EXISTS FOR (n:Claim) ON (n.claim_type)",
    "CREATE INDEX measurement_name IF NOT EXISTS FOR (n:Measurement) ON (n.name)",
    "CREATE INDEX mention_ocr_suspect IF NOT EXISTS FOR (n:Mention) ON (n.ocr_suspect)",
    "CREATE INDEX measurement_date IF NOT EXISTS FOR (n:Measurement) ON (n.measurement_date)",
    "CREATE INDEX run_config IF NOT EXISTS FOR (n:ExtractionRun) ON (n.config_fingerprint)",
    # Fulltext index for retrieval-layer keyword fallback search on claim text.
    "CREATE FULLTEXT INDEX claim_normalized_sentence IF NOT EXISTS FOR (n:Claim) ON EACH [n.normalized_sentence]",
    # Access control and institution isolation indexes.
    "CREATE INDEX document_access_level IF NOT EXISTS FOR (n:Document) ON (n.access_level)",
    "CREATE INDEX document_institution IF NOT EXISTS FOR (n:Document) ON (n.institution_id)",
    "CREATE INDEX document_deleted_at IF NOT EXISTS FOR (n:Document) ON (n.deleted_at)",
    "CREATE INDEX claim_quarantine IF NOT EXISTS FOR (n:Claim) ON (n.quarantine_status)",
    "CREATE INDEX document_quarantine IF NOT EXISTS FOR (n:Document) ON (n.quarantine_status)",
]

_DERIVATION_INDEX_STATEMENTS: list[str] = [
    "CREATE INDEX observation_type IF NOT EXISTS FOR (n:Observation) ON (n.observation_type)",
    "CREATE INDEX year_value IF NOT EXISTS FOR (n:Year) ON (n.year)",
    "CREATE INDEX observation_year_int IF NOT EXISTS FOR (n:Observation) ON (n.year)",
    "CREATE INDEX period_dates IF NOT EXISTS FOR (n:Period) ON (n.start_date, n.end_date)",
    "CREATE INDEX period_type IF NOT EXISTS FOR (n:Period) ON (n.period_type)",
    "CREATE INDEX event_type IF NOT EXISTS FOR (n:Event) ON (n.event_type)",
    "CREATE INDEX event_year_int IF NOT EXISTS FOR (n:Event) ON (n.year)",
]


def build_index_statements(*, include_derivation: bool = True) -> list[str]:
    """Return CREATE INDEX DDL for universal + optional derivation indexes."""
    if include_derivation:
        return list(_UNIVERSAL_INDEX_STATEMENTS) + list(_DERIVATION_INDEX_STATEMENTS)
    return list(_UNIVERSAL_INDEX_STATEMENTS)


# Back-compat module-level list used by existing callers and tests.
INDEX_STATEMENTS: list[str] = build_index_statements()


def build_constraint_statements(
    entity_labels: Iterable[str],
    *,
    include_derivation: bool = True,
) -> list[str]:
    """Return CREATE CONSTRAINT DDL for structural + domain entity labels."""
    statements: list[str] = []
    for label, property_name in build_id_constraints(
        entity_labels, include_derivation=include_derivation
    ).items():
        constraint_name = f"{label.lower()}_{property_name}"
        statements.append(
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
        )
    return statements

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
WHERE d.institution_id IN $institution_ids
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
WHERE d.institution_id IN $institution_ids
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
WHERE d.institution_id IN $institution_ids
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
WHERE d.institution_id IN $institution_ids
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

_CYPHER_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@functools.lru_cache(maxsize=32)
def build_temporal_with_anchor_query(anchor_label: str, anchor_relation: str) -> str:
    """Build a temporal-claims query anchored to a domain-specific entity.

    ``anchor_label`` is the Neo4j node label (e.g. ``Refuge``) and
    ``anchor_relation`` is the Document→anchor relation type
    (e.g. ``ABOUT_REFUGE``). Both must be valid Cypher identifiers since
    they are substituted directly into the query string.

    Results are memoized so that repeated calls with the same arguments
    return the same ``str`` object — this preserves identity-based dispatch
    in ``InMemoryQueryExecutor``.
    """
    if not _CYPHER_IDENT_RE.match(anchor_label):
        raise ValueError(f"invalid anchor_label for Cypher: {anchor_label!r}")
    if not _CYPHER_IDENT_RE.match(anchor_relation):
        raise ValueError(f"invalid anchor_relation for Cypher: {anchor_relation!r}")
    return f"""
MATCH (d:Document)-[:{anchor_relation}]->(ref:{anchor_label} {{entity_id: $anchor_id}})
WHERE d.institution_id IN $institution_ids
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
MATCH (d)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {{run_timestamp: latest_ts}})
MATCH (c:Claim {{run_id: lr.run_id}})
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
       '{anchor_relation}' AS traversal_rel_type
ORDER BY y.year, c.extraction_confidence DESC
LIMIT $limit
"""


@functools.lru_cache(maxsize=32)
def build_document_anchor_query(anchor_label: str, anchor_relation: str) -> str:
    """Build a document-level anchor claim query (no year/observation gate).

    Returns all claims from documents that have a ``(d)-[anchor_relation]->(:anchor_label)``
    edge to the supplied ``$anchor_id``. Unlike ``build_temporal_with_anchor_query``,
    observations and years are optional — claims without observation bindings still
    flow through. Used when a user's resolved entity is itself a corpus anchor
    (e.g. asking "What do you know about Spokane?" when the newspaper corpus is
    blanket-linked via ``ABOUT_PLACE`` to the Spokane Place node).
    """
    if not _CYPHER_IDENT_RE.match(anchor_label):
        raise ValueError(f"invalid anchor_label for Cypher: {anchor_label!r}")
    if not _CYPHER_IDENT_RE.match(anchor_relation):
        raise ValueError(f"invalid anchor_relation for Cypher: {anchor_relation!r}")
    return f"""
MATCH (d:Document)-[:{anchor_relation}]->(ref:{anchor_label} {{entity_id: $anchor_id}})
WHERE d.institution_id IN $institution_ids
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
MATCH (d)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {{run_timestamp: latest_ts}})
MATCH (c:Claim {{run_id: lr.run_id}})
WHERE ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)
           <-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
WHERE ($year_min IS NULL OR y IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y IS NULL OR y.year <= $year_max)
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       '{anchor_relation}' AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
LIMIT $limit
"""

MULTI_ENTITY_CLAIMS_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id IN $institution_ids
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
WHERE d.institution_id IN $institution_ids
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

@functools.lru_cache(maxsize=32)
def build_role_trend_query(role_label: str, role_relation: str) -> str:
    """Build a time-series trend query anchored on a role entity.

    ``role_label`` is the Neo4j node label (e.g. ``Species``) and
    ``role_relation`` is the Observation→role relation type
    (e.g. ``OF_SPECIES``). Caller passes ``$species_id`` as the param name
    regardless of label (back-compat).

    Both args are substituted directly into the query string and must be
    valid Cypher identifiers. Results are memoized by (label, relation) pair.
    """
    if not _CYPHER_IDENT_RE.match(role_label):
        raise ValueError(f"invalid role_label for Cypher: {role_label!r}")
    if not _CYPHER_IDENT_RE.match(role_relation):
        raise ValueError(f"invalid role_relation for Cypher: {role_relation!r}")
    return f"""
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id IN $institution_ids
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {{run_timestamp: latest_ts}})
MATCH (c:Claim {{run_id: lr.run_id}})-[:SUPPORTS]->(obs:Observation)-[:{role_relation}]->(sp:{role_label} {{entity_id: $species_id}})
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN sp.name AS species, y.year AS year,
       count(obs) AS observation_count,
       avg(c.extraction_confidence) AS avg_confidence,
       collect({{name: m.name, value: m.numeric_value, unit: m.unit, approximate: m.approximate}}) AS measurements
ORDER BY y.year
"""


# Back-compat alias — wildlife default. New callers should use
# ``build_role_trend_query(config.trend_role_label, config.trend_role_relation)``.
SPECIES_TREND_QUERY = build_role_trend_query("Species", "OF_SPECIES")

CORPUS_STATS_QUERY = """
CALL () {
  MATCH (d:Document)
  WHERE ($institution_ids IS NULL OR size($institution_ids) = 0 OR d.institution_id IN $institution_ids)
    AND d.deleted_at IS NULL
  RETURN count(d) AS n
} WITH n AS total_documents
CALL () {
  MATCH (d:Document)-[:HAS_PAGE]->(:Page)-[:HAS_PARAGRAPH]->(p:Paragraph)
  WHERE ($institution_ids IS NULL OR size($institution_ids) = 0 OR d.institution_id IN $institution_ids)
    AND d.deleted_at IS NULL
  RETURN count(p) AS n
} WITH total_documents, n AS total_paragraphs
RETURN total_paragraphs, total_documents
"""

@functools.lru_cache(maxsize=32)
def build_scope_condition_query(scope_label: str, scope_relation: str) -> str:
    """Build a condition query anchored on a secondary scope entity.

    ``scope_label`` is the Neo4j node label (e.g. ``Habitat``) and
    ``scope_relation`` is the Observation→scope relation type
    (e.g. ``IN_HABITAT``). Caller passes ``$habitat_id`` regardless of label.
    """
    if not _CYPHER_IDENT_RE.match(scope_label):
        raise ValueError(f"invalid scope_label for Cypher: {scope_label!r}")
    if not _CYPHER_IDENT_RE.match(scope_relation):
        raise ValueError(f"invalid scope_relation for Cypher: {scope_relation!r}")
    return f"""
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id IN $institution_ids
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {{run_timestamp: latest_ts}})
MATCH (c:Claim {{run_id: lr.run_id}})-[:SUPPORTS]->(obs:Observation)-[:{scope_relation}]->(h:{scope_label} {{entity_id: $habitat_id}})
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN h.name AS habitat, y.year AS year, sp.name AS species,
       count(obs) AS observation_count,
       avg(c.extraction_confidence) AS avg_confidence,
       collect({{name: m.name, value: m.numeric_value, unit: m.unit}}) AS measurements
ORDER BY y.year, sp.name
"""


# Back-compat alias — wildlife default.
HABITAT_CONDITION_QUERY = build_scope_condition_query("Habitat", "IN_HABITAT")

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
WHERE d.institution_id IN $institution_ids
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
WHERE d.institution_id IN $institution_ids
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
WHERE d.institution_id IN $institution_ids
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
RETURN d.doc_type AS doc_type, count(d) AS count
ORDER BY count DESC
"""

STATS_CLAIM_TYPE_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE d.institution_id IN $institution_ids
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN c.claim_type AS claim_type,
       count(c) AS count,
       avg(c.extraction_confidence) AS avg_confidence
ORDER BY count DESC
"""

STATS_CLAIM_TYPE_BY_INSTITUTION_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE d.institution_id IN $institution_ids
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN d.institution_id AS institution_id,
       c.claim_type AS claim_type,
       count(c) AS count
ORDER BY institution_id, count DESC
"""

STATS_ENTITY_TYPE_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)-[]->(e:Entity)
WHERE d.institution_id IN $institution_ids
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
RETURN [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type, count(DISTINCT e) AS count
ORDER BY count DESC
"""

STATS_TEMPORAL_COVERAGE_QUERY = """
MATCH (d:Document)
WHERE d.institution_id IN $institution_ids
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND d.report_year IS NOT NULL
RETURN d.report_year AS year, count(d) AS doc_count
ORDER BY year
"""

STATS_CONFIDENCE_DISTRIBUTION_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE d.institution_id IN $institution_ids
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
WHERE d.institution_id IN $institution_ids
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND d.report_year IS NOT NULL
OPTIONAL MATCH (d)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN d.report_year AS year, count(DISTINCT d) AS doc_count, count(c) AS claim_count
ORDER BY year
"""

# Per-corpus temporal density breakdown — used by the gap-analysis dashboard
# to render side-by-side panels per institution alongside the blended view.
GAP_TEMPORAL_DENSITY_BY_INSTITUTION_QUERY = """
MATCH (d:Document)
WHERE d.institution_id IN $institution_ids
  AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND d.report_year IS NOT NULL
OPTIONAL MATCH (d)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)
WHERE (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN d.institution_id AS institution_id,
       d.report_year AS year,
       count(DISTINCT d) AS doc_count,
       count(c) AS claim_count
ORDER BY institution_id, year
"""

# Wildlife-domain defaults preserved as named constants so the coupling is
# visible. Domains should pass their own values derived from DomainConfig
# rather than relying on these.
_DEFAULT_GAP_RELATION_TYPES: tuple[str, ...] = (
    "SUBJECT_OF_CLAIM", "SPECIES_FOCUS", "HABITAT_FOCUS",
    "LOCATION_FOCUS", "MANAGEMENT_TARGET",
)
_DEFAULT_GAP_ENTITY_LABELS: tuple[str, ...] = (
    "Species", "Person", "Organization", "Habitat", "Activity",
)
_DEFAULT_GAP_GEOGRAPHIC_LABELS: tuple[str, ...] = ("Place", "Refuge")
_DEFAULT_GAP_GEOGRAPHIC_RELATIONS: tuple[str, ...] = ("OCCURRED_AT", "LOCATION_FOCUS")


def build_gap_entity_depth_query(
    relation_types: Iterable[str] | None = None,
    entity_labels: Iterable[str] | None = None,
) -> str:
    """Build the GAP_ENTITY_DEPTH_QUERY with parameterized relation types and entity labels.

    When called with ``None``, falls back to wildlife-domain defaults for
    back-compat. Cross-domain callers should pass explicit lists derived
    from ``DomainConfig``.
    """
    if relation_types is None:
        relation_types = _DEFAULT_GAP_RELATION_TYPES
    if entity_labels is None:
        entity_labels = _DEFAULT_GAP_ENTITY_LABELS
    rel_list = ", ".join(f"'{r}'" for r in relation_types)
    label_filter = " OR ".join(f"e:{label}" for label in entity_labels)
    return f"""
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)-[:PRODUCED]->(c:Claim)-[rel]->(e:Entity)
WHERE d.institution_id IN $institution_ids AND d.deleted_at IS NULL
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


def build_gap_geographic_coverage_query(
    location_labels: Iterable[str] | None = None,
    location_relations: Iterable[str] | None = None,
) -> str:
    """Build the geographic-coverage gap query with parameterized labels.

    ``location_labels`` is the list of Neo4j labels that count as "place"
    nodes (e.g. ``["Place", "Refuge"]``). ``location_relations`` is the list
    of claim→location relation types (e.g. ``["OCCURRED_AT", "LOCATION_FOCUS"]``).
    Both fall back to wildlife-domain defaults when ``None``.
    """
    if location_labels is None:
        location_labels = _DEFAULT_GAP_GEOGRAPHIC_LABELS
    if location_relations is None:
        location_relations = _DEFAULT_GAP_GEOGRAPHIC_RELATIONS
    rel_list = ", ".join(f"'{r}'" for r in location_relations)
    label_filter = " OR ".join(f"e:{label}" for label in location_labels)
    return f"""
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)-[:PRODUCED]->(c:Claim)-[rel]->(e:Entity)
WHERE d.institution_id IN $institution_ids AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
  AND type(rel) IN [{rel_list}]
  AND ({label_filter})
WITH e, count(c) AS geo_claim_count,
     [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type
WHERE geo_claim_count <= $thin_threshold
RETURN e.entity_id AS entity_id, e.name AS name, entity_type,
       geo_claim_count AS location_specific_claims
ORDER BY geo_claim_count ASC
LIMIT $limit
"""


GAP_ENTITY_DEPTH_QUERY = build_gap_entity_depth_query()
GAP_GEOGRAPHIC_COVERAGE_QUERY = build_gap_geographic_coverage_query()

# ---------------------------------------------------------------------------
# Relationship Mapping Queries
# ---------------------------------------------------------------------------

ENTITY_SEARCH_QUERY = """
MATCH (e:Entity)
WHERE (toLower(e.name) CONTAINS toLower($query)
       OR toLower(e.normalized_form) CONTAINS toLower($query))
  AND ($institution_ids IS NULL OR size($institution_ids) = 0 OR EXISTS {
        MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(:Claim)-[]->(e)
        WHERE d.institution_id IN $institution_ids AND d.deleted_at IS NULL
      })
RETURN e.entity_id AS entity_id, e.name AS name,
       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type
ORDER BY e.name
LIMIT $limit
"""

ENTITY_DETAIL_QUERY = """
MATCH (e:Entity {entity_id: $entity_id})
OPTIONAL MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)-[rel]->(e)
WHERE d.institution_id IN $institution_ids AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
RETURN e.entity_id AS entity_id, e.name AS name,
       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type,
       count(DISTINCT c) AS claim_count
"""

ENTITY_NEIGHBORHOOD_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)-[r1]->(e1:Entity {entity_id: $entity_id})
WHERE d.institution_id IN $institution_ids AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
WITH c, e1, d
MATCH (c)-[r2]->(e2:Entity)
WHERE e2 <> e1
WITH e1, e2, c, d,
     type(r2) AS rel_type,
     c.source_sentence AS sentence,
     c.claim_id AS claim_id
WITH e1, e2,
     collect(DISTINCT rel_type) AS relationship_types,
     collect(DISTINCT claim_id)[..5] AS sample_claim_ids,
     collect(DISTINCT sentence)[..3] AS sample_sentences,
     collect(DISTINCT d.institution_id) AS institution_ids,
     count(DISTINCT c) AS co_occurrence_count
ORDER BY co_occurrence_count DESC
LIMIT $limit
RETURN e2.entity_id AS entity_id, e2.name AS name,
       [x IN labels(e2) WHERE x <> 'Entity'][0] AS entity_type,
       co_occurrence_count, relationship_types, sample_claim_ids, sample_sentences,
       institution_ids
"""
