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
    "CREATE INDEX document_doc_type IF NOT EXISTS FOR (n:Document) ON (n.doc_type)",
    "CREATE INDEX document_series IF NOT EXISTS FOR (n:Document) ON (n.series)",
    # Latest-run view support. Every retrieval template resolves the newest
    # ExtractionRun per Document and then pulls that run's claims via
    # `MATCH (c:Claim {run_id: lr.run_id})`. Without these two, that inner
    # lookup is a NodeByLabelScan over every Claim executed once per document:
    # profiled at 182M dbHits / 43.9s for a 250-row result on a 5.4k-document
    # corpus. With them, the same query is 1.9M dbHits / 1.0s -- a ~40x
    # improvement that takes an end-to-end /query from ~102s to ~60s.
    "CREATE INDEX claim_run_id IF NOT EXISTS FOR (n:Claim) ON (n.run_id)",
    "CREATE INDEX run_timestamp IF NOT EXISTS FOR (n:ExtractionRun) ON (n.run_timestamp)",
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
# Access-control WHERE fragments
# These are the canonical predicates applied by every retrieval query.
# Templates using the standard alias (`d` for Document, `c` for Claim) embed
# the {_DOC_ACCESS} and {_CLAIM_ACTIVE} tokens; _finalize() expands them at
# module-load time. For non-default aliases, call access_control_where()
# directly and concatenate.
# ---------------------------------------------------------------------------

def access_control_where(alias: str = "d", *, kind: str = "doc") -> str:
    """Return the standard access-control predicate body for *alias*.

    ``kind="doc"`` produces the 4-clause Document filter (institution_id,
    access_level, deleted_at, quarantine_status). ``kind="claim"`` produces
    the single Claim active-status predicate. The returned string does NOT
    include a leading WHERE/AND keyword — the caller emits that plus its
    own indentation.
    """
    if kind == "doc":
        return (
            f"{alias}.institution_id IN $institution_ids\n"
            f"  AND {alias}.access_level IN $permitted_levels\n"
            f"  AND {alias}.deleted_at IS NULL\n"
            f"  AND ({alias}.quarantine_status IS NULL OR {alias}.quarantine_status = 'active')"
        )
    if kind == "claim":
        return f"({alias}.quarantine_status IS NULL OR {alias}.quarantine_status = 'active')"
    raise ValueError(f"unknown access_control_where kind: {kind!r}")


_DOC_ACCESS = access_control_where("d", kind="doc")
_CLAIM_ACTIVE = access_control_where("c", kind="claim")


def _finalize(template: str) -> str:
    """Expand {_DOC_ACCESS} and {_CLAIM_ACTIVE} tokens in a Cypher template."""
    return template.replace("{_DOC_ACCESS}", _DOC_ACCESS).replace("{_CLAIM_ACTIVE}", _CLAIM_ACTIVE)


# ---------------------------------------------------------------------------
# Retrieval-layer query templates
# All queries anchor to the latest ExtractionRun per Document to guarantee
# freshness after re-extraction runs.
# ---------------------------------------------------------------------------

PROVENANCE_CHAIN_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE {_DOC_ACCESS}
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (sec:Section)-[:HAS_PARAGRAPH]->(para:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: lr.run_id})
WHERE c.claim_id = $claim_id
  AND {_CLAIM_ACTIVE}
MATCH (pg:Page)-[:HAS_SECTION]->(sec)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(om:Measurement)
OPTIONAL MATCH (c)-[:HAS_MEASUREMENT]->(dm:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
RETURN d, pg, sec, para, c, obs, sp, y,
       collect(DISTINCT om) + collect(DISTINCT dm) AS measurements
"""

# Tier 1 — count active claims linked to a single resolved entity, scoped by
# the same access-control filters as the full retrieval path.  Skips paragraph
# / page traversal because the answer is just a number.
COUNT_CLAIMS_FOR_ENTITY_QUERY = """
MATCH (e {entity_id: $entity_id})
MATCH (c:Claim)-[]->(e)
WHERE {_CLAIM_ACTIVE}
MATCH (c)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d:Document)
WHERE {_DOC_ACCESS}
RETURN count(DISTINCT c) AS n
"""

# Tier 1 — temporal coverage for a single resolved entity.  Returns the
# earliest/latest year and a count of distinct years, falling back to the
# document's report_year when an observation has no IN_YEAR link (some claim
# types are derived without an Observation→Year edge).
YEAR_RANGE_FOR_ENTITY_QUERY = """
MATCH (e {entity_id: $entity_id})
MATCH (c:Claim)-[]->(e)
WHERE {_CLAIM_ACTIVE}
MATCH (c)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d:Document)
WHERE {_DOC_ACCESS}
OPTIONAL MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(y:Year)
WITH d, c, coalesce(y.year, d.report_year) AS effective_year
WITH effective_year, count(DISTINCT c) AS claim_count
WHERE effective_year IS NOT NULL
RETURN min(effective_year) AS earliest_year,
       max(effective_year) AS latest_year,
       count(effective_year) AS distinct_years,
       sum(claim_count) AS total_claims
"""

# Tier 1 — list documents that contain at least one active claim linked to a
# single resolved entity.  Ranked by claim count descending, capped at 50.
LIST_DOCUMENTS_FOR_ENTITY_QUERY = """
MATCH (e {entity_id: $entity_id})
MATCH (c:Claim)-[]->(e)
WHERE {_CLAIM_ACTIVE}
MATCH (c)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d:Document)
WHERE {_DOC_ACCESS}
WITH d, count(DISTINCT c) AS claim_count
RETURN d.doc_id AS doc_id,
       d.title AS title,
       d.doc_type AS doc_type,
       d.date_start AS date_start,
       d.date_end AS date_end,
       d.report_year AS report_year,
       d.institution_id AS institution_id,
       d.lccn AS lccn,
       d.edition AS edition,
       d.loc_sequence AS loc_sequence,
       claim_count
ORDER BY claim_count DESC
LIMIT 50
"""

# Tier 1 — places where claims about a single resolved entity occurred.
# Returns ranked location nodes by claim count, capped at 25, filtered through
# the same document access-control structure as the rest of the retrieval layer.
def build_locations_for_entity_query(
    *,
    location_relation: str = "OCCURRED_AT",
    location_labels: tuple[str, ...] = ("Place", "Refuge"),
) -> str:
    """Build the Tier-1 "where did claims about X occur" query.

    ``location_relation`` and ``location_labels`` come from the domain's
    ``claim_location_relation`` + location entity types. The defaults reproduce
    the historical Turnbull query byte-for-byte. Returns a finalized query
    (access-control tokens already expanded) ready to execute.
    """
    label_pred = " OR ".join(f"p:{lbl}" for lbl in location_labels)
    return _finalize(f"""
MATCH (e {{entity_id: $entity_id}})
MATCH (c:Claim)-[]->(e)
WHERE {{_CLAIM_ACTIVE}}
MATCH (c)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d:Document)
WHERE {{_DOC_ACCESS}}
MATCH (c)-[:{location_relation}]->(p)
WHERE {label_pred}
RETURN p.entity_id AS place_id,
       p.name AS place_name,
       [x IN labels(p) WHERE x <> 'Entity'][0] AS place_type,
       count(DISTINCT c) AS claim_count
ORDER BY claim_count DESC
LIMIT 25
""")


LOCATIONS_FOR_ENTITY_QUERY = build_locations_for_entity_query()

# Tier 1 — entities that co-occur in claims with a single resolved entity.
# Distinct from ENTITY_NEIGHBORHOOD_QUERY (which uses a non-existent
# ExtractionRun-PRODUCED-Claim edge); this one uses the structural HAS_CLAIM
# chain that production retrieval depends on, so it actually returns rows.
# Returns ranked co-occurring entities with counts and entity types, capped at 25.
CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY = """
MATCH (anchor {entity_id: $entity_id})
MATCH (c:Claim)-[]->(anchor)
WHERE {_CLAIM_ACTIVE}
MATCH (c)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d:Document)
WHERE {_DOC_ACCESS}
MATCH (c)-[r2]->(other)
WHERE other.entity_id IS NOT NULL
  AND other.entity_id <> $entity_id
  AND any(lbl IN labels(other) WHERE lbl <> 'Entity')
RETURN other.entity_id AS entity_id,
       other.name AS name,
       [x IN labels(other) WHERE x <> 'Entity'][0] AS entity_type,
       count(DISTINCT c) AS co_occurrence_count
ORDER BY co_occurrence_count DESC
LIMIT 25
"""

# Tier 1 — top-N entities of a given label by claim count, scoped by access
# control.  Caller passes $entity_label (e.g. "Person", "Place", "Species")
# and $limit; we apply the label filter via a labels() check rather than
# direct interpolation to keep the template parameterised.
FREQUENCY_RANKING_FOR_TYPE_QUERY = """
MATCH (e:Entity)
WHERE $entity_label IN labels(e)
MATCH (c:Claim)-[]->(e)
WHERE {_CLAIM_ACTIVE}
MATCH (c)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d:Document)
WHERE {_DOC_ACCESS}
RETURN e.entity_id AS entity_id,
       e.name AS name,
       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type,
       count(DISTINCT c) AS claim_count
ORDER BY claim_count DESC
LIMIT $limit
"""


# Tier 1 — list documents matching a specific report_year.  Includes documents
# with no extracted claims (claim_count may be 0) so the user can see the full
# year roster.  Ranked by date_start ascending, capped at 50.
LIST_DOCUMENTS_FOR_YEAR_QUERY = """
MATCH (d:Document)
WHERE {_DOC_ACCESS}
  AND d.report_year = $target_year
OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)
              -[:HAS_PARAGRAPH]->(:Paragraph)-[:HAS_CLAIM]->(c:Claim)
WHERE c.quarantine_status IS NULL OR c.quarantine_status = 'active'
WITH d, count(DISTINCT c) AS claim_count
RETURN d.doc_id AS doc_id,
       d.title AS title,
       d.doc_type AS doc_type,
       d.date_start AS date_start,
       d.date_end AS date_end,
       d.report_year AS report_year,
       d.institution_id AS institution_id,
       d.lccn AS lccn,
       d.edition AS edition,
       d.loc_sequence AS loc_sequence,
       claim_count
ORDER BY d.date_start ASC, d.title ASC
LIMIT 50
"""


ENTITY_ANCHORED_CLAIMS_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE {_DOC_ACCESS}
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[rel]->(e:Entity {entity_id: $entity_id})
WHERE {_CLAIM_ACTIVE}
  AND (
    ($year_min IS NULL AND $year_max IS NULL)
    OR NOT EXISTS { (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(:Year) }
    OR EXISTS {
      MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(yr:Year)
      WHERE ($year_min IS NULL OR yr.year >= $year_min)
        AND ($year_max IS NULL OR yr.year <= $year_max)
    }
  )
WITH d, c, type(rel) AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
// Per-corpus stratification: each institution_id contributes up to $limit candidates
// before downstream truncation, so a higher-confidence corpus cannot monopolise the
// pool when both have claims attached to the anchor entity.
WITH d.institution_id AS corpus,
     collect({d: d, c: c, traversal_rel_type: traversal_rel_type}) AS rows
UNWIND rows[..$limit] AS row
WITH row.d AS d, row.c AS c, row.traversal_rel_type AS traversal_rel_type
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
WITH d, pg, sec, para, c, obs, sp, y, traversal_rel_type, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements, traversal_rel_type
ORDER BY c.extraction_confidence DESC
"""

FULLTEXT_CLAIMS_QUERY = """
CALL db.index.fulltext.queryNodes('claim_normalized_sentence', $search_text)
YIELD node AS c, score
WITH c, score
WHERE {_CLAIM_ACTIVE}
MATCH (d:Document)-[:PROCESSED_BY]->(lr:ExtractionRun {run_id: c.run_id})
WHERE {_DOC_ACCESS}
WITH c, d, score
ORDER BY score DESC
LIMIT $limit
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
WITH d, pg, sec, para, c, obs, sp, y, score, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements, score
ORDER BY score DESC
"""

TEMPORAL_CLAIMS_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE {_DOC_ACCESS}
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[:SUPPORTS]->(obs:Observation)-[:IN_YEAR]->(y:Year)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
  AND ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND {_CLAIM_ACTIVE}
WITH d, c, obs, y
ORDER BY c.extraction_confidence DESC
LIMIT $limit
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
RETURN d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements,
       'IN_YEAR' AS traversal_rel_type
ORDER BY y.year, c.extraction_confidence DESC
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
WHERE {_DOC_ACCESS}
MATCH (d)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {{run_timestamp: latest_ts}})
MATCH (c:Claim {{run_id: lr.run_id}})-[:SUPPORTS]->(obs_f:Observation)-[:IN_YEAR]->(y_f:Year)
WHERE ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND {_CLAIM_ACTIVE}
  AND ($year_min IS NULL OR y_f.year >= $year_min)
  AND ($year_max IS NULL OR y_f.year <= $year_max)
WITH DISTINCT d, c, y_f
ORDER BY y_f.year, c.extraction_confidence DESC
LIMIT $limit
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)
           <-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
MATCH (c)-[:SUPPORTS]->(obs:Observation)-[:IN_YEAR]->(y:Year)
WHERE y = y_f
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y,
     collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       '{anchor_relation}' AS traversal_rel_type
ORDER BY y.year, c.extraction_confidence DESC
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
WHERE {_DOC_ACCESS}
MATCH (d)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {{run_timestamp: latest_ts}})
MATCH (c:Claim {{run_id: lr.run_id}})
WHERE ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND {_CLAIM_ACTIVE}
  AND (
    ($year_min IS NULL AND $year_max IS NULL)
    OR NOT EXISTS {{ (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(:Year) }}
    OR EXISTS {{
      MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(yr:Year)
      WHERE ($year_min IS NULL OR yr.year >= $year_min)
        AND ($year_max IS NULL OR yr.year <= $year_max)
    }}
  )
WITH d, c
ORDER BY c.extraction_confidence DESC
LIMIT $limit
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)
           <-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       '{anchor_relation}' AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
"""

MULTI_ENTITY_CLAIMS_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE {_DOC_ACCESS}
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})-[rel]->(e:Entity)
WHERE e.entity_id IN $entity_ids
  AND ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND {_CLAIM_ACTIVE}
  AND (
    ($year_min IS NULL AND $year_max IS NULL)
    OR NOT EXISTS { (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(:Year) }
    OR EXISTS {
      MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(yr:Year)
      WHERE ($year_min IS NULL OR yr.year >= $year_min)
        AND ($year_max IS NULL OR yr.year <= $year_max)
    }
  )
WITH d, c, type(rel) AS traversal_rel_type,
     collect(DISTINCT e.entity_id) AS matched_entity_ids
ORDER BY size(matched_entity_ids) DESC, c.extraction_confidence DESC
LIMIT $limit
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y, traversal_rel_type, matched_entity_ids,
     collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements, traversal_rel_type,
       matched_entity_ids
ORDER BY size(matched_entity_ids) DESC, c.extraction_confidence DESC
"""

# Cross-corpus entity overlap — split into two parallel queries (NEWSPAPER + PATENT
# sides) that share the same row shape and are run together as a single cascade
# attempt.  Together they answer "find Persons who appear in both $corpus_a (e.g.
# Spokane Press as a newspaper-mention via cross_corpus_ner REFERS_TO/POSSIBLY_REFERS_TO
# edges) and $corpus_b (e.g. turnbull-composite GLO records as the SUBJECT_OF_CLAIM
# of a land_grant Claim) and surface bridging claims from each side."
#
# Schema constraints (from gemynd/ingest/enrichment/cross_corpus_ner.py):
#   • Newspaper Mention nodes link to Person via REFERS_TO / POSSIBLY_REFERS_TO.
#   • Mentions are attached to Paragraphs via (Paragraph)-[:CONTAINS_MENTION]->(Mention).
#   • Land patent claims link to Person via (Claim {claim_type:'land_grant'})-[:SUBJECT_OF_CLAIM]->(Person).
#   • Patent paragraphs do NOT have HAS_PARAGRAPH back to Document — the patent
#     side OPTIONAL MATCHes the chain so missing sec/pg/para fall through cleanly.
CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE d.institution_id = $corpus_a
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (d)-[:HAS_PAGE]->(pg:Page)-[:HAS_SECTION]->(sec:Section)
       -[:HAS_PARAGRAPH]->(para:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: lr.run_id})
WHERE ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND {_CLAIM_ACTIVE}
  AND (
    ($year_min IS NULL AND $year_max IS NULL)
    OR NOT EXISTS { (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(:Year) }
    OR EXISTS {
      MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(yr:Year)
      WHERE ($year_min IS NULL OR yr.year >= $year_min)
        AND ($year_max IS NULL OR yr.year <= $year_max)
    }
  )
  AND EXISTS {
    MATCH (para)-[:CONTAINS_MENTION]->(:Mention)
          -[:REFERS_TO|POSSIBLY_REFERS_TO]->(p:Person)
    WHERE EXISTS {
      MATCH (cb:Claim {claim_type: 'land_grant'})-[:SUBJECT_OF_CLAIM]->(p)
      MATCH (db:Document)-[:PROCESSED_BY]->(:ExtractionRun {run_id: cb.run_id})
      WHERE db.institution_id = $corpus_b
        AND db.access_level IN $permitted_levels
        AND db.deleted_at IS NULL
        AND (db.quarantine_status IS NULL OR db.quarantine_status = 'active')
        AND (cb.quarantine_status IS NULL OR cb.quarantine_status = 'active')
    }
  }
WITH d, pg, sec, para, c
ORDER BY c.extraction_confidence DESC
LIMIT $limit
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       'CROSS_CORPUS_MENTION' AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
"""

CROSS_CORPUS_OVERLAP_PATENT_QUERY = """
MATCH (c:Claim {claim_type: 'land_grant'})-[:SUBJECT_OF_CLAIM]->(p:Person)
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun {run_id: c.run_id})
WHERE d.institution_id = $corpus_b
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
  AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')
  AND {_CLAIM_ACTIVE}
  AND EXISTS {
    MATCH (da:Document)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)
          -[:HAS_PARAGRAPH]->(:Paragraph)-[:CONTAINS_MENTION]->(:Mention)
          -[:REFERS_TO|POSSIBLY_REFERS_TO]->(p)
    WHERE da.institution_id = $corpus_a
      AND da.access_level IN $permitted_levels
      AND da.deleted_at IS NULL
      AND (da.quarantine_status IS NULL OR da.quarantine_status = 'active')
  }
WITH d, c
ORDER BY c.extraction_confidence DESC
LIMIT $limit
OPTIONAL MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       'SUBJECT_OF_CLAIM' AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
"""

CLAIM_TYPE_SCOPED_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE {_DOC_ACCESS}
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})
WHERE c.claim_type IN $claim_types
  AND {_CLAIM_ACTIVE}
  AND ($entity_ids IS NULL OR EXISTS {
    MATCH (c)-[]->(e:Entity) WHERE e.entity_id IN $entity_ids
  })
  AND (
    ($year_min IS NULL AND $year_max IS NULL)
    OR NOT EXISTS { (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(:Year) }
    OR EXISTS {
      MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(yr:Year)
      WHERE ($year_min IS NULL OR yr.year >= $year_min)
        AND ($year_max IS NULL OR yr.year <= $year_max)
    }
  )
WITH d, c
ORDER BY c.extraction_confidence DESC
LIMIT $limit
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       c.claim_type AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
"""

DOC_TYPE_BROWSE_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE {_DOC_ACCESS}
  AND d.doc_type IN $doc_types
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})
WHERE {_CLAIM_ACTIVE}
  AND ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND (
    ($year_min IS NULL AND $year_max IS NULL)
    OR NOT EXISTS { (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(:Year) }
    OR EXISTS {
      MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(yr:Year)
      WHERE ($year_min IS NULL OR yr.year >= $year_min)
        AND ($year_max IS NULL OR yr.year <= $year_max)
    }
  )
WITH d, c
ORDER BY c.extraction_confidence DESC
LIMIT $limit
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       'DOC_TYPE_BROWSE' AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
"""

SERIES_BROWSE_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WHERE {_DOC_ACCESS}
  AND d.series IN $series
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {run_timestamp: latest_ts})
MATCH (c:Claim {run_id: lr.run_id})
WHERE {_CLAIM_ACTIVE}
  AND ($claim_types IS NULL OR c.claim_type IN $claim_types)
  AND (
    ($year_min IS NULL AND $year_max IS NULL)
    OR NOT EXISTS { (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(:Year) }
    OR EXISTS {
      MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(yr:Year)
      WHERE ($year_min IS NULL OR yr.year >= $year_min)
        AND ($year_max IS NULL OR yr.year <= $year_max)
    }
  )
WITH d, c
ORDER BY c.extraction_confidence DESC
LIMIT $limit
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
OPTIONAL MATCH (para)<-[:HAS_PARAGRAPH]-(sec:Section)<-[:HAS_SECTION]-(pg:Page)<-[:HAS_PAGE]-(d)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH d, pg, sec, para, c, obs, sp, y, collect(DISTINCT m) AS measurements
RETURN d, pg, sec, para, c, obs, sp, y, measurements,
       'SERIES_BROWSE' AS traversal_rel_type
ORDER BY c.extraction_confidence DESC
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
WHERE {_DOC_ACCESS}
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {{run_timestamp: latest_ts}})
MATCH (c:Claim {{run_id: lr.run_id}})-[:SUPPORTS]->(obs:Observation)-[:{role_relation}]->(sp:{role_label} {{entity_id: $species_id}})
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
  AND {_CLAIM_ACTIVE}
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
WHERE {_DOC_ACCESS}
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(lr:ExtractionRun {{run_timestamp: latest_ts}})
MATCH (c:Claim {{run_id: lr.run_id}})-[:SUPPORTS]->(obs:Observation)-[:{scope_relation}]->(h:{scope_label} {{entity_id: $habitat_id}})
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WHERE ($year_min IS NULL OR y.year >= $year_min)
  AND ($year_max IS NULL OR y.year <= $year_max)
  AND {_CLAIM_ACTIVE}
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
WHERE d.institution_id IN $institution_ids
SET d.deleted_at = $deleted_at, d.deleted_by = $deleted_by
RETURN d.doc_id AS doc_id, d.title AS title, d.institution_id AS institution_id
"""

RESTORE_DOCUMENT_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
WHERE d.institution_id IN $institution_ids
SET d.deleted_at = null, d.deleted_by = null
RETURN d.doc_id AS doc_id, d.title AS title, d.institution_id AS institution_id
"""

LIST_DOCUMENTS_QUERY = """
MATCH (d:Document)
WHERE d.institution_id IN $institution_ids
RETURN d.doc_id AS doc_id, d.title AS title, d.access_level AS access_level,
       d.institution_id AS institution_id,
       d.deleted_at AS deleted_at, properties(d).deleted_by AS deleted_by
ORDER BY d.institution_id, d.title
"""

# Paginated document listing for the /documents retrieval surface. Joins each
# Document to its highest-confidence active Claim to surface a snippet so the
# title-mangled-by-OCR newspaper rows remain identifiable, and aggregates a
# claim count per document. All filter parameters accept null to mean "off".
#
# $status accepts: 'active' (deleted_at IS NULL), 'deleted' (deleted_at IS NOT
# NULL), or 'all'.
#
# Year filtering uses an effective year derived from report_year first, then
# the leading 4 chars of date_start. Documents missing both are excluded only
# when year_min/year_max are non-null.
LIST_DOCUMENTS_PAGINATED_QUERY = """
MATCH (d:Document)
WHERE d.institution_id IN $institution_ids
  AND ($status = 'all'
       OR ($status = 'active' AND d.deleted_at IS NULL)
       OR ($status = 'deleted' AND d.deleted_at IS NOT NULL))
  AND ($institution_filter IS NULL OR d.institution_id IN $institution_filter)
  AND ($doc_types IS NULL OR d.doc_type IN $doc_types)
  AND ($search IS NULL OR toLower(coalesce(d.title, '')) CONTAINS toLower($search))
WITH d, coalesce(d.report_year,
                 CASE WHEN d.date_start IS NOT NULL AND size(d.date_start) >= 4
                      THEN toInteger(substring(d.date_start, 0, 4))
                      ELSE null END) AS effective_year
WHERE ($year_min IS NULL OR (effective_year IS NOT NULL AND effective_year >= $year_min))
  AND ($year_max IS NULL OR (effective_year IS NOT NULL AND effective_year <= $year_max))
OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)
              -[:HAS_PARAGRAPH]->(:Paragraph)-[:HAS_CLAIM]->(c:Claim)
WHERE c.quarantine_status IS NULL OR c.quarantine_status = 'active'
WITH d, effective_year, c
ORDER BY c.extraction_confidence DESC
WITH d, effective_year, collect(c) AS claims
RETURN d.doc_id AS doc_id,
       d.title AS title,
       d.doc_type AS doc_type,
       d.date_start AS date_start,
       d.date_end AS date_end,
       d.report_year AS report_year,
       effective_year AS effective_year,
       d.institution_id AS institution_id,
       d.archive_ref AS archive_ref,
       d.lccn AS lccn,
       d.edition AS edition,
       d.loc_sequence AS loc_sequence,
       d.access_level AS access_level,
       d.deleted_at AS deleted_at,
       size(claims) AS claim_count,
       claims[0].normalized_sentence AS snippet
ORDER BY coalesce(d.date_start, toString(d.report_year), '') DESC, d.doc_id
SKIP $offset
LIMIT $limit
"""

# Count of documents matching the same filters as
# LIST_DOCUMENTS_PAGINATED_QUERY — used to drive pagination totals.
COUNT_DOCUMENTS_QUERY = """
MATCH (d:Document)
WHERE d.institution_id IN $institution_ids
  AND ($status = 'all'
       OR ($status = 'active' AND d.deleted_at IS NULL)
       OR ($status = 'deleted' AND d.deleted_at IS NOT NULL))
  AND ($institution_filter IS NULL OR d.institution_id IN $institution_filter)
  AND ($doc_types IS NULL OR d.doc_type IN $doc_types)
  AND ($search IS NULL OR toLower(coalesce(d.title, '')) CONTAINS toLower($search))
WITH d, coalesce(d.report_year,
                 CASE WHEN d.date_start IS NOT NULL AND size(d.date_start) >= 4
                      THEN toInteger(substring(d.date_start, 0, 4))
                      ELSE null END) AS effective_year
WHERE ($year_min IS NULL OR (effective_year IS NOT NULL AND effective_year >= $year_min))
  AND ($year_max IS NULL OR (effective_year IS NOT NULL AND effective_year <= $year_max))
RETURN count(d) AS total
"""

# Facet values for the /documents filter rail: distinct doc_types and the
# observed year span. Driven off the same institution allowlist, no other
# filters applied so the facet options don't disappear when a filter is on.
DOCUMENT_FACETS_QUERY = """
MATCH (d:Document)
WHERE d.institution_id IN $institution_ids
WITH d, coalesce(d.report_year,
                 CASE WHEN d.date_start IS NOT NULL AND size(d.date_start) >= 4
                      THEN toInteger(substring(d.date_start, 0, 4))
                      ELSE null END) AS effective_year
RETURN collect(DISTINCT d.doc_type) AS doc_types,
       collect(DISTINCT d.institution_id) AS institutions,
       min(effective_year) AS year_min,
       max(effective_year) AS year_max,
       count(d) AS total_documents
"""

# Per-document claims listing for the document detail panel. Returns top
# 50 active claims ordered by extraction_confidence descending.
LIST_CLAIMS_FOR_DOCUMENT_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
WHERE d.institution_id IN $institution_ids
MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)
      -[:HAS_PARAGRAPH]->(:Paragraph)-[:HAS_CLAIM]->(c:Claim)
WHERE c.quarantine_status IS NULL OR c.quarantine_status = 'active'
RETURN c.claim_id AS claim_id,
       c.normalized_sentence AS sentence,
       c.claim_type AS claim_type,
       c.extraction_confidence AS confidence,
       c.epistemic_status AS epistemic_status
ORDER BY c.extraction_confidence DESC
LIMIT 50
"""

# OCR text fetch for the in-app source preview modal. Returns both the raw
# OCR string and the normalized clean_text so the modal can offer a
# clean/raw toggle. Either field may be empty depending on how the document
# was ingested; the route handler decides what to surface.
GET_DOCUMENT_SOURCE_TEXT_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
WHERE d.institution_id IN $institution_ids
RETURN d.doc_id AS doc_id,
       d.title AS title,
       d.institution_id AS institution_id,
       d.clean_text AS clean_text,
       d.raw_ocr_text AS raw_ocr_text,
       d.page_count AS page_count
"""

# Per-document paragraph reconstruction — used when the document-level
# clean_text / raw_ocr_text properties are empty (some ingestion paths only
# persist text at the paragraph granularity). Walks Document → Page →
# Section → Paragraph in graph order.
GET_DOCUMENT_PARAGRAPHS_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
WHERE d.institution_id IN $institution_ids
MATCH (d)-[:HAS_PAGE]->(p:Page)-[:HAS_SECTION]->(s:Section)
      -[:HAS_PARAGRAPH]->(para:Paragraph)
RETURN p.page_id AS page_id,
       p.page_number AS page_number,
       s.section_id AS section_id,
       s.heading AS section_heading,
       para.paragraph_id AS paragraph_id,
       para.clean_text AS clean_text,
       para.raw_ocr_text AS raw_ocr_text
ORDER BY p.page_number, s.section_id, para.paragraph_id
"""

# Document header for the detail panel — single-row metadata fetch.
GET_DOCUMENT_HEADER_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
WHERE d.institution_id IN $institution_ids
RETURN d.doc_id AS doc_id,
       d.title AS title,
       d.doc_type AS doc_type,
       d.date_start AS date_start,
       d.date_end AS date_end,
       d.report_year AS report_year,
       d.institution_id AS institution_id,
       d.archive_ref AS archive_ref,
       d.lccn AS lccn,
       d.edition AS edition,
       d.loc_sequence AS loc_sequence,
       d.access_level AS access_level,
       d.deleted_at AS deleted_at,
       properties(d).deleted_by AS deleted_by
"""

# ---------------------------------------------------------------------------
# Mode 1: entity-to-entity path-finding
#
# Researchers ask "how is A connected to B?" and the system answers in a
# cascade: direct co-assertion (one claim mentions both) → 1-hop (one
# intermediate entity) → 2-hop → 3-hop. The first non-empty layer wins.
# All queries apply the standard institution + access-level + soft-delete
# filter on every claim's owning document, plus an optional year window.
# ---------------------------------------------------------------------------

# Direct co-assertion: claims that name both entities. The dominant case
# in primary-source research — most "how are A and B connected" questions
# have a direct answer if one exists. Returns top 25 by extraction
# confidence so the strongest evidence ranks first.
CO_ASSERTING_CLAIMS_QUERY = """
MATCH (a {entity_id: $from_id})
MATCH (b {entity_id: $to_id})
MATCH (c:Claim)-[r1]->(a)
MATCH (c)-[r2]->(b)
WHERE {_CLAIM_ACTIVE}
MATCH (c)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d:Document)
WHERE d.institution_id IN $institution_ids
  AND d.access_level IN $permitted_levels
  AND d.deleted_at IS NULL
WITH a, b, c, r1, r2, d,
     coalesce(d.report_year,
              CASE WHEN d.date_start IS NOT NULL AND size(d.date_start) >= 4
                   THEN toInteger(substring(d.date_start, 0, 4))
                   ELSE null END) AS effective_year
WHERE ($year_min IS NULL OR (effective_year IS NOT NULL AND effective_year >= $year_min))
  AND ($year_max IS NULL OR (effective_year IS NOT NULL AND effective_year <= $year_max))
RETURN c.claim_id AS claim_id,
       c.normalized_sentence AS sentence,
       c.claim_type AS claim_type,
       c.extraction_confidence AS confidence,
       type(r1) AS from_relation,
       type(r2) AS to_relation,
       d.doc_id AS doc_id,
       d.title AS doc_title,
       d.date_start AS doc_date_start,
       d.report_year AS doc_report_year,
       d.institution_id AS institution_id
ORDER BY c.extraction_confidence DESC
LIMIT 25
"""

# Per-entity claim count, used for inverse-frequency path weighting and the
# 95th-percentile degree cap that filters paths through generic hub
# entities. Cached at app startup; recomputed only on TTL expiry.
#
# count(DISTINCT c) — without DISTINCT, an entity that's referenced multiple
# times by the same claim (e.g. via both LOCATION_FOCUS and TOPIC_OF_CLAIM)
# would be counted once per relation, inflating the degree above
# corpus_total_claims and producing negative inverse-frequency weights.
ENTITY_DEGREE_QUERY = """
MATCH (e:Entity)
OPTIONAL MATCH (e)<-[]-(c:Claim)
WHERE c.quarantine_status IS NULL OR c.quarantine_status = 'active'
RETURN e.entity_id AS entity_id, count(DISTINCT c) AS degree
"""

# 1-hop path: A ←c1→ I ←c2→ B (one intermediate entity, two claims). Each
# row is one candidate path. Multiple rows for the same intermediate are
# expected — Python collapses them into path-families before ranking.
PATHS_1_HOP_QUERY = """
MATCH (a {entity_id: $from_id})
MATCH (b {entity_id: $to_id})
MATCH (c1:Claim)-[r1]->(a)
WHERE c1.quarantine_status IS NULL OR c1.quarantine_status = 'active'
MATCH (c1)-[r2]->(i)
WHERE i.entity_id IS NOT NULL
  AND i.entity_id <> $from_id AND i.entity_id <> $to_id
  AND any(lbl IN labels(i) WHERE lbl <> 'Entity')
  AND none(lbl IN labels(i) WHERE lbl IN ['Period', 'Year', 'Date'])
  AND NOT i.entity_id IN $excluded_intermediate_ids
MATCH (c2:Claim)-[r3]->(i)
WHERE c2 <> c1
  AND (c2.quarantine_status IS NULL OR c2.quarantine_status = 'active')
MATCH (c2)-[r4]->(b)
MATCH (c1)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d1:Document)
WHERE d1.institution_id IN $institution_ids
  AND d1.access_level IN $permitted_levels
  AND d1.deleted_at IS NULL
MATCH (c2)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d2:Document)
WHERE d2.institution_id IN $institution_ids
  AND d2.access_level IN $permitted_levels
  AND d2.deleted_at IS NULL
WITH a, b, i, c1, c2, r1, r2, r3, r4, d1, d2,
     coalesce(d1.report_year,
              CASE WHEN d1.date_start IS NOT NULL AND size(d1.date_start) >= 4
                   THEN toInteger(substring(d1.date_start, 0, 4))
                   ELSE null END) AS y1,
     coalesce(d2.report_year,
              CASE WHEN d2.date_start IS NOT NULL AND size(d2.date_start) >= 4
                   THEN toInteger(substring(d2.date_start, 0, 4))
                   ELSE null END) AS y2
WHERE ($year_min IS NULL OR (y1 IS NOT NULL AND y1 >= $year_min AND y2 IS NOT NULL AND y2 >= $year_min))
  AND ($year_max IS NULL OR (y1 IS NOT NULL AND y1 <= $year_max AND y2 IS NOT NULL AND y2 <= $year_max))
RETURN i.entity_id AS i1_id, i.name AS i1_name,
       [x IN labels(i) WHERE x <> 'Entity'][0] AS i1_type,
       c1.claim_id AS e1_claim_id, c1.normalized_sentence AS e1_sentence,
       c1.extraction_confidence AS e1_conf,
       type(r1) AS e1_from_rel, type(r2) AS e1_to_rel,
       d1.doc_id AS e1_doc_id, d1.title AS e1_doc_title, d1.date_start AS e1_doc_date,
       c2.claim_id AS e2_claim_id, c2.normalized_sentence AS e2_sentence,
       c2.extraction_confidence AS e2_conf,
       type(r3) AS e2_from_rel, type(r4) AS e2_to_rel,
       d2.doc_id AS e2_doc_id, d2.title AS e2_doc_title, d2.date_start AS e2_doc_date
ORDER BY (c1.extraction_confidence + c2.extraction_confidence) DESC
LIMIT 100
"""

# 2-hop path: A ←c1→ I1 ←c2→ I2 ←c3→ B (two intermediates, three claims).
PATHS_2_HOP_QUERY = """
MATCH (a {entity_id: $from_id})
MATCH (b {entity_id: $to_id})
MATCH (c1:Claim)-[r1]->(a)
WHERE c1.quarantine_status IS NULL OR c1.quarantine_status = 'active'
MATCH (c1)-[r2]->(i1)
WHERE i1.entity_id IS NOT NULL
  AND i1.entity_id <> $from_id AND i1.entity_id <> $to_id
  AND any(lbl IN labels(i1) WHERE lbl <> 'Entity')
  AND none(lbl IN labels(i1) WHERE lbl IN ['Period', 'Year', 'Date'])
  AND NOT i1.entity_id IN $excluded_intermediate_ids
MATCH (c2:Claim)-[r3]->(i1)
WHERE c2 <> c1
  AND (c2.quarantine_status IS NULL OR c2.quarantine_status = 'active')
MATCH (c2)-[r4]->(i2)
WHERE i2.entity_id IS NOT NULL
  AND i2.entity_id <> $from_id AND i2.entity_id <> $to_id
  AND i2.entity_id <> i1.entity_id
  AND any(lbl IN labels(i2) WHERE lbl <> 'Entity')
  AND none(lbl IN labels(i2) WHERE lbl IN ['Period', 'Year', 'Date'])
  AND NOT i2.entity_id IN $excluded_intermediate_ids
MATCH (c3:Claim)-[r5]->(i2)
WHERE c3 <> c1 AND c3 <> c2
  AND (c3.quarantine_status IS NULL OR c3.quarantine_status = 'active')
MATCH (c3)-[r6]->(b)
MATCH (c1)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d1:Document)
WHERE d1.institution_id IN $institution_ids
  AND d1.access_level IN $permitted_levels
  AND d1.deleted_at IS NULL
MATCH (c2)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d2:Document)
WHERE d2.institution_id IN $institution_ids
  AND d2.access_level IN $permitted_levels
  AND d2.deleted_at IS NULL
MATCH (c3)<-[:HAS_CLAIM]-(:Paragraph)<-[:HAS_PARAGRAPH]-(:Section)
      <-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d3:Document)
WHERE d3.institution_id IN $institution_ids
  AND d3.access_level IN $permitted_levels
  AND d3.deleted_at IS NULL
WITH a, b, i1, i2, c1, c2, c3, r1, r2, r3, r4, r5, r6, d1, d2, d3,
     coalesce(d1.report_year,
              CASE WHEN d1.date_start IS NOT NULL AND size(d1.date_start) >= 4
                   THEN toInteger(substring(d1.date_start, 0, 4))
                   ELSE null END) AS y1,
     coalesce(d2.report_year,
              CASE WHEN d2.date_start IS NOT NULL AND size(d2.date_start) >= 4
                   THEN toInteger(substring(d2.date_start, 0, 4))
                   ELSE null END) AS y2,
     coalesce(d3.report_year,
              CASE WHEN d3.date_start IS NOT NULL AND size(d3.date_start) >= 4
                   THEN toInteger(substring(d3.date_start, 0, 4))
                   ELSE null END) AS y3
WHERE ($year_min IS NULL OR (y1 IS NOT NULL AND y1 >= $year_min AND y2 IS NOT NULL AND y2 >= $year_min AND y3 IS NOT NULL AND y3 >= $year_min))
  AND ($year_max IS NULL OR (y1 IS NOT NULL AND y1 <= $year_max AND y2 IS NOT NULL AND y2 <= $year_max AND y3 IS NOT NULL AND y3 <= $year_max))
RETURN i1.entity_id AS i1_id, i1.name AS i1_name,
       [x IN labels(i1) WHERE x <> 'Entity'][0] AS i1_type,
       i2.entity_id AS i2_id, i2.name AS i2_name,
       [x IN labels(i2) WHERE x <> 'Entity'][0] AS i2_type,
       c1.claim_id AS e1_claim_id, c1.normalized_sentence AS e1_sentence,
       c1.extraction_confidence AS e1_conf,
       type(r1) AS e1_from_rel, type(r2) AS e1_to_rel,
       d1.doc_id AS e1_doc_id, d1.title AS e1_doc_title, d1.date_start AS e1_doc_date,
       c2.claim_id AS e2_claim_id, c2.normalized_sentence AS e2_sentence,
       c2.extraction_confidence AS e2_conf,
       type(r3) AS e2_from_rel, type(r4) AS e2_to_rel,
       d2.doc_id AS e2_doc_id, d2.title AS e2_doc_title, d2.date_start AS e2_doc_date,
       c3.claim_id AS e3_claim_id, c3.normalized_sentence AS e3_sentence,
       c3.extraction_confidence AS e3_conf,
       type(r5) AS e3_from_rel, type(r6) AS e3_to_rel,
       d3.doc_id AS e3_doc_id, d3.title AS e3_doc_title, d3.date_start AS e3_doc_date
ORDER BY (c1.extraction_confidence + c2.extraction_confidence + c3.extraction_confidence) DESC
LIMIT 100
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
WHERE {_CLAIM_ACTIVE}
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
  AND {_CLAIM_ACTIVE}
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
  AND {_CLAIM_ACTIVE}
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
  AND {_CLAIM_ACTIVE}
RETURN
  count(c) AS total_claims,
  avg(c.extraction_confidence) AS avg_confidence,
  count(CASE WHEN c.extraction_confidence >= 0.85 THEN 1 ELSE null END) AS high_count,
  count(CASE WHEN c.extraction_confidence >= 0.70 AND c.extraction_confidence < 0.85 THEN 1 ELSE null END) AS medium_count,
  count(CASE WHEN c.extraction_confidence < 0.70 THEN 1 ELSE null END) AS low_count,
  count(CASE WHEN c.epistemic_status = 'uncertain' THEN 1 ELSE null END) AS uncertain_epistemic_count
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
WHERE {_CLAIM_ACTIVE}
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
WHERE {_CLAIM_ACTIVE}
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
  AND {_CLAIM_ACTIVE}
  AND type(rel) IN [{rel_list}]
  AND ({label_filter})
WITH e, count(c) AS primary_claim_count,
     [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type
WITH collect({{
       entity_id: e.entity_id,
       name: e.name,
       entity_type: entity_type,
       primary_claim_count: primary_claim_count
     }}) AS rows,
     percentileCont(primary_claim_count, 0.5) AS coverage_median
UNWIND rows AS row
WITH row, coverage_median
WHERE row.primary_claim_count <= $thin_threshold
RETURN row.entity_id AS entity_id, row.name AS name, row.entity_type AS entity_type,
       row.primary_claim_count AS primary_claim_count, coverage_median
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
  AND {_CLAIM_ACTIVE}
  AND type(rel) IN [{rel_list}]
  AND ({label_filter})
WITH e, count(c) AS geo_claim_count,
     [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type
WITH collect({{
       entity_id: e.entity_id,
       name: e.name,
       entity_type: entity_type,
       location_specific_claims: geo_claim_count
     }}) AS rows,
     percentileCont(geo_claim_count, 0.5) AS coverage_median
UNWIND rows AS row
WITH row, coverage_median
WHERE row.location_specific_claims <= $thin_threshold
RETURN row.entity_id AS entity_id, row.name AS name, row.entity_type AS entity_type,
       row.location_specific_claims AS location_specific_claims, coverage_median
ORDER BY location_specific_claims ASC
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
  AND {_CLAIM_ACTIVE}
RETURN e.entity_id AS entity_id, e.name AS name,
       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type,
       count(DISTINCT c) AS claim_count
"""

ENTITY_NEIGHBORHOOD_QUERY = """
MATCH (d:Document)-[:PROCESSED_BY]->(:ExtractionRun)-[:PRODUCED]->(c:Claim)-[r1]->(e1:Entity {entity_id: $entity_id})
WHERE d.institution_id IN $institution_ids AND d.deleted_at IS NULL
  AND d.access_level IN $permitted_levels
  AND {_CLAIM_ACTIVE}
  AND (
    ($year_min IS NULL AND $year_max IS NULL)
    OR NOT EXISTS { (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(:Year) }
    OR EXISTS {
      MATCH (c)-[:SUPPORTS]->(:Observation)-[:IN_YEAR]->(yr:Year)
      WHERE ($year_min IS NULL OR yr.year >= $year_min)
        AND ($year_max IS NULL OR yr.year <= $year_max)
    }
  )
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


# Expand {_DOC_ACCESS} and {_CLAIM_ACTIVE} tokens in every canonical-pattern
# query template at module-load time. Templates without tokens pass through
# unchanged. Done here once so individual templates stay readable as plain
# triple-quoted strings; the expanded text is byte-identical to the previous
# inlined predicates.
for _qname in list(globals()):
    _qval = globals().get(_qname)
    if (
        _qname.endswith("_QUERY")
        and isinstance(_qval, str)
        and ("{_DOC_ACCESS}" in _qval or "{_CLAIM_ACTIVE}" in _qval)
    ):
        globals()[_qname] = _finalize(_qval)
del _qname, _qval
