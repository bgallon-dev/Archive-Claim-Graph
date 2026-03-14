from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..claim_contract import CLAIM_ENTITY_RELATION_PRECEDENCE, LEGACY_ABOUT_RELATION
from ..models import SemanticBundle, StructureBundle

CLAIM_ENTITY_RELATION_CYPHER = ", ".join(
    f"'{relation}'" for relation in (LEGACY_ABOUT_RELATION, *CLAIM_ENTITY_RELATION_PRECEDENCE)
)

LATEST_VIEW_CYPHER = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, max(r.run_timestamp) AS latest_ts
MATCH (d)-[:PROCESSED_BY]->(latest:ExtractionRun {run_timestamp: latest_ts})
MATCH (p:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: latest.run_id})
OPTIONAL MATCH (c)-[:HAS_MEASUREMENT]->(m:Measurement)
OPTIONAL MATCH (c)-[cer]->(e:Entity)
WHERE type(cer) IN [""" + CLAIM_ENTITY_RELATION_CYPHER + """]
RETURN d.doc_id AS doc_id,
       latest.run_id AS run_id,
       c.claim_id AS claim_id,
       c.claim_type AS claim_type,
       p.paragraph_id AS paragraph_id,
       collect(DISTINCT m.name) AS measurement_names,
       collect(DISTINCT e.normalized_form) AS entities,
       collect(DISTINCT CASE WHEN e IS NULL THEN NULL ELSE {name: e.normalized_form, relation_type: type(cer)} END) AS entity_links
"""

AUDIT_VIEW_CYPHER = """
MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun)
MATCH (p:Paragraph)-[:HAS_CLAIM]->(c:Claim {run_id: r.run_id})
OPTIONAL MATCH (c)-[cer]->(e:Entity)
WHERE type(cer) IN [""" + CLAIM_ENTITY_RELATION_CYPHER + """]
RETURN d.doc_id AS doc_id,
       r.run_id AS run_id,
       r.run_timestamp AS run_timestamp,
       c.claim_id AS claim_id,
       c.claim_type AS claim_type,
       c.extraction_confidence AS extraction_confidence,
       c.source_sentence AS source_sentence,
       c.evidence_start AS evidence_start,
       c.evidence_end AS evidence_end,
       collect(DISTINCT CASE WHEN e IS NULL THEN NULL ELSE {name: e.normalized_form, relation_type: type(cer)} END) AS entity_links
ORDER BY d.doc_id, r.run_timestamp, c.claim_id
"""


def build_latest_view(pairs: list[tuple[StructureBundle, SemanticBundle]]) -> list[dict[str, Any]]:
    latest_by_doc: dict[str, tuple[StructureBundle, SemanticBundle]] = {}
    for structure, semantic in pairs:
        doc_id = structure.document.doc_id
        existing = latest_by_doc.get(doc_id)
        if not existing or semantic.extraction_run.run_timestamp > existing[1].extraction_run.run_timestamp:
            latest_by_doc[doc_id] = (structure, semantic)

    rows: list[dict[str, Any]] = []
    for structure, semantic in latest_by_doc.values():
        measures_by_claim: dict[str, list[str]] = defaultdict(list)
        entities_by_claim: dict[str, list[str]] = defaultdict(list)
        entity_links_by_claim: dict[str, list[dict[str, str]]] = defaultdict(list)
        entity_lookup = {entity.entity_id: entity for entity in semantic.entities}

        for measurement in semantic.measurements:
            measures_by_claim[measurement.claim_id].append(measurement.name)
        for link in semantic.claim_entity_links:
            entity = entity_lookup.get(link.entity_id)
            if entity:
                entities_by_claim[link.claim_id].append(entity.normalized_form)
                entity_links_by_claim[link.claim_id].append(
                    {"name": entity.normalized_form, "relation_type": link.relation_type}
                )

        for claim in semantic.claims:
            entity_links = sorted(
                {
                    (item["name"], item["relation_type"]): item
                    for item in entity_links_by_claim.get(claim.claim_id, [])
                }.values(),
                key=lambda row: (row["relation_type"], row["name"]),
            )
            rows.append(
                {
                    "doc_id": structure.document.doc_id,
                    "run_id": semantic.extraction_run.run_id,
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "paragraph_id": claim.paragraph_id,
                    "measurement_names": sorted(set(measures_by_claim.get(claim.claim_id, []))),
                    "entities": sorted(set(entities_by_claim.get(claim.claim_id, []))),
                    "entity_links": entity_links,
                }
            )
    return rows


def build_audit_view(pairs: list[tuple[StructureBundle, SemanticBundle]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for structure, semantic in pairs:
        entity_lookup = {entity.entity_id: entity for entity in semantic.entities}
        entity_links_by_claim: dict[str, list[dict[str, str]]] = defaultdict(list)
        for link in semantic.claim_entity_links:
            entity = entity_lookup.get(link.entity_id)
            if entity:
                entity_links_by_claim[link.claim_id].append(
                    {"name": entity.normalized_form, "relation_type": link.relation_type}
                )

        for claim in semantic.claims:
            entity_links = sorted(
                {
                    (item["name"], item["relation_type"]): item
                    for item in entity_links_by_claim.get(claim.claim_id, [])
                }.values(),
                key=lambda row: (row["relation_type"], row["name"]),
            )
            rows.append(
                {
                    "doc_id": structure.document.doc_id,
                    "run_id": semantic.extraction_run.run_id,
                    "run_timestamp": semantic.extraction_run.run_timestamp,
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "extraction_confidence": claim.extraction_confidence,
                    "source_sentence": claim.source_sentence,
                    "normalized_sentence": claim.normalized_sentence,
                    "evidence_start": claim.evidence_start,
                    "evidence_end": claim.evidence_end,
                    "entity_links": entity_links,
                }
            )
    rows.sort(key=lambda row: (row["doc_id"], row["run_timestamp"], row["claim_id"]))
    return rows
