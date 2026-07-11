from __future__ import annotations

from collections import defaultdict
from typing import Any

from gemynd.core.graph.cypher import (
    INDEX_STATEMENTS,
    build_constraint_statements,
    build_id_constraints,
)
from gemynd.core.models import SemanticBundle, StructureBundle


class InMemoryGraphWriter:
    """In-memory graph backend used for testing and development.

    ``node_store`` and ``rel_store`` are plain Python dicts that stage graph
    data; they are *not* Neo4j node properties or relationship objects.
    ``node_store`` is keyed ``{label: {node_id: props}}``.
    ``rel_store`` is keyed ``(start_label, start_id, rel_type, end_label, end_id, run_marker)``.
    """

    def __init__(
        self,
        *,
        entity_labels: frozenset[str],
        observation_role_edges: dict[str, tuple[str, str]] | None = None,
        event_role_edges: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._entity_labels = frozenset(entity_labels)
        self._id_constraints = build_id_constraints(self._entity_labels)
        self._schema_statements = (
            build_constraint_statements(self._entity_labels) + INDEX_STATEMENTS
        )
        self._observation_role_edges = dict(observation_role_edges or {})
        self._event_role_edges = dict(event_role_edges or {})
        self.node_store: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.rel_store: dict[tuple[str, str, str, str, str, str | None], dict[str, Any]] = {}
        self.schema_statements_executed: list[str] = []

    def create_schema(self) -> None:
        self.schema_statements_executed = list(self._schema_statements)

    def load_structure(self, structure: StructureBundle) -> None:
        self._merge_node("Document", "doc_id", structure.document.doc_id, structure.document.to_dict())

        for page in structure.pages:
            self._merge_node("Page", "page_id", page.page_id, page.node_props())
            self._merge_rel("Document", structure.document.doc_id, "HAS_PAGE", "Page", page.page_id)

        for section in structure.sections:
            self._merge_node("Section", "section_id", section.section_id, section.node_props())
            for page in structure.pages:
                if section.page_start <= page.page_number <= section.page_end:
                    self._merge_rel("Page", page.page_id, "HAS_SECTION", "Section", section.section_id)

        sorted_paragraphs = sorted(structure.paragraphs, key=lambda item: item.paragraph_index)
        for idx, paragraph in enumerate(sorted_paragraphs):
            self._merge_node("Paragraph", "paragraph_id", paragraph.paragraph_id, paragraph.node_props())
            if paragraph.section_id:
                self._merge_rel("Section", paragraph.section_id, "HAS_PARAGRAPH", "Paragraph", paragraph.paragraph_id)
            if idx > 0:
                previous = sorted_paragraphs[idx - 1]
                self._merge_rel("Paragraph", previous.paragraph_id, "NEXT", "Paragraph", paragraph.paragraph_id)

        for annotation in structure.annotations:
            self._merge_node("Annotation", "annotation_id", annotation.annotation_id, annotation.node_props())
            self._merge_rel("Page", annotation.page_id, "HAS_ANNOTATION", "Annotation", annotation.annotation_id)

    def load_semantic(self, structure: StructureBundle, semantic: SemanticBundle) -> None:
        run = semantic.extraction_run
        self._merge_node("ExtractionRun", "run_id", run.run_id, run.to_dict())
        self._merge_rel(
            "Document",
            structure.document.doc_id,
            "PROCESSED_BY",
            "ExtractionRun",
            run.run_id,
            props={"run_id": run.run_id, "run_timestamp": run.run_timestamp},
        )

        for claim in semantic.claims:
            self._merge_node("Claim", "claim_id", claim.claim_id, claim.node_props())
            self._merge_rel("Paragraph", claim.paragraph_id, "HAS_CLAIM", "Claim", claim.claim_id, props={"run_id": claim.run_id})
            self._merge_rel("Claim", claim.claim_id, "EVIDENCED_BY", "Paragraph", claim.paragraph_id, props={"run_id": claim.run_id})
            self._merge_rel("Claim", claim.claim_id, "EXTRACTED_IN", "ExtractionRun", claim.run_id, props={"run_id": claim.run_id})

        # Measurements claimed by observations get Observation->HAS_MEASUREMENT instead
        obs_claimed_measurement_ids = {link.measurement_id for link in semantic.observation_measurement_links}

        for measurement in semantic.measurements:
            self._merge_node("Measurement", "measurement_id", measurement.measurement_id, measurement.node_props())
            if measurement.measurement_id not in obs_claimed_measurement_ids:
                self._merge_rel(
                    "Claim",
                    measurement.claim_id,
                    "HAS_MEASUREMENT",
                    "Measurement",
                    measurement.measurement_id,
                    props={"run_id": measurement.run_id},
                )

        for mention in semantic.mentions:
            self._merge_node("Mention", "mention_id", mention.mention_id, mention.node_props())
            self._merge_rel(
                "Paragraph",
                mention.paragraph_id,
                "CONTAINS_MENTION",
                "Mention",
                mention.mention_id,
                props={"run_id": mention.run_id},
            )

        entity_type_lookup: dict[str, str] = {
            entity.entity_id: entity.entity_type for entity in semantic.entities
        }

        for entity in semantic.entities:
            props = entity.node_props()
            self._merge_node(entity.entity_type, "entity_id", entity.entity_id, props)
            if entity.entity_type in self._entity_labels:
                self._merge_node("Entity", "entity_id", entity.entity_id, props)
            if entity.entity_type == "Refuge":
                self._merge_node("Place", "entity_id", entity.entity_id, props)

        for resolution in semantic.entity_resolutions:
            end_label = self._find_entity_label(resolution.entity_id)
            if not end_label:
                continue
            self._merge_rel(
                "Mention",
                resolution.mention_id,
                resolution.relation_type,
                end_label,
                resolution.entity_id,
                props={"match_score": resolution.match_score},
            )

        for conf in semantic.entity_resolution_confirmations:
            end_label = self._find_entity_label(conf.entity_id)
            if end_label:
                self._merge_rel(
                    "Mention",
                    conf.mention_id,
                    conf.relation_type,
                    end_label,
                    conf.entity_id,
                    props={"confirmed_by": conf.confirmed_by, "confirmed_at": conf.confirmed_at},
                )

        for link in semantic.claim_entity_links:
            end_label = self._find_entity_label(link.entity_id)
            if end_label:
                self._merge_rel("Claim", link.claim_id, link.relation_type, end_label, link.entity_id)

        for link in semantic.claim_location_links:
            end_label = self._find_entity_label(link.entity_id)
            if end_label:
                self._merge_rel("Claim", link.claim_id, "OCCURRED_AT", end_label, link.entity_id)

        for link in semantic.claim_concept_links:
            self._merge_node("Concept", "concept_id", link.concept_id, {"concept_id": link.concept_id})
            self._merge_rel(
                "Claim", link.claim_id,
                "EXPRESSES",
                "Concept", link.concept_id,
                props={"confidence": link.confidence, "matched_rule": link.matched_rule},
            )

        for link in semantic.claim_period_links:
            self._merge_rel("Claim", link.claim_id, "OCCURRED_DURING", "Period", link.period_id)

        for link in semantic.document_anchor_links:
            end_label = entity_type_lookup.get(link.anchor_entity_id) or "Entity"
            self._merge_rel(
                "Document", link.doc_id, link.relation_type, end_label, link.anchor_entity_id,
            )

        for link in semantic.document_period_links:
            self._merge_rel("Document", link.doc_id, "COVERS_PERIOD", "Period", link.period_id)

        for link in semantic.document_signed_by_links:
            self._merge_rel("Document", link.doc_id, "SIGNED_BY", "Person", link.person_id)

        for link in semantic.person_affiliation_links:
            self._merge_rel("Person", link.person_id, "AFFILIATED_WITH", "Organization", link.organization_id)

        # --- Observation layer ---
        for obs in semantic.observations:
            self._merge_node("Observation", "observation_id", obs.observation_id, obs.node_props())
            self._merge_rel("Claim", obs.claim_id, "SUPPORTS", "Observation", obs.observation_id, props={"run_id": obs.run_id})
            self._merge_rel("Observation", obs.observation_id, "EVIDENCED_BY", "Paragraph", obs.paragraph_id, props={"run_id": obs.run_id})
            for role, entity_id in obs.role_entities.items():
                edge_spec = self._observation_role_edges.get(role)
                if not edge_spec or not entity_id:
                    continue
                target_label, edge_type = edge_spec
                self._merge_rel(
                    "Observation", obs.observation_id, edge_type, target_label, entity_id,
                )
            if obs.place_id:
                self._merge_rel("Observation", obs.observation_id, "AT_PLACE", "Place", obs.place_id)
            if obs.period_id:
                self._merge_rel("Observation", obs.observation_id, "DURING", "Period", obs.period_id)
            if obs.year_id:
                self._merge_rel("Observation", obs.observation_id, "IN_YEAR", "Year", obs.year_id)

        for link in semantic.observation_measurement_links:
            self._merge_rel("Observation", link.observation_id, "HAS_MEASUREMENT", "Measurement", link.measurement_id, props={"run_id": run.run_id})

        # --- Event layer ---
        for evt in semantic.events:
            self._merge_node("Event", "event_id", evt.event_id, evt.node_props())
            self._merge_rel("Claim", evt.claim_id, "TRIGGERED", "Event", evt.event_id, props={"run_id": evt.run_id})
            self._merge_rel("Event", evt.event_id, "SOURCED_FROM", "Paragraph", evt.paragraph_id, props={"run_id": evt.run_id})
            for role, entity_id in evt.role_entities.items():
                edge_spec = self._event_role_edges.get(role)
                if not edge_spec or not entity_id:
                    continue
                target_label, edge_type = edge_spec
                self._merge_rel(
                    "Event", evt.event_id, edge_type, target_label, entity_id,
                )
            if evt.place_id:
                self._merge_rel("Event", evt.event_id, "OCCURRED_AT", "Place", evt.place_id)
            if evt.period_id:
                self._merge_rel("Event", evt.event_id, "DURING", "Period", evt.period_id)
            if evt.year_id:
                self._merge_rel("Event", evt.event_id, "IN_YEAR", "Year", evt.year_id)

        for link in semantic.event_observation_links:
            self._merge_rel("Event", link.event_id, "PRODUCED", "Observation", link.observation_id)

        for link in semantic.event_measurement_links:
            self._merge_rel("Event", link.event_id, "PRODUCED_MEASUREMENT", "Measurement", link.measurement_id, props={"run_id": run.run_id})

        # --- Year layer ---
        for year in semantic.years:
            self._merge_node("Year", "year_id", year.year_id, year.to_dict())

        for link in semantic.document_year_links:
            self._merge_rel("Document", link.doc_id, "COVERS_YEAR", "Year", link.year_id)

        # --- Entity-hierarchy edges (child entity -> anchor entity) ---
        for link in semantic.entity_hierarchy_links:
            child_label = entity_type_lookup.get(link.child_entity_id) or "Entity"
            parent_label = entity_type_lookup.get(link.parent_entity_id) or "Entity"
            self._merge_rel(
                child_label, link.child_entity_id, link.relation_type,
                parent_label, link.parent_entity_id,
            )

        # --- Annotation targeting (semantic annotations) ---
        for annotation in structure.annotations:
            if annotation.target_claim_id:
                self._merge_rel("Annotation", annotation.annotation_id, "ABOUT", "Claim", annotation.target_claim_id)
            if annotation.target_measurement_id:
                rel_type = "CORRECTS" if annotation.corrects_measurement else "ABOUT"
                self._merge_rel("Annotation", annotation.annotation_id, rel_type, "Measurement", annotation.target_measurement_id)

    def _merge_node(self, label: str, id_key: str, node_id: str, props: dict[str, Any]) -> None:
        payload = dict(props)
        payload[id_key] = node_id
        if node_id in self.node_store[label]:
            self.node_store[label][node_id].update(payload)
        else:
            self.node_store[label][node_id] = payload

    def _merge_rel(
        self,
        start_label: str,
        start_id: str,
        rel_type: str,
        end_label: str,
        end_id: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        run_marker = None if props is None else str(props.get("run_id")) if props.get("run_id") else None
        key = (start_label, start_id, rel_type, end_label, end_id, run_marker)
        if key not in self.rel_store:
            self.rel_store[key] = dict(props or {})
        else:
            self.rel_store[key].update(props or {})

    def _find_entity_label(self, entity_id: str) -> str | None:
        for label in self._entity_labels:
            if entity_id in self.node_store.get(label, {}):
                return label
        return None

