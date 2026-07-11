from __future__ import annotations

from collections import defaultdict
from typing import Any

from gemynd.core.graph.cypher import (
    INDEX_STATEMENTS,
    build_constraint_statements,
    build_id_constraints,
)
from gemynd.core.models import SemanticBundle, StructureBundle

try:
    import neo4j as neo4j_pkg
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    neo4j_pkg = None  # type: ignore[assignment]
    GraphDatabase = None  # type: ignore[assignment]


class Neo4jGraphWriter:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        trust_mode: str = "system",
        ca_cert_path: str | None = None,
        *,
        entity_labels: frozenset[str],
        observation_role_edges: dict[str, tuple[str, str]] | None = None,
        event_role_edges: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        if GraphDatabase is None:  # pragma: no cover - optional dependency
            raise RuntimeError("neo4j package is not installed. Install with: pip install -e .[neo4j]")
        driver_kwargs = build_driver_kwargs(uri=uri, trust_mode=trust_mode, ca_cert_path=ca_cert_path)
        self._driver = GraphDatabase.driver(uri, auth=(user, password), **driver_kwargs)
        self._database = database
        self._uri = uri
        self._entity_labels = frozenset(entity_labels)
        self._id_constraints = build_id_constraints(self._entity_labels)
        self._schema_statements = (
            build_constraint_statements(self._entity_labels) + INDEX_STATEMENTS
        )
        self._observation_role_edges = dict(observation_role_edges or {})
        self._event_role_edges = dict(event_role_edges or {})

        try:
            self._driver.verify_connectivity()
        except Exception as exc:  # pragma: no cover - requires neo4j runtime
            message = format_connection_error(uri, exc)
            raise RuntimeError(message) from exc

    def close(self) -> None:
        self._driver.close()

    def create_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            for statement in self._schema_statements:
                session.run(statement).consume()

    def load_structure(self, structure: StructureBundle) -> None:
        self._upsert_nodes(
            label="Document",
            id_key="doc_id",
            rows=[{"id": structure.document.doc_id, "props": structure.document.to_dict()}],
        )
        self._upsert_nodes(
            label="Page",
            id_key="page_id",
            rows=[{"id": page.page_id, "props": page.node_props()} for page in structure.pages],
        )
        self._upsert_nodes(
            label="Section",
            id_key="section_id",
            rows=[{"id": section.section_id, "props": section.node_props()} for section in structure.sections],
        )
        self._upsert_nodes(
            label="Paragraph",
            id_key="paragraph_id",
            rows=[{"id": paragraph.paragraph_id, "props": paragraph.node_props()} for paragraph in structure.paragraphs],
        )
        self._upsert_nodes(
            label="Annotation",
            id_key="annotation_id",
            rows=[{"id": annotation.annotation_id, "props": annotation.node_props()} for annotation in structure.annotations],
        )

        self._upsert_relationships(
            "Document",
            "doc_id",
            "HAS_PAGE",
            "Page",
            "page_id",
            [
                {
                    "start_id": structure.document.doc_id,
                    "end_id": page.page_id,
                    "props": {},
                }
                for page in structure.pages
            ],
        )

        page_section_rows: list[dict[str, Any]] = []
        for section in structure.sections:
            for page in structure.pages:
                if section.page_start <= page.page_number <= section.page_end:
                    page_section_rows.append({"start_id": page.page_id, "end_id": section.section_id, "props": {}})
        self._upsert_relationships("Page", "page_id", "HAS_SECTION", "Section", "section_id", page_section_rows)

        section_paragraph_rows = [
            {"start_id": paragraph.section_id, "end_id": paragraph.paragraph_id, "props": {}}
            for paragraph in structure.paragraphs
            if paragraph.section_id
        ]
        self._upsert_relationships("Section", "section_id", "HAS_PARAGRAPH", "Paragraph", "paragraph_id", section_paragraph_rows)

        sorted_paragraphs = sorted(structure.paragraphs, key=lambda row: row.paragraph_index)
        next_rows = []
        for idx in range(1, len(sorted_paragraphs)):
            previous = sorted_paragraphs[idx - 1]
            current = sorted_paragraphs[idx]
            next_rows.append({"start_id": previous.paragraph_id, "end_id": current.paragraph_id, "props": {}})
        self._upsert_relationships("Paragraph", "paragraph_id", "NEXT", "Paragraph", "paragraph_id", next_rows)

        self._upsert_relationships(
            "Page",
            "page_id",
            "HAS_ANNOTATION",
            "Annotation",
            "annotation_id",
            [
                {
                    "start_id": annotation.page_id,
                    "end_id": annotation.annotation_id,
                    "props": {},
                }
                for annotation in structure.annotations
            ],
        )

    def load_semantic(self, structure: StructureBundle, semantic: SemanticBundle) -> None:
        run = semantic.extraction_run
        self._upsert_nodes(
            label="ExtractionRun",
            id_key="run_id",
            rows=[{"id": run.run_id, "props": run.to_dict()}],
        )
        self._upsert_relationships(
            "Document",
            "doc_id",
            "PROCESSED_BY",
            "ExtractionRun",
            "run_id",
            [
                {
                    "start_id": structure.document.doc_id,
                    "end_id": run.run_id,
                    "props": {"run_id": run.run_id, "run_timestamp": run.run_timestamp},
                }
            ],
        )

        self._upsert_nodes("Claim", "claim_id", [{"id": row.claim_id, "props": row.node_props()} for row in semantic.claims])
        self._upsert_relationships(
            "Paragraph",
            "paragraph_id",
            "HAS_CLAIM",
            "Claim",
            "claim_id",
            [{"start_id": row.paragraph_id, "end_id": row.claim_id, "props": {"run_id": row.run_id}} for row in semantic.claims],
        )
        self._upsert_relationships(
            "Claim",
            "claim_id",
            "EVIDENCED_BY",
            "Paragraph",
            "paragraph_id",
            [{"start_id": row.claim_id, "end_id": row.paragraph_id, "props": {"run_id": row.run_id}} for row in semantic.claims],
        )
        self._upsert_relationships(
            "ExtractionRun",
            "run_id",
            "PRODUCED",
            "Claim",
            "claim_id",
            [{"start_id": row.run_id, "end_id": row.claim_id, "props": {"run_id": row.run_id}} for row in semantic.claims],
        )

        # Measurements claimed by observations get Observation->HAS_MEASUREMENT instead
        obs_claimed_measurement_ids = {link.measurement_id for link in semantic.observation_measurement_links}

        self._upsert_nodes(
            "Measurement",
            "measurement_id",
            [{"id": row.measurement_id, "props": row.node_props()} for row in semantic.measurements],
        )
        non_obs_measurements = [row for row in semantic.measurements if row.measurement_id not in obs_claimed_measurement_ids]
        self._upsert_relationships(
            "Claim",
            "claim_id",
            "HAS_MEASUREMENT",
            "Measurement",
            "measurement_id",
            [
                {
                    "start_id": row.claim_id,
                    "end_id": row.measurement_id,
                    "props": {"run_id": row.run_id},
                }
                for row in non_obs_measurements
            ],
        )

        self._upsert_nodes("Mention", "mention_id", [{"id": row.mention_id, "props": row.node_props()} for row in semantic.mentions])
        self._upsert_relationships(
            "Paragraph",
            "paragraph_id",
            "CONTAINS_MENTION",
            "Mention",
            "mention_id",
            [
                {
                    "start_id": row.paragraph_id,
                    "end_id": row.mention_id,
                    "props": {"run_id": row.run_id},
                }
                for row in semantic.mentions
            ],
        )

        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in semantic.entities:
            by_label[entity.entity_type].append({"id": entity.entity_id, "props": entity.node_props()})
        for label, rows in by_label.items():
            self._upsert_nodes(label=label, id_key="entity_id", rows=rows)
            if label in self._entity_labels:
                self._add_labels(source_label=label, id_key="entity_id", new_label="Entity", rows=rows)
        if "Refuge" in by_label:
            self._add_labels(source_label="Refuge", id_key="entity_id", new_label="Place", rows=by_label["Refuge"])

        for relation_type in ("REFERS_TO", "POSSIBLY_REFERS_TO"):
            rows: list[dict[str, Any]] = []
            for row in semantic.entity_resolutions:
                if row.relation_type == relation_type:
                    target = next((entity for entity in semantic.entities if entity.entity_id == row.entity_id), None)
                    if not target:
                        continue
                    rows.append({"start_id": row.mention_id, "end_id": row.entity_id, "props": {"match_score": row.match_score}})
                    self._upsert_relationships(
                        "Mention",
                        "mention_id",
                        relation_type,
                        target.entity_type,
                        "entity_id",
                        rows,
                    )
                    rows = []

        # Human review confirmation/refutation edges
        entity_label_lookup_conf = {row.entity_id: row.entity_type for row in semantic.entities}
        for conf_type in ("CONFIRMED_AS", "REFUTED_BY"):
            conf_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for conf in semantic.entity_resolution_confirmations:
                if conf.relation_type == conf_type:
                    label = entity_label_lookup_conf.get(conf.entity_id)
                    if label:
                        conf_by_label[label].append({
                            "start_id": conf.mention_id,
                            "end_id": conf.entity_id,
                            "props": {"confirmed_by": conf.confirmed_by, "confirmed_at": conf.confirmed_at},
                        })
            for label, conf_rows in conf_by_label.items():
                self._upsert_relationships("Mention", "mention_id", conf_type, label, "entity_id", conf_rows)

        claim_entity_by_relation_and_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        claim_location_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        entity_label_lookup = {row.entity_id: row.entity_type for row in semantic.entities}
        for row in semantic.claim_entity_links:
            label = entity_label_lookup.get(row.entity_id)
            if label:
                claim_entity_by_relation_and_label[(row.relation_type, label)].append(
                    {"start_id": row.claim_id, "end_id": row.entity_id, "props": {}}
                )
        for row in semantic.claim_location_links:
            label = entity_label_lookup.get(row.entity_id)
            if label:
                claim_location_by_label[label].append({"start_id": row.claim_id, "end_id": row.entity_id, "props": {}})

        for (relation_type, label), rows in claim_entity_by_relation_and_label.items():
            self._upsert_relationships("Claim", "claim_id", relation_type, label, "entity_id", rows)
        for label, rows in claim_location_by_label.items():
            self._upsert_relationships("Claim", "claim_id", "OCCURRED_AT", label, "entity_id", rows)

        concept_ids_needed = {link.concept_id for link in semantic.claim_concept_links}
        if concept_ids_needed:
            self._upsert_nodes(
                "Concept",
                "concept_id",
                [{"id": cid, "props": {"concept_id": cid}} for cid in concept_ids_needed],
            )
        self._upsert_relationships(
            "Claim", "claim_id", "EXPRESSES", "Concept", "concept_id",
            [
                {
                    "start_id": link.claim_id,
                    "end_id": link.concept_id,
                    "props": {
                        "confidence": link.confidence,
                        "matched_rule": link.matched_rule,
                    },
                }
                for link in semantic.claim_concept_links
            ],
        )

        self._upsert_relationships(
            "Claim",
            "claim_id",
            "OCCURRED_DURING",
            "Period",
            "entity_id",
            [{"start_id": row.claim_id, "end_id": row.period_id, "props": {}} for row in semantic.claim_period_links],
        )
        anchor_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in semantic.document_anchor_links:
            anchor_label = entity_label_lookup.get(row.anchor_entity_id, "Entity")
            anchor_rows_by_key[(row.relation_type, anchor_label)].append(
                {"start_id": row.doc_id, "end_id": row.anchor_entity_id, "props": {}}
            )
        for (rel_type, end_label), rows in anchor_rows_by_key.items():
            self._upsert_relationships(
                "Document", "doc_id", rel_type, end_label, "entity_id", rows,
            )
        self._upsert_relationships(
            "Document",
            "doc_id",
            "COVERS_PERIOD",
            "Period",
            "entity_id",
            [{"start_id": row.doc_id, "end_id": row.period_id, "props": {}} for row in semantic.document_period_links],
        )
        self._upsert_relationships(
            "Document",
            "doc_id",
            "SIGNED_BY",
            "Person",
            "entity_id",
            [{"start_id": row.doc_id, "end_id": row.person_id, "props": {}} for row in semantic.document_signed_by_links],
        )
        self._upsert_relationships(
            "Person",
            "entity_id",
            "AFFILIATED_WITH",
            "Organization",
            "entity_id",
            [
                {
                    "start_id": row.person_id,
                    "end_id": row.organization_id,
                    "props": {},
                }
                for row in semantic.person_affiliation_links
            ],
        )

        # --- Observation layer ---
        self._upsert_nodes(
            "Observation",
            "observation_id",
            [{"id": obs.observation_id, "props": obs.node_props()} for obs in semantic.observations],
        )
        self._upsert_relationships(
            "Claim", "claim_id", "SUPPORTS", "Observation", "observation_id",
            [{"start_id": obs.claim_id, "end_id": obs.observation_id, "props": {"run_id": obs.run_id}} for obs in semantic.observations],
        )
        self._upsert_relationships(
            "Observation", "observation_id", "EVIDENCED_BY", "Paragraph", "paragraph_id",
            [{"start_id": obs.observation_id, "end_id": obs.paragraph_id, "props": {"run_id": obs.run_id}} for obs in semantic.observations],
        )
        # Observation -> role-entity relationships (batched by edge type and target label)
        obs_role_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for obs in semantic.observations:
            for role, entity_id in obs.role_entities.items():
                edge_spec = self._observation_role_edges.get(role)
                if not edge_spec or not entity_id:
                    continue
                target_label, edge_type = edge_spec
                obs_role_rows_by_key[(edge_type, target_label)].append(
                    {"start_id": obs.observation_id, "end_id": entity_id, "props": {}}
                )
        for (edge_type, target_label), rows in obs_role_rows_by_key.items():
            self._upsert_relationships(
                "Observation", "observation_id", edge_type, target_label, "entity_id", rows,
            )

        obs_place_rows = [{"start_id": obs.observation_id, "end_id": obs.place_id, "props": {}} for obs in semantic.observations if obs.place_id]
        self._upsert_relationships("Observation", "observation_id", "AT_PLACE", "Place", "entity_id", obs_place_rows)

        obs_period_rows = [{"start_id": obs.observation_id, "end_id": obs.period_id, "props": {}} for obs in semantic.observations if obs.period_id]
        self._upsert_relationships("Observation", "observation_id", "DURING", "Period", "entity_id", obs_period_rows)

        obs_year_rows = [{"start_id": obs.observation_id, "end_id": obs.year_id, "props": {}} for obs in semantic.observations if obs.year_id]
        self._upsert_relationships("Observation", "observation_id", "IN_YEAR", "Year", "year_id", obs_year_rows)

        self._upsert_relationships(
            "Observation", "observation_id", "HAS_MEASUREMENT", "Measurement", "measurement_id",
            [{"start_id": link.observation_id, "end_id": link.measurement_id, "props": {"run_id": run.run_id}} for link in semantic.observation_measurement_links],
        )

        # --- Event layer ---
        self._upsert_nodes(
            "Event",
            "event_id",
            [{"id": evt.event_id, "props": evt.node_props()} for evt in semantic.events],
        )
        self._upsert_relationships(
            "Claim", "claim_id", "TRIGGERED", "Event", "event_id",
            [{"start_id": evt.claim_id, "end_id": evt.event_id, "props": {"run_id": evt.run_id}} for evt in semantic.events],
        )
        self._upsert_relationships(
            "Event", "event_id", "SOURCED_FROM", "Paragraph", "paragraph_id",
            [{"start_id": evt.event_id, "end_id": evt.paragraph_id, "props": {"run_id": evt.run_id}} for evt in semantic.events],
        )
        evt_role_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for evt in semantic.events:
            for role, entity_id in evt.role_entities.items():
                edge_spec = self._event_role_edges.get(role)
                if not edge_spec or not entity_id:
                    continue
                target_label, edge_type = edge_spec
                evt_role_rows_by_key[(edge_type, target_label)].append(
                    {"start_id": evt.event_id, "end_id": entity_id, "props": {}}
                )
        for (edge_type, target_label), rows in evt_role_rows_by_key.items():
            self._upsert_relationships(
                "Event", "event_id", edge_type, target_label, "entity_id", rows,
            )

        evt_place_rows = [{"start_id": evt.event_id, "end_id": evt.place_id, "props": {}} for evt in semantic.events if evt.place_id]
        self._upsert_relationships("Event", "event_id", "OCCURRED_AT", "Place", "entity_id", evt_place_rows)

        evt_period_rows = [{"start_id": evt.event_id, "end_id": evt.period_id, "props": {}} for evt in semantic.events if evt.period_id]
        self._upsert_relationships("Event", "event_id", "DURING", "Period", "entity_id", evt_period_rows)

        evt_year_rows = [{"start_id": evt.event_id, "end_id": evt.year_id, "props": {}} for evt in semantic.events if evt.year_id]
        self._upsert_relationships("Event", "event_id", "IN_YEAR", "Year", "year_id", evt_year_rows)

        self._upsert_relationships(
            "Event", "event_id", "PRODUCED", "Observation", "observation_id",
            [{"start_id": lnk.event_id, "end_id": lnk.observation_id, "props": {}} for lnk in semantic.event_observation_links],
        )
        self._upsert_relationships(
            "Event", "event_id", "PRODUCED_MEASUREMENT", "Measurement", "measurement_id",
            [{"start_id": lnk.event_id, "end_id": lnk.measurement_id, "props": {"run_id": run.run_id}} for lnk in semantic.event_measurement_links],
        )

        # --- Year layer ---
        self._upsert_nodes("Year", "year_id", [{"id": y.year_id, "props": y.to_dict()} for y in semantic.years])
        self._upsert_relationships(
            "Document", "doc_id", "COVERS_YEAR", "Year", "year_id",
            [{"start_id": link.doc_id, "end_id": link.year_id, "props": {}} for link in semantic.document_year_links],
        )

        # --- Entity-hierarchy edges (child entity -> anchor entity) ---
        hierarchy_rows_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for link in semantic.entity_hierarchy_links:
            child_label = entity_label_lookup.get(link.child_entity_id, "Entity")
            parent_label = entity_label_lookup.get(link.parent_entity_id, "Entity")
            hierarchy_rows_by_key[(child_label, link.relation_type, parent_label)].append(
                {"start_id": link.child_entity_id, "end_id": link.parent_entity_id, "props": {}}
            )
        for (child_label, rel_type, parent_label), rows in hierarchy_rows_by_key.items():
            self._upsert_relationships(
                child_label, "entity_id", rel_type, parent_label, "entity_id", rows,
            )

        # --- Annotation targeting ---
        for annotation in structure.annotations:
            if annotation.target_claim_id:
                self._upsert_relationships(
                    "Annotation", "annotation_id", "ABOUT", "Claim", "claim_id",
                    [{"start_id": annotation.annotation_id, "end_id": annotation.target_claim_id, "props": {}}],
                )
            if annotation.target_measurement_id:
                rel_type = "CORRECTS" if annotation.corrects_measurement else "ABOUT"
                self._upsert_relationships(
                    "Annotation", "annotation_id", rel_type, "Measurement", "measurement_id",
                    [{"start_id": annotation.annotation_id, "end_id": annotation.target_measurement_id, "props": {}}],
                )

    def _upsert_nodes(self, label: str, id_key: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{label} {{{id_key}: row.id}}) "
            "SET n += row.props"
        )
        with self._driver.session(database=self._database) as session:
            session.run(query, rows=rows).consume()

    def _add_labels(self, source_label: str, id_key: str, new_label: str, rows: list[dict[str, Any]]) -> None:
        """Add a secondary label to existing nodes matched by *source_label*."""
        if not rows:
            return
        query = (
            f"UNWIND $rows AS row "
            f"MATCH (n:{source_label} {{{id_key}: row.id}}) "
            f"SET n:{new_label}"
        )
        with self._driver.session(database=self._database) as session:
            session.run(query, rows=rows).consume()

    def _upsert_relationships(
        self,
        start_label: str,
        start_key: str,
        rel_type: str,
        end_label: str,
        end_key: str,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return
        query = (
            "UNWIND $rows AS row "
            f"MATCH (a:{start_label} {{{start_key}: row.start_id}}) "
            f"MATCH (b:{end_label} {{{end_key}: row.end_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r += row.props"
        )
        with self._driver.session(database=self._database) as session:
            session.run(query, rows=rows).consume()


def build_driver_kwargs(uri: str, trust_mode: str, ca_cert_path: str | None) -> dict[str, Any]:
    if neo4j_pkg is None:  # pragma: no cover - optional dependency
        return {}

    normalized_uri = uri.lower().strip()
    normalized_mode = (trust_mode or "system").strip().lower()
    base_schemes = ("bolt://", "neo4j://")
    if not normalized_uri.startswith(base_schemes):
        return {}

    if normalized_mode == "all":
        return {"trusted_certificates": neo4j_pkg.TrustAll()}
    if normalized_mode == "custom":
        if not ca_cert_path:
            raise ValueError("NEO4J_CA_CERT is required when trust mode is 'custom'.")
        return {"trusted_certificates": neo4j_pkg.TrustCustomCAs(ca_cert_path)}
    if normalized_mode in {"system", ""}:
        return {"trusted_certificates": neo4j_pkg.TrustSystemCAs()}
    raise ValueError("NEO4J_TRUST must be one of: system, all, custom.")


def format_connection_error(uri: str, exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "ssl" in lowered or "certificate" in lowered or "self-signed" in lowered:
        return (
            f"Neo4j TLS handshake failed for URI '{uri}'. "
            "If this is a self-signed single instance, use 'bolt+ssc://host:7687'. "
            "If using 'bolt://' or 'neo4j://', set NEO4J_TRUST=all (dev only) or "
            "NEO4J_TRUST=custom with NEO4J_CA_CERT path."
        )
    return f"Neo4j connection failed for URI '{uri}': {text}"
