from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

from ..models import SemanticBundle, StructureBundle
from .cypher import ID_CONSTRAINTS, SCHEMA_STATEMENTS

try:
    import neo4j as neo4j_pkg
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    neo4j_pkg = None  # type: ignore[assignment]
    GraphDatabase = None  # type: ignore[assignment]


class GraphWriter(Protocol):
    def create_schema(self) -> None:
        ...

    def load_structure(self, structure: StructureBundle) -> None:
        ...

    def load_semantic(self, structure: StructureBundle, semantic: SemanticBundle) -> None:
        ...


DOMAIN_LABELS = {"Refuge", "Place", "Person", "Organization", "Species", "Activity", "Period"}


class InMemoryGraphWriter:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.relationships: dict[tuple[str, str, str, str, str, str | None], dict[str, Any]] = {}
        self.schema_statements_executed: list[str] = []

    def create_schema(self) -> None:
        self.schema_statements_executed = list(SCHEMA_STATEMENTS)

    def load_structure(self, structure: StructureBundle) -> None:
        self._merge_node("Document", "doc_id", structure.document.doc_id, structure.document.to_dict())

        for page in structure.pages:
            self._merge_node("Page", "page_id", page.page_id, page.to_dict())
            self._merge_rel("Document", structure.document.doc_id, "HAS_PAGE", "Page", page.page_id)

        for section in structure.sections:
            self._merge_node("Section", "section_id", section.section_id, section.to_dict())
            for page in structure.pages:
                if section.page_start <= page.page_number <= section.page_end:
                    self._merge_rel("Page", page.page_id, "HAS_SECTION", "Section", section.section_id)

        sorted_paragraphs = sorted(structure.paragraphs, key=lambda item: item.paragraph_index)
        for idx, paragraph in enumerate(sorted_paragraphs):
            self._merge_node("Paragraph", "paragraph_id", paragraph.paragraph_id, paragraph.to_dict())
            if paragraph.section_id:
                self._merge_rel("Section", paragraph.section_id, "HAS_PARAGRAPH", "Paragraph", paragraph.paragraph_id)
            if idx > 0:
                previous = sorted_paragraphs[idx - 1]
                self._merge_rel("Paragraph", previous.paragraph_id, "NEXT", "Paragraph", paragraph.paragraph_id)

        for annotation in structure.annotations:
            self._merge_node("Annotation", "annotation_id", annotation.annotation_id, annotation.to_dict())
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
            self._merge_node("Claim", "claim_id", claim.claim_id, claim.to_dict())
            self._merge_rel("Paragraph", claim.paragraph_id, "HAS_CLAIM", "Claim", claim.claim_id, props={"run_id": claim.run_id})
            self._merge_rel("Claim", claim.claim_id, "EVIDENCED_BY", "Paragraph", claim.paragraph_id, props={"run_id": claim.run_id})
            self._merge_rel("Claim", claim.claim_id, "EXTRACTED_IN", "ExtractionRun", claim.run_id, props={"run_id": claim.run_id})

        for measurement in semantic.measurements:
            self._merge_node("Measurement", "measurement_id", measurement.measurement_id, measurement.to_dict())
            self._merge_rel(
                "Claim",
                measurement.claim_id,
                "HAS_MEASUREMENT",
                "Measurement",
                measurement.measurement_id,
                props={"run_id": measurement.run_id},
            )

        for mention in semantic.mentions:
            self._merge_node("Mention", "mention_id", mention.mention_id, mention.to_dict())
            self._merge_rel(
                "Paragraph",
                mention.paragraph_id,
                "CONTAINS_MENTION",
                "Mention",
                mention.mention_id,
                props={"run_id": mention.run_id},
            )

        for entity in semantic.entities:
            self._merge_node(entity.label, "entity_id", entity.entity_id, entity.to_dict())
            if entity.label in DOMAIN_LABELS:
                self._merge_node("Entity", "entity_id", entity.entity_id, entity.to_dict())

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
                props={"score": resolution.score},
            )

        for link in semantic.claim_entity_links:
            end_label = self._find_entity_label(link.entity_id)
            if end_label:
                self._merge_rel("Claim", link.claim_id, "ABOUT", end_label, link.entity_id)

        for link in semantic.claim_location_links:
            end_label = self._find_entity_label(link.entity_id)
            if end_label:
                self._merge_rel("Claim", link.claim_id, "OCCURRED_AT", end_label, link.entity_id)

        for link in semantic.claim_period_links:
            self._merge_rel("Claim", link.claim_id, "OCCURRED_DURING", "Period", link.period_id)

        for link in semantic.document_refuge_links:
            self._merge_rel("Document", link.doc_id, "ABOUT_REFUGE", "Refuge", link.refuge_id)

        for link in semantic.document_period_links:
            self._merge_rel("Document", link.doc_id, "COVERS_PERIOD", "Period", link.period_id)

        for link in semantic.document_signed_by_links:
            self._merge_rel("Document", link.doc_id, "SIGNED_BY", "Person", link.person_id)

        for link in semantic.person_affiliation_links:
            self._merge_rel("Person", link.person_id, "AFFILIATED_WITH", "Organization", link.organization_id)

    def _merge_node(self, label: str, id_key: str, node_id: str, props: dict[str, Any]) -> None:
        payload = dict(props)
        payload[id_key] = node_id
        if node_id in self.nodes[label]:
            self.nodes[label][node_id].update(payload)
        else:
            self.nodes[label][node_id] = payload

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
        if key not in self.relationships:
            self.relationships[key] = dict(props or {})
        else:
            self.relationships[key].update(props or {})

    def _find_entity_label(self, entity_id: str) -> str | None:
        for label in DOMAIN_LABELS:
            if entity_id in self.nodes.get(label, {}):
                return label
        return None


class Neo4jGraphWriter:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        trust_mode: str = "system",
        ca_cert_path: str | None = None,
    ) -> None:
        if GraphDatabase is None:  # pragma: no cover - optional dependency
            raise RuntimeError("neo4j package is not installed. Install with: pip install -e .[neo4j]")
        driver_kwargs = _build_driver_kwargs(uri=uri, trust_mode=trust_mode, ca_cert_path=ca_cert_path)
        self._driver = GraphDatabase.driver(uri, auth=(user, password), **driver_kwargs)
        self._database = database
        self._uri = uri

        try:
            self._driver.verify_connectivity()
        except Exception as exc:  # pragma: no cover - requires neo4j runtime
            message = _format_connection_error(uri, exc)
            raise RuntimeError(message) from exc

    def close(self) -> None:
        self._driver.close()

    def create_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            for statement in SCHEMA_STATEMENTS:
                session.run(statement)

    def load_structure(self, structure: StructureBundle) -> None:
        self._upsert_nodes(
            label="Document",
            id_key="doc_id",
            rows=[{"id": structure.document.doc_id, "props": structure.document.to_dict()}],
        )
        self._upsert_nodes(
            label="Page",
            id_key="page_id",
            rows=[{"id": page.page_id, "props": page.to_dict()} for page in structure.pages],
        )
        self._upsert_nodes(
            label="Section",
            id_key="section_id",
            rows=[{"id": section.section_id, "props": section.to_dict()} for section in structure.sections],
        )
        self._upsert_nodes(
            label="Paragraph",
            id_key="paragraph_id",
            rows=[{"id": paragraph.paragraph_id, "props": paragraph.to_dict()} for paragraph in structure.paragraphs],
        )
        self._upsert_nodes(
            label="Annotation",
            id_key="annotation_id",
            rows=[{"id": annotation.annotation_id, "props": annotation.to_dict()} for annotation in structure.annotations],
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

        self._upsert_nodes("Claim", "claim_id", [{"id": row.claim_id, "props": row.to_dict()} for row in semantic.claims])
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
            "Claim",
            "claim_id",
            "EXTRACTED_IN",
            "ExtractionRun",
            "run_id",
            [{"start_id": row.claim_id, "end_id": row.run_id, "props": {"run_id": row.run_id}} for row in semantic.claims],
        )

        self._upsert_nodes(
            "Measurement",
            "measurement_id",
            [{"id": row.measurement_id, "props": row.to_dict()} for row in semantic.measurements],
        )
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
                for row in semantic.measurements
            ],
        )

        self._upsert_nodes("Mention", "mention_id", [{"id": row.mention_id, "props": row.to_dict()} for row in semantic.mentions])
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
            by_label[entity.label].append({"id": entity.entity_id, "props": entity.to_dict()})
        for label, rows in by_label.items():
            self._upsert_nodes(label=label, id_key="entity_id", rows=rows)
            if label in DOMAIN_LABELS:
                self._upsert_nodes(label="Entity", id_key="entity_id", rows=rows)

        for relation_type in ("REFERS_TO", "POSSIBLY_REFERS_TO"):
            rows = []
            for row in semantic.entity_resolutions:
                if row.relation_type == relation_type:
                    target = next((entity for entity in semantic.entities if entity.entity_id == row.entity_id), None)
                    if not target:
                        continue
                    rows.append({"start_id": row.mention_id, "end_id": row.entity_id, "props": {"score": row.score}})
                    self._upsert_relationships(
                        "Mention",
                        "mention_id",
                        relation_type,
                        target.label,
                        "entity_id",
                        rows,
                    )
                    rows = []

        claim_about_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        claim_location_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        entity_label_lookup = {row.entity_id: row.label for row in semantic.entities}
        for row in semantic.claim_entity_links:
            label = entity_label_lookup.get(row.entity_id)
            if label:
                claim_about_by_label[label].append({"start_id": row.claim_id, "end_id": row.entity_id, "props": {}})
        for row in semantic.claim_location_links:
            label = entity_label_lookup.get(row.entity_id)
            if label:
                claim_location_by_label[label].append({"start_id": row.claim_id, "end_id": row.entity_id, "props": {}})

        for label, rows in claim_about_by_label.items():
            self._upsert_relationships("Claim", "claim_id", "ABOUT", label, "entity_id", rows)
        for label, rows in claim_location_by_label.items():
            self._upsert_relationships("Claim", "claim_id", "OCCURRED_AT", label, "entity_id", rows)

        self._upsert_relationships(
            "Claim",
            "claim_id",
            "OCCURRED_DURING",
            "Period",
            "entity_id",
            [{"start_id": row.claim_id, "end_id": row.period_id, "props": {}} for row in semantic.claim_period_links],
        )
        self._upsert_relationships(
            "Document",
            "doc_id",
            "ABOUT_REFUGE",
            "Refuge",
            "entity_id",
            [{"start_id": row.doc_id, "end_id": row.refuge_id, "props": {}} for row in semantic.document_refuge_links],
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

    def _upsert_nodes(self, label: str, id_key: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{label} {{{id_key}: row.id}}) "
            "SET n += row.props"
        )
        with self._driver.session(database=self._database) as session:
            session.run(query, rows=rows)

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
            session.run(query, rows=rows)


def node_id_key_for_label(label: str) -> str:
    return ID_CONSTRAINTS.get(label, "entity_id")


def _build_driver_kwargs(uri: str, trust_mode: str, ca_cert_path: str | None) -> dict[str, Any]:
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


def _format_connection_error(uri: str, exc: Exception) -> str:
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
