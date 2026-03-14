from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    title: str
    doc_type: str = "narrative_report"
    series: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    report_year: int | None = None
    source_file: str | None = None
    archive_ref: str | None = None
    raw_ocr_text: str = ""
    clean_text: str = ""
    language: str = "en"
    file_hash: str = ""
    page_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentRecord":
        data = dict(payload)
        if "report_year" not in data and "year" in data:
            legacy_year = data.pop("year")
            if isinstance(legacy_year, str) and legacy_year.strip().isdigit():
                legacy_year = int(legacy_year.strip())
            data["report_year"] = legacy_year
        if "raw_ocr_text" not in data and "raw_text" in data:
            data["raw_ocr_text"] = data.pop("raw_text")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class ExtractionRunRecord:
    run_id: str
    ocr_engine: str = "unknown"
    ocr_version: str = "unknown"
    normalizer_version: str = "v1"
    ner_model: str = "hybrid-rules-llm"
    relation_model: str = "hybrid-rules-llm"
    run_timestamp: str = ""
    config_fingerprint: str = ""
    claim_type_schema_version: str = "v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractionRunRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(slots=True)
class PageRecord:
    page_id: str
    doc_id: str
    page_number: int
    raw_ocr_text: str
    clean_text: str
    ocr_confidence: float | None = None
    image_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node. Excludes doc_id, which is
        authoritative as the Document-[:HAS_PAGE]-> edge, to prevent property/edge drift."""
        d = asdict(self)
        d.pop("doc_id", None)
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageRecord":
        data = dict(payload)
        if "raw_ocr_text" not in data and "raw_text" in data:
            data["raw_ocr_text"] = data.pop("raw_text")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class SectionRecord:
    section_id: str
    doc_id: str
    heading: str
    section_number: int | None
    section_letter: str | None
    normalized_heading: str
    page_start: int
    page_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node. Excludes doc_id, which is
        authoritative as the Document-[:HAS_SECTION]-> edge, to prevent property/edge drift."""
        d = asdict(self)
        d.pop("doc_id", None)
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SectionRecord":
        return cls(**payload)


@dataclass(slots=True)
class ParagraphRecord:
    paragraph_id: str
    doc_id: str
    page_id: str
    section_id: str | None
    paragraph_index: int
    page_number: int
    raw_ocr_text: str
    clean_text: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        Excludes FK fields whose relationships are authoritative as graph edges:
          doc_id     → Document-[:HAS_PAGE]->Page-[:HAS_SECTION]->Section-[:HAS_PARAGRAPH]->Paragraph
          page_id    → structural chain above
          section_id → Section-[:HAS_PARAGRAPH]->Paragraph
        """
        _edge_fks = {"doc_id", "page_id", "section_id"}
        return {k: v for k, v in asdict(self).items() if k not in _edge_fks}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParagraphRecord":
        data = dict(payload)
        if "raw_ocr_text" not in data and "raw_text" in data:
            data["raw_ocr_text"] = data.pop("raw_text")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class AnnotationRecord:
    annotation_id: str
    doc_id: str
    page_id: str
    page_number: int
    kind: str
    text: str
    bbox: list[float] | None = None
    annotation_type: str | None = None
    comment: str | None = None
    created_by: str | None = None
    created_at: str | None = None
    status: str | None = None
    target_claim_id: str | None = None
    target_measurement_id: str | None = None
    corrects_measurement: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        Excludes FK fields whose relationships are authoritative as graph edges:
          doc_id  → Document structural chain
          page_id → Page-[:HAS_ANNOTATION]->Annotation
        """
        _edge_fks = {"doc_id", "page_id"}
        return {k: v for k, v in asdict(self).items() if k not in _edge_fks}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnnotationRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(slots=True)
class ClaimRecord:
    claim_id: str
    run_id: str
    paragraph_id: str
    claim_type: str
    source_sentence: str       # sentence as extracted from clean_text (normalized paragraph text)
    normalized_sentence: str   # lowercased, whitespace-collapsed form of source_sentence
    certainty: str             # "certain" | "uncertain" — categorical epistemic layer
    extraction_confidence: float  # [0.0–1.0] confidence of the claim extraction
    review_status: str = "unreviewed"
    notes: str = ""
    evidence_start: int | None = None
    evidence_end: int | None = None
    claim_date: str | None = None

    # Serialized bundles expose this as `epistemic_status`; `certainty` remains
    # the constructor field for backward-compatible Python callers.
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["epistemic_status"] = data.pop("certainty")
        return data

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        Excludes FK fields whose relationships are authoritative as graph edges:
          paragraph_id → Paragraph-[:HAS_CLAIM]->Claim and Claim-[:EVIDENCED_BY]->Paragraph
        """
        data = self.to_dict()
        data.pop("paragraph_id", None)
        return data

    @property
    def epistemic_status(self) -> str:
        return self.certainty

    @epistemic_status.setter
    def epistemic_status(self, value: str) -> None:
        self.certainty = value

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimRecord":
        data = dict(payload)
        if "source_sentence" not in data and "raw_sentence" in data:
            data["source_sentence"] = data.pop("raw_sentence")
        if "certainty" not in data and "epistemic_status" in data:
            data["certainty"] = data.pop("epistemic_status")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class MeasurementRecord:
    measurement_id: str
    claim_id: str
    run_id: str
    name: str
    raw_value: str
    numeric_value: float | None
    unit: str | None
    approximate: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    qualifier: str | None = None
    measurement_date: str | None = None
    methodology_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        Excludes FK fields whose relationships are authoritative as graph edges:
          claim_id → Claim-[:HAS_MEASUREMENT]->Measurement
        """
        _edge_fks = {"claim_id"}
        return {k: v for k, v in asdict(self).items() if k not in _edge_fks}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeasurementRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(slots=True)
class MentionRecord:
    mention_id: str
    run_id: str
    paragraph_id: str
    surface_form: str
    normalized_form: str
    start_offset: int
    end_offset: int
    detection_confidence: float  # [0.0–1.0] extractor's confidence in detecting this mention
    ocr_suspect: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        Excludes FK fields whose relationships are authoritative as graph edges:
          paragraph_id → Paragraph-[:CONTAINS_MENTION]->Mention
        """
        _edge_fks = {"paragraph_id"}
        return {k: v for k, v in asdict(self).items() if k not in _edge_fks}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MentionRecord":
        data = dict(payload)
        if "normalized_form" not in data and "normalized_name" in data:
            data["normalized_form"] = data.pop("normalized_name")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    entity_type: str       # entity class/category, e.g. "Species", "Place", "Refuge"
    name: str              # canonical display name
    normalized_form: str   # lowercased, normalized form for matching (mirrors MentionRecord.normalized_form)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        `entity_type` remains part of the serialized bundle contract so the
        pipeline can route entities to their domain labels, but the Neo4j label
        itself is authoritative once written to the graph.
        """
        data = asdict(self)
        data.pop("entity_type", None)
        nested = data.pop("properties", {})
        data.update(nested)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityRecord":
        data = dict(payload)
        if "normalized_form" not in data and "normalized_name" in data:
            data["normalized_form"] = data.pop("normalized_name")
        props = data.get("properties")
        if (
            data.get("entity_type") == "Period"
            and isinstance(props, dict)
            and "source_title" not in props
            and "label" in props
        ):
            remapped_props = dict(props)
            remapped_props["source_title"] = remapped_props.pop("label")
            data["properties"] = remapped_props
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class EntityResolutionRecord:
    mention_id: str
    entity_id: str
    relation_type: str
    match_score: float
    confirmed_by: str | None = None
    confirmed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityResolutionRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(slots=True)
class EntityResolutionConfirmationRecord:
    mention_id: str
    entity_id: str
    relation_type: str   # "CONFIRMED_AS" | "REFUTED_BY"
    confirmed_by: str
    confirmed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityResolutionConfirmationRecord":
        return cls(**payload)


@dataclass(slots=True)
class ClaimEntityLinkRecord:
    claim_id: str
    entity_id: str
    relation_type: str = "ABOUT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimEntityLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class ClaimLinkDiagnosticRecord:
    claim_id: str
    relation_type: str
    surface_form: str
    normalized_form: str
    diagnostic_code: str
    entity_type_hint: str | None = None
    candidate_count: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimLinkDiagnosticRecord":
        return cls(**payload)


@dataclass(slots=True)
class ClaimLocationLinkRecord:
    claim_id: str
    entity_id: str
    relation_type: str = "OCCURRED_AT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimLocationLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class ClaimPeriodLinkRecord:
    claim_id: str
    period_id: str
    relation_type: str = "OCCURRED_DURING"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimPeriodLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class DocumentRefugeLinkRecord:
    doc_id: str
    refuge_id: str
    relation_type: str = "ABOUT_REFUGE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentRefugeLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class DocumentPeriodLinkRecord:
    doc_id: str
    period_id: str
    relation_type: str = "COVERS_PERIOD"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentPeriodLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class DocumentSignedByLinkRecord:
    doc_id: str
    person_id: str
    relation_type: str = "SIGNED_BY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentSignedByLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class PersonAffiliationLinkRecord:
    person_id: str
    organization_id: str
    relation_type: str = "AFFILIATED_WITH"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersonAffiliationLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class ObservationRecord:
    observation_id: str
    run_id: str
    observation_type: str
    claim_id: str
    paragraph_id: str
    species_id: str | None = None
    refuge_id: str | None = None
    place_id: str | None = None
    period_id: str | None = None
    year_id: str | None = None
    habitat_id: str | None = None
    survey_method_id: str | None = None
    value_text: str | None = None
    confidence: float = 0.0  # pipeline confidence carried forward from the supporting claim extraction
    is_estimate: bool = False
    review_status: str = "unreviewed"
    # Derivation contract fields
    source_claim_type: str = ""
    year: int | None = None
    year_source: str = "unknown"   # "claim_date" | "document_primary_year" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        Excludes FK fields whose relationships are authoritative as graph edges:
          species_id    → OF_SPECIES (Observation→Species)
          refuge_id     → AT_REFUGE (Observation→Refuge)
          place_id      → AT_PLACE (Observation→Place)
          period_id     → DURING (Observation→Period)
          year_id       → IN_YEAR (Observation→Year)
          habitat_id    → IN_HABITAT (Observation→Habitat)
          survey_method_id → USED_METHOD (Observation→SurveyMethod)
          paragraph_id  → EVIDENCED_BY (Observation→Paragraph)
          claim_id      → SUPPORTS (Claim→Observation)
        """
        _edge_fks = {
            "species_id", "refuge_id", "place_id", "period_id",
            "year_id", "habitat_id", "survey_method_id",
            "paragraph_id", "claim_id",
        }
        return {k: v for k, v in asdict(self).items() if k not in _edge_fks}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(slots=True)
class YearRecord:
    year_id: str
    year: int
    year_label: str  # human-readable label, e.g. "1938" or "FY-1972"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "YearRecord":
        data = dict(payload)
        if "year_label" not in data and "label" in data:
            data["year_label"] = data.pop("label")
        return cls(**data)


@dataclass(slots=True)
class ObservationMeasurementLinkRecord:
    observation_id: str
    measurement_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationMeasurementLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class EventRecord:
    event_id: str
    run_id: str
    event_type: str            # "SurveyEvent", "FireEvent", etc.
    claim_id: str
    paragraph_id: str
    species_id: str | None = None
    refuge_id: str | None = None
    place_id: str | None = None
    period_id: str | None = None
    year_id: str | None = None
    habitat_id: str | None = None
    survey_method_id: str | None = None
    source_claim_type: str = ""
    year: int | None = None
    year_source: str = "unknown"
    confidence: float = 0.0  # pipeline confidence carried forward from the supporting claim extraction
    review_status: str = "unreviewed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        Excludes FK fields whose relationships are authoritative as graph edges:
          species_id    → INVOLVED_SPECIES (Event→Species)
          refuge_id     → OCCURRED_AT (Event→Refuge)
          place_id      → OCCURRED_AT (Event→Place)
          period_id     → DURING (Event→Period)
          year_id       → IN_YEAR (Event→Year)
          habitat_id    → IN_HABITAT (Event→Habitat)
          survey_method_id → USED_METHOD (Event→SurveyMethod)
          paragraph_id  → SOURCED_FROM (Event→Paragraph)
          claim_id      → TRIGGERED (Claim→Event)
        """
        _edge_fks = {
            "species_id", "refuge_id", "place_id", "period_id",
            "year_id", "habitat_id", "survey_method_id",
            "paragraph_id", "claim_id",
        }
        return {k: v for k, v in asdict(self).items() if k not in _edge_fks}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(slots=True)
class EventObservationLinkRecord:
    event_id: str
    observation_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventObservationLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class EventMeasurementLinkRecord:
    event_id: str
    measurement_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventMeasurementLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class DocumentYearLinkRecord:
    doc_id: str
    year_id: str
    relation_type: str = "COVERS_YEAR"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentYearLinkRecord":
        return cls(**payload)


@dataclass(slots=True)
class PlaceRefugeLinkRecord:
    place_id: str
    refuge_id: str
    relation_type: str = "LOCATED_IN_REFUGE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlaceRefugeLinkRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in payload.items() if k in known}
        # Remap legacy PART_OF to LOCATED_IN_REFUGE on load.
        if data.get("relation_type") == "PART_OF":
            data["relation_type"] = "LOCATED_IN_REFUGE"
        return cls(**data)


@dataclass(slots=True)
class StructureBundle:
    document: DocumentRecord
    pages: list[PageRecord]
    sections: list[SectionRecord]
    paragraphs: list[ParagraphRecord]
    annotations: list[AnnotationRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "pages": [item.to_dict() for item in self.pages],
            "sections": [item.to_dict() for item in self.sections],
            "paragraphs": [item.to_dict() for item in self.paragraphs],
            "annotations": [item.to_dict() for item in self.annotations],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureBundle":
        return cls(
            document=DocumentRecord.from_dict(payload["document"]),
            pages=[PageRecord.from_dict(row) for row in payload["pages"]],
            sections=[SectionRecord.from_dict(row) for row in payload["sections"]],
            paragraphs=[ParagraphRecord.from_dict(row) for row in payload["paragraphs"]],
            annotations=[AnnotationRecord.from_dict(row) for row in payload["annotations"]],
        )


@dataclass(slots=True)
class SemanticBundle:
    extraction_run: ExtractionRunRecord
    claims: list[ClaimRecord]
    measurements: list[MeasurementRecord]
    mentions: list[MentionRecord]
    entities: list[EntityRecord]
    entity_resolutions: list[EntityResolutionRecord]
    claim_entity_links: list[ClaimEntityLinkRecord]
    claim_link_diagnostics: list[ClaimLinkDiagnosticRecord]
    claim_location_links: list[ClaimLocationLinkRecord]
    claim_period_links: list[ClaimPeriodLinkRecord]
    document_refuge_links: list[DocumentRefugeLinkRecord]
    document_period_links: list[DocumentPeriodLinkRecord]
    document_signed_by_links: list[DocumentSignedByLinkRecord]
    person_affiliation_links: list[PersonAffiliationLinkRecord]
    observations: list[ObservationRecord] = field(default_factory=list)
    years: list[YearRecord] = field(default_factory=list)
    observation_measurement_links: list[ObservationMeasurementLinkRecord] = field(default_factory=list)
    document_year_links: list[DocumentYearLinkRecord] = field(default_factory=list)
    place_refuge_links: list[PlaceRefugeLinkRecord] = field(default_factory=list)
    entity_resolution_confirmations: list[EntityResolutionConfirmationRecord] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    event_observation_links: list[EventObservationLinkRecord] = field(default_factory=list)
    event_measurement_links: list[EventMeasurementLinkRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_run": self.extraction_run.to_dict(),
            "claims": [item.to_dict() for item in self.claims],
            "measurements": [item.to_dict() for item in self.measurements],
            "mentions": [item.to_dict() for item in self.mentions],
            "entities": [item.to_dict() for item in self.entities],
            "entity_resolutions": [item.to_dict() for item in self.entity_resolutions],
            "claim_entity_links": [item.to_dict() for item in self.claim_entity_links],
            "claim_link_diagnostics": [item.to_dict() for item in self.claim_link_diagnostics],
            "claim_location_links": [item.to_dict() for item in self.claim_location_links],
            "claim_period_links": [item.to_dict() for item in self.claim_period_links],
            "document_refuge_links": [item.to_dict() for item in self.document_refuge_links],
            "document_period_links": [item.to_dict() for item in self.document_period_links],
            "document_signed_by_links": [item.to_dict() for item in self.document_signed_by_links],
            "person_affiliation_links": [item.to_dict() for item in self.person_affiliation_links],
            "observations": [item.to_dict() for item in self.observations],
            "years": [item.to_dict() for item in self.years],
            "observation_measurement_links": [item.to_dict() for item in self.observation_measurement_links],
            "document_year_links": [item.to_dict() for item in self.document_year_links],
            "place_refuge_links": [item.to_dict() for item in self.place_refuge_links],
            "entity_resolution_confirmations": [item.to_dict() for item in self.entity_resolution_confirmations],
            "events": [item.to_dict() for item in self.events],
            "event_observation_links": [item.to_dict() for item in self.event_observation_links],
            "event_measurement_links": [item.to_dict() for item in self.event_measurement_links],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticBundle":
        return cls(
            extraction_run=ExtractionRunRecord.from_dict(payload["extraction_run"]),
            claims=[ClaimRecord.from_dict(row) for row in payload["claims"]],
            measurements=[MeasurementRecord.from_dict(row) for row in payload["measurements"]],
            mentions=[MentionRecord.from_dict(row) for row in payload["mentions"]],
            entities=[EntityRecord.from_dict(row) for row in payload["entities"]],
            entity_resolutions=[EntityResolutionRecord.from_dict(row) for row in payload["entity_resolutions"]],
            claim_entity_links=[ClaimEntityLinkRecord.from_dict(row) for row in payload["claim_entity_links"]],
            claim_link_diagnostics=[ClaimLinkDiagnosticRecord.from_dict(row) for row in payload.get("claim_link_diagnostics", [])],
            claim_location_links=[ClaimLocationLinkRecord.from_dict(row) for row in payload["claim_location_links"]],
            claim_period_links=[ClaimPeriodLinkRecord.from_dict(row) for row in payload["claim_period_links"]],
            document_refuge_links=[DocumentRefugeLinkRecord.from_dict(row) for row in payload["document_refuge_links"]],
            document_period_links=[DocumentPeriodLinkRecord.from_dict(row) for row in payload["document_period_links"]],
            document_signed_by_links=[DocumentSignedByLinkRecord.from_dict(row) for row in payload["document_signed_by_links"]],
            person_affiliation_links=[PersonAffiliationLinkRecord.from_dict(row) for row in payload["person_affiliation_links"]],
            observations=[ObservationRecord.from_dict(row) for row in payload.get("observations", [])],
            years=[YearRecord.from_dict(row) for row in payload.get("years", [])],
            observation_measurement_links=[ObservationMeasurementLinkRecord.from_dict(row) for row in payload.get("observation_measurement_links", [])],
            document_year_links=[DocumentYearLinkRecord.from_dict(row) for row in payload.get("document_year_links", [])],
            place_refuge_links=[PlaceRefugeLinkRecord.from_dict(row) for row in payload.get("place_refuge_links", [])],
            entity_resolution_confirmations=[EntityResolutionConfirmationRecord.from_dict(row) for row in payload.get("entity_resolution_confirmations", [])],
            events=[EventRecord.from_dict(row) for row in payload.get("events", [])],
            event_observation_links=[EventObservationLinkRecord.from_dict(row) for row in payload.get("event_observation_links", [])],
            event_measurement_links=[EventMeasurementLinkRecord.from_dict(row) for row in payload.get("event_measurement_links", [])],
        )
