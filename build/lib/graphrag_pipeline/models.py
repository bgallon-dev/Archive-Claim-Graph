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
        return cls(**payload)


@dataclass(slots=True)
class ExtractionRunRecord:
    run_id: str
    ocr_engine: str = "unknown"
    ocr_version: str = "unknown"
    normalizer_version: str = "v1"
    ner_model: str = "hybrid-rules-llm"
    relation_model: str = "hybrid-rules-llm"
    run_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractionRunRecord":
        return cls(**payload)


@dataclass(slots=True)
class PageRecord:
    page_id: str
    doc_id: str
    page_number: int
    raw_text: str
    clean_text: str
    ocr_confidence: float | None = None
    image_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageRecord":
        return cls(**payload)


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
    raw_text: str
    clean_text: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParagraphRecord":
        return cls(**payload)


@dataclass(slots=True)
class AnnotationRecord:
    annotation_id: str
    doc_id: str
    page_id: str
    page_number: int
    kind: str
    text: str
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnnotationRecord":
        return cls(**payload)


@dataclass(slots=True)
class ClaimRecord:
    claim_id: str
    run_id: str
    paragraph_id: str
    claim_type: str
    raw_sentence: str
    normalized_sentence: str
    certainty: str
    extraction_confidence: float
    review_status: str = "unreviewed"
    notes: str = ""
    evidence_start: int | None = None
    evidence_end: int | None = None
    claim_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimRecord":
        return cls(**payload)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeasurementRecord":
        return cls(**payload)


@dataclass(slots=True)
class MentionRecord:
    mention_id: str
    run_id: str
    paragraph_id: str
    surface_form: str
    normalized_form: str
    start_offset: int
    end_offset: int
    confidence: float
    ocr_suspect: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MentionRecord":
        return cls(**payload)


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    label: str
    name: str
    normalized_name: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityRecord":
        return cls(**payload)


@dataclass(slots=True)
class EntityResolutionRecord:
    mention_id: str
    entity_id: str
    relation_type: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityResolutionRecord":
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
    claim_location_links: list[ClaimLocationLinkRecord]
    claim_period_links: list[ClaimPeriodLinkRecord]
    document_refuge_links: list[DocumentRefugeLinkRecord]
    document_period_links: list[DocumentPeriodLinkRecord]
    document_signed_by_links: list[DocumentSignedByLinkRecord]
    person_affiliation_links: list[PersonAffiliationLinkRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_run": self.extraction_run.to_dict(),
            "claims": [item.to_dict() for item in self.claims],
            "measurements": [item.to_dict() for item in self.measurements],
            "mentions": [item.to_dict() for item in self.mentions],
            "entities": [item.to_dict() for item in self.entities],
            "entity_resolutions": [item.to_dict() for item in self.entity_resolutions],
            "claim_entity_links": [item.to_dict() for item in self.claim_entity_links],
            "claim_location_links": [item.to_dict() for item in self.claim_location_links],
            "claim_period_links": [item.to_dict() for item in self.claim_period_links],
            "document_refuge_links": [item.to_dict() for item in self.document_refuge_links],
            "document_period_links": [item.to_dict() for item in self.document_period_links],
            "document_signed_by_links": [item.to_dict() for item in self.document_signed_by_links],
            "person_affiliation_links": [item.to_dict() for item in self.person_affiliation_links],
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
            claim_location_links=[ClaimLocationLinkRecord.from_dict(row) for row in payload["claim_location_links"]],
            claim_period_links=[ClaimPeriodLinkRecord.from_dict(row) for row in payload["claim_period_links"]],
            document_refuge_links=[DocumentRefugeLinkRecord.from_dict(row) for row in payload["document_refuge_links"]],
            document_period_links=[DocumentPeriodLinkRecord.from_dict(row) for row in payload["document_period_links"]],
            document_signed_by_links=[DocumentSignedByLinkRecord.from_dict(row) for row in payload["document_signed_by_links"]],
            person_affiliation_links=[PersonAffiliationLinkRecord.from_dict(row) for row in payload["person_affiliation_links"]],
        )
