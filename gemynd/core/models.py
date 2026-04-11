from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

_R = TypeVar("_R", bound="_BaseRecord")


class _BaseRecord:
    """Shared plumbing for all record dataclasses.

    Provides default ``to_dict``, ``from_dict``, and ``node_props`` so
    subclasses only override what genuinely differs.

    ``_EDGE_FKS``
        Class-level frozenset of field names that are FK references stored as
        graph edges rather than node properties.  ``node_props()`` excludes
        these keys.  Defaults to empty (all fields become node props).
    """

    __slots__ = ()
    _EDGE_FKS: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def node_props(self) -> dict[str, Any]:
        """Return properties suitable for a Neo4j node, excluding FK edges."""
        fks = type(self)._EDGE_FKS
        return {k: v for k, v in asdict(self).items() if k not in fks}  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls: type[_R], payload: dict[str, Any]) -> _R:
        """Construct from a dict, silently dropping unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(slots=True)
class DocumentRecord(_BaseRecord):
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
    access_level: str = "public"        # "public" | "staff_only" | "restricted" | "indigenous_restricted"
    institution_id: str = ""             # tenant identifier for multi-institution isolation
    donor_restricted: bool = False      # reproduction restrictions from donor agreement
    deleted_at: str | None = None       # ISO-8601 UTC timestamp when soft-deleted; None = active
    deleted_by: str | None = None       # identity string of the user who performed deletion


@dataclass(slots=True)
class ExtractionRunRecord(_BaseRecord):
    run_id: str
    ocr_engine: str = "unknown"
    ocr_version: str = "unknown"
    normalizer_version: str = "v1"
    ner_model: str = "hybrid-rules-llm"
    relation_model: str = "hybrid-rules-llm"
    run_timestamp: str = ""
    config_fingerprint: str = ""
    claim_type_schema_version: str = "v2"


@dataclass(slots=True)
class PageRecord(_BaseRecord):
    _EDGE_FKS = frozenset({"doc_id"})

    page_id: str
    doc_id: str
    page_number: int
    raw_ocr_text: str
    clean_text: str
    ocr_confidence: float | None = None
    image_ref: str | None = None


@dataclass(slots=True)
class SectionRecord(_BaseRecord):
    _EDGE_FKS = frozenset({"doc_id"})

    section_id: str
    doc_id: str
    heading: str
    section_number: int | None
    section_letter: str | None
    normalized_heading: str
    page_start: int
    page_end: int


@dataclass(slots=True)
class ParagraphRecord(_BaseRecord):
    _EDGE_FKS = frozenset({"doc_id", "page_id", "section_id"})

    paragraph_id: str
    doc_id: str
    page_id: str
    section_id: str | None
    paragraph_index: int
    page_number: int
    raw_ocr_text: str
    clean_text: str
    char_count: int


@dataclass(slots=True)
class AnnotationRecord(_BaseRecord):
    _EDGE_FKS = frozenset({"doc_id", "page_id"})

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


@dataclass(slots=True)
class ClaimRecord(_BaseRecord):
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
    quarantine_status: str = "active"       # "active" | "quarantined" | "reviewed_restricted" | "reviewed_cleared"
    quarantine_reason: str | None = None    # issue_class that triggered quarantine
    quarantine_timestamp: str | None = None # ISO-8601 UTC timestamp of quarantine event

    # Serialized bundles expose this as `epistemic_status`; `certainty` remains
    # the constructor field for backward-compatible Python callers.
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["epistemic_status"] = data.pop("certainty")
        return data

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        Uses ``to_dict`` (not ``asdict``) so the ``certainty`` → ``epistemic_status``
        rename is applied, then drops ``paragraph_id`` which is authoritative as
        the Paragraph-[:HAS_CLAIM]->Claim edge.
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
        if "certainty" not in data and "epistemic_status" in data:
            data["certainty"] = data.pop("epistemic_status")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class MeasurementRecord(_BaseRecord):
    _EDGE_FKS = frozenset({"claim_id"})

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


@dataclass(slots=True)
class MentionRecord(_BaseRecord):
    _EDGE_FKS = frozenset({"paragraph_id"})

    mention_id: str
    run_id: str
    paragraph_id: str
    surface_form: str
    normalized_form: str
    start_offset: int
    end_offset: int
    detection_confidence: float  # [0.0–1.0] extractor's confidence in detecting this mention
    ocr_suspect: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MentionRecord":
        data = dict(payload)
        if "normalized_form" not in data and "normalized_name" in data:
            data["normalized_form"] = data.pop("normalized_name")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class EntityRecord(_BaseRecord):
    entity_id: str
    entity_type: str       # entity class/category, e.g. "Species", "Place", "Refuge"
    name: str              # canonical display name
    normalized_form: str   # lowercased, normalized form for matching (mirrors MentionRecord.normalized_form)
    properties: dict[str, Any] = field(default_factory=dict)

    def node_props(self) -> dict[str, Any]:
        """Properties to persist on the Neo4j node.

        ``entity_type`` remains part of the serialized bundle contract so the
        pipeline can route entities to their domain labels, but the Neo4j label
        itself is authoritative once written to the graph.  ``properties`` is
        flattened inline rather than stored as a nested map.
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
class EntityResolutionRecord(_BaseRecord):
    mention_id: str
    entity_id: str
    relation_type: str
    match_score: float
    confirmed_by: str | None = None
    confirmed_at: str | None = None


@dataclass(slots=True)
class EntityResolutionConfirmationRecord(_BaseRecord):
    mention_id: str
    entity_id: str
    relation_type: str   # "CONFIRMED_AS" | "REFUTED_BY"
    confirmed_by: str
    confirmed_at: str


@dataclass(slots=True)
class ClaimEntityLinkRecord(_BaseRecord):
    claim_id: str
    entity_id: str
    relation_type: str = "ABOUT"


@dataclass(slots=True)
class ClaimLinkDiagnosticRecord(_BaseRecord):
    claim_id: str
    relation_type: str
    surface_form: str
    normalized_form: str
    diagnostic_code: str
    entity_type_hint: str | None = None
    candidate_count: int = 0
    detail: str = ""


@dataclass(slots=True)
class ClaimLocationLinkRecord(_BaseRecord):
    claim_id: str
    entity_id: str
    relation_type: str = "OCCURRED_AT"


@dataclass(slots=True)
class ClaimPeriodLinkRecord(_BaseRecord):
    claim_id: str
    period_id: str
    relation_type: str = "OCCURRED_DURING"


@dataclass(slots=True)
class DocumentAnchorLinkRecord(_BaseRecord):
    doc_id: str
    anchor_entity_id: str
    relation_type: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentAnchorLinkRecord":
        # Accept legacy key "refuge_id" → "anchor_entity_id" for
        # round-tripping pre-Phase-8 serialized bundles.
        data = dict(payload)
        if "anchor_entity_id" not in data and "refuge_id" in data:
            data["anchor_entity_id"] = data.pop("refuge_id")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class DocumentPeriodLinkRecord(_BaseRecord):
    doc_id: str
    period_id: str
    relation_type: str = "COVERS_PERIOD"


@dataclass(slots=True)
class DocumentSignedByLinkRecord(_BaseRecord):
    doc_id: str
    person_id: str
    relation_type: str = "SIGNED_BY"


@dataclass(slots=True)
class PersonAffiliationLinkRecord(_BaseRecord):
    person_id: str
    organization_id: str
    relation_type: str = "AFFILIATED_WITH"


_LEGACY_WILDLIFE_ROLE_KEYS: dict[str, str] = {
    "species_id": "species",
    "refuge_id": "refuge",
    "habitat_id": "habitat",
    "survey_method_id": "survey_method",
}


def _pack_legacy_role_keys(payload: dict[str, Any]) -> dict[str, Any]:
    """Move legacy wildlife FK keys into a ``role_entities`` dict.

    Pre-Phase-8 serialized observation/event records carried
    ``species_id`` / ``refuge_id`` / ``habitat_id`` / ``survey_method_id``
    as typed fields.  Packing them into ``role_entities`` preserves
    round-trip compatibility with bundles on disk and in test fixtures.
    """
    data = dict(payload)
    role_entities: dict[str, str] = dict(data.get("role_entities") or {})
    for legacy_key, role_name in _LEGACY_WILDLIFE_ROLE_KEYS.items():
        if legacy_key in data:
            value = data.pop(legacy_key)
            if value is not None and role_name not in role_entities:
                role_entities[role_name] = value
    data["role_entities"] = role_entities
    return data


@dataclass(slots=True)
class ObservationRecord(_BaseRecord):
    _EDGE_FKS = frozenset({
        "place_id", "period_id", "year_id",
        "paragraph_id", "claim_id", "role_entities",
    })

    observation_id: str
    run_id: str
    observation_type: str
    claim_id: str
    paragraph_id: str
    place_id: str | None = None
    period_id: str | None = None
    year_id: str | None = None
    role_entities: dict[str, str] = field(default_factory=dict)
    value_text: str | None = None
    confidence: float = 0.0  # pipeline confidence carried forward from the supporting claim extraction
    is_estimate: bool = False
    review_status: str = "unreviewed"
    # Derivation contract fields
    source_claim_type: str = ""
    year: int | None = None
    year_source: str = "unknown"   # "claim_date" | "document_primary_year" | "unknown"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationRecord":
        data = _pack_legacy_role_keys(payload)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class YearRecord(_BaseRecord):
    year_id: str
    year: int
    year_label: str  # human-readable label, e.g. "1938" or "FY-1972"


@dataclass(slots=True)
class ObservationMeasurementLinkRecord(_BaseRecord):
    observation_id: str
    measurement_id: str


@dataclass(slots=True)
class EventRecord(_BaseRecord):
    _EDGE_FKS = frozenset({
        "place_id", "period_id", "year_id",
        "paragraph_id", "claim_id", "role_entities",
    })

    event_id: str
    run_id: str
    event_type: str            # "SurveyEvent", "FireEvent", etc.
    claim_id: str
    paragraph_id: str
    place_id: str | None = None
    period_id: str | None = None
    year_id: str | None = None
    role_entities: dict[str, str] = field(default_factory=dict)
    source_claim_type: str = ""
    year: int | None = None
    year_source: str = "unknown"
    confidence: float = 0.0  # pipeline confidence carried forward from the supporting claim extraction
    review_status: str = "unreviewed"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventRecord":
        data = _pack_legacy_role_keys(payload)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class EventObservationLinkRecord(_BaseRecord):
    event_id: str
    observation_id: str


@dataclass(slots=True)
class EventMeasurementLinkRecord(_BaseRecord):
    event_id: str
    measurement_id: str


@dataclass(slots=True)
class DocumentYearLinkRecord(_BaseRecord):
    doc_id: str
    year_id: str
    relation_type: str = "COVERS_YEAR"


@dataclass(slots=True)
class EntityHierarchyLinkRecord(_BaseRecord):
    child_entity_id: str
    parent_entity_id: str
    relation_type: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityHierarchyLinkRecord":
        # Accept legacy keys "place_id"/"refuge_id" for round-tripping
        # pre-Phase-8 serialized bundles.
        data = dict(payload)
        if "child_entity_id" not in data and "place_id" in data:
            data["child_entity_id"] = data.pop("place_id")
        if "parent_entity_id" not in data and "refuge_id" in data:
            data["parent_entity_id"] = data.pop("refuge_id")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


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
class ClaimConceptLinkRecord(_BaseRecord):
    claim_id: str
    concept_id: str
    confidence: float = 0.0
    matched_rule: str = ""


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
    document_anchor_links: list[DocumentAnchorLinkRecord]
    document_period_links: list[DocumentPeriodLinkRecord]
    document_signed_by_links: list[DocumentSignedByLinkRecord]
    person_affiliation_links: list[PersonAffiliationLinkRecord]
    observations: list[ObservationRecord] = field(default_factory=list)
    years: list[YearRecord] = field(default_factory=list)
    observation_measurement_links: list[ObservationMeasurementLinkRecord] = field(default_factory=list)
    document_year_links: list[DocumentYearLinkRecord] = field(default_factory=list)
    entity_hierarchy_links: list[EntityHierarchyLinkRecord] = field(default_factory=list)
    entity_resolution_confirmations: list[EntityResolutionConfirmationRecord] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    event_observation_links: list[EventObservationLinkRecord] = field(default_factory=list)
    event_measurement_links: list[EventMeasurementLinkRecord] = field(default_factory=list)
    claim_concept_links: list[ClaimConceptLinkRecord] = field(default_factory=list)

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
            "document_anchor_links": [item.to_dict() for item in self.document_anchor_links],
            "document_period_links": [item.to_dict() for item in self.document_period_links],
            "document_signed_by_links": [item.to_dict() for item in self.document_signed_by_links],
            "person_affiliation_links": [item.to_dict() for item in self.person_affiliation_links],
            "observations": [item.to_dict() for item in self.observations],
            "years": [item.to_dict() for item in self.years],
            "observation_measurement_links": [item.to_dict() for item in self.observation_measurement_links],
            "document_year_links": [item.to_dict() for item in self.document_year_links],
            "entity_hierarchy_links": [item.to_dict() for item in self.entity_hierarchy_links],
            "entity_resolution_confirmations": [item.to_dict() for item in self.entity_resolution_confirmations],
            "events": [item.to_dict() for item in self.events],
            "event_observation_links": [item.to_dict() for item in self.event_observation_links],
            "event_measurement_links": [item.to_dict() for item in self.event_measurement_links],
            "claim_concept_links": [item.to_dict() for item in self.claim_concept_links],
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
            document_anchor_links=[
                DocumentAnchorLinkRecord.from_dict(row)
                for row in payload.get("document_anchor_links", payload.get("document_refuge_links", []))
            ],
            document_period_links=[DocumentPeriodLinkRecord.from_dict(row) for row in payload["document_period_links"]],
            document_signed_by_links=[DocumentSignedByLinkRecord.from_dict(row) for row in payload["document_signed_by_links"]],
            person_affiliation_links=[PersonAffiliationLinkRecord.from_dict(row) for row in payload["person_affiliation_links"]],
            observations=[ObservationRecord.from_dict(row) for row in payload.get("observations", [])],
            years=[YearRecord.from_dict(row) for row in payload.get("years", [])],
            observation_measurement_links=[ObservationMeasurementLinkRecord.from_dict(row) for row in payload.get("observation_measurement_links", [])],
            document_year_links=[DocumentYearLinkRecord.from_dict(row) for row in payload.get("document_year_links", [])],
            entity_hierarchy_links=[
                EntityHierarchyLinkRecord.from_dict(row)
                for row in payload.get("entity_hierarchy_links", payload.get("place_refuge_links", []))
            ],
            entity_resolution_confirmations=[EntityResolutionConfirmationRecord.from_dict(row) for row in payload.get("entity_resolution_confirmations", [])],
            events=[EventRecord.from_dict(row) for row in payload.get("events", [])],
            event_observation_links=[EventObservationLinkRecord.from_dict(row) for row in payload.get("event_observation_links", [])],
            event_measurement_links=[EventMeasurementLinkRecord.from_dict(row) for row in payload.get("event_measurement_links", [])],
            claim_concept_links=[ClaimConceptLinkRecord.from_dict(row) for row in payload.get("claim_concept_links", [])],
        )
