from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class QueryIntent:
    """Output of Layer 0: query intent classifier."""

    bucket: Literal["analytical", "conversational", "hybrid"]
    classifier_confidence: float
    # Surface forms of entities detected in the query text.
    entities: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    # Claim types inferred from query vocabulary (optional hint for Cypher filtering).
    claim_types: list[str] = field(default_factory=list)


@dataclass
class ResolvedEntity:
    """A single entity resolved from a surface form mention."""

    surface_form: str
    entity_id: str
    entity_type: str
    resolution_confidence: float
    resolution_relation: str  # "REFERS_TO" or "POSSIBLY_REFERS_TO"


@dataclass
class EntityContext:
    """Output of Layer 1: entity resolution gateway."""

    # High-confidence resolutions (REFERS_TO, score >= 0.85 with uniqueness gap).
    resolved: list[ResolvedEntity] = field(default_factory=list)
    # Low-confidence matches (POSSIBLY_REFERS_TO); surfaced to caller as clarification candidates.
    ambiguous: list[str] = field(default_factory=list)
    # No match found above maybe_threshold.
    unresolved: list[str] = field(default_factory=list)


@dataclass
class AnalyticalResult:
    """Structured result from Layer 2A: Cypher Query Builder."""

    query_name: str
    columns: list[str]
    rows: list[dict[str, Any]]

    def to_summary_text(self) -> str:
        """Render rows as a compact text table for inclusion in LLM context."""
        if not self.rows:
            return f"[{self.query_name}: no results]"
        header = " | ".join(self.columns)
        lines = [header, "-" * len(header)]
        for row in self.rows:
            lines.append(" | ".join(str(row.get(col, "")) for col in self.columns))
        return "\n".join(lines)


@dataclass
class ProvenanceBlock:
    """One claim with its full documentary provenance chain.

    Assembled by Layer 2B: Provenance Context Assembler and serialised into
    the context window passed to the synthesis engine.
    """

    doc_title: str
    doc_date_start: str | None
    doc_date_end: str | None
    page_number: int | None
    paragraph_id: str
    claim_id: str
    claim_type: str
    extraction_confidence: float
    epistemic_status: str
    source_sentence: str
    observation_type: str | None = None
    species_name: str | None = None
    year: int | None = None
    # Each entry: {"name": str, "value": float|None, "unit": str|None, "approximate": bool}
    measurements: list[dict[str, Any]] = field(default_factory=list)
    # Relationship types traversed from anchor entity to this claim (entity-anchored path only).
    traversal_rel_types: list[str] = field(default_factory=list)


@dataclass
class RetrievalStats:
    """Coverage metadata populated by the context assembler and web layer."""

    candidates_retrieved: int     # rows from Neo4j before OCR filter + budget cap
    ocr_dropped: int              # claims dropped by the OCR corruption filter
    claims_in_context: int        # claims actually sent to synthesis
    paragraphs_in_context: int    # unique paragraph_ids in context blocks
    documents_in_context: int     # unique doc_titles in context blocks
    corpus_total_paragraphs: int  # cached at startup
    corpus_total_documents: int   # cached at startup


@dataclass
class SynthesisResult:
    """Typed response from Layer 3: Synthesis Engine."""

    answer: str
    confidence_assessment: str
    supporting_claim_ids: list[str]
    caveats: list[str]
    # Minimum extraction_confidence across supporting claims (populated by assembler).
    min_extraction_confidence: float | None = None
    # Present when the analytical path executed (analytical or hybrid queries).
    analytical_result: AnalyticalResult | None = None
    # Surface forms that resolved only to POSSIBLY_REFERS_TO (ambiguous).
    ambiguous_entities: list[str] = field(default_factory=list)
    # Coverage metadata for the retrieval pass.
    retrieval_stats: RetrievalStats | None = None
