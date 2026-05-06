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
    # Document types inferred from query vocabulary (e.g. "land_grant", "glo_record").
    doc_type_hints: list[str] = field(default_factory=list)
    # Document series inferred from query vocabulary (e.g. "BLM GLO Records").
    series_hints: list[str] = field(default_factory=list)
    # True when the query asks for entities (persons, places) appearing in BOTH
    # of two named corpora — e.g. "individuals named in Spokane newspaper coverage
    # who also appear as patentees in GLO records".  When set, the retrieval
    # cascade routes through CROSS_CORPUS_OVERLAP_*_QUERY before falling back to
    # the standard entity / fulltext modes.
    cross_corpus_overlap: bool = False


@dataclass
class ResolvedEntity:
    """A single entity resolved from a surface form mention."""

    surface_form: str
    entity_id: str
    entity_type: str
    resolution_confidence: float
    resolution_relation: str  # "REFERS_TO" or "POSSIBLY_REFERS_TO"


@dataclass
class Tier1Intent:
    """Detected structural-lookup intent that bypasses the synthesis engine.

    Tier 1 covers questions whose answer is fully derivable from the graph
    without LLM reasoning (counts, bare lists, year ranges).  When a query
    matches a Tier 1 pattern AND its required filters resolve cleanly, the
    request handler runs a focused Cypher and returns a deterministic answer
    in well under a second.

    Operations:
      - count_claims_for_entity            "How many claims mention X?"
      - year_range_for_entity              "What years cover X?" / "When was X documented?"
      - list_documents_for_entity          "List documents about X"
      - list_documents_for_year            "List documents from 1909"
      - locations_for_entity               "Where does X appear?" / "What places ..."
      - co_occurring_entities_for_entity   "Who appears with X?" / "What entities ..."
      - frequency_ranking_for_type         "Top 10 people by claim count" / "Most common species"
    """

    operation: Literal[
        "count_claims_for_entity",
        "year_range_for_entity",
        "list_documents_for_entity",
        "list_documents_for_year",
        "locations_for_entity",
        "co_occurring_entities_for_entity",
        "frequency_ranking_for_type",
    ]
    entity_id: str = ""             # populated for entity-anchored operations
    entity_name: str = ""           # surface form, or year/type label otherwise
    entity_type: str = ""           # entity label, "Year", or target type for ranking
    target_year: int | None = None  # populated for year-anchored operations
    target_label: str = ""          # populated for type-anchored ops (frequency_ranking)
    limit: int = 25                 # row cap for list / ranking ops


@dataclass
class Tier1Result:
    """Structured output of a Tier 1 lookup, ready for response serialisation.

    Different operations populate different subsets of fields; the frontend
    introspects ``operation`` and renders the appropriate shape (count badge,
    document table, year range, etc.).
    """

    operation: str
    entity_name: str
    answer_text: str
    count: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    # Populated by year_range_for_entity:
    earliest_year: int | None = None
    latest_year: int | None = None
    distinct_years: int | None = None
    # Populated by list_documents_for_year:
    target_year: int | None = None


@dataclass
class EntityContext:
    """Output of Layer 1: entity resolution gateway."""

    # High-confidence resolutions (REFERS_TO, score >= 0.85 with uniqueness gap).
    resolved: list[ResolvedEntity] = field(default_factory=list)
    # Low-confidence matches (POSSIBLY_REFERS_TO); surfaced to caller as clarification candidates.
    ambiguous: list[str] = field(default_factory=list)
    # No match found above maybe_threshold.
    unresolved: list[str] = field(default_factory=list)
    # POSSIBLY_REFERS_TO matches retained with graph IDs so the retrieval plan
    # can optionally include them when a strong anchor already exists.
    maybe_resolved: list[ResolvedEntity] = field(default_factory=list)


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
    # Access classification and donor restriction flags, populated from the Document node.
    access_level: str = "public"
    donor_restricted: bool = False
    # Document identifier; used for archivist annotation lookup in context assembly.
    doc_id: str = ""
    institution_id: str | None = None
    # Archival series (e.g. "BLM GLO Records") for citation provenance disambiguation
    # in composite collections.  None when the document is not part of a named series.
    series: str | None = None


@dataclass
class RetrievalPlanAttempt:
    """One rung of the relaxation ladder attempted by the router."""

    label: str
    template_names: list[str]
    row_count: int
    duration_ms: float = 0.0


@dataclass
class RetrievalPlanTrace:
    """Structured trace of router decisions for a single query."""

    attempts: list[RetrievalPlanAttempt] = field(default_factory=list)
    resolved_entity_ids: list[str] = field(default_factory=list)
    selected_label: str = ""


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
    plan_trace: RetrievalPlanTrace | None = None


@dataclass
class SidecarCell:
    """One clickable (or static) cell in the structural sidecar."""

    label: str
    value: int | str
    # Natural-language Tier 1 phrasing injected into the chat input on click.
    # None for static (display-only) cells.
    template: str | None = None
    # Advisory hint for telemetry; the client does not dispatch on this.
    tier1_op: str | None = None


@dataclass
class StructuralView:
    """Compact, structurally-derived view of a synthesis-path retrieval.

    Computed from `ProvenanceBlock` blocks plus one targeted co-occurrence
    Cypher and the institution-scoped quarantine count.  Rendered as a
    horizontal strip between the answer prose and the confidence strip.
    None when fewer than 5 blocks were retrieved (the strip degenerates to
    redundant info below that threshold).
    """

    claim_count: SidecarCell
    document_count: SidecarCell
    paragraph_count: SidecarCell
    year_range: SidecarCell
    # Decade buckets sorted ascending; each entry: {"decade": int, "n": int}.
    year_buckets: list[dict[str, int]] = field(default_factory=list)
    # Each entry: {name, entity_id, entity_type, n, template, tier1_op}.
    top_entities: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {name, n, template, tier1_op}.
    top_claim_types: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {name, n, template} (no Tier 1 op binding).
    top_series: list[dict[str, Any]] = field(default_factory=list)
    # Counts bucketed against thresholds [0.85, 0.70].
    confidence_dist: dict[str, int] = field(default_factory=dict)
    # Counter over ProvenanceBlock.epistemic_status values.
    epistemic_dist: dict[str, int] = field(default_factory=dict)
    # Cells frontend renders only when value > 0.
    quarantined: SidecarCell | None = None
    ocr_dropped: SidecarCell | None = None


RetrievalPath = Literal["rule_based", "llm_cypher", "fulltext_fallback", "empty"]

# Closed enum for the synthesizer's declared organizing principle.  Stored as
# a literal so the schema gate can reject malformed values rather than letting
# free-form structure labels through.
OrganizingStructure = Literal["chronological", "causal", "thematic", "mixed"]


@dataclass
class AnswerSentence:
    """One sentence of synthesis output, paired with its supporting evidence.

    The schema gate guarantees `is_inferred=True` OR `len(supporting_claim_ids) >= 1`
    for every sentence — a synthesis answer cannot smuggle uncited assertions
    past the validator.
    """

    text: str
    supporting_claim_ids: list[str] = field(default_factory=list)
    is_inferred: bool = False


@dataclass
class SectionOutline:
    """One section of the synthesis answer, when the model chooses to use sections."""

    heading: str
    claim_ids: list[str] = field(default_factory=list)


@dataclass
class CorpusContribution:
    """Per-corpus block count and year range, populated by the deterministic
    provenance builder from the actual retrieved blocks (not the model)."""

    institution_id: str
    block_count: int
    year_min: int | None = None
    year_max: int | None = None


@dataclass
class CorpusDeprioritization:
    """Why a searched corpus didn't contribute to the synthesis context."""

    institution_id: str
    reason: str  # e.g. "no matches", "below tier threshold", "budget cap"
    candidate_count: int = 0  # blocks the corpus offered before drop, when known


@dataclass
class RetrievalProvenance:
    """Deterministic preamble describing what the retrieval pass actually did.

    Built server-side from `permitted_institution_ids` + the assembled
    provenance blocks; the synthesis model never participates in writing it.
    Lets a reader distinguish "we found nothing in GLO" from "we found GLO
    material but ranked CCP higher" from "GLO was not searched."
    """

    searched: list[str] = field(default_factory=list)
    contributed: list[CorpusContribution] = field(default_factory=list)
    deprioritized: list[CorpusDeprioritization] = field(default_factory=list)


@dataclass
class DivergenceWarning:
    """Server-detected mismatch between the model's declared organizing
    structure and a structural signal in the retrieved blocks (e.g., a tight
    temporal cluster but the model picked thematic).  Surfaced inline to the
    user *and* logged for reviewer audit."""

    declared_structure: OrganizingStructure
    detected_signal: str  # e.g. "tight_temporal_cluster", "monotonic_dated_sequence"
    message: str  # one-line user-facing explanation


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
    # Which retrieval path produced the provenance context. UI flags
    # llm_cypher answers as "lower trust" for archivist reviewers.
    retrieval_path: RetrievalPath = "rule_based"
    # Sentence-level decomposition of the answer.  Each entry is gated by the
    # schema validator: every sentence is either cited or marked inferred.
    # Empty when synthesis fell back to the legacy prose path (e.g. a model
    # that declined to follow the structured schema after one retry).
    answer_sentences: list[AnswerSentence] = field(default_factory=list)
    # Closed enum the model picks before drafting; None when synthesis fell back.
    organizing_structure: OrganizingStructure | None = None
    section_outline: list[SectionOutline] = field(default_factory=list)
    # Server-built; the model never writes this.
    retrieval_provenance: RetrievalProvenance | None = None
    # Server-detected mismatch between the model's structural choice and the
    # data; None when no divergence was detected.
    structure_divergence: DivergenceWarning | None = None
