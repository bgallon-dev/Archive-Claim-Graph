"""Layer 2B — Provenance Context Assembler (Conversational Path).

Retrieval cascade:
  1. Entity-anchored graph traversal (ENTITY_ANCHORED_CLAIMS_QUERY) when at
     least one high-confidence entity was resolved by the gateway.
  2. Fulltext keyword fallback (FULLTEXT_CLAIMS_QUERY) when entity resolution
     returned no REFERS_TO matches.

Each retrieved row is mapped to a ProvenanceBlock, ranked by
extraction_confidence, and budget-capped.  Blocks are then serialised into
the structured text format expected by the synthesis engine.
"""
from __future__ import annotations

import collections
import logging
import re

from gemynd.core.graph.cypher import (
    CLAIM_TYPE_SCOPED_QUERY,
    DOC_TYPE_BROWSE_QUERY,
    ENTITY_ANCHORED_CLAIMS_QUERY,
    FULLTEXT_CLAIMS_QUERY,
    MULTI_ENTITY_CLAIMS_QUERY,
    SERIES_BROWSE_QUERY,
    TEMPORAL_CLAIMS_QUERY,
    build_document_anchor_query,
    build_temporal_with_anchor_query,
)
from .executor import Neo4jQueryExecutor
from .models import (
    EntityContext,
    ProvenanceBlock,
    QueryIntent,
    RetrievalPath,
    RetrievalPlanAttempt,
    RetrievalPlanTrace,
)

_log = logging.getLogger(__name__)

# Default context window budgets.
_BUDGET_CONVERSATIONAL = 30

_BUDGET_HYBRID = 4

# Minimum rows an attempt must produce to count as "satisfying" the query.
# Falling below this threshold triggers the next relaxation step.
_MIN_RESULTS_THRESHOLD = 1

# Confidence below which we are willing to spend an LLM call on Cypher
# generation. Confident-but-zero answers (e.g. "no claims about X") are
# semantically meaningful and should not trip the fallback.
_LLM_FALLBACK_CONFIDENCE_CEILING = 0.7


# A single rung of the relaxation ladder: a label + one or more (cypher, params)
# pairs that are executed and merged as a single logical attempt.
PlanAttempt = tuple[str, list[tuple[str, dict]]]

# Fallback intent map used when no config-driven query_intent_map is provided.
_FALLBACK_INTENT_MAP: dict[str, list[str]] = {
    "predator": ["predator_control", "management_action"],
    "habitat": ["habitat_condition", "weather_observation"],
    "population": ["population_estimate", "species_presence", "species_absence"],
    "breeding": ["breeding_activity"],
    "migration": ["migration_timing"],
    "fire": ["fire_incident"],
    "management": ["management_action", "development_activity", "economic_use"],
}


# Lucene special characters that must be escaped to prevent query injection
# when user text is passed to db.index.fulltext.queryNodes().
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
_FULLTEXT_MAX_LEN = 500


def _sanitize_fulltext(text: str) -> str:
    """Escape Lucene special characters and enforce a length cap.

    Prevents field-qualified queries (e.g. normalized_sentence:*) and expensive
    fuzzy/wildcard operators from reaching the Neo4j fulltext index.
    """
    return _LUCENE_SPECIAL.sub(r'\\\1', text.strip()[:_FULLTEXT_MAX_LEN])


# Function words stripped during fulltext term extraction. Intentionally narrow:
# we keep domain nouns that the classifier used to drop (stand, patterns, …) so
# they still contribute to fulltext scoring when no entity anchor resolved.
_FULLTEXT_STOPWORDS: frozenset[str] = frozenset({
    "what", "when", "where", "which", "while", "were", "have", "been", "does", "did",
    "that", "this", "these", "those", "their", "there", "then", "than",
    "with", "from", "about", "over", "under", "into", "onto", "upon",
    "describe", "explain", "tell", "show", "give", "find", "list",
    "during", "across",
    "year", "years", "time", "times", "date", "dates",
})


def _extract_fulltext_terms(text: str, max_terms: int = 5) -> str:
    """Extract 3–5 content-bearing terms and OR-join them as a Lucene query.

    Falls back to sanitized raw text if no terms survive filtering. Preserves
    Lucene escaping and the 500-char cap that _sanitize_fulltext guarantees.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text)
    if not tokens:
        return _sanitize_fulltext(text)

    def _rank(tok: str) -> tuple[int, int]:
        # Capitalized tokens first, then longer tokens.
        return (0 if tok[0].isupper() else 1, -len(tok))

    seen: set[str] = set()
    ranked: list[str] = []
    for tok in sorted((t for t in tokens if t.lower() not in _FULLTEXT_STOPWORDS), key=_rank):
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append(tok)
        if len(ranked) >= max_terms:
            break

    if not ranked:
        return _sanitize_fulltext(text)

    joined = " OR ".join(_sanitize_fulltext(t) for t in ranked)
    return joined[:_FULLTEXT_MAX_LEN]


def _infer_claim_types(
    query_text: str,
    intent_map: dict[str, list[str]] | None = None,
) -> list[str] | None:
    """Map query vocabulary to claim_type filters. Returns None if no signal."""
    lowered = query_text.lower()
    matched: set[str] = set()
    _map = intent_map if intent_map is not None else _FALLBACK_INTENT_MAP
    for keyword, types in _map.items():
        if keyword in lowered:
            matched.update(types)
    return list(matched) if matched else None


def _select_retrieval_plan(
    query_text: str,
    entity_context: "EntityContext",
    year_min: int | None,
    year_max: int | None,
    budget: int,
    permitted_levels: list[str] | None = None,
    institution_ids: list[str] | None = None,
    *,
    intent_map: dict[str, list[str]] | None = None,
    anchor_temporal_plans: list[tuple[str, str]] | None = None,
    anchor_document_plans: dict[str, str] | None = None,
    doc_types: list[str] | None = None,
    series: list[str] | None = None,
) -> list[PlanAttempt]:
    """Return an ordered list of relaxation attempts matching the query shape.

    Each attempt is a ``(label, [(cypher, params), ...])`` pair. The assembler
    runs the attempts in order: the first that yields at least
    ``_MIN_RESULTS_THRESHOLD`` rows wins. Subsequent attempts drop progressively
    more filters (claim_type → year bounds → degrade template) to recover from
    over-constrained primary plans.

    Priority cascade of primary plans:
    1. Temporal — year bounds present, no entity anchor
    2. Multi-entity comparative — 2+ resolved entities
    3. Single entity + claim_type signal — entity-anchored over-fetch + post-filter
    4. Claim-type scoped — vocabulary signal, no entity
    5. Single entity fallback — 1 entity, no vocabulary signal
    6. Fulltext — nothing else matched
    """
    inferred_claim_types = _infer_claim_types(query_text, intent_map=intent_map)
    resolved_ids = list(dict.fromkeys(e.entity_id for e in entity_context.resolved))
    maybe_ids = [
        e.entity_id
        for e in getattr(entity_context, "maybe_resolved", [])
        if e.entity_id not in set(resolved_ids)
    ]
    maybe_ids = list(dict.fromkeys(maybe_ids))
    has_years = year_min is not None or year_max is not None
    has_entities = len(resolved_ids) > 0
    has_claim_types = bool(inferred_claim_types)

    _permitted = permitted_levels if permitted_levels is not None else ["public"]
    _ids = list(institution_ids) if institution_ids else []
    _access_params = {"permitted_levels": _permitted, "institution_ids": _ids}

    def _temporal_queries(
        ymin: int | None, ymax: int | None, ctypes: list[str] | None,
    ) -> list[tuple[str, dict]]:
        if len(resolved_ids) == 0 and anchor_temporal_plans:
            queries: list[tuple[str, dict]] = []
            for anchor_id, anchor_cypher in anchor_temporal_plans:
                queries.append((
                    anchor_cypher,
                    {
                        "anchor_id": anchor_id,
                        "year_min": ymin,
                        "year_max": ymax,
                        "claim_types": ctypes,
                        "limit": budget * 3,
                        **_access_params,
                    },
                ))
            if queries:
                return queries
        return [(TEMPORAL_CLAIMS_QUERY, {
            "year_min": ymin,
            "year_max": ymax,
            "claim_types": ctypes,
            "limit": budget * 3,
            **_access_params,
        })]

    def _entity_anchored_queries(
        ymin: int | None, ymax: int | None,
    ) -> list[tuple[str, dict]]:
        queries: list[tuple[str, dict]] = [(ENTITY_ANCHORED_CLAIMS_QUERY, {
            "entity_id": resolved_ids[0], "year_min": ymin,
            "year_max": ymax, "limit": budget * 3,
            **_access_params,
        })]
        if anchor_document_plans:
            for resolved_id in resolved_ids:
                doc_cypher = anchor_document_plans.get(resolved_id)
                if doc_cypher is not None:
                    queries.append((doc_cypher, {
                        "anchor_id": resolved_id,
                        "year_min": ymin,
                        "year_max": ymax,
                        "claim_types": inferred_claim_types,
                        "limit": budget * 3,
                        **_access_params,
                    }))
        return queries

    def _fulltext_query(text: str) -> list[tuple[str, dict]]:
        return [(FULLTEXT_CLAIMS_QUERY, {
            "search_text": _extract_fulltext_terms(text),
            "limit": budget * 2,
            **_access_params,
        })]

    attempts: list[PlanAttempt] = []

    # When a doc_type scope is active (from NL detection or explicit API param),
    # try a pure browse of documents of that type FIRST.  If no documents with
    # that doc_type exist the attempt returns 0 rows and the cascade falls
    # through to the normal entity/fulltext path unchanged — no regression.
    if doc_types:
        _doc_types_list = list(doc_types)
        attempts.append(("doc_type_browse", [(DOC_TYPE_BROWSE_QUERY, {
            "doc_types": _doc_types_list,
            "claim_types": inferred_claim_types,
            "year_min": year_min,
            "year_max": year_max,
            "limit": budget * 3,
            **_access_params,
        })]))
        # Relax claim_type and year constraints before giving up on doc_type scope.
        if inferred_claim_types:
            attempts.append(("doc_type_browse:drop_claim_type", [(DOC_TYPE_BROWSE_QUERY, {
                "doc_types": _doc_types_list,
                "claim_types": None,
                "year_min": year_min,
                "year_max": year_max,
                "limit": budget * 3,
                **_access_params,
            })]))
        if year_min is not None or year_max is not None:
            attempts.append(("doc_type_browse:drop_years", [(DOC_TYPE_BROWSE_QUERY, {
                "doc_types": _doc_types_list,
                "claim_types": None,
                "year_min": None,
                "year_max": None,
                "limit": budget * 3,
                **_access_params,
            })]))

    if series:
        _series_list = list(series)
        attempts.append(("series_browse", [(SERIES_BROWSE_QUERY, {
            "series": _series_list,
            "claim_types": inferred_claim_types,
            "year_min": year_min,
            "year_max": year_max,
            "limit": budget * 3,
            **_access_params,
        })]))
        if inferred_claim_types:
            attempts.append(("series_browse:drop_claim_type", [(SERIES_BROWSE_QUERY, {
                "series": _series_list,
                "claim_types": None,
                "year_min": year_min,
                "year_max": year_max,
                "limit": budget * 3,
                **_access_params,
            })]))
        if year_min is not None or year_max is not None:
            attempts.append(("series_browse:drop_years", [(SERIES_BROWSE_QUERY, {
                "series": _series_list,
                "claim_types": None,
                "year_min": None,
                "year_max": None,
                "limit": budget * 3,
                **_access_params,
            })]))

    if has_years and len(resolved_ids) <= 1:
        # --- Primary: temporal (with or without corpus anchor) ---
        attempts.append(("temporal", _temporal_queries(year_min, year_max, inferred_claim_types)))
        # Relax claim_type first, then year bounds.
        if inferred_claim_types:
            attempts.append((
                "temporal:drop_claim_type",
                _temporal_queries(year_min, year_max, None),
            ))
        attempts.append((
            "temporal:drop_years",
            _temporal_queries(None, None, None),
        ))
    elif len(resolved_ids) >= 2:
        # --- Primary: multi-entity comparative ---
        attempts.append(("multi_entity", [(MULTI_ENTITY_CLAIMS_QUERY, {
            "entity_ids": resolved_ids, "claim_types": inferred_claim_types,
            "year_min": year_min, "year_max": year_max, "limit": budget * 2,
            **_access_params,
        })]))
        if has_years:
            attempts.append(("multi_entity:drop_years", [(MULTI_ENTITY_CLAIMS_QUERY, {
                "entity_ids": resolved_ids, "claim_types": inferred_claim_types,
                "year_min": None, "year_max": None, "limit": budget * 2,
                **_access_params,
            })]))
        if inferred_claim_types:
            attempts.append(("multi_entity:drop_claim_type", [(MULTI_ENTITY_CLAIMS_QUERY, {
                "entity_ids": resolved_ids, "claim_types": None,
                "year_min": None, "year_max": None, "limit": budget * 2,
                **_access_params,
            })]))
        # Degrade to single-entity scans, one per resolved id.
        degraded: list[tuple[str, dict]] = []
        for rid in resolved_ids:
            degraded.append((ENTITY_ANCHORED_CLAIMS_QUERY, {
                "entity_id": rid, "year_min": None, "year_max": None,
                "limit": budget * 2, **_access_params,
            }))
        attempts.append(("multi_entity:degrade_single", degraded))
    elif has_entities and has_claim_types:
        # --- Primary: entity-anchored over-fetch + post-filter ---
        attempts.append(("entity_anchored", _entity_anchored_queries(year_min, year_max)))
        if has_years:
            attempts.append((
                "entity_anchored:drop_years",
                _entity_anchored_queries(None, None),
            ))
        # If we have one strong anchor and one or more POSSIBLY_REFERS_TO
        # candidates, try a multi-entity comparative using the combined set
        # before falling back to claim_type-only.
        if len(resolved_ids) == 1 and maybe_ids:
            combined = resolved_ids + maybe_ids
            attempts.append(("entity_anchored:include_maybe", [(MULTI_ENTITY_CLAIMS_QUERY, {
                "entity_ids": combined, "claim_types": inferred_claim_types,
                "year_min": year_min, "year_max": year_max, "limit": budget * 2,
                **_access_params,
            })]))
        # Fall through to claim-type-scoped: the keyword signal is the only
        # thing left once the entity anchor is exhausted.
        attempts.append(("entity_anchored:drop_entity_anchor", [(CLAIM_TYPE_SCOPED_QUERY, {
            "claim_types": inferred_claim_types, "entity_ids": None,
            "year_min": None, "year_max": None, "limit": budget * 2,
            **_access_params,
        })]))
    elif has_claim_types and not has_entities:
        attempts.append(("claim_type_scoped", [(CLAIM_TYPE_SCOPED_QUERY, {
            "claim_types": inferred_claim_types, "entity_ids": None,
            "year_min": year_min, "year_max": year_max, "limit": budget * 2,
            **_access_params,
        })]))
        if has_years:
            attempts.append(("claim_type_scoped:drop_years", [(CLAIM_TYPE_SCOPED_QUERY, {
                "claim_types": inferred_claim_types, "entity_ids": None,
                "year_min": None, "year_max": None, "limit": budget * 2,
                **_access_params,
            })]))
        attempts.append(("claim_type_scoped:fulltext", _fulltext_query(query_text)))
    elif has_entities:
        attempts.append(("entity_only", _entity_anchored_queries(year_min, year_max)))
        if has_years:
            attempts.append((
                "entity_only:drop_years",
                _entity_anchored_queries(None, None),
            ))
        if len(resolved_ids) == 1 and maybe_ids:
            combined = resolved_ids + maybe_ids
            attempts.append(("entity_only:include_maybe", [(MULTI_ENTITY_CLAIMS_QUERY, {
                "entity_ids": combined, "claim_types": None,
                "year_min": year_min, "year_max": year_max, "limit": budget * 2,
                **_access_params,
            })]))
    else:
        attempts.append(("fulltext", _fulltext_query(query_text)))
        # Token-level OR fallback: drop Lucene's AND-by-default and require
        # any 4+ char token to match. Reuses _sanitize_fulltext per token.
        tokens = [
            t for t in re.findall(r"[A-Za-z]{4,}", query_text)
            if t.lower() not in {"what", "when", "where", "which", "does", "did", "have", "been"}
        ]
        if tokens:
            or_text = " OR ".join(_sanitize_fulltext(t) for t in tokens[:8])
            attempts.append(("fulltext:token_or", [(FULLTEXT_CLAIMS_QUERY, {
                "search_text": or_text, "limit": budget * 2,
                **_access_params,
            })]))

    return attempts


# ---------------------------------------------------------------------------
# Prompt injection guard
# Detects sequences in archival document text that resemble LLM instructions.
# A sentence that matches is replaced with a placeholder rather than being
# forwarded to the synthesis model verbatim.
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_RE = re.compile(
    r"(ignore\s+(previous|prior|above)\s+instructions?"
    r"|you\s+are\s+now"
    r"|new\s+instructions?"
    r"|system\s*:"
    r"|assistant\s*:"
    r"|<\s*/?\s*(?:system|assistant|user)\s*>)",
    re.IGNORECASE,
)


def _sanitize_claim_text(sentence: str) -> str:
    """Return *sentence* or a redaction placeholder if injection patterns are found.

    XML-escapes the content so it cannot break the <claim_text> delimiters.
    """
    if _PROMPT_INJECTION_RE.search(sentence):
        return "[CLAIM CONTENT REDACTED: potential injection pattern detected]"
    # Escape XML/HTML characters to prevent delimiter break-out.
    return (
        sentence
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Regex that matches characters indicating OCR structural corruption.
# Caret (^) is the primary signal; control characters (non-whitespace) are secondary.
_OCR_GARBAGE_RE = re.compile(r"[\^\x00-\x08\x0b\x0c\x0e-\x1f]")

# Matches the newspaper OCR failure mode where every character is space-separated,
# producing sequences like "G O N TIN u E D" or "N E w S".  We require at least
# 5 consecutive single-character tokens to avoid false positives on legitimate
# initialisms (e.g. "U.S.A." or "P.O. Box").
_OCR_SPACED_CHARS_RE = re.compile(r'(?<!\S)(?:[A-Za-z] ){5,}')


def _is_ocr_corrupted(sentence: str) -> bool:
    """Return True if *sentence* contains characters indicating OCR corruption."""
    if _OCR_GARBAGE_RE.search(sentence):
        return True
    if _OCR_SPACED_CHARS_RE.search(sentence):
        return True
    return False


def _safe_str(value: object) -> str:
    return "" if value is None else str(value)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _row_to_block(row: dict) -> ProvenanceBlock | None:
    """Convert a raw executor result row to a ProvenanceBlock.

    Returns *None* if the row is missing the minimum required fields.
    """
    c = row.get("c") or {}
    d = row.get("d") or {}
    pg = row.get("pg") or {}
    para = row.get("para") or {}
    obs = row.get("obs") or {}
    sp = row.get("sp") or {}
    y = row.get("y") or {}

    claim_id = c.get("claim_id") or ""
    if not claim_id:
        return None

    source_sentence = _safe_str(c.get("source_sentence"))
    if _is_ocr_corrupted(source_sentence):
        return None

    # Measurements come as a list of dicts from the COLLECT in Cypher.
    raw_measurements = row.get("measurements") or []
    measurements: list[dict] = []
    for m in raw_measurements:
        if not m:
            continue
        measurements.append(
            {
                "name": _safe_str(m.get("name")),
                "value": m.get("numeric_value"),
                "unit": _safe_str(m.get("unit")),
                "approximate": bool(m.get("approximate", False)),
            }
        )

    return ProvenanceBlock(
        doc_title=_safe_str(d.get("title")),
        doc_date_start=_safe_str(d.get("start_date") or d.get("date_start")) or None,
        doc_date_end=_safe_str(d.get("end_date") or d.get("date_end")) or None,
        page_number=pg.get("page_number"),
        paragraph_id=_safe_str(para.get("paragraph_id")),
        claim_id=claim_id,
        claim_type=_safe_str(c.get("claim_type")),
        extraction_confidence=_safe_float(c.get("extraction_confidence"), 0.0),
        epistemic_status=_safe_str(c.get("certainty") or c.get("epistemic_status") or "unknown"),
        source_sentence=source_sentence,
        observation_type=_safe_str(obs.get("observation_type")) or None,
        species_name=_safe_str(sp.get("name")) or None,
        year=y.get("year"),
        measurements=measurements,
        access_level=_safe_str(d.get("access_level") or "public"),
        donor_restricted=bool(d.get("donor_restricted", False)),
        doc_id=_safe_str(d.get("doc_id")),
        institution_id=_safe_str(d.get("institution_id")) or None,
    )


def _serialise_block(
    block: ProvenanceBlock,
    query_intent_signal: str = "",
    archivist_note: "dict | None" = None,
) -> str:
    """Render a ProvenanceBlock to the structured text format."""
    date_range = ""
    if block.doc_date_start or block.doc_date_end:
        date_range = f", {block.doc_date_start or '?'}–{block.doc_date_end or '?'}"

    retrieval_note = ""
    if block.traversal_rel_types:
        retrieval_note = f"\n  RETRIEVED_VIA: {', '.join(block.traversal_rel_types)}"

    confidence_tier = (
        "HIGH" if block.extraction_confidence >= 0.85
        else "MEDIUM" if block.extraction_confidence >= 0.70
        else "LOW"
    )

    lines: list[str] = [
        f"DOCUMENT: {block.doc_title}{date_range}{retrieval_note}",
    ]
    if archivist_note:
        note_text = (archivist_note.get("note_text") or "").strip()
        if note_text:
            # Archivist notes come from role-gated write endpoints (archivist/admin only).
            # They are trusted content; no prompt-injection guard is applied here.
            lines.append(f"  ARCHIVIST NOTE: {note_text}")
    lines += [
        f"  CONFIDENCE_TIER: {confidence_tier} ({block.extraction_confidence:.2f})",
        f"  PAGE: {block.page_number if block.page_number is not None else '?'}",
        f"    PARAGRAPH: {block.paragraph_id}",
        (
            f"      CLAIM [{block.claim_type}, "
            f"epistemic={block.epistemic_status}]:"
        ),
        f"        <claim_text>{_sanitize_claim_text(block.source_sentence)}</claim_text>",
    ]

    if block.observation_type:
        lines.append(f"      OBSERVATION [{block.observation_type}]:")
        obs_parts: list[str] = []
        if block.species_name:
            obs_parts.append(f"species={block.species_name}")
        if block.year is not None:
            obs_parts.append(f"year={block.year}")
        if obs_parts:
            lines.append(f"        {', '.join(obs_parts)}")
        for m in block.measurements:
            val = m.get("value")
            approx = " (approximate=True)" if m.get("approximate") else ""
            lines.append(
                f"        MEASUREMENT: {m.get('name')}={val} {m.get('unit') or ''}{approx}".rstrip()
            )

    return "\n".join(lines)


class ProvenanceContextAssembler:
    """Assemble provenance context for conversational and hybrid queries.

    Parameters
    ----------
    executor:
        A live Neo4jQueryExecutor.
    budget_conversational:
        Maximum number of claims to include for conversational queries.
    budget_hybrid:
        Maximum number of claims to include when the analytical path also
        executes (to conserve context window space).
    """

    def __init__(
        self,
        executor: Neo4jQueryExecutor,
        budget_conversational: int = _BUDGET_CONVERSATIONAL,
        budget_hybrid: int = _BUDGET_HYBRID,
        annotation_store: object | None = None,
        *,
        query_intent_map: dict[str, list[str]] | None = None,
        institution_ids: list[str] | None = None,
        anchors: dict[str, tuple[str | None, str | None, str | None]] | None = None,
        llm_cypher_generator: object | None = None,
    ) -> None:
        self._executor = executor
        self._budget_conversational = budget_conversational
        self._budget_hybrid = budget_hybrid
        self._query_intent_map = query_intent_map or _FALLBACK_INTENT_MAP
        self._institution_ids = list(institution_ids) if institution_ids else []
        # Optional text-to-Cypher fallback. None disables the path entirely.
        self._llm_cypher_generator = llm_cypher_generator
        # Per-institution anchors: {inst_id: (entity_id, entity_type, relation)}
        self._anchors: dict[str, tuple[str | None, str | None, str | None]] = dict(anchors or {})

        # Pre-build per-corpus anchor-temporal cypher templates, in the same
        # order as institution_ids so result iteration is deterministic.
        self._anchor_temporal_plans: list[tuple[str, str]] = []
        # Per-corpus document-level anchor plans, keyed by anchor entity_id so
        # the router can look up "is this resolved entity a corpus anchor?"
        self._anchor_document_plans: dict[str, str] = {}
        # Map of cypher template → human label for telemetry.
        self._template_names: dict[str, str] = {
            TEMPORAL_CLAIMS_QUERY: "TEMPORAL",
            MULTI_ENTITY_CLAIMS_QUERY: "MULTI_ENTITY",
            CLAIM_TYPE_SCOPED_QUERY: "CLAIM_TYPE_SCOPED",
            ENTITY_ANCHORED_CLAIMS_QUERY: "ENTITY_ANCHORED",
            FULLTEXT_CLAIMS_QUERY: "FULLTEXT",
            DOC_TYPE_BROWSE_QUERY: "DOC_TYPE_BROWSE",
            SERIES_BROWSE_QUERY: "SERIES_BROWSE",
        }
        for inst_id in self._institution_ids:
            spec = self._anchors.get(inst_id)
            if not spec:
                continue
            entity_id, entity_type, relation = spec
            # Graph-based fallback: resolve anchor entity from graph when
            # not supplied via config (e.g. seed_entities had no match).
            if entity_id is None and entity_type and re.match(r'^[A-Za-z_]+$', entity_type):
                try:
                    rows = executor.run(
                        f"MATCH (d:Document)-->(r:{entity_type})"
                        " WHERE d.institution_id = $inst_id"
                        " RETURN r.entity_id AS eid LIMIT 1",
                        {"inst_id": inst_id},
                    )
                    if rows:
                        entity_id = rows[0]["eid"]
                        self._anchors[inst_id] = (entity_id, entity_type, relation)
                except Exception:
                    pass
            if entity_id and entity_type and relation:
                cypher = build_temporal_with_anchor_query(entity_type, relation)
                self._anchor_temporal_plans.append((entity_id, cypher))
                self._template_names[cypher] = f"TEMPORAL_ANCHOR[{inst_id}]"
                doc_cypher = build_document_anchor_query(entity_type, relation)
                self._anchor_document_plans[entity_id] = doc_cypher
                self._template_names[doc_cypher] = f"DOC_ANCHOR[{inst_id}]"

        # Optional AnnotationStore; when set, archivist notes are injected into context.
        self._annotation_store = annotation_store
        # Populated after each assemble() call; read by the web layer for coverage stats.
        self._last_candidate_count: int = 0
        self._last_ocr_dropped: int = 0
        self._last_context_count: int = 0
        self._last_plan_trace: RetrievalPlanTrace | None = None
        # Tracks which cascade rung produced the rows that fed synthesis.
        # Read by the web layer to set SynthesisResult.retrieval_path.
        self._last_retrieval_path: RetrievalPath = "rule_based"

    def assemble(
        self,
        query_text: str,
        entity_context: EntityContext,
        year_min: int | None = None,
        year_max: int | None = None,
        is_hybrid: bool = False,
        permitted_levels: list[str] | None = None,
        institution_ids: list[str] | None = None,
        intent: QueryIntent | None = None,
        doc_types: list[str] | None = None,
        series: list[str] | None = None,
    ) -> tuple[list[ProvenanceBlock], str]:
        """Retrieve and serialise provenance blocks for *query_text*.

        Returns
        -------
        blocks:
            Ordered list of ProvenanceBlock (ranked by confidence, capped).
        context_text:
            Serialised string ready for injection into the synthesis prompt.
        """
        budget = self._budget_hybrid if is_hybrid else self._budget_conversational
        inferred_claim_types = _infer_claim_types(query_text, intent_map=self._query_intent_map)
        effective_ids = list(institution_ids) if institution_ids else list(self._institution_ids)

        # Merge caller-supplied doc_types with any detected in the query intent.
        effective_doc_types: list[str] | None = None
        intent_doc_hints = list(getattr(intent, "doc_type_hints", None) or [])
        caller_types = list(doc_types) if doc_types else []
        merged_types = list(dict.fromkeys(caller_types + intent_doc_hints))
        if merged_types:
            effective_doc_types = merged_types

        # Merge caller-supplied series with any detected in the query intent.
        effective_series: list[str] | None = None
        intent_series_hints = list(getattr(intent, "series_hints", None) or [])
        caller_series = list(series) if series else []
        merged_series = list(dict.fromkeys(caller_series + intent_series_hints))
        if merged_series:
            effective_series = merged_series

        attempts = _select_retrieval_plan(
            query_text, entity_context, year_min, year_max, budget,
            permitted_levels=permitted_levels,
            institution_ids=effective_ids,
            intent_map=self._query_intent_map,
            anchor_temporal_plans=self._anchor_temporal_plans,
            anchor_document_plans=self._anchor_document_plans,
            doc_types=effective_doc_types,
            series=effective_series,
        )

        rows: list[dict] = []
        claim_rel_types: dict[str, list[str]] = {}
        seen_claim_ids: set[str] = set()
        last_template: str = FULLTEXT_CLAIMS_QUERY

        resolved_ids_for_trace = list(
            dict.fromkeys(e.entity_id for e in entity_context.resolved)
        )
        trace = RetrievalPlanTrace(
            attempts=[],
            resolved_entity_ids=resolved_ids_for_trace,
            selected_label="",
        )

        for label, queries in attempts:
            attempt_rows: list[dict] = []
            attempt_seen: set[str] = set()
            attempt_rel_types: dict[str, list[str]] = {}
            for template, params in queries:
                raw = self._executor.run(template, params)
                for r in raw:
                    c = r.get("c") or {}
                    cid = c.get("claim_id")
                    if cid:
                        rel_type = r.get("traversal_rel_type")
                        if rel_type and rel_type not in attempt_rel_types.get(cid, []):
                            attempt_rel_types.setdefault(cid, []).append(rel_type)
                    if cid and cid not in attempt_seen:
                        attempt_seen.add(cid)
                        attempt_rows.append(r)
                    elif not cid:
                        attempt_rows.append(r)

            trace.attempts.append(RetrievalPlanAttempt(
                label=label,
                template_names=[
                    self._template_names.get(t, "unknown") for t, _ in queries
                ],
                row_count=len(attempt_rows),
            ))

            if len(attempt_rows) >= _MIN_RESULTS_THRESHOLD:
                rows = attempt_rows
                claim_rel_types = attempt_rel_types
                seen_claim_ids = attempt_seen
                last_template = queries[-1][0] if queries else FULLTEXT_CLAIMS_QUERY
                trace.selected_label = label
                break
        else:
            # No attempt produced rows; keep the final attempt's (empty) state
            # so downstream logic sees an empty-but-valid result.
            if attempts:
                trace.selected_label = attempts[-1][0]
                last_template = (
                    attempts[-1][1][-1][0] if attempts[-1][1] else FULLTEXT_CLAIMS_QUERY
                )

        # Determine which retrieval path produced the rows. Defaults to
        # rule_based; the LLM fallback below may override to llm_cypher,
        # and the empty case below to fulltext_fallback or empty.
        retrieval_path: RetrievalPath = "rule_based"

        # --- B1: text-to-Cypher LLM fallback ---------------------------------
        # Fires only when the rule-based cascade returned zero rows AND the
        # classifier is not highly confident in the intent (low-confidence
        # classifications are the ones most likely to have routed wrong).
        # Confident-but-zero answers are semantically meaningful and should
        # not be papered over with an LLM call.
        if (
            not rows
            and self._llm_cypher_generator is not None
            and (intent is None or intent.classifier_confidence < _LLM_FALLBACK_CONFIDENCE_CEILING)
        ):
            llm_cypher = None
            try:
                llm_cypher = self._llm_cypher_generator.generate(query_text, intent)
            except Exception as exc:
                _log.info("LLM cypher generation raised: %s", exc)
            if llm_cypher:
                llm_params = {
                    "institution_ids": effective_ids,
                    "permitted_levels": permitted_levels if permitted_levels is not None else ["public"],
                    "entity_id": (
                        entity_context.resolved[0].entity_id
                        if entity_context.resolved
                        else ""
                    ),
                    "entity_ids": [e.entity_id for e in entity_context.resolved],
                    "anchor_id": (
                        entity_context.resolved[0].entity_id
                        if entity_context.resolved
                        else ""
                    ),
                    "claim_types": inferred_claim_types,
                    "year_min": year_min,
                    "year_max": year_max,
                    "limit": budget * 2,
                    "search_text": _sanitize_fulltext(query_text),
                }
                try:
                    llm_rows = self._executor.run(llm_cypher, llm_params)
                except Exception as exc:
                    _log.info("LLM cypher execution failed: %s", exc)
                    llm_rows = []
                attempt_seen: set[str] = set()
                attempt_rows: list[dict] = []
                for r in llm_rows:
                    cid = (r.get("c") or {}).get("claim_id")
                    if cid and cid not in attempt_seen:
                        attempt_seen.add(cid)
                        attempt_rows.append(r)
                    elif not cid:
                        attempt_rows.append(r)
                trace.attempts.append(RetrievalPlanAttempt(
                    label="llm_cypher",
                    template_names=["LLM_CYPHER"],
                    row_count=len(attempt_rows),
                ))
                if attempt_rows:
                    rows = attempt_rows
                    seen_claim_ids = attempt_seen
                    last_template = llm_cypher
                    trace.selected_label = "llm_cypher"
                    retrieval_path = "llm_cypher"
            else:
                trace.attempts.append(RetrievalPlanAttempt(
                    label="llm_cypher:declined",
                    template_names=["LLM_CYPHER"],
                    row_count=0,
                ))

        # Post-filter by claim_type when entity-anchored path over-fetches (Case 3).
        if last_template == ENTITY_ANCHORED_CLAIMS_QUERY and inferred_claim_types:
            filtered = [
                r for r in rows
                if (r.get("c") or {}).get("claim_type") in inferred_claim_types
            ]
            rows = filtered or rows  # fall back to unfiltered if filter removes everything

        # Diagnostic: log claim_type distribution to stderr for depth/breadth analysis.
        claim_types_in_result = collections.Counter(
            (r.get("c") or {}).get("claim_type", "unknown") for r in rows
        )
        _log.debug(
            "retrieval: selected=%s attempts=%s n_rows=%d claim_type_dist=%s",
            trace.selected_label,
            [(a.label, a.row_count) for a in trace.attempts],
            len(rows),
            dict(claim_types_in_result),
        )

        self._last_candidate_count = len(rows)
        self._last_plan_trace = trace
        if not rows:
            retrieval_path = "empty"
        elif retrieval_path == "rule_based" and last_template == FULLTEXT_CLAIMS_QUERY:
            retrieval_path = "fulltext_fallback"
        self._last_retrieval_path = retrieval_path

        # Convert rows → blocks, filter None (including OCR-corrupted), rank by confidence, cap.
        blocks_raw: list[ProvenanceBlock] = []
        for row in rows:
            block = _row_to_block(row)
            if block is not None:
                block.traversal_rel_types = claim_rel_types.get(block.claim_id, [])
                blocks_raw.append(block)

        self._last_ocr_dropped = self._last_candidate_count - len(blocks_raw)
        blocks_raw.sort(key=lambda b: b.extraction_confidence, reverse=True)
        blocks = blocks_raw[:budget]
        self._last_context_count = len(blocks)

        # Batch-lookup archivist notes for documents present in context.
        _notes: dict[str, dict] = {}
        if self._annotation_store is not None:
            _doc_ids = list(dict.fromkeys(b.doc_id for b in blocks if b.doc_id))
            if _doc_ids:
                _notes = self._annotation_store.get_notes_for_docs(_doc_ids)

        context_text = "\n\n".join(
            _serialise_block(b, archivist_note=_notes.get(b.doc_id) if b.doc_id else None)
            for b in blocks
        )
        return blocks, context_text

    def chain_for_claim(
        self,
        claim_id: str,
        permitted_levels: list[str] | None = None,
        institution_ids: list[str] | None = None,
    ) -> list[ProvenanceBlock]:
        """Return the full provenance chain for a single known *claim_id*.

        Used by the ``POST /query/provenance`` endpoint.
        """
        from ..core.graph.cypher import PROVENANCE_CHAIN_QUERY

        effective_ids = list(institution_ids) if institution_ids else list(self._institution_ids)
        rows = self._executor.run(PROVENANCE_CHAIN_QUERY, {
            "claim_id": claim_id,
            "permitted_levels": permitted_levels if permitted_levels is not None else ["public"],
            "institution_ids": effective_ids,
        })
        blocks = [_row_to_block(r) for r in rows]
        return [b for b in blocks if b is not None]
