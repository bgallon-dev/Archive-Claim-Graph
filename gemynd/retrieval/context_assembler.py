"""Layer 2B — Provenance Context Assembler (Conversational Path).

Retrieval cascade:
  1. Entity-anchored graph traversal (ENTITY_ANCHORED_CLAIMS_QUERY) when at
     least one high-confidence entity was resolved by the gateway.
  2. Fulltext keyword fallback (FULLTEXT_CLAIMS_QUERY) when entity resolution
     returned no REFERS_TO matches.

Each retrieved row is mapped to a ProvenanceBlock, ordered by the configured
ContextRanker (graph proximity / KGR temporal rerank / legacy confidence —
see context_ranker.py), and budget-capped.  Blocks are then serialised into
the structured text format expected by the synthesis engine.
"""
from __future__ import annotations

import collections
import logging
import os
import re
import time

from gemynd.core.graph.cypher import (
    CLAIM_TYPE_SCOPED_QUERY,
    CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY,
    COUNT_CLAIMS_FOR_ENTITY_QUERY,
    CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY,
    CROSS_CORPUS_OVERLAP_PATENT_QUERY,
    DOC_TYPE_BROWSE_QUERY,
    ENTITY_ANCHORED_CLAIMS_QUERY,
    FREQUENCY_RANKING_FOR_TYPE_QUERY,
    FULLTEXT_CLAIMS_QUERY,
    LIST_DOCUMENTS_FOR_ENTITY_QUERY,
    LIST_DOCUMENTS_FOR_YEAR_QUERY,
    LOCATIONS_FOR_ENTITY_QUERY,
    MULTI_ENTITY_CLAIMS_QUERY,
    SERIES_BROWSE_QUERY,
    TEMPORAL_CLAIMS_QUERY,
    YEAR_RANGE_FOR_ENTITY_QUERY,
    build_document_anchor_query,
    build_temporal_with_anchor_query,
)
from .context_ranker import ConfidenceRanker, ContextRanker
from .executor import Neo4jQueryExecutor
from .models import (
    EntityContext,
    ProvenanceBlock,
    QueryIntent,
    RetrievalPath,
    RetrievalPlanAttempt,
    RetrievalPlanTrace,
    SidecarCell,
    StructuralView,
    Tier1Intent,
    Tier1Result,
)

_log = logging.getLogger(__name__)

# Default context window budgets.  Sized so the Cypher templates'
# `LIMIT budget*3` over-fetch (used for confidence-sort headroom) stays
# tractable for Neo4j — values much above 1000 push LIMIT past 3000 and
# routinely defunct the Bolt connection on entity-anchored traversals.
# The char-budget below provides the true context-window protection;
# this constant is mostly belt-and-suspenders against pathological cases.
_BUDGET_CONVERSATIONAL = 1000

_BUDGET_HYBRID = 1000

# Hard ceiling on the serialised context_text size (in characters). Blocks
# beyond this budget are dropped after count-cap stratification, in confidence-
# descending order so the most-trusted claims survive truncation. Keep this
# conservative by default because large archival contexts make synthesis slow
# and brittle; operators can raise it deliberately for larger-context models.
_CONTEXT_CHAR_BUDGET = int(os.environ.get("SYNTHESIS_CONTEXT_CHAR_BUDGET", "250000"))

# Maximum rows any single retrieval Cypher should materialize before Python
# ranking/stratification. Larger context budgets are still honored up to the
# rows Neo4j can safely return, but broad anchor queries must not request
# thousands of provenance rows with optional expansions in one transaction.
_MAX_CYPHER_FETCH_LIMIT = 250

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


def _bounded_query_limit(budget: int, multiplier: int = 1) -> int:
    requested = max(1, int(budget) * int(multiplier))
    return min(requested, _MAX_CYPHER_FETCH_LIMIT)


def _is_neo4j_resource_error(exc: Exception) -> bool:
    """Return True for transient Neo4j resource exhaustion errors."""
    text = " ".join(
        str(part)
        for part in (
            type(exc).__name__,
            getattr(exc, "code", ""),
            getattr(exc, "neo4j_code", ""),
            str(exc),
        )
        if part
    )
    markers = (
        "OutOfMemoryError",
        "TransientError",
        "resource exhaustion",
        "failed to allocate",
        "not enough memory",
    )
    return any(marker in text for marker in markers)


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


def _enumerate_cross_corpus_pairs(
    institution_ids: list[str],
    doc_types: list[str] | None,
    series: list[str] | None,
) -> list[tuple[str, str]]:
    """Return ordered (corpus_a, corpus_b) pairs for the cross-corpus overlap
    cascade.  corpus_a is the newspaper (mention) side; corpus_b is the patent
    (land_grant subject) side.

    All ordered pairs (a, b) with a ≠ b are returned.  The Cypher templates
    self-filter — a wrong-direction query (e.g. corpus_a points at the patent
    corpus) finds no CONTAINS_MENTION→Mention→Person bridge and returns zero
    rows, so the cascade naturally moves to the next attempt.  This avoids
    coupling cascade ordering to ad-hoc institution_id naming.

    `doc_types` and `series` are accepted for symmetry with the rest of
    `_select_retrieval_plan`'s parameter shape; they're not yet used to bias
    pair ordering because the available alias maps don't carry institution_id
    bindings.  Future work: extend `corpus_registry.yaml` with a corpus→
    doc_type/series back-map and prefer the ordered pair whose corpus_a carries
    a newspaper hint.
    """
    if len(institution_ids) < 2:
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a in institution_ids:
        for b in institution_ids:
            if a == b:
                continue
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((a, b))
    return pairs


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
    cross_corpus_overlap: bool = False,
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
                        "limit": _bounded_query_limit(budget, 3),
                        **_access_params,
                    },
                ))
            if queries:
                return queries
        return [(TEMPORAL_CLAIMS_QUERY, {
            "year_min": ymin,
            "year_max": ymax,
            "claim_types": ctypes,
            "limit": _bounded_query_limit(budget, 3),
            **_access_params,
        })]

    def _entity_anchored_queries(
        ymin: int | None, ymax: int | None,
    ) -> list[tuple[str, dict]]:
        queries: list[tuple[str, dict]] = [(ENTITY_ANCHORED_CLAIMS_QUERY, {
            "entity_id": resolved_ids[0], "year_min": ymin,
            "year_max": ymax, "limit": _bounded_query_limit(budget, 3),
            **_access_params,
        })]
        if anchor_document_plans:
            for resolved_id in resolved_ids:
                doc_cypher = anchor_document_plans.get(resolved_id)
                if doc_cypher is not None:
                    # Do NOT apply inferred_claim_types here.  The doc-anchor
                    # path's purpose is to surface ALL claims from documents
                    # structurally linked to the anchor entity (e.g. Turnbull
                    # NWR docs linked via ABOUT_REFUGE).  Claim-type filtering
                    # defeats this when the inferred type is corpus-specific
                    # — e.g. "turnbull_prehistory" exists in the newspaper
                    # corpus but not in Turnbull NWR docs, so filtering by it
                    # removes 100% of Turnbull NWR content from the doc-anchor
                    # path.  Topical filtering still happens via the post-
                    # filter on entity-anchored rows (line ~1093).
                    queries.append((doc_cypher, {
                        "anchor_id": resolved_id,
                        "year_min": ymin,
                        "year_max": ymax,
                        "claim_types": None,
                        "limit": _bounded_query_limit(budget, 3),
                        **_access_params,
                    }))
        return queries

    def _fulltext_query(text: str) -> list[tuple[str, dict]]:
        return [(FULLTEXT_CLAIMS_QUERY, {
            "search_text": _extract_fulltext_terms(text),
            "limit": _bounded_query_limit(budget, 2),
            **_access_params,
        })]

    attempts: list[PlanAttempt] = []

    # Cross-corpus entity overlap takes precedence when the classifier flagged
    # the query as asking for entities present in two corpora.  Requires the
    # deployment to span at least two institution_ids; otherwise there's
    # nothing to overlap and we fall through to the standard cascade.
    if cross_corpus_overlap and len(_ids) >= 2:
        for corpus_a, corpus_b in _enumerate_cross_corpus_pairs(_ids, doc_types, series):
            overlap_params_a = {
                "corpus_a": corpus_a, "corpus_b": corpus_b,
                "claim_types": inferred_claim_types,
                "year_min": year_min, "year_max": year_max,
                "limit": _bounded_query_limit(budget, 2),
                "permitted_levels": _permitted,
            }
            overlap_params_b = {
                "corpus_a": corpus_a, "corpus_b": corpus_b,
                "limit": _bounded_query_limit(budget, 2),
                "permitted_levels": _permitted,
            }
            attempts.append((
                f"cross_corpus_overlap:{corpus_a}+{corpus_b}",
                [
                    (CROSS_CORPUS_OVERLAP_NEWSPAPER_QUERY, overlap_params_a),
                    (CROSS_CORPUS_OVERLAP_PATENT_QUERY, overlap_params_b),
                ],
            ))

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
            "limit": _bounded_query_limit(budget, 3),
            **_access_params,
        })]))
        # Relax claim_type and year constraints before giving up on doc_type scope.
        if inferred_claim_types:
            attempts.append(("doc_type_browse:drop_claim_type", [(DOC_TYPE_BROWSE_QUERY, {
                "doc_types": _doc_types_list,
                "claim_types": None,
                "year_min": year_min,
                "year_max": year_max,
                "limit": _bounded_query_limit(budget, 3),
                **_access_params,
            })]))
        if year_min is not None or year_max is not None:
            attempts.append(("doc_type_browse:drop_years", [(DOC_TYPE_BROWSE_QUERY, {
                "doc_types": _doc_types_list,
                "claim_types": None,
                "year_min": None,
                "year_max": None,
                "limit": _bounded_query_limit(budget, 3),
                **_access_params,
            })]))

    if series:
        _series_list = list(series)
        attempts.append(("series_browse", [(SERIES_BROWSE_QUERY, {
            "series": _series_list,
            "claim_types": inferred_claim_types,
            "year_min": year_min,
            "year_max": year_max,
            "limit": _bounded_query_limit(budget, 3),
            **_access_params,
        })]))
        if inferred_claim_types:
            attempts.append(("series_browse:drop_claim_type", [(SERIES_BROWSE_QUERY, {
                "series": _series_list,
                "claim_types": None,
                "year_min": year_min,
                "year_max": year_max,
                "limit": _bounded_query_limit(budget, 3),
                **_access_params,
            })]))
        if year_min is not None or year_max is not None:
            attempts.append(("series_browse:drop_years", [(SERIES_BROWSE_QUERY, {
                "series": _series_list,
                "claim_types": None,
                "year_min": None,
                "year_max": None,
                "limit": _bounded_query_limit(budget, 3),
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
            "year_min": year_min, "year_max": year_max, "limit": _bounded_query_limit(budget, 2),
            **_access_params,
        })]))
        if has_years:
            attempts.append(("multi_entity:drop_years", [(MULTI_ENTITY_CLAIMS_QUERY, {
                "entity_ids": resolved_ids, "claim_types": inferred_claim_types,
                "year_min": None, "year_max": None, "limit": _bounded_query_limit(budget, 2),
                **_access_params,
            })]))
        if inferred_claim_types:
            attempts.append(("multi_entity:drop_claim_type", [(MULTI_ENTITY_CLAIMS_QUERY, {
                "entity_ids": resolved_ids, "claim_types": None,
                "year_min": None, "year_max": None, "limit": _bounded_query_limit(budget, 2),
                **_access_params,
            })]))
        # Degrade to single-entity scans, one per resolved id.
        degraded: list[tuple[str, dict]] = []
        for rid in resolved_ids:
            degraded.append((ENTITY_ANCHORED_CLAIMS_QUERY, {
                "entity_id": rid, "year_min": None, "year_max": None,
                "limit": _bounded_query_limit(budget, 2), **_access_params,
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
                "year_min": year_min, "year_max": year_max, "limit": _bounded_query_limit(budget, 2),
                **_access_params,
            })]))
        # Fall through to claim-type-scoped: the keyword signal is the only
        # thing left once the entity anchor is exhausted.
        attempts.append(("entity_anchored:drop_entity_anchor", [(CLAIM_TYPE_SCOPED_QUERY, {
            "claim_types": inferred_claim_types, "entity_ids": None,
            "year_min": None, "year_max": None, "limit": _bounded_query_limit(budget, 2),
            **_access_params,
        })]))
    elif has_claim_types and not has_entities:
        attempts.append(("claim_type_scoped", [(CLAIM_TYPE_SCOPED_QUERY, {
            "claim_types": inferred_claim_types, "entity_ids": None,
            "year_min": year_min, "year_max": year_max, "limit": _bounded_query_limit(budget, 2),
            **_access_params,
        })]))
        if has_years:
            attempts.append(("claim_type_scoped:drop_years", [(CLAIM_TYPE_SCOPED_QUERY, {
                "claim_types": inferred_claim_types, "entity_ids": None,
                "year_min": None, "year_max": None, "limit": _bounded_query_limit(budget, 2),
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
                "year_min": year_min, "year_max": year_max, "limit": _bounded_query_limit(budget, 2),
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
                "search_text": or_text, "limit": _bounded_query_limit(budget, 2),
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
        series=_safe_str(d.get("series")) or None,
    )


def _block_year(block: ProvenanceBlock) -> int | None:
    """Return the best available year for a block: observation year, then doc start year."""
    if block.year is not None:
        return block.year
    if block.doc_date_start:
        try:
            return int(block.doc_date_start[:4])
        except (ValueError, IndexError):
            pass
    return None


def _temporally_diversified_take(
    blocks: list[ProvenanceBlock],
    budget: int,
    *,
    min_year_buckets: int = 5,
) -> list[ProvenanceBlock]:
    """Return up to *budget* blocks with proportional coverage across years.

    When the result set spans fewer than *min_year_buckets* distinct non-None
    years, falls back to simple top-N by confidence (blocks must already be
    sorted descending before calling).  Undated blocks (year=None) share one
    bucket and receive a proportional share.
    """
    by_year: dict[int | None, list[ProvenanceBlock]] = {}
    for b in blocks:
        by_year.setdefault(_block_year(b), []).append(b)

    dated_keys = [y for y in by_year if y is not None]
    if len(dated_keys) < min_year_buckets:
        return blocks[:budget]

    total = len(blocks)
    quotas: dict[int | None, int] = {
        year: max(1, round(budget * len(year_blocks) / total))
        for year, year_blocks in by_year.items()
    }

    # Trim any overflow (from rounding + min-1 floor) from the largest bucket.
    overflow = sum(quotas.values()) - budget
    if overflow > 0:
        for year in sorted(quotas, key=lambda y: -quotas[y]):
            if overflow == 0:
                break
            take = min(overflow, quotas[year] - 1)
            quotas[year] -= take
            overflow -= take

    selected: list[ProvenanceBlock] = []
    for year, year_blocks in by_year.items():
        selected.extend(year_blocks[:quotas[year]])
    return selected


def _stratified_budget_cap(
    blocks: list[ProvenanceBlock],
    budget: int,
    *,
    min_corpus_threshold: int = 5,
) -> list[ProvenanceBlock]:
    """Apply budget cap with per-corpus and per-year diversification.

    When multiple corpora are present and at least two each have
    *min_corpus_threshold* blocks, the budget is split equally across them
    (largest corpus absorbs the remainder).  Within each corpus's quota,
    temporal diversification is applied via :func:`_temporally_diversified_take`.

    Falls back to pure temporal diversification when only one corpus is
    significant (including single-corpus deployments).

    Assumes *blocks* is already ranked best-first by the context ranker.
    """
    by_corpus: dict[str, list[ProvenanceBlock]] = {}
    for b in blocks:
        by_corpus.setdefault(b.institution_id or "", []).append(b)

    significant = {k: v for k, v in by_corpus.items() if len(v) >= min_corpus_threshold}
    if len(significant) <= 1:
        return _temporally_diversified_take(blocks, budget)

    # Sort corpora largest-first so the remainder goes to the biggest corpus.
    ordered = sorted(significant.items(), key=lambda kv: -len(kv[1]))
    n = len(ordered)
    quota_per = budget // n
    remainder = budget - quota_per * n

    selected: list[ProvenanceBlock] = []
    for i, (_, corpus_blocks) in enumerate(ordered):
        quota = quota_per + (remainder if i == 0 else 0)
        selected.extend(_temporally_diversified_take(corpus_blocks, quota))
    return selected


def _estimate_block_chars(block: ProvenanceBlock) -> int:
    """Cheap upper-bound estimate of how many chars *block* will serialise to.

    Slightly over-estimates so the cumulative-sum trim in
    :func:`_apply_context_char_budget` undershoots rather than blows the
    Anthropic input budget.
    """
    overhead = 320  # DOCUMENT/CONFIDENCE/PAGE/PARAGRAPH/CLAIM scaffolding
    return (
        overhead
        + len(block.doc_title or "")
        + len(block.source_sentence or "")
        + (60 if block.traversal_rel_types else 0)
        + sum(len(str(m.get("name", ""))) + 30 for m in (block.measurements or []))
    )


def _apply_context_char_budget(
    blocks: list[ProvenanceBlock],
    char_budget: int = _CONTEXT_CHAR_BUDGET,
) -> list[ProvenanceBlock]:
    """Drop blocks (in iteration order) once cumulative serialised size would
    exceed *char_budget*.  Caller is responsible for sorting *blocks* before
    calling — we trim from the tail so confidence-descending input keeps the
    highest-confidence claims.
    """
    out: list[ProvenanceBlock] = []
    cum = 0
    for b in blocks:
        cost = _estimate_block_chars(b)
        if cum + cost > char_budget:
            break
        out.append(b)
        cum += cost
    return out


def _build_retrieval_summary(
    candidates: list[ProvenanceBlock],
    selected: list[ProvenanceBlock],
) -> str:
    """Return a bracketed header line describing truncation, or '' if not truncated."""
    if len(selected) >= len(candidates):
        return ""

    corpus_counts: dict[str, int] = {}
    corpus_years: dict[str, list[int]] = {}
    for b in selected:
        corp = b.institution_id or "unknown"
        corpus_counts[corp] = corpus_counts.get(corp, 0) + 1
        y = _block_year(b)
        if y is not None:
            corpus_years.setdefault(corp, []).append(y)

    parts: list[str] = []
    for corp in sorted(corpus_counts):
        n = corpus_counts[corp]
        years = corpus_years.get(corp, [])
        if years:
            parts.append(f"{n} from {corp} ({min(years)}–{max(years)})")
        else:
            parts.append(f"{n} from {corp}")

    breakdown = ", ".join(parts)
    return (
        f"[RETRIEVAL SUMMARY: {len(selected)} of {len(candidates)} candidate claims"
        f" — {breakdown}]"
    )


# Cypher: given a set of claim_ids, return the top-N entities that participate
# (directly or via observation/event nodes) in those claims, excluding the
# query's resolved anchors.  Permitted-levels filter mirrors the rest of the
# retrieval layer — sensitivity-gated content must never leak via this path.
_TOP_CO_OCCURRING_ENTITIES_QUERY = """
MATCH (c:Claim)-[*1..2]-(e:Entity)
WHERE c.claim_id IN $claim_ids
  AND NOT e.entity_id IN $exclude_entity_ids
  AND ($institution_ids IS NULL OR c.institution_id IN $institution_ids)
  AND ($permitted_levels IS NULL OR c.access_level IN $permitted_levels)
RETURN e.entity_id AS entity_id,
       coalesce(e.name, e.entity_id) AS name,
       [label IN labels(e) WHERE label <> 'Entity'][0] AS entity_type,
       count(DISTINCT c) AS n
ORDER BY n DESC, name ASC
LIMIT $limit
"""


def _compute_top_co_occurring_entities(
    executor: Neo4jQueryExecutor,
    claim_ids: list[str],
    exclude_entity_ids: frozenset[str],
    institution_ids: list[str] | None,
    permitted_levels: list[str] | None,
    limit: int = 5,
) -> list[dict]:
    """Return up to *limit* entities most frequently co-occurring with the given claims.

    Runs a single targeted Cypher query against the executor.  Used by the
    structural sidecar to populate its "top entities" cell with graph-truth
    co-occurrence rather than a sentence-string approximation.

    Returns ``[]`` on any executor failure — the sidecar gracefully omits the
    cell rather than failing the response.  The exception is logged but never
    raised into the request path.
    """
    if not claim_ids:
        return []
    try:
        rows = executor.run(
            _TOP_CO_OCCURRING_ENTITIES_QUERY,
            {
                "claim_ids": list(claim_ids),
                "exclude_entity_ids": list(exclude_entity_ids),
                "institution_ids": institution_ids,
                "permitted_levels": permitted_levels,
                "limit": int(limit),
            },
        )
    except Exception:
        _log.exception("top co-occurring entities query failed; sidecar cell will be empty")
        return []

    out: list[dict] = []
    for row in rows or []:
        name = row.get("name")
        eid = row.get("entity_id")
        if not name or not eid:
            continue
        out.append(
            {
                "name": name,
                "entity_id": eid,
                "entity_type": row.get("entity_type") or "",
                "n": int(row.get("n") or 0),
            }
        )
    return out


# Confidence buckets — must stay aligned with web/app.py:_conf_tier_class
# so the sidecar's distribution numbers match the per-claim badges in the
# sources table.  high ≥ 0.85, medium ≥ 0.70, low otherwise.
_CONFIDENCE_HIGH_CUTOFF = 0.85
_CONFIDENCE_MEDIUM_CUTOFF = 0.70


def _derive_structural_view(
    blocks: list[ProvenanceBlock],
    entity_context: EntityContext,
    top_co_entities: list[dict],
    quarantined_count: int,
    ocr_dropped: int,
    threshold: int = 5,
) -> StructuralView | None:
    """Return a StructuralView summarising the retrieved claim set, or None.

    Returns None when ``len(blocks) < threshold`` — below that, the strip
    degenerates into single-cell "1 doc, 1 year" noise that adds visual
    cost without information.

    The caller is responsible for the (graph-truth) ``top_co_entities`` lookup;
    this function is otherwise pure on its inputs.  ``quarantined_count`` and
    ``ocr_dropped`` are surfaced as their own cells so users see them at the
    answer level rather than buried in a coverage footer.
    """
    if len(blocks) < threshold:
        return None

    # Anchor for natural-language templates.  When no entity resolved, anchor-
    # bound cells fall back to template=None and render as static.
    anchor_name: str | None = None
    if entity_context.resolved:
        anchor_name = entity_context.resolved[0].surface_form or None

    def _anchor_template(template_with_anchor: str) -> str | None:
        if not anchor_name:
            return None
        return template_with_anchor.format(anchor=anchor_name)

    # ── counts ─────────────────────────────────────────────────────────────
    n_claims = len(blocks)
    n_docs = len({b.doc_title for b in blocks if b.doc_title})
    n_paragraphs = len({b.paragraph_id for b in blocks if b.paragraph_id})

    claim_count = SidecarCell(
        label="claims",
        value=n_claims,
        template=_anchor_template("How many claims mention {anchor}?"),
        tier1_op="count_claims_for_entity" if anchor_name else None,
    )
    document_count = SidecarCell(
        label="documents",
        value=n_docs,
        template=_anchor_template("List documents about {anchor}"),
        tier1_op="list_documents_for_entity" if anchor_name else None,
    )
    paragraph_count = SidecarCell(label="paragraphs", value=n_paragraphs)

    # ── year range / decade buckets ────────────────────────────────────────
    years = [y for y in (_block_year(b) for b in blocks) if y is not None]
    if years:
        year_label = f"{min(years)}–{max(years)}"
    else:
        year_label = "no dates"
    year_range = SidecarCell(
        label="years",
        value=year_label,
        template=_anchor_template("What years was {anchor} documented?") if years else None,
        tier1_op="year_range_for_entity" if (anchor_name and years) else None,
    )
    decade_counter: collections.Counter = collections.Counter((y // 10) * 10 for y in years)
    year_buckets = [
        {"decade": int(decade), "n": int(count)}
        for decade, count in sorted(decade_counter.items())
    ]

    # ── top co-occurring entities (graph-truth, computed by caller) ────────
    top_entities: list[dict] = []
    for ent in top_co_entities[:5]:
        name = ent.get("name")
        if not name:
            continue
        top_entities.append(
            {
                "name": name,
                "entity_id": ent.get("entity_id"),
                "entity_type": ent.get("entity_type") or "",
                "n": ent.get("n", 0),
                "template": f"Who appears with {name}?",
                "tier1_op": "co_occurring_entities_for_entity",
            }
        )

    # ── top claim_types (the doc_type analogue — closest typed dimension) ──
    claim_type_counter = collections.Counter(b.claim_type for b in blocks if b.claim_type)
    top_claim_types = [
        {
            "name": ct,
            "n": int(n),
            "template": f"Most common entities of type {ct}",
            "tier1_op": "frequency_ranking_for_type",
        }
        for ct, n in claim_type_counter.most_common(5)
    ]

    # ── top series (drops corpora that don't use named series) ─────────────
    series_counter = collections.Counter(b.series for b in blocks if b.series)
    top_series = [
        {
            "name": s,
            "n": int(n),
            "template": f"List documents in series {s}",
        }
        for s, n in series_counter.most_common(3)
    ]

    # ── confidence distribution ────────────────────────────────────────────
    conf_dist = {"high": 0, "medium": 0, "low": 0}
    for b in blocks:
        c = b.extraction_confidence
        if c >= _CONFIDENCE_HIGH_CUTOFF:
            conf_dist["high"] += 1
        elif c >= _CONFIDENCE_MEDIUM_CUTOFF:
            conf_dist["medium"] += 1
        else:
            conf_dist["low"] += 1

    # ── epistemic distribution ─────────────────────────────────────────────
    epistemic_dist = dict(collections.Counter(b.epistemic_status for b in blocks if b.epistemic_status))

    # ── quarantine + OCR-drop cells (frontend hides when value == 0) ───────
    quarantined_cell = SidecarCell(
        label="under sensitivity review",
        value=int(quarantined_count),
        template="Show quarantined claims for this query" if quarantined_count > 0 else None,
    )
    ocr_dropped_cell = SidecarCell(
        label="OCR-corrupted (excluded)",
        value=int(ocr_dropped),
    )

    return StructuralView(
        claim_count=claim_count,
        document_count=document_count,
        paragraph_count=paragraph_count,
        year_range=year_range,
        year_buckets=year_buckets,
        top_entities=top_entities,
        top_claim_types=top_claim_types,
        top_series=top_series,
        confidence_dist=conf_dist,
        epistemic_dist=epistemic_dist,
        quarantined=quarantined_cell,
        ocr_dropped=ocr_dropped_cell,
    )


_OCR_TITLE_SPACED_CAPS_RE = re.compile(r"[A-Z] [A-Z] [A-Z]")


def _is_ocr_garbage_title(title: str) -> bool:
    """Heuristic: True when *title* looks like OCR noise rather than a real headline.

    Three signals (any one trips it):
      * upper-case ratio above 55% — mostly all-caps OCR headline
      * three or more single-letter caps separated by single spaces (e.g. "G O N")
      * length above 120 chars — runaway capture of a whole headline as title
    """
    if not title:
        return False
    upper_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
    return (
        upper_ratio > 0.55
        or bool(_OCR_TITLE_SPACED_CAPS_RE.search(title))
        or len(title) > 120
    )


def _scrub_doc_list_titles(rows: "list[dict]") -> "list[dict]":
    """Tier 1 list-operation hygiene.

    The list_documents_* Cypher templates return raw Document.title values,
    which in newspaper corpora frequently come back as OCR-degraded headlines
    (e.g. "SPOKANE TELLS NO HARD LUCK HISTORY Cit..."). Apply the same
    detector used by the synthesis context: replace the garbage title with
    None so the frontend can render a clean fallback ("title unavailable")
    while still showing the surrounding date / type / institution columns
    that let the user identify the document.

    The original raw title is preserved on the row as ``title_raw`` for
    debugging / archivist annotation flows that need to see what was
    actually stored.
    """
    out: list[dict] = []
    for row in rows:
        new_row = dict(row)
        title = (new_row.get("title") or "").strip()
        if title and _is_ocr_garbage_title(title):
            new_row["title_raw"] = title
            new_row["title"] = None
        out.append(new_row)
    return out


def _display_doc_title(block: ProvenanceBlock) -> str:
    """Title used in the synthesis context block.

    Returns the raw title when usable; otherwise a structured fallback that
    keeps the document distinguishable to the model even when OCR mangled the
    headline (the date and page already render on adjacent lines, so the
    fallback only needs to preserve the institution).
    """
    title = block.doc_title or ""
    if title and not _is_ocr_garbage_title(title):
        return title
    if block.institution_id:
        return f"[{block.institution_id} document — title OCR degraded]"
    return "[document title unavailable]"


# A numeral with non-numeric characters embedded between digits — "3(X)0",
# "1O5", "2,0O0". Excludes the separators that legitimately appear inside
# numbers (. , : / -) so "3.5", "1,000", "1938-39" and "10:30" never match.
_CORRUPT_NUMERAL_RE = re.compile(r"\b\d+[^\d\s.,:/\-]{1,3}\d+\b")


def detect_corrupt_numerals(sentence: str) -> list[str]:
    """Return numeral-like tokens in *sentence* that look OCR-damaged.

    Surfacing this per-claim beats a general prompt rule about OCR: the model
    is told about the specific token at the point it reads it. Measured on
    gemma-4-e4b, a general instruction was ignored and "3(X)0 mallards" was
    reported as a third population estimate distinct from the 3,000 recorded
    elsewhere — a discrepancy that does not exist in the archive.
    """
    return _CORRUPT_NUMERAL_RE.findall(sentence or "")


def cite_token_for_index(index: int) -> str:
    """Short citation handle for the block at *index* (0-based) -> ``C1``, ``C2``…

    Single source of truth for the mapping. ``synthesis._resolve_citation_tokens``
    rebuilds it from the same ordered block list to translate a model's
    citations back to claim ids; if this format changes, that resolver follows
    automatically because it calls this function rather than reimplementing it.
    """
    return f"C{index + 1}"


def _serialise_block(
    block: ProvenanceBlock,
    query_intent_signal: str = "",
    archivist_note: "dict | None" = None,
    cite_token: str | None = None,
) -> str:
    """Render a ProvenanceBlock to the structured text format.

    *cite_token* is a short stable handle (``C1``, ``C2``, …) the model may
    cite instead of transcribing the full CLAIM_ID. Claim ids are ~55
    characters of opaque run-prefix plus hash; small local models reliably
    truncate them to the run prefix, which is ambiguous (several claims share
    one run) and therefore unrepairable. A two-token handle removes that
    failure mode entirely and cuts output tokens per citation. Raw CLAIM_ID
    citation still works — the resolver in synthesis.py accepts both.
    """
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
        f"DOCUMENT: {_display_doc_title(block)}{date_range}{retrieval_note}",
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
        *(
            [f"      CITE: {cite_token}   (short handle for CLAIM_ID below)"]
            if cite_token else []
        ),
        f"      CLAIM_ID: {block.claim_id}",
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
        context_ranker: "ContextRanker | None" = None,
        locations_query: str | None = None,
    ) -> None:
        self._executor = executor
        # Tier-1 "where did claims about X occur" query. Defaults to the module
        # constant (OCCURRED_AT → Place|Refuge); app construction passes a
        # config-built query so a domain's own location relation/labels apply.
        self._locations_query = locations_query or LOCATIONS_FOR_ENTITY_QUERY
        # Candidate ordering strategy applied before the count/char caps.
        # Defaults to the legacy confidence sort; app construction passes
        # build_context_ranker() output (proximity / KGR-temporal).
        self._context_ranker: ContextRanker = context_ranker or ConfidenceRanker()
        self._budget_conversational = budget_conversational
        self._budget_hybrid = budget_hybrid
        # `is not None`, NOT `or`: a domain that declares no intent keywords
        # loads as an empty dict, and an empty dict must stay empty. With `or`
        # it fell through to the Turnbull wildlife map below, silently scoping
        # another institution's queries to claim types their corpus does not
        # contain ("population" -> species_presence) with no error. The
        # fallback now applies only when a caller passes nothing at all.
        self._query_intent_map = (
            query_intent_map if query_intent_map is not None else _FALLBACK_INTENT_MAP
        )
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
        # Wall-clock duration (ms) of the LLM-cypher generator call.  None when
        # the rule-based cascade satisfied the query and the LLM fallback never
        # fired.  Read by the web layer for per-stage instrumentation.
        self._last_llm_cypher_ms: float | None = None

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
            Ordered list of ProvenanceBlock (best-first per the configured
            context ranker, capped).
        context_text:
            Serialised string ready for injection into the synthesis prompt.
        """
        # Reset per-call instrumentation; these track stage timings the web
        # layer reads after assemble() returns.
        self._last_llm_cypher_ms = None
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
            cross_corpus_overlap=bool(getattr(intent, "cross_corpus_overlap", False)),
        )

        rows: list[dict] = []
        claim_rel_types: dict[str, list[str]] = {}
        seen_claim_ids: set[str] = set()
        last_template: str = FULLTEXT_CLAIMS_QUERY
        resource_exhausted = False

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
            _t_rung = time.perf_counter()
            for template, params in queries:
                try:
                    raw = self._executor.run(template, params)
                except Exception as exc:
                    if _is_neo4j_resource_error(exc):
                        resource_exhausted = True
                        _log.warning(
                            "retrieval query skipped after Neo4j resource error: label=%s template=%s limit=%s error=%s",
                            label,
                            self._template_names.get(template, "unknown"),
                            params.get("limit"),
                            exc,
                        )
                        continue
                    raise
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
                duration_ms=(time.perf_counter() - _t_rung) * 1000,
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
            and not resource_exhausted
            and self._llm_cypher_generator is not None
            and (intent is None or intent.classifier_confidence < _LLM_FALLBACK_CONFIDENCE_CEILING)
        ):
            llm_cypher = None
            _t0_llm = time.perf_counter()
            try:
                llm_cypher = self._llm_cypher_generator.generate(query_text, intent)
            except Exception as exc:
                _log.info("LLM cypher generation raised: %s", exc)
            self._last_llm_cypher_ms = (time.perf_counter() - _t0_llm) * 1000
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
                    "limit": _bounded_query_limit(budget, 2),
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
                    duration_ms=(time.perf_counter() - _t0_llm) * 1000,
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
                    duration_ms=(time.perf_counter() - _t0_llm) * 1000,
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
        blocks_raw = self._context_ranker.rank(
            blocks_raw,
            entity_context=entity_context,
            year_min=year_min,
            year_max=year_max,
        )
        blocks = _stratified_budget_cap(blocks_raw, budget)
        # Hard ceiling on serialised context size, enforced after the count
        # cap so the ranker's best-first order is preserved.  Without this an
        # uncapped retrieval (10K blocks) blows the 1M-token Anthropic input
        # limit on dense corpora.
        blocks = _apply_context_char_budget(blocks)
        self._last_context_count = len(blocks)

        # Batch-lookup archivist notes for documents present in context.
        _notes: dict[str, dict] = {}
        if self._annotation_store is not None:
            _doc_ids = list(dict.fromkeys(b.doc_id for b in blocks if b.doc_id))
            if _doc_ids:
                _notes = self._annotation_store.get_notes_for_docs(_doc_ids)

        # cite_token is positional over `blocks`, and `blocks` is what we
        # return — synthesis rebuilds the same C<N> -> claim_id mapping from
        # that list, so the two stay aligned by construction.
        context_text = "\n\n".join(
            _serialise_block(
                b,
                archivist_note=_notes.get(b.doc_id) if b.doc_id else None,
                cite_token=cite_token_for_index(i),
            )
            for i, b in enumerate(blocks)
        )
        summary = _build_retrieval_summary(blocks_raw, blocks)
        if summary:
            context_text = summary + "\n\n" + context_text
        return blocks, context_text

    def tier_1_lookup(
        self,
        intent: Tier1Intent,
        *,
        institution_ids: list[str] | None = None,
        permitted_levels: list[str] | None = None,
    ) -> Tier1Result:
        """Run a structural-lookup Cypher and return a deterministic result.

        Used by the request handler when ``detect_tier1_intent`` returns a
        non-None intent — short-circuits the full retrieval + synthesis pipeline
        for queries whose answer is fully derivable from the graph.
        """
        effective_ids = list(institution_ids) if institution_ids else list(self._institution_ids)
        effective_levels = list(permitted_levels) if permitted_levels else ["public"]

        if intent.operation == "count_claims_for_entity":
            rows = self._executor.run(
                COUNT_CLAIMS_FOR_ENTITY_QUERY,
                {
                    "entity_id": intent.entity_id,
                    "institution_ids": effective_ids,
                    "permitted_levels": effective_levels,
                },
            )
            n = int(rows[0]["n"]) if rows else 0
            plural = "s" if n != 1 else ""
            return Tier1Result(
                operation=intent.operation,
                entity_name=intent.entity_name,
                count=n,
                answer_text=(
                    f"{intent.entity_name} appears in {n} claim{plural} in this collection."
                ),
            )

        if intent.operation == "year_range_for_entity":
            rows = self._executor.run(
                YEAR_RANGE_FOR_ENTITY_QUERY,
                {
                    "entity_id": intent.entity_id,
                    "institution_ids": effective_ids,
                    "permitted_levels": effective_levels,
                },
            )
            row = rows[0] if rows else {}
            earliest = row.get("earliest_year")
            latest = row.get("latest_year")
            distinct = int(row.get("distinct_years") or 0)
            total = int(row.get("total_claims") or 0)
            if earliest is None or latest is None:
                answer = (
                    f"No dated claims found for {intent.entity_name} in this collection."
                )
            elif earliest == latest:
                answer = (
                    f"{intent.entity_name} is documented in {earliest} "
                    f"({total} claim{'s' if total != 1 else ''})."
                )
            else:
                answer = (
                    f"{intent.entity_name} is documented from {earliest} to {latest} "
                    f"across {distinct} distinct year{'s' if distinct != 1 else ''} "
                    f"({total} claim{'s' if total != 1 else ''} total)."
                )
            return Tier1Result(
                operation=intent.operation,
                entity_name=intent.entity_name,
                count=total,
                earliest_year=earliest,
                latest_year=latest,
                distinct_years=distinct,
                answer_text=answer,
            )

        if intent.operation == "list_documents_for_entity":
            rows = self._executor.run(
                LIST_DOCUMENTS_FOR_ENTITY_QUERY,
                {
                    "entity_id": intent.entity_id,
                    "institution_ids": effective_ids,
                    "permitted_levels": effective_levels,
                },
            )
            items = _scrub_doc_list_titles(rows)
            n = len(items)
            if n == 0:
                answer = f"No documents found for {intent.entity_name} in this collection."
            else:
                top = items[0]
                top_title = top.get("title") or top.get("doc_id") or "(untitled)"
                answer = (
                    f"{intent.entity_name} appears in {n} document{'s' if n != 1 else ''}. "
                    f"Top by claim count: \"{top_title}\" ({top.get('claim_count', 0)} claims)."
                )
            return Tier1Result(
                operation=intent.operation,
                entity_name=intent.entity_name,
                count=n,
                items=items,
                answer_text=answer,
            )

        if intent.operation == "list_documents_for_year":
            rows = self._executor.run(
                LIST_DOCUMENTS_FOR_YEAR_QUERY,
                {
                    "target_year": intent.target_year,
                    "institution_ids": effective_ids,
                    "permitted_levels": effective_levels,
                },
            )
            items = _scrub_doc_list_titles(rows)
            n = len(items)
            year_label = str(intent.target_year)
            if n == 0:
                answer = f"No documents found in this collection for {year_label}."
            else:
                answer = (
                    f"{n} document{'s' if n != 1 else ''} from {year_label} "
                    f"in this collection (sorted by date)."
                )
            return Tier1Result(
                operation=intent.operation,
                entity_name=year_label,
                count=n,
                items=items,
                target_year=intent.target_year,
                answer_text=answer,
            )

        if intent.operation == "locations_for_entity":
            rows = self._executor.run(
                self._locations_query,
                {
                    "entity_id": intent.entity_id,
                    "institution_ids": effective_ids,
                    "permitted_levels": effective_levels,
                },
            )
            items = list(rows)
            n = len(items)
            if n == 0:
                answer = (
                    f"No locations recorded for {intent.entity_name} in this collection."
                )
            else:
                top = items[0]
                top_name = top.get("place_name") or "(unnamed place)"
                answer = (
                    f"{intent.entity_name} appears in {n} location{'s' if n != 1 else ''}. "
                    f"Top by claim count: {top_name} ({top.get('claim_count', 0)} claims)."
                )
            return Tier1Result(
                operation=intent.operation,
                entity_name=intent.entity_name,
                count=n,
                items=items,
                answer_text=answer,
            )

        if intent.operation == "co_occurring_entities_for_entity":
            rows = self._executor.run(
                CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY,
                {
                    "entity_id": intent.entity_id,
                    "institution_ids": effective_ids,
                    "permitted_levels": effective_levels,
                },
            )
            items = list(rows)
            n = len(items)
            if n == 0:
                answer = (
                    f"No co-occurring entities found for {intent.entity_name} "
                    f"in this collection."
                )
            else:
                top = items[0]
                top_name = top.get("name") or "(unnamed entity)"
                top_type = top.get("entity_type") or ""
                type_suffix = f" [{top_type}]" if top_type else ""
                answer = (
                    f"{intent.entity_name} co-occurs with {n} other "
                    f"entit{'ies' if n != 1 else 'y'} in this collection. "
                    f"Top: {top_name}{type_suffix} "
                    f"({top.get('co_occurrence_count', 0)} shared claims)."
                )
            return Tier1Result(
                operation=intent.operation,
                entity_name=intent.entity_name,
                count=n,
                items=items,
                answer_text=answer,
            )

        if intent.operation == "frequency_ranking_for_type":
            rows = self._executor.run(
                FREQUENCY_RANKING_FOR_TYPE_QUERY,
                {
                    "entity_label": intent.target_label,
                    "institution_ids": effective_ids,
                    "permitted_levels": effective_levels,
                    "limit": intent.limit,
                },
            )
            items = list(rows)
            n = len(items)
            type_label = intent.entity_name or intent.target_label.lower()
            if n == 0:
                answer = (
                    f"No {type_label} found in this collection."
                )
            else:
                top = items[0]
                top_name = top.get("name") or "(unnamed)"
                answer = (
                    f"Top {n} {type_label} by claim count. "
                    f"Leading: {top_name} ({top.get('claim_count', 0)} claims)."
                )
            return Tier1Result(
                operation=intent.operation,
                entity_name=type_label,
                count=n,
                items=items,
                answer_text=answer,
            )

        raise ValueError(f"Unknown Tier 1 operation: {intent.operation}")

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
