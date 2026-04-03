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
    ENTITY_ANCHORED_CLAIMS_QUERY,
    FULLTEXT_CLAIMS_QUERY,
    MULTI_ENTITY_CLAIMS_QUERY,
    TEMPORAL_CLAIMS_QUERY,
    TEMPORAL_CLAIMS_QUERY_WITH_REFUGE,
)
from .executor import Neo4jQueryExecutor
from .models import EntityContext, ProvenanceBlock

_log = logging.getLogger(__name__)

# Default context window budgets.
_BUDGET_CONVERSATIONAL = 20

_BUDGET_HYBRID = 4

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


def _select_retrieval_strategy(
    query_text: str,
    entity_context: "EntityContext",
    year_min: int | None,
    year_max: int | None,
    budget: int,
    permitted_levels: list[str] | None = None,
    institution_id: str | None = None,
    *,
    intent_map: dict[str, list[str]] | None = None,
    anchor_entity_id: str | None = None,
) -> tuple[str, dict]:
    """Select the Cypher template and parameters best matched to this query shape.

    Priority cascade:
    1. Temporal — year bounds present, no entity anchor
    2. Multi-entity comparative — 2+ resolved entities
    3. Single entity + claim_type signal — entity-anchored over-fetch + post-filter
    4. Claim-type scoped — vocabulary signal, no entity
    5. Single entity fallback — 1 entity, no vocabulary signal
    6. Fulltext — nothing else matched
    """
    inferred_claim_types = _infer_claim_types(query_text, intent_map=intent_map)
    resolved_ids = list(dict.fromkeys(e.entity_id for e in entity_context.resolved))
    has_years = year_min is not None or year_max is not None
    has_entities = len(resolved_ids) > 0
    has_claim_types = bool(inferred_claim_types)

    _permitted = permitted_levels if permitted_levels is not None else ["public"]
    _access_params = {"permitted_levels": _permitted, "institution_id": institution_id or ""}

    if has_years and len(resolved_ids) <= 1:
        # When no entity resolved, anchor to the configured domain anchor
        # so the temporal query doesn't scan the full corpus
        if len(resolved_ids) == 0:
            refuge_id = anchor_entity_id
            if refuge_id:
                return TEMPORAL_CLAIMS_QUERY_WITH_REFUGE, {
                    "refuge_id": refuge_id,
                    "year_min": year_min,
                    "year_max": year_max,
                    "claim_types": inferred_claim_types,
                    "limit": budget * 3,
                    **_access_params,
                }
        return TEMPORAL_CLAIMS_QUERY, {
            "year_min": year_min,
            "year_max": year_max,
            "claim_types": inferred_claim_types,
            "limit": budget * 3,
            **_access_params,
        }
    if len(resolved_ids) >= 2:
        return MULTI_ENTITY_CLAIMS_QUERY, {
            "entity_ids": resolved_ids, "claim_types": inferred_claim_types,
            "year_min": year_min, "year_max": year_max, "limit": budget * 2,
            **_access_params,
        }
    if has_entities and has_claim_types:
        # Over-fetch; post-filter by claim_type in assemble()
        return ENTITY_ANCHORED_CLAIMS_QUERY, {
            "entity_id": resolved_ids[0], "year_min": year_min,
            "year_max": year_max, "limit": budget * 3,
            **_access_params,
        }
    if has_claim_types and not has_entities:
        return CLAIM_TYPE_SCOPED_QUERY, {
            "claim_types": inferred_claim_types, "entity_ids": None,
            "year_min": year_min, "year_max": year_max, "limit": budget * 2,
            **_access_params,
        }
    if has_entities:
        return ENTITY_ANCHORED_CLAIMS_QUERY, {
            "entity_id": resolved_ids[0], "year_min": year_min,
            "year_max": year_max, "limit": budget * 2,
            **_access_params,
        }
    return FULLTEXT_CLAIMS_QUERY, {
        "search_text": _sanitize_fulltext(query_text), "limit": budget * 2,
        **_access_params,
    }


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


def _is_ocr_corrupted(sentence: str) -> bool:
    """Return True if *sentence* contains characters indicating OCR corruption."""
    return bool(_OCR_GARBAGE_RE.search(sentence))


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
        institution_id: str | None = None,
        anchor_entity_id: str | None = None,
        anchor_entity_type: str | None = None,
    ) -> None:
        self._executor = executor
        self._budget_conversational = budget_conversational
        self._budget_hybrid = budget_hybrid
        self._query_intent_map = query_intent_map or _FALLBACK_INTENT_MAP
        self._institution_id = institution_id
        self._anchor_entity_id = anchor_entity_id
        # Optional AnnotationStore; when set, archivist notes are injected into context.
        self._annotation_store = annotation_store
        # Populated after each assemble() call; read by the web layer for coverage stats.
        self._last_candidate_count: int = 0
        self._last_ocr_dropped: int = 0
        self._last_context_count: int = 0
        # Graph-based fallback: resolve anchor entity from graph when not
        # supplied via config (e.g. seed_entities had no match).
        if self._anchor_entity_id is None and anchor_entity_type:
            if re.match(r'^[A-Za-z_]+$', anchor_entity_type):
                try:
                    rows = executor.run(
                        f"MATCH (:Document)-->(r:{anchor_entity_type})"
                        " RETURN r.entity_id AS eid LIMIT 1"
                    )
                    if rows:
                        self._anchor_entity_id = rows[0]["eid"]
                except Exception:
                    pass

    def assemble(
        self,
        query_text: str,
        entity_context: EntityContext,
        year_min: int | None = None,
        year_max: int | None = None,
        is_hybrid: bool = False,
        permitted_levels: list[str] | None = None,
        institution_id: str | None = None,
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

        template, params = _select_retrieval_strategy(
            query_text, entity_context, year_min, year_max, budget,
            permitted_levels=permitted_levels,
            institution_id=institution_id or self._institution_id,
            intent_map=self._query_intent_map,
            anchor_entity_id=self._anchor_entity_id,
        )

        rows: list[dict] = []
        claim_rel_types: dict[str, list[str]] = {}

        if template == MULTI_ENTITY_CLAIMS_QUERY:
            seen_claim_ids: set[str] = set()
            for r in self._executor.run(template, params):
                c = r.get("c") or {}
                cid = c.get("claim_id")
                if cid:
                    rel_type = r.get("traversal_rel_type")
                    if rel_type and rel_type not in claim_rel_types.get(cid, []):
                        claim_rel_types.setdefault(cid, []).append(rel_type)
                if cid and cid not in seen_claim_ids:
                    seen_claim_ids.add(cid)
                    rows.append(r)
        else:
            raw = self._executor.run(template, params)
            for r in raw:
                c = r.get("c") or {}
                cid = c.get("claim_id")
                if cid:
                    rel_type = r.get("traversal_rel_type")
                    if rel_type and rel_type not in claim_rel_types.get(cid, []):
                        claim_rel_types.setdefault(cid, []).append(rel_type)
            rows = raw

        # Post-filter by claim_type when entity-anchored path over-fetches (Case 3).
        if template == ENTITY_ANCHORED_CLAIMS_QUERY and inferred_claim_types:
            filtered = [
                r for r in rows
                if (r.get("c") or {}).get("claim_type") in inferred_claim_types
            ]
            rows = filtered or rows  # fall back to unfiltered if filter removes everything

        # Diagnostic: log claim_type distribution to stderr for depth/breadth analysis.
        _TEMPLATE_NAMES = {
            TEMPORAL_CLAIMS_QUERY: "TEMPORAL",
            TEMPORAL_CLAIMS_QUERY_WITH_REFUGE: "TEMPORAL_REFUGE",
            MULTI_ENTITY_CLAIMS_QUERY: "MULTI_ENTITY",
            CLAIM_TYPE_SCOPED_QUERY: "CLAIM_TYPE_SCOPED",
            ENTITY_ANCHORED_CLAIMS_QUERY: "ENTITY_ANCHORED",
            FULLTEXT_CLAIMS_QUERY: "FULLTEXT",
        }
        claim_types_in_result = collections.Counter(
            (r.get("c") or {}).get("claim_type", "unknown") for r in rows
        )
        _log.debug(
            "retrieval: template=%s n_rows=%d claim_type_dist=%s",
            _TEMPLATE_NAMES.get(template, "unknown"),
            len(rows),
            dict(claim_types_in_result),
        )

        self._last_candidate_count = len(rows)

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
        institution_id: str | None = None,
    ) -> list[ProvenanceBlock]:
        """Return the full provenance chain for a single known *claim_id*.

        Used by the ``POST /query/provenance`` endpoint.
        """
        from ..core.graph.cypher import PROVENANCE_CHAIN_QUERY

        rows = self._executor.run(PROVENANCE_CHAIN_QUERY, {
            "claim_id": claim_id,
            "permitted_levels": permitted_levels if permitted_levels is not None else ["public"],
            "institution_id": institution_id or self._institution_id or "",
        })
        blocks = [_row_to_block(r) for r in rows]
        return [b for b in blocks if b is not None]
