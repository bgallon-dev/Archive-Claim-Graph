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
import re
import sys

from ..graph.cypher import (
    CLAIM_TYPE_SCOPED_QUERY,
    ENTITY_ANCHORED_CLAIMS_QUERY,
    FULLTEXT_CLAIMS_QUERY,
    MULTI_ENTITY_CLAIMS_QUERY,
    TEMPORAL_CLAIMS_QUERY,
    TEMPORAL_CLAIMS_QUERY_WITH_REFUGE,
)
from .executor import Neo4jQueryExecutor
from .models import EntityContext, ProvenanceBlock

# Default context window budgets.
_BUDGET_CONVERSATIONAL = 20

# Cached Turnbull Refuge entity_id — resolved once at first use.
_TURNBULL_REFUGE_ID: str | None = None


def _get_default_refuge_id() -> str | None:
    """Return the canonical Refuge entity_id for Turnbull.
    Resolved once and cached — avoids repeated seed entity loading."""
    global _TURNBULL_REFUGE_ID
    if _TURNBULL_REFUGE_ID is not None:
        return _TURNBULL_REFUGE_ID
    try:
        from ..resolver import default_seed_entities
        seeds = default_seed_entities()
        refuge = next(
            (e for e in seeds
             if e.entity_type == "Refuge"
             and "turnbull" in e.normalized_form),
            None,
        )
        if refuge:
            _TURNBULL_REFUGE_ID = refuge.entity_id
    except Exception:
        pass
    return _TURNBULL_REFUGE_ID
_BUDGET_HYBRID = 4

# Maps query vocabulary keywords to claim_type lists for retrieval scoping.
_QUERY_INTENT_TO_CLAIM_TYPES: dict[str, list[str]] = {
    "predator": ["predator_control", "management_action"],
    "habitat": ["habitat_condition", "weather_observation"],
    "population": ["population_estimate", "species_presence", "species_absence"],
    "breeding": ["breeding_activity"],
    "migration": ["migration_timing"],
    "fire": ["fire_incident"],
    "management": ["management_action", "development_activity", "economic_use"],
}


def _infer_claim_types(query_text: str) -> list[str] | None:
    """Map query vocabulary to claim_type filters. Returns None if no signal."""
    lowered = query_text.lower()
    matched: set[str] = set()
    for keyword, types in _QUERY_INTENT_TO_CLAIM_TYPES.items():
        if keyword in lowered:
            matched.update(types)
    return list(matched) if matched else None


def _select_retrieval_strategy(
    query_text: str,
    entity_context: "EntityContext",
    year_min: int | None,
    year_max: int | None,
    budget: int,
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
    inferred_claim_types = _infer_claim_types(query_text)
    resolved_ids = list(dict.fromkeys(e.entity_id for e in entity_context.resolved))
    has_years = year_min is not None or year_max is not None
    has_entities = len(resolved_ids) > 0
    has_claim_types = bool(inferred_claim_types)

    if has_years and len(resolved_ids) <= 1:
        # When no entity resolved, anchor to the default refuge
        # so the temporal query doesn't scan the full corpus
        if len(resolved_ids) == 0:
            refuge_id = _get_default_refuge_id()
            if refuge_id:
                return TEMPORAL_CLAIMS_QUERY_WITH_REFUGE, {
                    "refuge_id": refuge_id,
                    "year_min": year_min,
                    "year_max": year_max,
                    "claim_types": inferred_claim_types,
                    "limit": budget * 3,
                }
        return TEMPORAL_CLAIMS_QUERY, {
            "year_min": year_min,
            "year_max": year_max,
            "claim_types": inferred_claim_types,
            "limit": budget * 3,
        }
    if len(resolved_ids) >= 2:
        return MULTI_ENTITY_CLAIMS_QUERY, {
            "entity_ids": resolved_ids, "claim_types": inferred_claim_types,
            "year_min": year_min, "year_max": year_max, "limit": budget * 2,
        }
    if has_entities and has_claim_types:
        # Over-fetch; post-filter by claim_type in assemble()
        return ENTITY_ANCHORED_CLAIMS_QUERY, {
            "entity_id": resolved_ids[0], "year_min": year_min,
            "year_max": year_max, "limit": budget * 3,
        }
    if has_claim_types and not has_entities:
        return CLAIM_TYPE_SCOPED_QUERY, {
            "claim_types": inferred_claim_types, "entity_ids": None,
            "year_min": year_min, "year_max": year_max, "limit": budget * 2,
        }
    if has_entities:
        return ENTITY_ANCHORED_CLAIMS_QUERY, {
            "entity_id": resolved_ids[0], "year_min": year_min,
            "year_max": year_max, "limit": budget * 2,
        }
    return FULLTEXT_CLAIMS_QUERY, {
        "search_text": query_text, "limit": budget * 2,
    }


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
    )


def _serialise_block(block: ProvenanceBlock, query_intent_signal: str = "") -> str:
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
        f"  CONFIDENCE_TIER: {confidence_tier} ({block.extraction_confidence:.2f})",
        f"  PAGE: {block.page_number if block.page_number is not None else '?'}",
        f"    PARAGRAPH: {block.paragraph_id}",
        (
            f"      CLAIM [{block.claim_type}, "
            f"epistemic={block.epistemic_status}]:"
        ),
        f'        "{block.source_sentence}"',
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
    ) -> None:
        self._executor = executor
        self._budget_conversational = budget_conversational
        self._budget_hybrid = budget_hybrid
        # Populated after each assemble() call; read by the web layer for coverage stats.
        self._last_candidate_count: int = 0
        self._last_ocr_dropped: int = 0
        self._last_context_count: int = 0
        # Seed the refuge ID cache from the graph at startup — graph-authoritative,
        # avoids dependency on seed entity CSV at query time.
        global _TURNBULL_REFUGE_ID
        if _TURNBULL_REFUGE_ID is None:
            try:
                rows = executor.run(
                    "MATCH (:Document)-[:ABOUT_REFUGE]->(r:Refuge)"
                    " RETURN r.entity_id AS eid LIMIT 1"
                )
                if rows:
                    _TURNBULL_REFUGE_ID = rows[0]["eid"]
            except Exception:
                pass

    def assemble(
        self,
        query_text: str,
        entity_context: EntityContext,
        year_min: int | None = None,
        year_max: int | None = None,
        is_hybrid: bool = False,
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
        inferred_claim_types = _infer_claim_types(query_text)

        template, params = _select_retrieval_strategy(
            query_text, entity_context, year_min, year_max, budget
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
            MULTI_ENTITY_CLAIMS_QUERY: "MULTI_ENTITY",
            CLAIM_TYPE_SCOPED_QUERY: "CLAIM_TYPE_SCOPED",
            ENTITY_ANCHORED_CLAIMS_QUERY: "ENTITY_ANCHORED",
            FULLTEXT_CLAIMS_QUERY: "FULLTEXT",
        }
        claim_types_in_result = collections.Counter(
            (r.get("c") or {}).get("claim_type", "unknown") for r in rows
        )
        print(
            f"DEBUG retrieval: template={_TEMPLATE_NAMES.get(template, 'unknown')}, "
            f"n_rows={len(rows)}, claim_type_dist={dict(claim_types_in_result)}",
            file=sys.stderr,
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

        context_text = "\n\n".join(_serialise_block(b) for b in blocks)
        return blocks, context_text

    def chain_for_claim(self, claim_id: str) -> list[ProvenanceBlock]:
        """Return the full provenance chain for a single known *claim_id*.

        Used by the ``POST /query/provenance`` endpoint.
        """
        from ..graph.cypher import PROVENANCE_CHAIN_QUERY

        rows = self._executor.run(PROVENANCE_CHAIN_QUERY, {"claim_id": claim_id})
        blocks = [_row_to_block(r) for r in rows]
        return [b for b in blocks if b is not None]
