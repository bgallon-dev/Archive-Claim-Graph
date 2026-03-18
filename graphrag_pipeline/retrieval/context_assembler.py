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

import re

from ..graph.cypher import ENTITY_ANCHORED_CLAIMS_QUERY, FULLTEXT_CLAIMS_QUERY
from .executor import Neo4jQueryExecutor
from .models import EntityContext, ProvenanceBlock

# Default context window budgets.
_BUDGET_CONVERSATIONAL = 20
_BUDGET_HYBRID = 4

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


def _serialise_block(block: ProvenanceBlock) -> str:
    """Render a ProvenanceBlock to the structured text format."""
    date_range = ""
    if block.doc_date_start or block.doc_date_end:
        date_range = f", {block.doc_date_start or '?'}–{block.doc_date_end or '?'}"

    lines: list[str] = [
        f"DOCUMENT: {block.doc_title}{date_range}",
        f"  PAGE: {block.page_number if block.page_number is not None else '?'}",
        f"    PARAGRAPH: {block.paragraph_id}",
        (
            f"      CLAIM [{block.claim_type}, "
            f"confidence={block.extraction_confidence:.2f}, "
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

        rows: list[dict] = []

        if entity_context.resolved:
            # Entity-anchored path: one query per resolved entity.
            seen_claim_ids: set[str] = set()
            for re_obj in entity_context.resolved:
                entity_rows = self._executor.run(
                    ENTITY_ANCHORED_CLAIMS_QUERY,
                    {
                        "entity_id": re_obj.entity_id,
                        "year_min": year_min,
                        "year_max": year_max,
                        "limit": budget * 2,  # over-fetch before dedup + cap
                    },
                )
                for r in entity_rows:
                    c = r.get("c") or {}
                    cid = c.get("claim_id")
                    if cid and cid not in seen_claim_ids:
                        seen_claim_ids.add(cid)
                        rows.append(r)
        else:
            # Fulltext fallback path.
            rows = self._executor.run(
                FULLTEXT_CLAIMS_QUERY,
                {"search_text": query_text, "limit": budget * 2},
            )

        self._last_candidate_count = len(rows)

        # Convert rows → blocks, filter None (including OCR-corrupted), rank by confidence, cap.
        blocks_raw: list[ProvenanceBlock] = []
        for row in rows:
            block = _row_to_block(row)
            if block is not None:
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
