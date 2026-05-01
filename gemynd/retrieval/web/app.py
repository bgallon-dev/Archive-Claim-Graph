"""Layer 4 — FastAPI Retrieval API.

Exposes the retrieval pipeline as HTTP endpoints.

Endpoints
---------
POST /query
    Accept a natural-language query, route through all four layers, and return
    a synthesised answer with provenance metadata.

POST /query/provenance
    Accept a claim_id and return the full raw provenance chain without LLM
    synthesis — useful for citation verification and the review web app.

Usage
-----
Start via the CLI:
    gemynd query-serve --port 8788
"""
from __future__ import annotations

import dataclasses
import hashlib as _hashlib
import logging
import os
import re
import sqlite3 as _sqlite3
import statistics as _statistics
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field, field_validator
except Exception:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "fastapi/pydantic not installed. Install with: pip install -e .[retrieval]"
    )

from .auth import UserContext, require_user
from gemynd.auth.dependencies import NeedsLoginException, require_admin, require_archivist_or_admin
from gemynd.auth.router import create_auth_router
from ..classifier import classify_query, detect_tier1_intent
from ..context_assembler import ProvenanceContextAssembler, _serialise_block
from ..conversation_log import (
    ClaimInteraction,
    ConversationLogger,
    LogRecord,
    make_conversation_id,
    start_retention_purge,
)
from gemynd.core.graph.cypher import (
    CORPUS_STATS_QUERY,
    COUNT_QUARANTINED_CLAIMS_QUERY,
    SOFT_DELETE_DOCUMENT_QUERY,
    RESTORE_DOCUMENT_QUERY,
    LIST_DOCUMENTS_QUERY,
    LIST_DOCUMENTS_PAGINATED_QUERY,
    COUNT_DOCUMENTS_QUERY,
    DOCUMENT_FACETS_QUERY,
    LIST_CLAIMS_FOR_DOCUMENT_QUERY,
    GET_DOCUMENT_HEADER_QUERY,
    GET_DOCUMENT_SOURCE_TEXT_QUERY,
    GET_DOCUMENT_PARAGRAPHS_QUERY,
    ENTITY_DEGREE_QUERY,
    STATS_DOC_OVERVIEW_QUERY,
    STATS_DOC_TYPE_QUERY,
    STATS_CLAIM_TYPE_QUERY,
    STATS_CLAIM_TYPE_BY_INSTITUTION_QUERY,
    STATS_ENTITY_TYPE_QUERY,
    STATS_TEMPORAL_COVERAGE_QUERY,
    STATS_CONFIDENCE_DISTRIBUTION_QUERY,
    GAP_TEMPORAL_DENSITY_QUERY,
    GAP_ENTITY_DEPTH_QUERY,
    GAP_GEOGRAPHIC_COVERAGE_QUERY,
    ENTITY_SEARCH_QUERY,
    ENTITY_DETAIL_QUERY,
    ENTITY_NEIGHBORHOOD_QUERY,
)
from gemynd.core.resolver import DictionaryFuzzyResolver, ResolutionPolicy
from ..context_assembler import (
    _infer_claim_types,
    _is_ocr_garbage_title,
    _scrub_doc_list_titles,
)
from ..entity_gateway import EntityResolutionGateway
from ..executor import Neo4jQueryExecutor
from ..models import RetrievalStats, SynthesisResult
from ..query_builder import CypherQueryBuilder
from ..synthesis import SynthesisEngine


# Allowlist for entity_id URL path parameters.
# Accepts both hash-based IDs (person_a1b2c3d4) and metadata-supplied readable
# IDs (patent_person_robert_mill). Constrained to lowercase alphanumeric + underscore.
_ENTITY_ID_RE = re.compile(r'^[a-z][a-z0-9_]{3,79}$')

def _citation_label(b: "ProvenanceBlock") -> str:  # type: ignore[name-defined]
    """Build a human-readable citation from structured block fields.

    Prefers institution + date + page over the raw OCR doc_title, which is
    often garbage in newspaper corpora (e.g. 'G O N TIN u E D h o').

    When ``series`` is set, the provenance segment is rendered as
    "<series> via <institution>" to disambiguate composite collections (e.g. BLM
    GLO patent records ingested under the Turnbull institution_id).
    """
    parts: list[str] = []
    series = (b.series or "").strip() if hasattr(b, "series") else ""
    institution_pretty = b.institution_id.replace("_", " ").title() if b.institution_id else ""
    if series and institution_pretty:
        parts.append(f"{series} via {institution_pretty}")
    elif series:
        parts.append(series)
    elif institution_pretty:
        parts.append(institution_pretty)
    if b.doc_date_start:
        parts.append(b.doc_date_start)
    if b.page_number is not None:
        parts.append(f"pp{b.page_number}")
    if parts:
        return " — ".join(parts)
    # Fallback to doc_title, but suppress OCR garbage via the shared detector.
    title = b.doc_title or ""
    if title and not _is_ocr_garbage_title(title):
        return title
    return b.doc_id or ""


def _build_source_links(
    doc: dict[str, Any],
    source_link_configs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Render per-document external links from a corpus's source_link templates.

    Each template declares ``requires`` — fields that must be non-empty on the
    document for the link to render. Documents missing any required field
    simply don't contribute that link (no error, no broken URL).

    Returns a list of {label, url, note} dicts ready for template iteration.
    """
    out: list[dict[str, str]] = []
    for cfg in source_link_configs:
        required = cfg.get("requires") or []
        values: dict[str, str] = {}
        ok = True
        for field in required:
            v = doc.get(field)
            if v in (None, ""):
                ok = False
                break
            values[field] = str(v)
        if not ok:
            continue
        try:
            url = cfg["template"].format(**values)
        except (KeyError, IndexError):
            continue
        out.append({
            "label": cfg.get("label", "View source"),
            "url": url,
            "note": cfg.get("note", ""),
        })
    return out


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    text: str = Field(..., description="Natural-language query")
    year_range: tuple[int, int] | None = Field(
        None, description="Optional year bounds [min, max]"
    )
    entity_hints: list[str] = Field(
        default_factory=list,
        description="Additional entity surface forms to help resolution",
    )
    mode: Literal["analytical", "conversational", "auto"] = Field(
        "auto", description="Force a specific retrieval mode or use auto-classification"
    )
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Prior Q&A turns (client-managed). Empty list = stateless behaviour.",
    )
    session_id: str | None = Field(
        None,
        max_length=128,
        pattern=r'^[a-zA-Z0-9_\-]{1,128}$',
        description="Client-generated session ID grouping multi-turn exchanges.",
    )
    doc_types: list[str] | None = Field(
        None,
        description=(
            "Restrict retrieval to documents whose doc_type matches one of these values "
            "(e.g. ['land_patent']). Supplements any doc_type hints detected from the query text."
        ),
    )
    series: list[str] | None = Field(
        None,
        description=(
            "Restrict retrieval to documents belonging to one of these archival series "
            "(e.g. ['BLM GLO Records']). Supplements any series hints detected from the query text."
        ),
    )

    @field_validator("conversation_history")
    @classmethod
    def cap_history(cls, v: list[ConversationTurn]) -> list[ConversationTurn]:
        if len(v) > 10:
            raise ValueError("conversation_history may not exceed 10 turns")
        return v


class ProvenanceRequest(BaseModel):
    claim_id: str = Field(..., description="Canonical claim_id to retrieve provenance for")


class QueryResponse(BaseModel):
    answer: str
    confidence_assessment: str
    supporting_claim_ids: list[str]
    caveats: list[str]
    min_extraction_confidence: float | None = None
    analytical_result: dict[str, Any] | None = None
    ambiguous_entities: list[str] = Field(default_factory=list)
    retrieval_stats: dict[str, Any] | None = None
    synthesis_available: bool = True  # False when API call failed; provenance blocks still returned
    quarantined_claims_count: int = 0  # claims excluded from this answer due to sensitivity review
    supporting_claims_detail: list[dict[str, Any]] = Field(default_factory=list)
    # Each entry: {claim_id, confidence, epistemic_status, claim_type, doc_id, doc_title}
    user_can_annotate: bool = False  # True when the requesting user has archivist/admin role
    # Tier 1 fast-path output: present when the query was answered directly from
    # the graph without LLM synthesis. UI renders structured fields (count, items)
    # rather than parsing answer prose. None for normal synthesis-path responses.
    tier_1_result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Collection statistics HTML renderer
# ---------------------------------------------------------------------------

_TEMPLATES_DIR: Path = Path(__file__).parent / "templates"

def _conf_tier_class(avg: float | None) -> str:
    if avg is None:
        return ""
    if avg >= 0.85:
        return "tier-high"
    if avg >= 0.70:
        return "tier-medium"
    return "tier-low"


_DOC_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,80}$')
# UUID v4 as produced by str(uuid.uuid4())
_SAVED_ID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


# ---------------------------------------------------------------------------
# Gap detection helpers and HTML renderer
# ---------------------------------------------------------------------------

def _temporal_year_data(temporal_rows: list[dict]) -> dict[int, dict[str, int]]:
    """Return year-keyed document and claim counts, coercing Neo4j values to ints."""
    year_data: dict[int, dict[str, int]] = {}
    for row in temporal_rows:
        if row.get("year") is None:
            continue
        year = int(row["year"])
        year_data[year] = {
            "doc_count": int(row.get("doc_count") or 0),
            "claim_count": int(row.get("claim_count") or 0),
        }
    return year_data


def _temporal_gap_threshold(temporal_rows: list[dict]) -> float:
    year_data = _temporal_year_data(temporal_rows)
    if not year_data:
        return 0.0
    min_year, max_year = min(year_data), max(year_data)
    active = [
        year_data.get(y, {}).get("doc_count", 0)
        for y in range(min_year, max_year + 1)
        if year_data.get(y, {}).get("doc_count", 0) > 0
    ]
    if not active:
        return 0.0
    return float(max(1, _statistics.median(active) * 0.2))


def _compute_temporal_gaps(temporal_rows: list[dict]) -> list[dict]:
    """Return consecutive sparse-year periods that fall between active coverage."""
    if not temporal_rows:
        return []
    year_data = {year: row["doc_count"] for year, row in _temporal_year_data(temporal_rows).items()}
    all_years = sorted(year_data)
    if len(all_years) < 2:
        return []
    min_year, max_year = all_years[0], all_years[-1]
    full = {y: year_data.get(y, 0) for y in range(min_year, max_year + 1)}
    active = [v for v in full.values() if v > 0]
    if not active:
        return []
    gap_threshold = _temporal_gap_threshold(temporal_rows)

    gaps: list[dict] = []
    current_gap: list[int] = []
    for y in range(min_year, max_year + 1):
        if full[y] <= gap_threshold:
            current_gap.append(y)
        else:
            if current_gap:
                before_docs = full.get(current_gap[0] - 1, 0)
                after_docs = full[y]
                if before_docs > gap_threshold or after_docs > gap_threshold:
                    gaps.append({"start": current_gap[0], "end": current_gap[-1],
                                 "years_count": len(current_gap),
                                 "before_docs": before_docs, "after_docs": after_docs})
            current_gap = []
    if current_gap:
        before_docs = full.get(current_gap[0] - 1, 0)
        if before_docs > gap_threshold:
            gaps.append({"start": current_gap[0], "end": current_gap[-1],
                         "years_count": len(current_gap),
                         "before_docs": before_docs, "after_docs": 0})
    return gaps


def _build_temporal_timeline(temporal_rows: list[dict]) -> dict:
    year_data = _temporal_year_data(temporal_rows)
    if not year_data:
        return {"years": [], "max_doc_count": 0, "start_year": None, "end_year": None}

    min_year, max_year = min(year_data), max(year_data)
    max_doc_count = max((row["doc_count"] for row in year_data.values()), default=0)
    gap_threshold = _temporal_gap_threshold(temporal_rows)
    years = []
    for year in range(min_year, max_year + 1):
        doc_count = year_data.get(year, {}).get("doc_count", 0)
        claim_count = year_data.get(year, {}).get("claim_count", 0)
        if year == min_year or year == max_year or year % 10 == 0:
            tick_label = str(year)
        else:
            tick_label = ""
        opacity = "0.00"
        if max_doc_count and doc_count:
            opacity = f"{0.18 + (0.82 * (doc_count / max_doc_count)):.2f}"
        years.append({
            "year": year,
            "doc_count": doc_count,
            "claim_count": claim_count,
            "is_gap": doc_count <= gap_threshold,
            "opacity": opacity,
            "tick_label": tick_label,
        })

    return {
        "years": years,
        "max_doc_count": max_doc_count,
        "start_year": min_year,
        "end_year": max_year,
    }


_RESTRICTED_ACCESS_LEVELS = frozenset({"restricted", "indigenous_restricted", "staff_only"})


def _safe_log_claim_id(claim_id: str, access_level: str) -> str:
    """Return an opaque stable hash for claims from restricted documents.

    Preserves the ability to see access patterns (e.g. same claim retrieved
    N times) without exposing the actual claim_id of sensitive content in the
    conversation log.

    Hash length: 12 hex chars = 48 bits of SHA-256.  This is a deliberate
    tradeoff — 48 bits is sufficient to distinguish individual claims within
    a single institution's corpus (<<2^24 claims expected) while keeping log
    rows compact.  It is NOT collision-resistant at scale; do not use these
    hashed IDs as primary keys or for cryptographic purposes.
    """
    if access_level in _RESTRICTED_ACCESS_LEVELS:
        return "redacted_" + _hashlib.sha256(claim_id.encode()).hexdigest()[:12]
    return claim_id


def _read_query_signal_gaps(conv_log_db: str, limit: int = 10) -> list[dict]:
    """Return query buckets with high frequency but thin claim context from conversation log."""
    try:
        con = _sqlite3.connect(conv_log_db, check_same_thread=False)
        con.execute("PRAGMA busy_timeout=5000")
        try:
            rows = con.execute(
                """
                SELECT c.bucket,
                       COUNT(DISTINCT c.conversation_id) AS query_count,
                       AVG(CAST(r.claims_in_context AS REAL)) AS avg_claims
                FROM conversation c
                JOIN retrieval_event r ON c.conversation_id = r.conversation_id
                GROUP BY c.bucket
                HAVING avg_claims < 5.0
                ORDER BY query_count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            con.close()
        return [{"bucket": r[0], "query_count": r[1], "avg_claims": round(r[2], 1)} for r in rows]
    except Exception:
        return []


def _build_stats_context(
    overview: dict,
    doc_types: list[dict],
    claim_types: list[dict],
    entity_types: list[dict],
    temporal: list[dict],
    confidence: dict,
) -> dict:
    total_docs = overview.get("total_docs") or 0
    earliest = overview.get("earliest_year") or "—"
    latest = overview.get("latest_year") or "—"
    total_pages = overview.get("total_pages") or 0
    donor_restricted = overview.get("donor_restricted_count") or 0

    total_claims = confidence.get("total_claims") or 0
    avg_conf = confidence.get("avg_confidence")
    high = confidence.get("high_count") or 0
    medium = confidence.get("medium_count") or 0
    low = confidence.get("low_count") or 0
    uncertain_epistemic = confidence.get("uncertain_epistemic_count") or 0
    total_entities = sum(r.get("count", 0) for r in entity_types)
    claims_per_doc = total_claims / total_docs if total_docs else 0.0
    entities_per_doc = total_entities / total_docs if total_docs else 0.0

    date_span = f"{earliest} – {latest}" if earliest != "—" else "—"

    temporal_rows: list[dict] = []
    if temporal:
        year_map = {int(r["year"]): int(r["doc_count"]) for r in temporal if r.get("year") is not None}
        if year_map:
            min_year = min(year_map)
            max_year = max(year_map)
            span = max_year - min_year
            if span > 30:
                decade_map: dict[int, int] = {}
                for y, c in year_map.items():
                    d = (y // 10) * 10
                    decade_map[d] = decade_map.get(d, 0) + c
                start_decade = (min_year // 10) * 10
                end_decade = (max_year // 10) * 10
                for d in range(start_decade, end_decade + 1, 10):
                    cnt = decade_map.get(d, 0)
                    decade_counts = [year_map.get(y, 0) for y in range(d, d + 10)]
                    decade_peak = max(decade_counts) if decade_counts else 0
                    segments = []
                    for y, year_count in zip(range(d, d + 10), decade_counts):
                        opacity = 0.0
                        if decade_peak and year_count:
                            opacity = 0.22 + (0.78 * (year_count / decade_peak))
                        segments.append({
                            "year": y,
                            "count": year_count,
                            "opacity": f"{opacity:.2f}",
                        })
                    temporal_rows.append({
                        "label": f"{d}s",
                        "count": cnt,
                        "is_gap": cnt == 0,
                        "segments": segments,
                    })
            else:
                for y in range(min_year, max_year + 1):
                    cnt = year_map.get(y, 0)
                    temporal_rows.append({
                        "label": str(y),
                        "count": cnt,
                        "is_gap": cnt == 0,
                        "segments": [],
                    })

    safe_total = max(total_claims, 1)
    temporal_max_count = max((row["count"] for row in temporal_rows), default=0)
    temporal_max_label = ""
    for row in temporal_rows:
        if temporal_max_count:
            row["width_pct"] = round(row["count"] / temporal_max_count * 100)
        else:
            row["width_pct"] = 0
        if row["count"] == temporal_max_count and temporal_max_count and not temporal_max_label:
            temporal_max_label = row["label"]

    high_pct = high * 100 // safe_total
    medium_pct = medium * 100 // safe_total
    low_pct = low * 100 // safe_total
    confidence_rows = [
        {"label": "High", "count": high, "pct": high_pct, "color": "var(--bark-700)"},
        {"label": "Medium", "count": medium, "pct": medium_pct, "color": "var(--accent-amber-text)"},
        {"label": "Low", "count": low, "pct": low_pct, "color": "var(--accent-red-text)"},
    ]
    if total_claims <= 0:
        confidence_qualifier = "No active claims are available for confidence calibration."
    elif low_pct >= 25:
        confidence_qualifier = "Elevated low-confidence share; review low-confidence claims before treating extracted patterns as stable."
    elif high_pct >= 70 and low_pct <= 10:
        confidence_qualifier = "Strong local confidence profile; most active claims meet the high-confidence threshold."
    elif medium_pct + low_pct >= 40:
        confidence_qualifier = "Mixed local confidence profile; use claim-level confidence when comparing evidence."
    else:
        confidence_qualifier = "Confidence distribution is calibrated to local thresholds only; no external benchmark is configured."
    conf_avg_str = f"{avg_conf:.2f}" if avg_conf is not None else "—"

    return {
        "total_docs": total_docs,
        "total_claims": total_claims,
        "total_entities": total_entities,
        "date_span": date_span,
        "total_pages": total_pages,
        "donor_restricted": donor_restricted,
        "claims_per_doc": f"{claims_per_doc:.1f}",
        "entities_per_doc": f"{entities_per_doc:.1f}",
        "doc_types": doc_types,
        "claim_types": claim_types,
        "entity_types": entity_types,
        "temporal_rows": temporal_rows,
        "temporal_max_count": temporal_max_count,
        "temporal_max_label": temporal_max_label,
        "high": high,
        "medium": medium,
        "low": low,
        "high_pct": high_pct,
        "medium_pct": medium_pct,
        "low_pct": low_pct,
        "confidence_rows": confidence_rows,
        "confidence_qualifier": confidence_qualifier,
        "uncertain_epistemic": uncertain_epistemic,
        "conf_avg_str": conf_avg_str,
    }


def _compute_topic_rows(
    actual_shares: dict[str, float],
    expected_shares: dict[str, float],
) -> list[dict]:
    """Return gap-annotated rows for a (actual, expected) claim-share pair."""
    rows: list[dict] = []
    for claim_type, expected in sorted(expected_shares.items(), key=lambda x: -x[1]):
        actual = actual_shares.get(claim_type, 0.0)
        actual_pct = round(actual * 100, 1)
        expected_pct = round(expected * 100, 1)
        width_scale = max(actual_pct, expected_pct)
        actual_width_pct = round(actual_pct / width_scale * 100, 1) if width_scale else 0.0
        expected_width_pct = round(expected_pct / width_scale * 100, 1) if width_scale else 0.0
        status = _topic_status(actual, expected)
        rows.append({
            "claim_type": claim_type,
            "actual_pct": actual_pct,
            "expected_pct": expected_pct,
            "actual_width_pct": actual_width_pct,
            "expected_width_pct": expected_width_pct,
            "delta_pct": round(actual_pct - expected_pct, 1),
            "status": status,
        })
    return rows


def _topic_status(actual: float, expected: float) -> str:
    if expected <= 0:
        return "ok"
    if actual < expected * 0.25:
        return "gap"
    if actual < expected * 0.50:
        return "thin"
    return "ok"


def _worst_topic_status(statuses: list[str]) -> str:
    rank = {"ok": 0, "thin": 1, "gap": 2}
    return max(statuses or ["ok"], key=lambda status: rank.get(status, 0))


def _coverage_median(rows: list[dict], count_key: str) -> float:
    for row in rows:
        for key in ("coverage_median", "median_claim_count", "median_count"):
            value = row.get(key)
            if value is not None:
                return float(value)
    counts = [int(row.get(count_key) or 0) for row in rows]
    return float(_statistics.median(counts)) if counts else 0.0


def _claim_count_label(count: int) -> str:
    return "1 claim" if count == 1 else f"{count} claims"


def _enrich_thin_rows(rows: list[dict], count_key: str, median: float) -> list[dict]:
    enriched = []
    for row in rows:
        count = int(row.get(count_key) or 0)
        copy = dict(row)
        copy["claim_count_label"] = _claim_count_label(count)
        copy["median_pct"] = round(count / median * 100) if median > 0 else None
        enriched.append(copy)
    return enriched


def _build_per_corpus_comparison(
    expected_shares_by_corpus: dict[str, dict[str, float]],
    counts_by_corpus: dict[str, dict[str, int]],
) -> dict:
    corpora = []
    actual_shares_by_corpus: dict[str, dict[str, float]] = {}
    for inst in sorted(expected_shares_by_corpus):
        counts = counts_by_corpus.get(inst, {})
        total = sum(counts.values())
        actual_shares_by_corpus[inst] = {
            claim_type: count / total
            for claim_type, count in counts.items()
        } if total else {}
        corpora.append({"institution_id": inst, "total_claims": total})

    claim_types = sorted(
        {
            claim_type
            for shares in expected_shares_by_corpus.values()
            for claim_type in shares
        } | {
            claim_type
            for counts in counts_by_corpus.values()
            for claim_type in counts
        }
    )

    rows = []
    for claim_type in claim_types:
        cells = []
        statuses = []
        for corpus in corpora:
            inst = corpus["institution_id"]
            expected = expected_shares_by_corpus.get(inst, {}).get(claim_type, 0.0)
            actual = actual_shares_by_corpus.get(inst, {}).get(claim_type, 0.0)
            status = _topic_status(actual, expected)
            statuses.append(status)
            actual_pct = round(actual * 100, 1)
            expected_pct = round(expected * 100, 1)
            width_scale = max(actual_pct, expected_pct)
            cells.append({
                "institution_id": inst,
                "actual_pct": actual_pct,
                "expected_pct": expected_pct,
                "actual_width_pct": round(actual_pct / width_scale * 100, 1) if width_scale else 0.0,
                "expected_width_pct": round(expected_pct / width_scale * 100, 1) if width_scale else 0.0,
                "status": status,
            })
        rows.append({
            "claim_type": claim_type,
            "cells": cells,
            "status": _worst_topic_status(statuses),
        })

    return {"corpora": corpora, "rows": rows}


def _build_gaps_context(
    temporal: list[dict],
    entity_depth: list[dict],
    geo_coverage: list[dict],
    claim_types: list[dict],
    conv_gaps: list[dict],
    per_corpus_expected_shares: dict[str, dict[str, float]] | None = None,
    per_corpus_claim_types: list[dict] | None = None,
) -> dict:
    temporal_gaps = _compute_temporal_gaps(temporal)
    temporal_timeline = _build_temporal_timeline(temporal)
    temporal_gap_threshold = _temporal_gap_threshold(temporal)
    if temporal_timeline["start_year"] is not None:
        year_span = f"{temporal_timeline['start_year']}-{temporal_timeline['end_year']}"
    else:
        year_span = "unknown"

    entity_depth_median = _coverage_median(entity_depth, "primary_claim_count")
    enriched_entity_depth = _enrich_thin_rows(
        entity_depth,
        "primary_claim_count",
        entity_depth_median,
    )
    entity_depth_by_type: dict[str, list[dict]] = {}
    for row in enriched_entity_depth:
        entity_depth_by_type.setdefault(row["entity_type"], []).append(row)

    geo_coverage_median = _coverage_median(geo_coverage, "location_specific_claims")
    enriched_geo_coverage = _enrich_thin_rows(
        geo_coverage,
        "location_specific_claims",
        geo_coverage_median,
    )

    # Blended topical share: union of claim_types across corpora, share
    # computed against the unioned total.
    total_claims = sum(r.get("count", 0) for r in claim_types)
    actual_shares: dict[str, float] = {}
    unclassified_count = 0
    if total_claims > 0:
        for r in claim_types:
            ct = r.get("claim_type") or ""
            actual_shares[ct] = r.get("count", 0) / total_claims
            if ct == "unclassified_assertion":
                unclassified_count = r.get("count", 0)

    show_unclassified_banner = total_claims > 0 and unclassified_count / total_claims > 0.10
    unclassified_pct = round(unclassified_count / total_claims * 100) if total_claims > 0 else 0

    # Per-corpus claim-type totals, keyed by institution_id → {claim_type: count}
    per_corpus_counts: dict[str, dict[str, int]] = {}
    for r in per_corpus_claim_types or []:
        inst = r.get("institution_id") or ""
        ct = r.get("claim_type") or ""
        per_corpus_counts.setdefault(inst, {})[ct] = int(r.get("count", 0))

    # Blend the expected shares across corpora weighted by document count.
    # If a corpus has no weight signal in per_corpus_counts, weight it equally.
    _shares_by_corpus = per_corpus_expected_shares or {}
    corpus_weights: dict[str, float] = {}
    for inst in _shares_by_corpus:
        corpus_weights[inst] = float(sum(per_corpus_counts.get(inst, {}).values()))
    total_weight = sum(corpus_weights.values())
    blended_expected: dict[str, float] = {}
    if total_weight > 0:
        for inst, shares in _shares_by_corpus.items():
            w = corpus_weights.get(inst, 0.0) / total_weight
            for ct, share in shares.items():
                blended_expected[ct] = blended_expected.get(ct, 0.0) + share * w
    elif _shares_by_corpus:
        # Unweighted average fallback when no counts are available yet.
        n = len(_shares_by_corpus)
        for shares in _shares_by_corpus.values():
            for ct, share in shares.items():
                blended_expected[ct] = blended_expected.get(ct, 0.0) + share / n

    topic_rows = _compute_topic_rows(actual_shares, blended_expected)

    # Per-corpus panel data for the gaps template.
    per_corpus_panels: list[dict] = []
    for inst, shares in sorted(_shares_by_corpus.items()):
        counts = per_corpus_counts.get(inst, {})
        corpus_total = sum(counts.values())
        corpus_actual = {
            ct: (cnt / corpus_total if corpus_total else 0.0)
            for ct, cnt in counts.items()
        }
        per_corpus_panels.append({
            "institution_id": inst,
            "total_claims": corpus_total,
            "topic_rows": _compute_topic_rows(corpus_actual, shares),
        })
    per_corpus_comparison = _build_per_corpus_comparison(_shares_by_corpus, per_corpus_counts)

    return {
        "temporal_gaps": temporal_gaps,
        "temporal_timeline": temporal_timeline,
        "temporal_gap_threshold": temporal_gap_threshold,
        "year_span": year_span,
        "entity_depth_median": entity_depth_median,
        "entity_depth_by_type": sorted(entity_depth_by_type.items()),
        "geo_coverage_median": geo_coverage_median,
        "geo_coverage": enriched_geo_coverage,
        "show_unclassified_banner": show_unclassified_banner,
        "unclassified_pct": unclassified_pct,
        "topic_rows": topic_rows,
        "per_corpus_panels": per_corpus_panels,
        "per_corpus_comparison": per_corpus_comparison,
        "conv_gaps": conv_gaps,
    }


def _build_history_context(
    rows: list[dict],
    total: int,
    saved: list[dict],
    q: str,
    bucket: str,
    limit: int,
    offset: int,
) -> dict:
    import urllib.parse as _urlparse

    def _rerun_url(
        query_text: str,
        rerun_bucket: str,
        year_min: object,
        year_max: object,
        conversation_id: str,
    ) -> str:
        params: dict[str, str] = {"q": query_text}
        if rerun_bucket:
            params["bucket"] = rerun_bucket
        if year_min is not None and str(year_min) != "":
            params["year_min"] = str(year_min)
        if year_max is not None and str(year_max) != "":
            params["year_max"] = str(year_max)
        if conversation_id:
            params["conversation_id"] = conversation_id
        return "/?" + _urlparse.urlencode(params)

    prev_offset = max(0, offset - limit)
    next_offset = offset + limit
    has_prev = offset > 0
    has_next = next_offset < total

    base_qs = ""
    if q:
        base_qs += f"&q={_urlparse.quote(q)}"
    if bucket:
        base_qs += f"&bucket={_urlparse.quote(bucket)}"

    row_ctxs = []
    for row in rows:
        qtext = row.get("query_text") or ""
        qtrun = qtext[:120] + ("\u2026" if len(qtext) > 120 else "")
        bkt = row.get("bucket") or ""
        yr_min = row.get("year_min")
        yr_max = row.get("year_max")
        if yr_min and yr_max:
            yr_str = f"{yr_min}\u2013{yr_max}"
        elif yr_min:
            yr_str = str(yr_min)
        elif yr_max:
            yr_str = str(yr_max)
        else:
            yr_str = "\u2014"
        ts = (row.get("created_at") or "")[:16]
        conv_id = row.get("conversation_id") or ""
        rerun_url = _rerun_url(qtext, bkt, yr_min, yr_max, conv_id)
        row_ctxs.append({
            "ts": ts,
            "qtext": qtext,
            "qtrun": qtrun,
            "bkt": bkt,
            "yr_str": yr_str,
            "yr_min": str(yr_min) if yr_min is not None else "",
            "yr_max": str(yr_max) if yr_max is not None else "",
            "conv_id": conv_id,
            "rerun_url": rerun_url,
        })

    saved_ctxs = []
    for s in saved:
        slabel = s.get("label") or ""
        sqtext = s.get("query_text") or ""
        sqtrun = sqtext[:100] + ("\u2026" if len(sqtext) > 100 else "")
        sbkt = s.get("bucket") or ""
        syr_min = s.get("year_min")
        syr_max = s.get("year_max")
        if syr_min and syr_max:
            syr_str = f"{syr_min}\u2013{syr_max}"
        elif syr_min:
            syr_str = str(syr_min)
        elif syr_max:
            syr_str = str(syr_max)
        else:
            syr_str = "\u2014"
        screated_by = s.get("created_by") or ""
        sts = (s.get("created_at") or "")[:16]
        saved_id = s.get("saved_id") or ""
        sconv_id = s.get("conversation_id") or ""
        srerun_url = _rerun_url(sqtext, sbkt, syr_min, syr_max, sconv_id)
        saved_ctxs.append({
            "label": slabel,
            "sqtext": sqtext,
            "sqtrun": sqtrun,
            "sbkt": sbkt,
            "syr_str": syr_str,
            "screated_by": screated_by,
            "sts": sts,
            "saved_id": saved_id,
            "sconv_id": sconv_id,
            "srerun_url": srerun_url,
        })

    bucket_options = [
        {"value": "", "label": "All types", "selected": bucket == ""},
        {"value": "conversational", "label": "Conversational", "selected": bucket == "conversational"},
        {"value": "analytical", "label": "Analytical", "selected": bucket == "analytical"},
        {"value": "hybrid", "label": "Hybrid", "selected": bucket == "hybrid"},
    ]

    shown_end = min(offset + limit, total)
    pagination = None
    if has_prev or has_next:
        skip_back_offset = max(0, offset - (limit * 5))
        skip_forward_offset = offset + (limit * 5)
        if total > 0:
            last_page_offset = ((total - 1) // limit) * limit
            skip_forward_offset = min(skip_forward_offset, last_page_offset)
        pagination = {
            "has_prev": has_prev,
            "has_next": has_next,
            "prev_url": f"/history?offset={prev_offset}&limit={limit}{base_qs}",
            "next_url": f"/history?offset={next_offset}&limit={limit}{base_qs}",
            "has_skip_back": offset - (limit * 5) >= 0,
            "has_skip_forward": offset + (limit * 5) < total,
            "skip_back_url": f"/history?offset={skip_back_offset}&limit={limit}{base_qs}",
            "skip_forward_url": f"/history?offset={skip_forward_offset}&limit={limit}{base_qs}",
            "shown_start": offset + 1,
            "shown_end": shown_end,
            "total": total,
        }

    return {
        "q": q,
        "bucket": bucket,
        "rows": row_ctxs,
        "saved": saved_ctxs,
        "bucket_options": bucket_options,
        "pagination": pagination,
    }


def _apply_composite_scope(
    user: UserContext,
    composite: Any,
    *,
    admin_only: bool,
) -> UserContext:
    """Rebind ``user.permitted_institution_ids`` to every registered corpus.

    When ``admin_only`` is True, non-admin callers are returned unchanged to
    preserve tenant isolation. The update is in-place on a fresh per-request
    ``UserContext`` so there is no cross-request mutation hazard.
    """
    if admin_only and user.role != "admin":
        return user
    if composite is not None:
        user.permitted_institution_ids = list(composite.institution_ids)
    return user


def _corpus_labels_for_composite(composite: Any) -> dict[str, str]:
    """Return display labels keyed by corpus/institution id."""
    labels: dict[str, str] = {}
    if composite is None:
        return labels
    for member in getattr(composite, "members", []) or []:
        institution_id = getattr(member, "institution_id", "")
        if not institution_id:
            continue
        labels[institution_id] = (
            getattr(member, "synthesis_context_label", None) or institution_id
        )
    return labels


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str = "neo4j",
    neo4j_trust: str = "system",
    neo4j_ca_cert: str | None = None,
    anthropic_api_key: str | None = None,
    max_tokens: int = 1000,
    domain_dir: str | None = None,
    annotation_db_path: str | None = None,
) -> FastAPI:
    """Construct and return the FastAPI retrieval application.

    All parameters fall back to environment variables so the app can be
    started from the CLI without explicit arguments.
    """
    from gemynd.shared.logging_config import setup_logging
    setup_logging()
    _log = logging.getLogger(__name__)

    # Resolve connection parameters from env if not provided explicitly.
    uri = neo4j_uri or os.environ.get("NEO4J_URI", "")
    user = neo4j_user or os.environ.get("NEO4J_USER", "")
    password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "")
    database = neo4j_database or os.environ.get("NEO4J_DATABASE", "neo4j")
    trust = neo4j_trust or os.environ.get("NEO4J_TRUST", "system")
    ca_cert = neo4j_ca_cert or os.environ.get("NEO4J_CA_CERT")
    api_key = anthropic_api_key or os.environ.get("Anthropic_API_Key") or os.environ.get("ANTHROPIC_API_KEY")

    # Shared pipeline components (initialised at startup, closed at shutdown).
    state: dict[str, Any] = {}

    from gemynd.shared.rate_limit import TokenBucketLimiter
    _query_limiter = TokenBucketLimiter(
        max_calls=int(os.environ.get("QUERY_RATE_LIMIT_MAX", "20")),
        period_seconds=float(os.environ.get("QUERY_RATE_LIMIT_PERIOD", "60")),
    )

    def _try_init(factory: Any) -> Any:
        """Attempt to construct a store; return None on failure."""
        try:
            return factory()
        except Exception:
            _log.warning("Store unavailable", exc_info=True)
            return None

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[misc]
        # --- Domain config: load every registered corpus from the
        #     corpus_registry.yaml, then union them into a composite view.
        #     The --domain-dir CLI flag still takes precedence when supplied
        #     (single-corpus mode), but by default we blend all registered
        #     corpora so the retrieval tools see the entire graph at once.
        from gemynd.core.domain_config import (
            load_all_registered_corpora,
            load_domain_config,
            merge_domain_configs,
        )
        if domain_dir:
            _members = [load_domain_config(Path(domain_dir))]
        else:
            _members = load_all_registered_corpora()
        if not _members:
            raise RuntimeError(
                "retrieval web app requires at least one registered corpus "
                "(data/corpus_registry.yaml) or an explicit --domain-dir"
            )
        _composite = merge_domain_configs(_members)
        _domain_config = _composite.default_member  # back-compat for call sites
        state["domain_config"] = _domain_config
        state["composite_config"] = _composite

        # Per-corpus source_link templates for the /documents retrieval surface.
        # Loaded from data/corpus_registry.yaml; entries with no source_links
        # block simply contribute an empty list. Skipped silently when the
        # registry isn't on disk (e.g. tests using --domain-dir).
        from gemynd.core.corpus_registry import load_registry as _load_corpus_registry
        try:
            _corpus_entries = _load_corpus_registry()
        except Exception:
            _log.warning("Could not load corpus_registry for source_links", exc_info=True)
            _corpus_entries = []
        state["source_links_by_corpus"] = {
            entry.corpus_id: [dict(sl) for sl in entry.source_links]
            for entry in _corpus_entries
        }

        # Pre-compute classifier inputs so each /query call avoids rebuilding
        # the seed vocab set and year bounds on every request.
        state["classifier_seed_vocab"] = frozenset(
            e.normalized_form.lower()
            for e in _composite.seed_entities
            if e.normalized_form
        )
        _yv = _domain_config.year_validation or {}
        _year_lo = _yv.get("min_year")
        _year_hi = _yv.get("max_year")
        state["classifier_year_bounds"] = (
            (int(_year_lo), int(_year_hi))
            if _year_lo is not None and _year_hi is not None
            else None
        )
        state["analytical_dispatch"] = _composite.analytical_queries

        # --- Neo4j executor ---
        executor = Neo4jQueryExecutor(
            uri=uri,
            user=user,
            password=password,
            database=database,
            trust_mode=trust,
            ca_cert_path=ca_cert,
            entity_labels=_composite.entity_labels,
        )
        executor.ensure_schema()
        state["executor"] = executor

        corpus_rows = executor.run(
            CORPUS_STATS_QUERY, {"institution_ids": _composite.institution_ids}
        )
        state["corpus_stats"] = corpus_rows[0] if corpus_rows else {"total_paragraphs": 0, "total_documents": 0}

        # --- Path-finder caches (Mode 1) ---
        # Per-entity claim count, used for inverse-frequency path weighting and
        # the degree-cap that rejects paths through generic hub entities.
        # Plus total active-claim count for the inverse-frequency normalizer.
        # Failures are logged but non-fatal: the path-finder degrades to
        # uniform weights with degree {} rather than blocking app startup.
        try:
            _degree_rows = executor.run(ENTITY_DEGREE_QUERY, {})
            state["entity_degree"] = {
                r["entity_id"]: int(r.get("degree") or 0)
                for r in _degree_rows
                if r.get("entity_id")
            }
        except Exception:
            _log.warning("Could not load ENTITY_DEGREE_QUERY for path-finder", exc_info=True)
            state["entity_degree"] = {}
        try:
            _total_rows = executor.run(
                "MATCH (c:Claim) "
                "WHERE c.quarantine_status IS NULL OR c.quarantine_status = 'active' "
                "RETURN count(c) AS total",
                {},
            )
            state["total_active_claims"] = int(_total_rows[0]["total"]) if _total_rows else 0
        except Exception:
            _log.warning("Could not load total_active_claims for path-finder", exc_info=True)
            state["total_active_claims"] = 0
        _nonzero_degree = sum(1 for v in state["entity_degree"].values() if v > 0)
        _log.info(
            "Path-finder caches loaded: %d entities (%d with non-zero degree), "
            "%d active claims corpus-wide",
            len(state["entity_degree"]), _nonzero_degree, state["total_active_claims"],
        )

        # --- DatabaseManager: centralised SQLite lifecycle ---
        from gemynd.shared.database_manager import DatabaseManager
        from gemynd.shared.settings import Settings
        _settings = Settings.from_env()
        if annotation_db_path:
            _settings = dataclasses.replace(_settings, annotation_db=annotation_db_path)
        db_mgr = DatabaseManager(_settings)
        db_mgr.run_migrations()
        state["db_manager"] = db_mgr

        # --- Persistent-connection stores (receive managed connections) ---
        from gemynd.review.store import ReviewStore
        state["review_store"] = _try_init(
            lambda: ReviewStore(_settings.review_db, conn=db_mgr.get_connection("review"))
        )

        _annotation_store = None
        if _settings.annotation_db:
            from gemynd.ingest.annotation import AnnotationStore
            _annotation_store = _try_init(
                lambda: AnnotationStore(_settings.annotation_db, conn=db_mgr.get_connection("annotation"))
            )
        state["annotation_store"] = _annotation_store

        from gemynd.retrieval.conversation_log import QueryHistoryStore as _QHSInit
        state["history_store"] = _QHSInit(
            _settings.conv_log_db, conn=db_mgr.get_connection("conv_log"),
        )

        from gemynd.shared.token_tracker import TokenUsageLogger, TokenUsageStore, load_pricing
        state["token_store"] = TokenUsageStore(
            _settings.token_usage_db, conn=db_mgr.get_connection("token_usage"),
        )

        # --- On-demand / daemon stores (skip_init, manager ran migrations) ---
        from gemynd.auth.store import UserStore
        state["user_store"] = _try_init(
            lambda: UserStore(_settings.users_db, skip_init=True)
        )

        from gemynd.ingest.store import IngestStore
        state["ingest_store"] = _try_init(
            lambda: IngestStore(_settings.ingest_db, skip_init=True)
        )

        from .write_audit_log import WriteAuditLogger
        state["write_audit"] = WriteAuditLogger(_settings.write_audit_db, skip_init=True)

        # Daemon writers: own their thread connections, skip schema init.
        _token_pricing = load_pricing()
        _token_logger = TokenUsageLogger(
            db_mgr.get_path("token_usage"), _token_pricing, skip_init=True,
        )
        state["token_logger"] = _token_logger
        state["conv_logger"] = ConversationLogger(
            db_mgr.get_path("conv_log"), skip_init=True,
        )
        # 90-day retention sweep for cached conversation turns.  Daemon thread,
        # fire-and-forget — runs once at startup, then every 24h.
        if state.get("history_store") is not None:
            start_retention_purge(state["history_store"], days=90)

        # --- Pipeline components ---
        state["query_builder"] = CypherQueryBuilder(
            executor,
            institution_ids=_composite.institution_ids,
        )
        # --- Optional text-to-Cypher LLM fallback ---
        # Built only when (a) the composite domain opts in via
        # `enable_llm_cypher` and (b) an Anthropic API key is configured.
        # Schema and cache are shared with the validator inside the generator.
        _llm_cypher_generator: Any = None
        if _composite.enable_llm_cypher and api_key:
            try:
                from gemynd.retrieval.cypher_validator import CypherSchema
                from gemynd.retrieval.cypher_cache import SqliteCypherCache
                from gemynd.retrieval.llm_cypher import LLMCypherGenerator
                from gemynd.shared.token_tracker import MeteredAnthropicClient
                import anthropic as _anthropic_pkg

                _anchor_rels: set[str] = set()
                for _m in _composite.members:
                    if _m.anchor_relation:
                        _anchor_rels.add(_m.anchor_relation)
                _llm_schema = CypherSchema.from_domain(
                    domain_labels=_composite.entity_labels,
                    domain_edges=frozenset(_composite.claim_entity_relation_precedence),
                    anchor_relations=frozenset(_anchor_rels),
                )
                _raw_anthropic = _anthropic_pkg.Anthropic(api_key=api_key)
                _llm_client = MeteredAnthropicClient(
                    client=_raw_anthropic,
                    logger=_token_logger,
                    caller="llm_cypher",
                    institution_id=_composite.institution_ids[0] if _composite.institution_ids else "",
                )
                _llm_cypher_generator = LLMCypherGenerator(
                    client=_llm_client,
                    domain_config=_composite,
                    schema=_llm_schema,
                    cache=SqliteCypherCache(db_mgr.get_connection("conv_log")),
                )
                _log.info("LLM-Cypher fallback enabled")
            except Exception:
                _log.warning("LLM-Cypher fallback disabled (init failed)", exc_info=True)
                _llm_cypher_generator = None

        state["assembler"] = ProvenanceContextAssembler(
            executor,
            annotation_store=_annotation_store,
            query_intent_map=_composite.query_intent_to_claim_types,
            institution_ids=_composite.institution_ids,
            anchors={
                inst_id: _composite.anchor_for(inst_id)
                for inst_id in _composite.institution_ids
            },
            llm_cypher_generator=_llm_cypher_generator,
        )
        # Build the gateway with a resolver whose vocabulary spans EVERY
        # registered corpus. Without this, ``EntityResolutionGateway()`` would
        # fall back to ``default_seed_entities()`` which reads only the default
        # resources_dir (Turnbull), leaving Spokane surface forms unresolvable
        # at query time and causing cross-corpus queries to return nothing.
        state["gateway"] = EntityResolutionGateway(
            resolver=DictionaryFuzzyResolver(
                config=_composite,
                policy=ResolutionPolicy(),
            ),
            executor=executor,
        )
        synthesis_ctx = _composite.synthesis_context()
        # Surface key-loading state at startup so a missing/short key is visible
        # before the first query rather than buried in a per-request exception.
        if api_key:
            _log.info(
                "synthesis: Anthropic API key loaded (length=%d, prefix=%s...)",
                len(api_key),
                api_key[:7],
            )
        else:
            _log.warning(
                "synthesis: no Anthropic API key found (checked Anthropic_API_Key, "
                "ANTHROPIC_API_KEY env vars and CLI param). All synthesis calls will fail."
            )
        state["synthesis"] = SynthesisEngine(
            api_key=api_key,
            max_tokens=max_tokens,
            synthesis_context=synthesis_ctx,
            timeout=float(os.environ.get("SYNTHESIS_TIMEOUT", "60")),
            token_logger=_token_logger,
        )

        yield
        executor.close()
        db_mgr.close_all()

    app = FastAPI(
        title="Turnbull GraphRAG Retrieval API",
        description="Natural-language query interface over the Turnbull refuge archive graph.",
        lifespan=lifespan,
    )

    # Auth router (login/logout/me/user management at /auth/...)
    from fastapi.responses import RedirectResponse as _RedirectResponse
    from fastapi.templating import Jinja2Templates
    _shared_dir = str(Path(__file__).parent.parent.parent / "shared_templates")
    _templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    _templates.env.loader = __import__("jinja2").FileSystemLoader(
        [str(_TEMPLATES_DIR), _shared_dir]
    )
    _templates.env.filters["conf_tier_class"] = _conf_tier_class

    def _user_ctx(user: UserContext) -> dict[str, Any]:
        """Build template context vars from the authenticated user."""
        email = user.email or ""
        name = email.split("@")[0] if email else "User"
        initials = (name[:2]).upper() if name else "U"
        return {
            "user_role": user.role,
            "user_display_name": name,
            "user_initials": initials,
            "is_admin": user.role in ("admin", "indigenous_admin"),
        }

    def require_user_scoped(user: UserContext = Depends(require_user)) -> UserContext:
        """Expand ``permitted_institution_ids`` to every registered corpus for
        admin callers. Non-admin roles remain pinned to their home
        institution for tenant isolation.
        """
        return _apply_composite_scope(user, state.get("composite_config"), admin_only=True)

    def require_admin_scoped(user: UserContext = Depends(require_admin)) -> UserContext:
        """Admin-gated counterpart of :func:`require_user_scoped`; callers are
        already guaranteed to be admins, so the composite corpus list is
        always applied.
        """
        return _apply_composite_scope(user, state.get("composite_config"), admin_only=False)

    from gemynd.auth.setup import is_setup_needed

    users_db_path = os.environ.get("USERS_DB", "data/users.db")
    app.include_router(create_auth_router(users_db_path), prefix="/auth")

    @app.exception_handler(NeedsLoginException)
    async def _needs_login_handler(request: Request, exc: NeedsLoginException):
        return _RedirectResponse(url=exc.redirect_url, status_code=303)

    from fastapi import HTTPException as _HTTPException
    from fastapi.responses import JSONResponse as _JSONResponse

    @app.exception_handler(Exception)
    async def _internal_error_handler(request: Request, exc: Exception) -> _JSONResponse:
        # Re-raise HTTPException so FastAPI's built-in handler produces the
        # correct status code and detail — only swallow truly unhandled errors.
        if isinstance(exc, _HTTPException):
            raise exc
        _log.error(
            "Unhandled exception on %s %s: %s",
            request.method, request.url, exc, exc_info=True,
        )
        return _JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred."},
        )

    import uuid as _uuid
    from gemynd.shared.response_headers import authenticated_no_store
    from gemynd.shared.csrf import GemyndCSRFMiddleware

    # Innermost middleware: sets Cache-Control on authenticated responses.
    # Decorated FIRST so it wraps closest to the handler — the cache headers
    # are set on the way out before any outer middleware can override them.
    @app.middleware("http")
    async def _no_store(request: Request, call_next):
        return await authenticated_no_store(request, call_next)

    # CSRF middleware sits between cache-headers (innermost) and request-id
    # (outermost).  Bearer-only requests are exempted at the request level
    # by GemyndCSRFMiddleware itself.  Falls back to JWT_SECRET_KEY if
    # CSRF_SECRET_KEY is unset so dev environments don't break — both are
    # private HMAC secrets and can be rotated independently in production.
    _csrf_secret = os.environ.get("CSRF_SECRET_KEY") or os.environ.get("JWT_SECRET_KEY", "")
    _cookie_secure = os.environ.get("COOKIE_SECURE", "true").lower() != "false"
    app.add_middleware(
        GemyndCSRFMiddleware,
        secret=_csrf_secret,
        cookie_name="csrftoken",
        cookie_samesite="strict",
        cookie_secure=_cookie_secure,
        cookie_httponly=False,
        header_name="x-csrftoken",
    )

    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.middleware("http")
    async def _setup_guard(request: Request, call_next):
        if not request.url.path.startswith("/auth/setup") and request.url.path != "/health":
            if is_setup_needed(users_db_path):
                return _RedirectResponse(url="/auth/setup", status_code=303)
        return await call_next(request)

    # ------------------------------------------------------------------
    # GET /health — liveness + Neo4j connectivity check
    # NOTE: This endpoint MUST remain unauthenticated. It is used by uptime
    # monitors and load balancers that have no session credentials. Do not
    # add require_user or any auth dependency here.
    # ------------------------------------------------------------------
    @app.get("/health", include_in_schema=True)
    def health():
        ts = datetime.now(timezone.utc).isoformat()
        executor = state.get("executor")
        if executor is None:
            return _JSONResponse(
                status_code=503,
                content={"status": "starting", "neo4j": "not_ready", "timestamp": ts},
            )
        try:
            executor.run("RETURN 1 AS ok", {})
            neo4j_status = "connected"
        except Exception:
            neo4j_status = "unavailable"
        neo4j_ok = neo4j_status == "connected"
        return _JSONResponse(
            status_code=200 if neo4j_ok else 503,
            content={"status": "ok" if neo4j_ok else "degraded", "neo4j": neo4j_status, "timestamp": ts},
        )

    # ------------------------------------------------------------------
    # GET / — Chat UI
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def chat_ui(request: Request, user: UserContext = Depends(require_user)) -> HTMLResponse:
        composite = state.get("composite_config")
        members: list[dict[str, Any]] = []
        if composite is not None:
            for m in composite.members:
                members.append({
                    "institution_id": m.institution_id,
                    "label": getattr(m, "synthesis_context_label", None) or m.institution_id,
                    "topic_hints": getattr(m, "welcome_topic_hints", []) or [],
                })
        bounds = state.get("classifier_year_bounds") or (None, None)
        return _templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "composite_members": members,
                "corpus_year_min": bounds[0],
                "corpus_year_max": bounds[1],
                **_user_ctx(user),
            },
        )

    # ------------------------------------------------------------------
    # POST /query
    # ------------------------------------------------------------------
    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, request: Request, user: UserContext = Depends(require_user_scoped)) -> QueryResponse:
        # Rate limiting: admins are exempt; all other users are limited to
        # QUERY_RATE_LIMIT_MAX requests per QUERY_RATE_LIMIT_PERIOD seconds.
        if user.role != "admin":
            _rate_key = user.user_id or user.institution_id
            if not _query_limiter.is_allowed(_rate_key):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Maximum 20 requests per minute.",
                    headers={"Retry-After": "60"},
                )

        gateway: EntityResolutionGateway = state["gateway"]
        assembler: ProvenanceContextAssembler = state["assembler"]
        builder: CypherQueryBuilder = state["query_builder"]
        engine: SynthesisEngine = state["synthesis"]

        created_at = datetime.now(timezone.utc).isoformat()
        # Always generate a unique per-turn conversation_id so every turn gets
        # its own row in the conversation + retrieval_event tables.  Previously
        # this was set to req.session_id, which collapsed all turns onto the
        # same primary key — the INSERT OR IGNORE in _write_record then silently
        # dropped entity resolution data for turns 2+ in a session.
        # session_id is stored separately in LogRecord for grouping multi-turn
        # sessions in the analytics queries.
        conversation_id = make_conversation_id(req.text, created_at)
        # Assumes symmetric history (alternating user/assistant pairs), which holds
        # for the browser client (history only accumulates on success). Non-browser
        # clients sending asymmetric history will get an approximate turn_number.
        turn_number = len(req.conversation_history) // 2 + 1

        # Layer 0: classify intent.
        _dc = state.get("domain_config")
        _seed_vocab = state.get("classifier_seed_vocab")
        _year_bounds = state.get("classifier_year_bounds")
        intent = classify_query(
            text=req.text,
            year_range=req.year_range,
            entity_hints=req.entity_hints or [],
            extra_stopwords=_dc.extraction_stopwords if _dc else None,
            year_bounds=_year_bounds,
            seed_vocab=_seed_vocab,
        )
        if req.mode != "auto":
            # Caller overrides bucket.
            intent = intent.__class__(  # type: ignore[call-arg]
                bucket=req.mode if req.mode != "auto" else intent.bucket,
                classifier_confidence=intent.classifier_confidence,
                entities=intent.entities,
                year_min=intent.year_min,
                year_max=intent.year_max,
                claim_types=intent.claim_types,
                doc_type_hints=intent.doc_type_hints,
                series_hints=intent.series_hints,
            )

        # Layer 1: entity resolution.
        _intent_map = getattr(state.get("composite_config"), "query_intent_to_claim_types", None)
        _inferred_ct = _infer_claim_types(req.text, intent_map=_intent_map)
        entity_ctx = gateway.resolve(
            surface_forms=intent.entities,
            entity_hints=req.entity_hints,
            inferred_claim_types=_inferred_ct,
        )
        _log.debug("entities extracted: %s", intent.entities)
        _log.debug("resolved: %s", [(r.surface_form, r.entity_id, r.resolution_confidence) for r in entity_ctx.resolved])
        _log.debug("unresolved: %s", entity_ctx.unresolved)

        # ------------------------------------------------------------------
        # Tier 1 fast-path: structural-lookup queries that bypass synthesis.
        # When the classifier detects a "how many ..." pattern AND entity
        # resolution returns exactly one high-confidence entity, we run a
        # focused Cypher and return a deterministic answer in well under a
        # second (vs. 30–120s for the full synthesis path).
        # ------------------------------------------------------------------
        tier_1_intent = detect_tier1_intent(
            req.text,
            entity_ctx,
            year_min=intent.year_min,
            year_max=intent.year_max,
        )
        if tier_1_intent is not None and req.mode != "analytical":
            _t0 = time.perf_counter()
            try:
                tier_1_result = assembler.tier_1_lookup(
                    tier_1_intent,
                    institution_ids=user.permitted_institution_ids,
                    permitted_levels=user.permitted_levels,
                )
                tier_1_ok = True
            except Exception as exc:
                # On any failure, fall through to the normal synthesis path
                # rather than returning a broken short-circuit response.
                _log.exception("tier_1 lookup failed (%s); falling through to synthesis", exc)
                tier_1_intent = None
                tier_1_ok = False

            if tier_1_intent is not None and tier_1_ok:
                tier_1_ms = round((time.perf_counter() - _t0) * 1000, 1)
                _tier1_caveat_by_op = {
                    "count_claims_for_entity": (
                        "Direct graph count — reflects active claims linked to "
                        "the resolved entity under the requesting user's "
                        "permission scope."
                    ),
                    "year_range_for_entity": (
                        "Direct graph lookup — year range reflects observation "
                        "years for active claims, falling back to the document's "
                        "report_year when an observation has no year link."
                    ),
                    "list_documents_for_entity": (
                        "Direct graph lookup — documents are ranked by the count "
                        "of active claims linking them to the resolved entity, "
                        "capped at 50."
                    ),
                    "list_documents_for_year": (
                        "Direct graph lookup — documents whose report_year "
                        "matches the requested year, sorted by date_start, "
                        "capped at 50."
                    ),
                    "locations_for_entity": (
                        "Direct graph lookup — places linked to the entity via "
                        "OCCURRED_AT, ranked by claim count, capped at 25."
                    ),
                    "co_occurring_entities_for_entity": (
                        "Direct graph lookup — other entities appearing in the "
                        "same active claims as the resolved entity, ranked by "
                        "shared-claim count, capped at 25."
                    ),
                    "frequency_ranking_for_type": (
                        "Direct graph lookup — entities of the requested type, "
                        "ranked by total active claim count under the user's "
                        "permission scope."
                    ),
                }
                tier_1_response = QueryResponse(
                    answer=tier_1_result.answer_text,
                    confidence_assessment=(
                        "Direct graph lookup — no LLM synthesis was performed."
                    ),
                    supporting_claim_ids=[],
                    caveats=[
                        _tier1_caveat_by_op.get(
                            tier_1_result.operation,
                            "Direct graph lookup — no LLM synthesis was performed.",
                        ),
                    ],
                    ambiguous_entities=entity_ctx.ambiguous,
                    synthesis_available=True,
                    tier_1_result={
                        "operation": tier_1_result.operation,
                        "entity_name": tier_1_result.entity_name,
                        "count": tier_1_result.count,
                        "items": tier_1_result.items,
                        "earliest_year": tier_1_result.earliest_year,
                        "latest_year": tier_1_result.latest_year,
                        "distinct_years": tier_1_result.distinct_years,
                        "target_year": tier_1_result.target_year,
                    },
                )

                # Log Tier 1 queries to the conversation log so timing analytics
                # include them and the chat UI's history view can replay them.
                conv_logger: ConversationLogger | None = state.get("conv_logger")
                if conv_logger is not None:
                    try:
                        _tier1_resp_json = tier_1_response.model_dump_json()
                    except Exception:
                        _tier1_resp_json = None
                    conv_logger.enqueue(LogRecord(
                        conversation_id=conversation_id,
                        query_text=req.text,
                        bucket=intent.bucket,
                        classifier_confidence=intent.classifier_confidence,
                        year_min=intent.year_min,
                        year_max=intent.year_max,
                        retrieval_path="tier_1",
                        created_at=created_at,
                        entity_ids_resolved=[e.entity_id for e in entity_ctx.resolved],
                        entity_types_resolved=[e.entity_type for e in entity_ctx.resolved],
                        candidates_retrieved=tier_1_result.count,
                        ocr_dropped=0,
                        claims_in_context=0,
                        session_id=req.session_id,
                        turn_number=turn_number,
                        request_id=getattr(request.state, "request_id", None),
                        user_id=user.user_id,
                        response_json=_tier1_resp_json,
                        claim_interactions=[],
                        timing_ms={"tier_1_ms": tier_1_ms},
                    ))
                return tier_1_response

        is_hybrid = intent.bucket == "hybrid"
        analytical_result = None

        # Layer 2A: analytical path.
        if intent.bucket in ("analytical", "hybrid"):
            analytical_dispatch: dict[str, dict[str, str]] = state.get(
                "analytical_dispatch", {}
            )
            # Walk resolved entities in order; fire the first one whose type
            # is registered in the domain's analytical_queries map. Domains
            # without analytical_queries get no analytical path at all.
            for resolved_entity in entity_ctx.resolved:
                spec = analytical_dispatch.get(resolved_entity.entity_type)
                if spec is None:
                    continue
                builder_name = spec.get("builder")
                param_name = spec.get("param")
                method = getattr(builder, builder_name, None) if builder_name else None
                if method is None or not param_name:
                    _log.warning(
                        "analytical_queries entry for %r references unknown builder %r",
                        resolved_entity.entity_type, builder_name,
                    )
                    continue
                analytical_result = method(
                    **{param_name: resolved_entity.entity_id},
                    year_min=intent.year_min,
                    year_max=intent.year_max,
                    permitted_levels=user.permitted_levels,
                    institution_ids=user.permitted_institution_ids,
                )
                break

        # Layer 2B: conversational path. Wall-clock instrumentation feeds
        # conversation_log.timing_json so we can attribute end-to-end latency
        # to retrieval vs. synthesis vs. LLM-cypher fallback after the fact.
        _stage_timings: dict[str, Any] = {}
        _t0 = time.perf_counter()
        blocks, context_text = assembler.assemble(
            query_text=req.text,
            entity_context=entity_ctx,
            year_min=intent.year_min,
            year_max=intent.year_max,
            is_hybrid=is_hybrid,
            permitted_levels=user.permitted_levels,
            institution_ids=user.permitted_institution_ids,
            intent=intent,
            doc_types=req.doc_types,
            series=req.series,
        )
        _stage_timings["assemble_ms"] = round((time.perf_counter() - _t0) * 1000, 1)
        if assembler._last_llm_cypher_ms is not None:
            _stage_timings["llm_cypher_ms"] = round(assembler._last_llm_cypher_ms, 1)
        # Capture which retrieval rung fired so timing analytics can correlate
        # latency with Cypher path (entity_anchored vs claim_type_scoped vs
        # fulltext, etc.).  Stored as the cascade's selected_label string.
        _plan_trace = assembler._last_plan_trace
        if _plan_trace and _plan_trace.selected_label:
            # Store as a JSON string field so SQL `json_extract` works.
            _stage_timings["selected_rung"] = _plan_trace.selected_label  # type: ignore[assignment]

        # Build coverage stats before synthesis.
        corpus = state["corpus_stats"]
        retrieval_stats = RetrievalStats(
            candidates_retrieved=assembler._last_candidate_count,
            ocr_dropped=assembler._last_ocr_dropped,
            claims_in_context=assembler._last_context_count,
            paragraphs_in_context=len({b.paragraph_id for b in blocks}),
            documents_in_context=len({b.doc_title for b in blocks}),
            corpus_total_paragraphs=int(corpus.get("total_paragraphs", 0)),
            corpus_total_documents=int(corpus.get("total_documents", 0)),
            plan_trace=assembler._last_plan_trace,
        )

        # Layer 3: synthesis.
        synthesis_available = True
        _t0 = time.perf_counter()
        try:
            result: SynthesisResult = engine.synthesise(
                query=req.text,
                provenance_blocks=blocks,
                context_text=context_text,
                analytical_result=analytical_result,
                conversation_history=[t.model_dump() for t in req.conversation_history],
            )
        except Exception as exc:
            # API unavailable, rate-limited, or response unparseable.
            # Return retrieved provenance blocks with a fallback answer so the
            # archivist can read source sentences directly without a 500 error.
            # Log with traceback so operators can diagnose the underlying cause —
            # the bare-except previously swallowed authentication failures, model
            # name typos, network errors, and JSON parse failures indistinguishably.
            _log.exception(
                "synthesis failed (%s: %s); returning fallback response with %d retrieved blocks",
                type(exc).__name__,
                exc,
                len(blocks),
            )
            synthesis_available = False
            result = SynthesisResult(
                answer=(
                    "AI synthesis is currently unavailable. "
                    "Retrieved source sentences are provided via supporting_claim_ids for direct review."
                ),
                confidence_assessment="",
                supporting_claim_ids=[b.claim_id for b in blocks],
                caveats=[
                    "AI synthesis was unavailable at query time. Source provenance blocks were retrieved successfully.",
                    f"Synthesis error: {type(exc).__name__}: {exc}",
                ],
            )
        # Capture synthesis wall-clock on both success and failure paths so the
        # timing data distinguishes "API was slow" from "API errored fast".
        _stage_timings["synthesise_ms"] = round((time.perf_counter() - _t0) * 1000, 1)
        _stage_timings["synthesise_ok"] = 1.0 if synthesis_available else 0.0
        # Stamp the retrieval path the assembler chose for this query so the
        # UI can flag llm_cypher answers as lower-trust.
        result.retrieval_path = assembler._last_retrieval_path

        # Count quarantined claims for this institution to include in response.
        quarantined_claims_count = 0
        try:
            _executor: Neo4jQueryExecutor = state["executor"]
            _qrows = _executor.run(
                COUNT_QUARANTINED_CLAIMS_QUERY,
                {
                    "institution_ids": user.permitted_institution_ids,
                    "permitted_levels": user.permitted_levels,
                },
            )
            quarantined_claims_count = int((_qrows[0].get("quarantined_count") or 0) if _qrows else 0)
        except Exception:
            pass

        final_caveats = list(result.caveats)
        if quarantined_claims_count > 0:
            final_caveats.append(
                f"{quarantined_claims_count} potentially relevant claim(s) are currently under "
                "sensitivity review and have been excluded from this answer."
            )

        # Build per-claim detail for uncertainty surfacing in the UI.
        claims_detail = [
            {
                "claim_id": b.claim_id,
                "source_sentence": b.source_sentence,
                "confidence": b.extraction_confidence,
                "epistemic_status": b.epistemic_status,
                "claim_type": b.claim_type,
                "doc_id": b.doc_id,
                "doc_title": _citation_label(b),
                "doc_title_raw": b.doc_title,
            }
            for b in blocks
        ]

        response = QueryResponse(
            answer=result.answer,
            confidence_assessment=result.confidence_assessment,
            supporting_claim_ids=result.supporting_claim_ids,
            caveats=final_caveats,
            min_extraction_confidence=result.min_extraction_confidence,
            analytical_result=(
                {
                    "query_name": analytical_result.query_name,
                    "columns": analytical_result.columns,
                    "rows": analytical_result.rows,
                }
                if analytical_result
                else None
            ),
            ambiguous_entities=entity_ctx.ambiguous,
            retrieval_stats=dataclasses.asdict(retrieval_stats),
            synthesis_available=synthesis_available,
            quarantined_claims_count=quarantined_claims_count,
            supporting_claims_detail=claims_detail,
            user_can_annotate=user.role in ("admin", "archivist"),
        )

        # Fire-and-forget log write.  Deferred until after the QueryResponse is
        # built so `response_json` captures the exact payload the client will
        # see (post-PII-redaction, inclusive of quarantine caveat and claim
        # detail).  Only authenticated human users populate the conversation
        # cache — bearer-token API clients have `user_id=None` and are
        # excluded from the chat-history sidebar.
        conv_logger: ConversationLogger | None = state.get("conv_logger")
        if conv_logger is not None:
            cited_ids = set(result.supporting_claim_ids)
            _cache_response_json: str | None = None
            if user.user_id is not None:
                try:
                    _cache_response_json = response.model_dump_json()
                except Exception:  # noqa: BLE001 — never block the response
                    _cache_response_json = None
            conv_logger.enqueue(LogRecord(
                conversation_id=conversation_id,
                query_text=req.text,
                bucket=intent.bucket,
                classifier_confidence=intent.classifier_confidence,
                year_min=intent.year_min,
                year_max=intent.year_max,
                retrieval_path="entity_anchored" if entity_ctx.resolved else "fulltext",
                created_at=created_at,
                entity_ids_resolved=[e.entity_id for e in entity_ctx.resolved],
                entity_types_resolved=[e.entity_type for e in entity_ctx.resolved],
                candidates_retrieved=retrieval_stats.candidates_retrieved,
                ocr_dropped=retrieval_stats.ocr_dropped,
                claims_in_context=retrieval_stats.claims_in_context,
                session_id=req.session_id,
                turn_number=turn_number,
                request_id=getattr(request.state, "request_id", None),
                user_id=user.user_id,
                response_json=_cache_response_json,
                claim_interactions=[
                    ClaimInteraction(
                        claim_id=_safe_log_claim_id(b.claim_id, b.access_level),
                        claim_type=b.claim_type,
                        traversal_rel_types=b.traversal_rel_types,
                        was_cited=b.claim_id in cited_ids,
                        extraction_confidence=b.extraction_confidence,
                    )
                    for b in blocks
                ],
                timing_ms=_stage_timings,
            ))

        return response

    # ------------------------------------------------------------------
    # POST /query/provenance
    # ------------------------------------------------------------------
    @app.post("/query/provenance")
    def query_provenance(req: ProvenanceRequest, user: UserContext = Depends(require_user_scoped)) -> dict[str, Any]:
        """Return the full raw provenance chain for a known *claim_id*.

        Does not invoke the synthesis engine — useful for citation
        verification and the review web interface.
        """
        assembler: ProvenanceContextAssembler = state["assembler"]
        blocks = assembler.chain_for_claim(
            req.claim_id,
            permitted_levels=user.permitted_levels,
            institution_ids=user.permitted_institution_ids,
        )
        if not blocks:
            raise HTTPException(status_code=404, detail=f"No provenance found for claim_id={req.claim_id!r}")
        return {
            "claim_id": req.claim_id,
            "provenance_blocks": [
                {
                    "doc_title": b.doc_title,
                    "doc_date_start": b.doc_date_start,
                    "doc_date_end": b.doc_date_end,
                    "page_number": b.page_number,
                    "paragraph_id": b.paragraph_id,
                    "claim_id": b.claim_id,
                    "claim_type": b.claim_type,
                    "extraction_confidence": b.extraction_confidence,
                    "epistemic_status": b.epistemic_status,
                    "source_sentence": b.source_sentence,
                    "observation_type": b.observation_type,
                    "species_name": b.species_name,
                    "year": b.year,
                    "measurements": b.measurements,
                }
                for b in blocks
            ],
            "serialised_context": "\n\n".join(_serialise_block(b) for b in blocks),
        }

    # ------------------------------------------------------------------
    # GET /stats  — collection statistics dashboard
    # ------------------------------------------------------------------
    @app.get("/stats", response_class=HTMLResponse, include_in_schema=False)
    def collection_stats(request: Request, user: UserContext = Depends(require_user_scoped)):
        executor: Neo4jQueryExecutor = state["executor"]
        params = {
            "institution_ids": user.permitted_institution_ids,
            "permitted_levels": user.permitted_levels,
        }
        overview = (executor.run(STATS_DOC_OVERVIEW_QUERY, params) or [{}])[0]
        doc_types = executor.run(STATS_DOC_TYPE_QUERY, params) or []
        claim_types = executor.run(STATS_CLAIM_TYPE_QUERY, params) or []
        entity_types = executor.run(STATS_ENTITY_TYPE_QUERY, params) or []
        temporal = executor.run(STATS_TEMPORAL_COVERAGE_QUERY, params) or []
        confidence = (executor.run(STATS_CONFIDENCE_DISTRIBUTION_QUERY, params) or [{}])[0]
        ctx = _build_stats_context(overview, doc_types, claim_types, entity_types, temporal, confidence)
        return _templates.TemplateResponse("stats.html", {"request": request, "active_page": "stats", **_user_ctx(user), **ctx})

    # ------------------------------------------------------------------
    # GET /gaps  — collection gap analysis dashboard
    # ------------------------------------------------------------------
    @app.get("/gaps", response_class=HTMLResponse, include_in_schema=False)
    def collection_gaps(request: Request, user: UserContext = Depends(require_admin_scoped)):
        executor: Neo4jQueryExecutor = state["executor"]
        params = {
            "institution_ids": user.permitted_institution_ids,
            "permitted_levels": user.permitted_levels,
        }
        thin_params = {**params, "thin_threshold": 3, "limit": 50}

        temporal = executor.run(GAP_TEMPORAL_DENSITY_QUERY, params) or []
        entity_depth = executor.run(GAP_ENTITY_DEPTH_QUERY, thin_params) or []
        geo_coverage = executor.run(GAP_GEOGRAPHIC_COVERAGE_QUERY, thin_params) or []
        claim_types = executor.run(STATS_CLAIM_TYPE_QUERY, params) or []
        per_corpus_claim_types = executor.run(
            STATS_CLAIM_TYPE_BY_INSTITUTION_QUERY, params
        ) or []

        conv_gaps: list[dict] = []
        conv_log_db = os.environ.get("CONV_LOG_DB", "")
        if conv_log_db and os.path.exists(conv_log_db):
            conv_gaps = _read_query_signal_gaps(conv_log_db)

        _composite = state.get("composite_config")
        _per_corpus_shares = _composite.expected_claim_shares() if _composite else {}
        ctx = _build_gaps_context(
            temporal, entity_depth, geo_coverage, claim_types, conv_gaps,
            per_corpus_expected_shares=_per_corpus_shares,
            per_corpus_claim_types=per_corpus_claim_types,
        )
        return _templates.TemplateResponse("gaps.html", {"request": request, "active_page": "gaps", **_user_ctx(user), **ctx})

    # ------------------------------------------------------------------
    # GET /map  — relationship mapping explorer
    # ------------------------------------------------------------------
    @app.get("/map", response_class=HTMLResponse, include_in_schema=False)
    def relationship_map(request: Request, user: UserContext = Depends(require_user)):
        return _templates.TemplateResponse("map.html", {"request": request, "active_page": "map", **_user_ctx(user)})

    # ------------------------------------------------------------------
    # GET /api/entities/search  — entity autocomplete
    # ------------------------------------------------------------------
    @app.get("/api/entities/search")
    def entity_search(
        q: str = Query("", min_length=0),
        limit: int = Query(15, le=50),
        user: UserContext = Depends(require_user_scoped),
    ) -> list[dict]:
        if not q.strip():
            return []
        executor: Neo4jQueryExecutor = state["executor"]
        return executor.run(
            ENTITY_SEARCH_QUERY,
            {
                "query": q.strip(),
                "limit": limit,
                "institution_ids": user.permitted_institution_ids,
            },
        ) or []

    # ------------------------------------------------------------------
    # GET /api/entity/{entity_id}/neighborhood  — graph neighborhood data
    # ------------------------------------------------------------------
    @app.get("/api/entity/{entity_id}/neighborhood")
    def entity_neighborhood(
        entity_id: str,
        limit: int = Query(15, le=30),
        user: UserContext = Depends(require_user_scoped),
    ) -> dict[str, Any]:
        if not _ENTITY_ID_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity_id format")
        executor: Neo4jQueryExecutor = state["executor"]
        params = {
            "institution_ids": user.permitted_institution_ids,
            "permitted_levels": user.permitted_levels,
        }

        center_rows = executor.run(ENTITY_DETAIL_QUERY, {**params, "entity_id": entity_id}) or [{}]
        center = center_rows[0] if center_rows else {}
        if not center.get("entity_id"):
            raise HTTPException(status_code=404, detail="Entity not found")

        neighbors = executor.run(
            ENTITY_NEIGHBORHOOD_QUERY, {**params, "entity_id": entity_id, "limit": limit}
        ) or []

        nodes: list[dict] = [{"data": {
            "id": center["entity_id"], "label": center["name"],
            "entity_type": center.get("entity_type", ""),
            "claim_count": center.get("claim_count", 0),
            "institution_ids": [],
            "is_center": True,
        }}]
        edges: list[dict] = []
        for n in neighbors:
            nodes.append({"data": {
                "id": n["entity_id"], "label": n["name"],
                "entity_type": n.get("entity_type", ""),
                "claim_count": 0,
                "institution_ids": n.get("institution_ids") or [],
                "is_center": False,
            }})
            edges.append({"data": {
                "id": f"{entity_id}__{n['entity_id']}",
                "source": entity_id, "target": n["entity_id"],
                "weight": n["co_occurrence_count"],
                "relationship_types": n.get("relationship_types") or [],
                "sample_sentences": n.get("sample_sentences") or [],
                "sample_claim_ids": n.get("sample_claim_ids") or [],
                "institution_ids": n.get("institution_ids") or [],
                "source_label": center["name"],
                "target_label": n["name"],
            }})

        return {"nodes": nodes, "edges": edges, "center": center}

    # ------------------------------------------------------------------
    # GET /paths           — Mode 1 page shell
    # GET /paths/results   — HTMX fragment: cascade-rendered HTML
    # ------------------------------------------------------------------
    @app.get("/paths", response_class=HTMLResponse, include_in_schema=False)
    def paths_page(
        request: Request,
        user: UserContext = Depends(require_user_scoped),
    ):
        bounds = state.get("classifier_year_bounds") or (None, None)
        return _templates.TemplateResponse("paths.html", {
            "request": request,
            "active_page": "paths",
            "corpus_year_min": bounds[0],
            "corpus_year_max": bounds[1],
            **_user_ctx(user),
        })

    @app.get("/paths/results", response_class=HTMLResponse, include_in_schema=False)
    def paths_results(
        request: Request,
        from_id: str = Query(..., alias="from"),
        to_id: str = Query(..., alias="to"),
        year_min: str | None = Query(None),
        year_max: str | None = Query(None),
        max_hops: int = Query(2, ge=1, le=2),
        include_paths: bool = Query(False),
        user: UserContext = Depends(require_user_scoped),
    ):
        if not _ENTITY_ID_RE.match(from_id):
            raise HTTPException(status_code=400, detail="Invalid from entity_id")
        if not _ENTITY_ID_RE.match(to_id):
            raise HTTPException(status_code=400, detail="Invalid to entity_id")
        if from_id == to_id:
            return _templates.TemplateResponse(
                "fragments/path_results.html",
                {"request": request, "error": "Pick two different entities.",
                 **_user_ctx(user)},
            )
        from gemynd.retrieval.web.path_finder import find_paths, path_result_to_dict

        def _parse_year(v: str | None) -> int | None:
            if not v:
                return None
            try:
                return int(v.strip())
            except (ValueError, AttributeError):
                return None

        executor: Neo4jQueryExecutor = state["executor"]
        try:
            result = find_paths(
                executor.run,
                from_id=from_id, to_id=to_id,
                institution_ids=user.permitted_institution_ids,
                permitted_levels=user.permitted_levels,
                year_min=_parse_year(year_min), year_max=_parse_year(year_max),
                max_hops=max_hops,
                degree=state.get("entity_degree", {}),
                corpus_total_claims=int(state.get("total_active_claims", 0)),
                include_paths_when_co_asserted=include_paths,
            )
        except Exception as exc:
            # Most likely a Neo4j TransientError when the join cardinality
            # exceeds the configured transaction memory budget. Surface as a
            # graceful in-page message rather than a 500 — this happens for
            # extreme hub-to-hub queries where even the Cypher-level hub
            # exclusion can't fully bound the search.
            from neo4j.exceptions import TransientError
            if isinstance(exc, TransientError):
                _log.warning("Path query exhausted Neo4j memory: %s", exc)
                return _templates.TemplateResponse(
                    "fragments/path_results.html",
                    {"request": request,
                     "error": ("This path search exceeded the database's "
                               "memory budget. Try narrowing the year range, "
                               "or pick more specific endpoints."),
                     **_user_ctx(user)},
                )
            raise
        # Fetch endpoint metadata for display (names, types).
        head_rows = executor.run(
            "MATCH (e {entity_id: $entity_id}) "
            "RETURN e.entity_id AS entity_id, e.name AS name, "
            "       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type",
            {"entity_id": from_id},
        )
        from_meta = head_rows[0] if head_rows else {"entity_id": from_id, "name": from_id, "entity_type": ""}
        head_rows = executor.run(
            "MATCH (e {entity_id: $entity_id}) "
            "RETURN e.entity_id AS entity_id, e.name AS name, "
            "       [x IN labels(e) WHERE x <> 'Entity'][0] AS entity_type",
            {"entity_id": to_id},
        )
        to_meta = head_rows[0] if head_rows else {"entity_id": to_id, "name": to_id, "entity_type": ""}

        return _templates.TemplateResponse("fragments/path_results.html", {
            "request": request,
            "from_meta": from_meta,
            "to_meta": to_meta,
            "result": path_result_to_dict(result),
            "year_min": _parse_year(year_min),
            "year_max": _parse_year(year_max),
            "include_paths": include_paths,
            "error": None,
            **_user_ctx(user),
        })

    # ------------------------------------------------------------------
    # GET /paths/edge  — HTMX side-panel: all claims linking two entities
    # ------------------------------------------------------------------
    @app.get("/paths/edge", response_class=HTMLResponse, include_in_schema=False)
    def paths_edge_panel(
        request: Request,
        from_id: str = Query(..., alias="from"),
        to_id: str = Query(..., alias="to"),
        year_min: str | None = Query(None),
        year_max: str | None = Query(None),
        user: UserContext = Depends(require_user_scoped),
    ):
        if not _ENTITY_ID_RE.match(from_id):
            raise HTTPException(status_code=400, detail="Invalid from entity_id")
        if not _ENTITY_ID_RE.match(to_id):
            raise HTTPException(status_code=400, detail="Invalid to entity_id")

        def _parse_year(v: str | None) -> int | None:
            if not v:
                return None
            try:
                return int(v.strip())
            except (ValueError, AttributeError):
                return None

        from gemynd.core.graph.cypher import CO_ASSERTING_CLAIMS_QUERY
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(CO_ASSERTING_CLAIMS_QUERY, {
            "from_id": from_id, "to_id": to_id,
            "institution_ids": user.permitted_institution_ids,
            "permitted_levels": user.permitted_levels,
            "year_min": _parse_year(year_min),
            "year_max": _parse_year(year_max),
        })
        # Endpoint name lookup for the panel title.
        head_rows = executor.run(
            "MATCH (e {entity_id: $entity_id}) "
            "RETURN e.name AS name",
            {"entity_id": from_id},
        )
        from_name = (head_rows[0]["name"] if head_rows else from_id) or from_id
        head_rows = executor.run(
            "MATCH (e {entity_id: $entity_id}) "
            "RETURN e.name AS name",
            {"entity_id": to_id},
        )
        to_name = (head_rows[0]["name"] if head_rows else to_id) or to_id
        return _templates.TemplateResponse("fragments/path_edge_panel.html", {
            "request": request,
            "from_id": from_id, "to_id": to_id,
            "from_name": from_name, "to_name": to_name,
            "claims": rows,
            "year_min": _parse_year(year_min),
            "year_max": _parse_year(year_max),
            **_user_ctx(user),
        })

    # ------------------------------------------------------------------
    # GET /api/path  — Mode 1: cascade-shaped path-finding between two entities
    # ------------------------------------------------------------------
    @app.get("/api/path")
    def api_path(
        from_id: str = Query(..., alias="from"),
        to_id: str = Query(..., alias="to"),
        year_min: int | None = Query(None),
        year_max: int | None = Query(None),
        max_hops: int = Query(2, ge=1, le=2),
        include_paths: bool = Query(False),
        user: UserContext = Depends(require_user_scoped),
    ) -> dict[str, Any]:
        if not _ENTITY_ID_RE.match(from_id):
            raise HTTPException(status_code=400, detail="Invalid from entity_id")
        if not _ENTITY_ID_RE.match(to_id):
            raise HTTPException(status_code=400, detail="Invalid to entity_id")
        if from_id == to_id:
            raise HTTPException(status_code=400, detail="from and to must be different entities")
        from gemynd.retrieval.web.path_finder import find_paths, path_result_to_dict
        executor: Neo4jQueryExecutor = state["executor"]
        result = find_paths(
            executor.run,
            from_id=from_id, to_id=to_id,
            institution_ids=user.permitted_institution_ids,
            permitted_levels=user.permitted_levels,
            year_min=year_min, year_max=year_max,
            max_hops=max_hops,
            degree=state.get("entity_degree", {}),
            corpus_total_claims=int(state.get("total_active_claims", 0)),
            include_paths_when_co_asserted=include_paths,
        )
        return {
            "endpoints": {"from": from_id, "to": to_id},
            **path_result_to_dict(result),
        }

    # ------------------------------------------------------------------
    # GET /documents  — faceted retrieval surface for all logged-in users
    # GET /documents/list — HTMX partial: filtered/paginated row block
    # GET /document/{doc_id}/panel — HTMX partial: per-document detail panel
    # ------------------------------------------------------------------
    _DOCS_PAGE_SIZE = 50

    def _list_documents_json(user: UserContext = Depends(require_admin_scoped)) -> dict[str, Any]:
        """JSON list for /v1/documents API."""
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(
            LIST_DOCUMENTS_QUERY,
            {"institution_ids": user.permitted_institution_ids},
        )
        return {"institution_ids": user.permitted_institution_ids, "documents": rows}

    def _truncate_snippet(text: str | None, length: int = 160) -> str:
        if not text:
            return ""
        s = " ".join(text.split())
        return s if len(s) <= length else s[: length - 1].rstrip() + "…"

    def _decorate_document_rows(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply OCR-title scrubbing, snippet truncation, and per-corpus
        source-link rendering to a batch of document rows."""
        scrubbed = _scrub_doc_list_titles(rows)
        links_by_corpus: dict[str, list[dict[str, Any]]] = state.get(
            "source_links_by_corpus", {}
        ) or {}
        out: list[dict[str, Any]] = []
        for row in scrubbed:
            new_row = dict(row)
            new_row["snippet"] = _truncate_snippet(row.get("snippet"))
            corpus_id = (row.get("institution_id") or "").strip()
            cfgs = links_by_corpus.get(corpus_id, [])
            new_row["source_links"] = _build_source_links(row, cfgs)
            out.append(new_row)
        return out

    def _parse_docs_filters(
        q: str,
        institution: list[str] | None,
        doc_type: list[str] | None,
        year_min: str | None,
        year_max: str | None,
        status: str,
    ) -> dict[str, Any]:
        """Normalize raw query-string filter values into Cypher parameters.

        ``year_min`` / ``year_max`` arrive as strings because the form ships
        empty inputs as ``year_min=`` (which would 422 if typed as int). Empty
        and unparseable values collapse to None.
        """
        def _parse_year(v: str | None) -> int | None:
            if v is None:
                return None
            stripped = v.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                return None

        valid_status = status if status in ("active", "deleted", "all") else "active"
        return {
            "search": (q or "").strip() or None,
            "institution_filter": [i for i in (institution or []) if i] or None,
            "doc_types": [d for d in (doc_type or []) if d] or None,
            "year_min": _parse_year(year_min),
            "year_max": _parse_year(year_max),
            "status": valid_status,
        }

    def _docs_list_context(
        user: UserContext,
        request: Request,
        filters: dict[str, Any],
        page: int,
    ) -> dict[str, Any]:
        """Run the paginated query + count and assemble the row-fragment context."""
        executor: Neo4jQueryExecutor = state["executor"]
        offset = max(0, (page - 1) * _DOCS_PAGE_SIZE)
        params = {
            "institution_ids": user.permitted_institution_ids,
            **filters,
            "offset": offset,
            "limit": _DOCS_PAGE_SIZE,
        }
        rows = executor.run(LIST_DOCUMENTS_PAGINATED_QUERY, params)
        decorated = _decorate_document_rows(rows)
        count_rows = executor.run(
            COUNT_DOCUMENTS_QUERY,
            {k: v for k, v in params.items() if k not in ("offset", "limit")},
        )
        total = int(count_rows[0]["total"]) if count_rows else 0
        total_pages = max(1, (total + _DOCS_PAGE_SIZE - 1) // _DOCS_PAGE_SIZE)
        return {
            "request": request,
            "documents": decorated,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "page_size": _DOCS_PAGE_SIZE,
            "filters": filters,
            "corpus_labels": _corpus_labels_for_composite(
                state.get("composite_config")
            ),
            **_user_ctx(user),
        }

    @app.get("/documents", response_class=HTMLResponse, include_in_schema=False)
    def list_documents(
        request: Request,
        q: str = Query(""),
        institution: list[str] | None = Query(None),
        doc_type: list[str] | None = Query(None),
        year_min: str | None = Query(None),
        year_max: str | None = Query(None),
        status: str = Query("active"),
        page: int = Query(1, ge=1),
        user: UserContext = Depends(require_user_scoped),
    ):
        executor: Neo4jQueryExecutor = state["executor"]
        facets = executor.run(
            DOCUMENT_FACETS_QUERY,
            {"institution_ids": user.permitted_institution_ids},
        )
        facet_row = facets[0] if facets else {
            "doc_types": [], "institutions": [],
            "year_min": None, "year_max": None, "total_documents": 0,
        }
        filters = _parse_docs_filters(q, institution, doc_type, year_min, year_max, status)
        ctx = _docs_list_context(user, request, filters, page)
        ctx.update({
            "active_page": "documents",
            "facets": {
                "doc_types": sorted([dt for dt in (facet_row.get("doc_types") or []) if dt]),
                "institutions": sorted([i for i in (facet_row.get("institutions") or []) if i]),
                "year_min": facet_row.get("year_min"),
                "year_max": facet_row.get("year_max"),
                "total_documents": facet_row.get("total_documents") or 0,
            },
            "institution_ids": user.permitted_institution_ids,
        })
        return _templates.TemplateResponse("documents.html", ctx)

    @app.get("/documents/list", response_class=HTMLResponse, include_in_schema=False)
    def list_documents_partial(
        request: Request,
        q: str = Query(""),
        institution: list[str] | None = Query(None),
        doc_type: list[str] | None = Query(None),
        year_min: str | None = Query(None),
        year_max: str | None = Query(None),
        status: str = Query("active"),
        page: int = Query(1, ge=1),
        user: UserContext = Depends(require_user_scoped),
    ):
        filters = _parse_docs_filters(q, institution, doc_type, year_min, year_max, status)
        ctx = _docs_list_context(user, request, filters, page)
        return _templates.TemplateResponse("fragments/documents_list.html", ctx)

    def _fetch_document_ocr_text(
        doc_id: str,
        user: UserContext,
    ) -> tuple[str, str]:
        """Return (clean_text, raw_ocr_text) for a document.

        Reads the Document-level properties first; falls back to walking
        paragraphs in graph order when those are empty (some ingestion paths
        persist OCR only at paragraph granularity).
        """
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(
            GET_DOCUMENT_SOURCE_TEXT_QUERY,
            {"doc_id": doc_id, "institution_ids": user.permitted_institution_ids},
        )
        if not rows:
            return "", ""
        head = rows[0]
        clean_text = (head.get("clean_text") or "").strip()
        raw_ocr_text = (head.get("raw_ocr_text") or "").strip()
        if clean_text or raw_ocr_text:
            return clean_text, raw_ocr_text
        para_rows = executor.run(
            GET_DOCUMENT_PARAGRAPHS_QUERY,
            {"doc_id": doc_id, "institution_ids": user.permitted_institution_ids},
        )
        clean_chunks: list[str] = []
        raw_chunks: list[str] = []
        for p in para_rows:
            ct = (p.get("clean_text") or "").strip()
            rt = (p.get("raw_ocr_text") or "").strip()
            if ct:
                clean_chunks.append(ct)
            if rt:
                raw_chunks.append(rt)
        return "\n\n".join(clean_chunks), "\n\n".join(raw_chunks)

    def _render_document_modal(
        doc_id: str,
        request: Request,
        user: UserContext,
        tab: str = "ocr",
        view: str = "clean",
    ) -> HTMLResponse:
        """Render the unified per-document modal with the requested tab active."""
        executor: Neo4jQueryExecutor = state["executor"]
        header_rows = executor.run(
            GET_DOCUMENT_HEADER_QUERY,
            {"doc_id": doc_id, "institution_ids": user.permitted_institution_ids},
        )
        if not header_rows:
            raise HTTPException(status_code=404, detail="Document not found")
        decorated = _decorate_document_rows(header_rows)
        doc = decorated[0]

        active_tab = tab if tab in ("ocr", "claims", "notes") else "ocr"

        # OCR tab content
        clean_text = ""
        raw_ocr_text = ""
        has_clean = False
        has_raw = False
        ocr_body = ""
        view_mode = "clean"
        if active_tab == "ocr":
            clean_text, raw_ocr_text = _fetch_document_ocr_text(doc_id, user)
            has_clean = bool(clean_text)
            has_raw = bool(raw_ocr_text)
            view_mode = view if view in ("clean", "raw") else "clean"
            if view_mode == "raw" and not has_raw and has_clean:
                view_mode = "clean"
            if view_mode == "clean" and not has_clean and has_raw:
                view_mode = "raw"
            ocr_body = clean_text if view_mode == "clean" else raw_ocr_text

        # Claims tab content
        claims: list[dict[str, Any]] = []
        if active_tab == "claims":
            claims = executor.run(
                LIST_CLAIMS_FOR_DOCUMENT_QUERY,
                {"doc_id": doc_id, "institution_ids": user.permitted_institution_ids},
            )

        # Claim count is needed by the tab nav badge regardless of active tab.
        # Cheap to compute as a side query; in practice deferring until claim
        # tab is opened would mean the badge says "Claims" not "Claims (N)".
        count_rows = executor.run(
            "MATCH (d:Document {doc_id: $doc_id}) "
            "WHERE d.institution_id IN $institution_ids "
            "OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)"
            "-[:HAS_PARAGRAPH]->(:Paragraph)-[:HAS_CLAIM]->(c:Claim) "
            "WHERE c.quarantine_status IS NULL OR c.quarantine_status = 'active' "
            "RETURN count(c) AS n",
            {"doc_id": doc_id, "institution_ids": user.permitted_institution_ids},
        )
        claim_count = int(count_rows[0]["n"]) if count_rows else 0

        return _templates.TemplateResponse(
            "fragments/document_modal.html",
            {
                "request": request,
                "doc": doc,
                "active_tab": active_tab,
                "claims": claims,
                "claim_count": claim_count,
                "ocr_body": ocr_body,
                "view_mode": view_mode,
                "has_clean": has_clean,
                "has_raw": has_raw,
                **_user_ctx(user),
            },
        )

    @app.get("/document/{doc_id}/modal",
             response_class=HTMLResponse, include_in_schema=False)
    def document_modal(
        doc_id: str,
        request: Request,
        tab: str = Query("ocr"),
        view: str = Query("clean"),
        user: UserContext = Depends(require_user_scoped),
    ):
        if not _DOC_ID_RE.match(doc_id):
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
        return _render_document_modal(doc_id, request, user, tab=tab, view=view)

    # ------------------------------------------------------------------
    # GET /document/{doc_id}/notes-panel  — HTMX partial: archivist notes
    # POST /document/{doc_id}/notes       — save/update a document note
    # ------------------------------------------------------------------
    @app.get("/document/{doc_id}/notes-panel")
    async def get_notes_panel(
        doc_id: str,
        request: Request,
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> HTMLResponse:
        if not _DOC_ID_RE.match(doc_id):
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
        store = state.get("annotation_store")
        if store is None:
            return _templates.TemplateResponse(
                "fragments/notes_panel_unavailable.html", {"request": request, "doc_id": doc_id}
            )
        current = store.get_current_note(doc_id)
        history = store.get_note_history(doc_id, limit=5)
        older = [h for h in history if not h.get("is_current")]
        return _templates.TemplateResponse(
            "fragments/notes_panel.html",
            {"request": request, "doc_id": doc_id, "current": current, "older": older},
        )

    @app.post("/document/{doc_id}/notes")
    async def post_document_note(
        doc_id: str,
        request: Request,
        note_text: str = Form(...),
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> HTMLResponse:
        if not _DOC_ID_RE.match(doc_id):
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
        stripped = note_text.strip()
        if not stripped:
            return _templates.TemplateResponse(
                "fragments/notes_panel_error.html",
                {"request": request, "doc_id": doc_id, "msg": "Note cannot be empty."},
            )
        if len(stripped) > 2000:
            return _templates.TemplateResponse(
                "fragments/notes_panel_error.html",
                {"request": request, "doc_id": doc_id, "msg": "Note exceeds 2000 characters."},
            )
        store = state.get("annotation_store")
        if store is None:
            raise HTTPException(status_code=503, detail="Annotation store not configured.")
        store.upsert_note(doc_id, stripped, created_by=user.identity)
        current = store.get_current_note(doc_id)
        history = store.get_note_history(doc_id, limit=5)
        older = [h for h in history if not h.get("is_current")]
        return _templates.TemplateResponse(
            "fragments/notes_panel.html",
            {"request": request, "doc_id": doc_id, "current": current, "older": older},
        )

    # ------------------------------------------------------------------
    # GET /history             — query history & saved searches page
    # POST /history/saved      — save a search
    # POST /history/saved/{id}/delete  — delete a saved search
    # GET /api/history         — JSON history list
    # GET /api/history/saved   — JSON saved searches
    # ------------------------------------------------------------------
    from gemynd.retrieval.conversation_log import QueryHistoryStore as _QHS

    @app.get("/history", response_class=HTMLResponse, include_in_schema=False)
    def query_history_page(
        request: Request,
        q: str = Query(""),
        bucket: str = Query(""),
        limit: int = Query(50, le=200),
        offset: int = Query(0, ge=0),
        user: UserContext = Depends(require_user),
    ) -> HTMLResponse:
        store: _QHS | None = state.get("history_store")
        if store is None:
            return HTMLResponse(
                "<p style='font-family:sans-serif;padding:2rem'>"
                "Query history not available. Set the <code>CONV_LOG_DB</code> "
                "environment variable to enable it.</p>"
            )
        rows = store.list_queries(limit=limit, offset=offset, q=q, bucket=bucket)
        total = store.count_queries(q=q, bucket=bucket)
        saved = store.get_saved_searches(created_by=user.identity)
        ctx = _build_history_context(rows, total, saved, q, bucket, limit, offset)
        return _templates.TemplateResponse("history.html", {"request": request, "active_page": "history", **_user_ctx(user), **ctx})

    @app.post("/history/saved", include_in_schema=False)
    async def save_query(
        query_text: str = Form(...),
        label: str = Form(""),
        bucket: str = Form(""),
        year_min: str = Form(""),
        year_max: str = Form(""),
        conversation_id: str = Form(""),
        user: UserContext = Depends(require_user),
    ):
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        _yr_min: int | None = None
        _yr_max: int | None = None
        try:
            _yr_min = int(year_min) if year_min.strip() else None
        except ValueError:
            pass
        try:
            _yr_max = int(year_max) if year_max.strip() else None
        except ValueError:
            pass
        store.save_search(
            query_text=query_text.strip()[:2000],
            label=label.strip()[:200],
            bucket=bucket,
            year_min=_yr_min,
            year_max=_yr_max,
            created_by=user.identity,
            conversation_id=conversation_id.strip() or None,
        )
        return _RedirectResponse(url="/history", status_code=303)

    @app.post("/history/saved/{saved_id}/delete", include_in_schema=False)
    async def delete_saved_query(
        saved_id: str,
        user: UserContext = Depends(require_user),
    ):
        if not _SAVED_ID_RE.match(saved_id):
            raise HTTPException(status_code=400, detail="Invalid saved_id format")
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        store.delete_saved_search(saved_id, created_by=user.identity)
        return _RedirectResponse(url="/history", status_code=303)

    @app.get("/api/history")
    def api_history(
        q: str = Query(""),
        bucket: str = Query(""),
        limit: int = Query(50, le=200),
        offset: int = Query(0, ge=0),
        user: UserContext = Depends(require_user),
    ) -> dict[str, Any]:
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        rows = store.list_queries(limit=limit, offset=offset, q=q, bucket=bucket)
        total = store.count_queries(q=q, bucket=bucket)
        return {"total": total, "offset": offset, "limit": limit, "queries": rows}

    @app.get("/api/history/saved")
    def api_history_saved(
        user: UserContext = Depends(require_user),
    ) -> list[dict[str, Any]]:
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        return store.get_saved_searches(created_by=user.identity)

    # ------------------------------------------------------------------
    # Conversation cache — per-user multi-turn chat history
    # ------------------------------------------------------------------
    # `session_id` path params accept only the subset of characters allowed
    # by the QueryRequest.session_id regex (alphanumerics, underscore, hyphen,
    # max 128 chars). Anything else is 400.
    _SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,128}$')

    @app.get("/api/conversations")
    def api_list_conversations(
        limit: int = Query(50, le=200),
        offset: int = Query(0, ge=0),
        user: UserContext = Depends(require_user),
    ) -> dict[str, Any]:
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        if user.user_id is None:
            # Bearer-token API clients have no persisted history.
            return {"conversations": []}
        convs = store.list_user_sessions(user.user_id, limit=limit, offset=offset)
        return {"conversations": convs}

    @app.get("/api/conversations/{session_id}")
    def api_get_conversation(
        session_id: str,
        user: UserContext = Depends(require_user),
    ) -> dict[str, Any]:
        if not _SESSION_ID_RE.match(session_id):
            raise HTTPException(status_code=400, detail="Invalid session_id format")
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        if user.user_id is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        turns = store.get_session_turns(user.user_id, session_id)
        if not turns:
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"session_id": session_id, "turns": turns}

    @app.delete("/api/conversations/{session_id}")
    def api_delete_conversation(
        session_id: str,
        user: UserContext = Depends(require_user),
    ) -> dict[str, Any]:
        if not _SESSION_ID_RE.match(session_id):
            raise HTTPException(status_code=400, detail="Invalid session_id format")
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        if user.user_id is None:
            return {"deleted": 0}
        deleted = store.delete_session(user.user_id, session_id)
        return {"deleted": deleted}

    # ------------------------------------------------------------------
    # GET /api/token-stats  — admin: current-period token usage summary
    # ------------------------------------------------------------------
    @app.get("/api/token-stats")
    def api_token_stats(
        user: UserContext = Depends(require_admin),
    ) -> dict[str, Any]:
        from gemynd.shared.token_tracker import TokenUsageStore
        store: TokenUsageStore | None = state.get("token_store")
        if store is None:
            raise HTTPException(status_code=503, detail="Token usage tracking not configured.")
        inst = user.institution_id
        return {
            "today": store.today_summary(inst),
            "this_month": store.month_summary(inst),
            "budget_status": store.budget_status(inst),
        }

    # ------------------------------------------------------------------
    # GET /api/token-stats/history  — admin: daily aggregates for date range
    # ------------------------------------------------------------------
    @app.get("/api/token-stats/history")
    def api_token_stats_history(
        start: str = Query(default="", alias="from"),
        end: str = Query(default="", alias="to"),
        caller: str = Query(default=""),
        user: UserContext = Depends(require_admin),
    ) -> list[dict[str, Any]]:
        from gemynd.shared.token_tracker import TokenUsageStore
        store: TokenUsageStore | None = state.get("token_store")
        if store is None:
            raise HTTPException(status_code=503, detail="Token usage tracking not configured.")
        if not start:
            start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end:
            end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return store.history(start, end, caller=caller, institution_id=user.institution_id)

    # ------------------------------------------------------------------
    # DELETE /document/{doc_id}  — admin: soft-delete a document
    # ------------------------------------------------------------------
    @app.delete("/document/{doc_id}")
    def delete_document(
        doc_id: str,
        request: Request,
        confirm: bool = Query(default=False),
        user: UserContext = Depends(require_admin_scoped),
    ) -> dict[str, Any]:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Set ?confirm=true to acknowledge this soft-deletion. The document can be restored.",
            )
        executor: Neo4jQueryExecutor = state["executor"]
        deleted_at = datetime.now(timezone.utc).isoformat()
        deleted_by = user.identity
        rows = executor.run(SOFT_DELETE_DOCUMENT_QUERY, {
            "doc_id": doc_id,
            "institution_ids": user.permitted_institution_ids,
            "deleted_at": deleted_at,
            "deleted_by": deleted_by,
        })
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id!r} not found in institutions {user.permitted_institution_ids!r}",
            )
        write_audit: Any = state.get("write_audit")
        if write_audit is not None:
            write_audit.log(
                event_type="soft_delete",
                doc_id=doc_id,
                doc_title=rows[0].get("title"),
                institution_id=rows[0].get("institution_id"),
                performed_by=deleted_by,
            )
        if request.headers.get("HX-Request") == "true":
            return _render_document_modal(doc_id, request, user)
        return {"status": "deleted", "doc_id": doc_id, "deleted_at": deleted_at, "deleted_by": deleted_by}

    # ------------------------------------------------------------------
    # POST /document/{doc_id}/restore  — admin: restore a soft-deleted document
    # ------------------------------------------------------------------
    @app.post("/document/{doc_id}/restore")
    def restore_document(
        doc_id: str,
        request: Request,
        user: UserContext = Depends(require_admin_scoped),
    ) -> dict[str, Any]:
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(RESTORE_DOCUMENT_QUERY, {
            "doc_id": doc_id,
            "institution_ids": user.permitted_institution_ids,
        })
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id!r} not found in institutions {user.permitted_institution_ids!r}",
            )
        restored_at = datetime.now(timezone.utc).isoformat()
        performed_by = user.identity
        write_audit: Any = state.get("write_audit")
        if write_audit is not None:
            write_audit.log(
                event_type="restore",
                doc_id=doc_id,
                doc_title=rows[0].get("title"),
                institution_id=rows[0].get("institution_id"),
                performed_by=performed_by,
            )
        if request.headers.get("HX-Request") == "true":
            return _render_document_modal(doc_id, request, user)
        return {"status": "restored", "doc_id": doc_id, "restored_at": restored_at}

    # ------------------------------------------------------------------
    # GET /review  — admin: merged review queue (dashboard + list)
    # POST /review/proposals/{id}/accept|reject|defer — HTMX actions
    # ------------------------------------------------------------------

    _ISSUE_CLASS_LABELS: dict[str, str] = {
        "header_contamination": "Header contamination",
        "boilerplate_contamination": "Boilerplate contamination",
        "short_generic_token": "Short generic token",
        "ocr_garbage_mention": "OCR garbage mention",
        "ocr_spelling_variant": "OCR spelling variant",
        "duplicate_entity_alias": "Duplicate entity alias",
        "missing_species_focus": "Missing species link",
        "missing_event_location": "Missing location link",
        "method_overtrigger": "Method overtrigger",
        "pii_exposure": "PII exposure",
        "indigenous_sensitivity": "Indigenous cultural sensitivity",
        "living_person_reference": "Living person reference",
    }

    _SENSITIVITY_CLASSES = {"pii_exposure", "indigenous_sensitivity", "living_person_reference"}

    _QUEUE_LABELS: dict[str, str] = {
        "sensitivity": "Sensitivity",
        "ocr_entity": "OCR / Entity",
        "junk_mention": "Junk mention",
        "builder_repair": "Builder repair",
    }

    _JUNK_CLASSES = {
        "header_contamination", "boilerplate_contamination",
        "short_generic_token", "ocr_garbage_mention",
    }

    def _review_card_diagnostic_html(issue_class: str, evidence: dict, *, is_sensitivity: bool) -> str:
        import html as _html

        def trunc(value: object, limit: int = 180) -> str:
            text = str(value or "").strip()
            if len(text) > limit:
                text = text[: limit - 1].rstrip() + "\u2026"
            return _html.escape(text)

        if is_sensitivity:
            matched = evidence.get("matched_term") or evidence.get("matched_pattern")
            if matched:
                return f"Matched term: <strong>{trunc(matched, 80)}</strong>"
            redacted = evidence.get("redacted_sentence") or evidence.get("source_sentence")
            if redacted:
                return f"Evidence: <em>{trunc(redacted)}</em>"
            return ""

        sentence = evidence.get("source_sentence")
        if sentence:
            return f"Source: <em>{trunc(sentence)}</em>"
        return ""

    def _target_summary(targets: list[Any], limit: int = 8) -> dict[str, Any]:
        counts: dict[str, int] = {}
        sample = []
        for target in targets:
            kind = getattr(target, "target_kind", "")
            role = getattr(target, "target_role", "")
            key = f"{kind}:{role}" if role else kind
            counts[key] = counts.get(key, 0) + 1
            if len(sample) < limit:
                target_id = getattr(target, "target_id", "")
                sample.append({
                    "target_kind": kind,
                    "target_id": target_id,
                    "target_role": role,
                    "exists_in_snapshot": getattr(target, "exists_in_snapshot", False),
                })
        top_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4]
        return {
            "total": len(targets),
            "sample": sample,
            "remaining": max(0, len(targets) - len(sample)),
            "counts": [{"label": label, "count": count} for label, count in top_counts],
        }

    def _review_objects(
        issue_class: str,
        evidence: dict,
        patch_spec: dict,
        targets: list[Any],
        limit: int = 5,
    ) -> dict[str, Any]:
        def text(value: object) -> str:
            return str(value or "").strip()

        def truncate(value: object, limit_chars: int = 220) -> str:
            value_text = " ".join(text(value).split())
            if len(value_text) > limit_chars:
                return value_text[: limit_chars - 1].rstrip() + "\u2026"
            return value_text

        def entity_name(value: object) -> str:
            return text(value.get("name")) if isinstance(value, dict) else ""

        def add_object(
            items: list[dict[str, str]],
            *,
            kind: str,
            label: object,
            detail: object = "",
            snippet: object = "",
            meta: object = "",
        ) -> None:
            label_text = text(label)
            if not label_text:
                return
            items.append({
                "kind": text(kind),
                "label": truncate(label_text, 90),
                "detail": truncate(detail, 120),
                "snippet": truncate(snippet, 260),
                "meta": truncate(meta, 140),
            })

        items: list[dict[str, str]] = []
        total = 0

        if issue_class in _JUNK_CLASSES:
            mentions = [m for m in evidence.get("affected_mentions", []) if isinstance(m, dict)]
            total = len(mentions) or len(targets)
            for mention in mentions[:limit]:
                page_number = mention.get("page_number")
                source_file = text(mention.get("source_file"))
                page_label = f"page {page_number}" if page_number else ""
                detail = page_label or source_file
                if page_label and source_file:
                    detail = f"{page_label} / {source_file}"
                add_object(
                    items,
                    kind="Mention",
                    label=mention.get("surface_form") or mention.get("normalized_form") or mention.get("mention_id"),
                    detail=detail,
                    snippet=mention.get("paragraph_clean_text") or mention.get("paragraph_raw_ocr_text"),
                    meta=mention.get("mention_id"),
                )

        elif issue_class == "ocr_spelling_variant":
            variants = [v for v in evidence.get("merge_entities", []) if isinstance(v, dict)]
            canon = evidence.get("canonical_entity", {}) if isinstance(evidence.get("canonical_entity"), dict) else {}
            total = len(variants) or len(targets)
            canon_name = entity_name(canon) or patch_spec.get("canonical_name", "")
            for variant in variants[:limit]:
                add_object(
                    items,
                    kind="Entity",
                    label=entity_name(variant) or variant.get("entity_id"),
                    detail=f"merge into {canon_name}" if canon_name else "merge source",
                    meta=variant.get("entity_id"),
                )

        elif issue_class == "duplicate_entity_alias":
            alias = evidence.get("alias_entity", {}) if isinstance(evidence.get("alias_entity"), dict) else {}
            canon = evidence.get("canonical_entity", {}) if isinstance(evidence.get("canonical_entity"), dict) else {}
            total = 1 if alias or canon else len(targets)
            add_object(
                items,
                kind="Alias",
                label=entity_name(alias) or patch_spec.get("alias_entity_id"),
                detail=f"resolve to {entity_name(canon) or patch_spec.get('canonical_name', '')}",
                meta=alias.get("entity_id"),
            )

        elif issue_class in ("missing_species_focus", "missing_event_location"):
            candidate_key = "candidate_species" if issue_class == "missing_species_focus" else "candidate_location"
            candidate = evidence.get(candidate_key, {}) if isinstance(evidence.get(candidate_key), dict) else {}
            relation = "SPECIES_FOCUS" if issue_class == "missing_species_focus" else "OCCURRED_AT"
            total = 1 if candidate or evidence.get("source_sentence") else len(targets)
            add_object(
                items,
                kind="Claim link",
                label=entity_name(candidate) or candidate.get("entity_id") or relation,
                detail=f"proposed {relation}",
                snippet=evidence.get("source_sentence"),
                meta=evidence.get("claim_id") or candidate.get("entity_type"),
            )

        elif issue_class == "method_overtrigger":
            method = evidence.get("method_entity", {}) if isinstance(evidence.get("method_entity"), dict) else {}
            total = 1 if method or evidence.get("source_sentence") else len(targets)
            add_object(
                items,
                kind="Claim link",
                label=entity_name(method) or method.get("entity_id") or "Method focus",
                detail="overtriggered METHOD_FOCUS",
                snippet=evidence.get("source_sentence"),
                meta=evidence.get("claim_id") or evidence.get("compatibility"),
            )

        elif issue_class in _SENSITIVITY_CLASSES:
            names = evidence.get("person_names", [])
            if isinstance(names, list) and names:
                label = ", ".join(text(name) for name in names[:3] if text(name))
            else:
                label = evidence.get("matched_term") or evidence.get("matched_pattern") or issue_class
            total = 1 if label or evidence.get("source_sentence") or evidence.get("redacted_sentence") else len(targets)
            add_object(
                items,
                kind="Sensitive claim",
                label=label,
                detail=evidence.get("matched_pattern") or issue_class,
                snippet=evidence.get("redacted_sentence") or evidence.get("source_sentence"),
                meta=evidence.get("claim_id") or evidence.get("most_recent_year"),
            )

        if not items:
            total = len(targets)
            for target in targets[:limit]:
                kind = text(getattr(target, "target_kind", "Target"))
                role = text(getattr(target, "target_role", ""))
                target_id = text(getattr(target, "target_id", ""))
                add_object(
                    items,
                    kind=kind or "Target",
                    label=target_id,
                    detail=role,
                    meta="in snapshot" if getattr(target, "exists_in_snapshot", False) else "not in snapshot",
                )

        total = total or len(items)
        return {
            "items": items,
            "total": total,
            "remaining": max(0, total - len(items)),
        }

    def _build_source_comparison_context(
        issue_class: str, evidence: dict,
    ) -> dict | None:
        """Build source comparison context for the detail template.

        Returns *None* when the evidence lacks the enriched fields needed to
        render a meaningful comparison.
        """
        import html as _html
        from gemynd.review.web.diff_utils import char_diff_html, highlight_in_text

        def esc(v: object) -> str:
            return _html.escape(str(v))

        # -- OCR spelling variant / duplicate alias: side-by-side char diff --
        if issue_class == "ocr_spelling_variant":
            canon = evidence.get("canonical_entity", {})
            variants = evidence.get("merge_entities", [])
            if not canon or not variants:
                return None
            # Build diff for each variant vs canonical
            left_parts: list[str] = []
            for v in variants[:5]:
                v_html, c_html = char_diff_html(v.get("name", ""), canon.get("name", ""))
                count = evidence.get("merge_mention_counts", {}).get(v.get("entity_id", ""), 0)
                left_parts.append(
                    f'<span class="source-entity-name">{v_html}</span>'
                    f' <span style="font-size:11px;color:var(--text-secondary)">({count} mentions)</span>'
                )
            canon_count = evidence.get("canonical_mention_count", 0)
            _, canon_html = char_diff_html(
                variants[0].get("name", ""), canon.get("name", ""),
            )
            ctx_paras = []
            for cp in evidence.get("context_paragraphs", [])[:3]:
                para_text = cp.get("clean_text") or cp.get("raw_ocr_text", "")
                # Highlight all mention surface forms
                marked = esc(para_text)
                for sf in cp.get("mention_surface_forms", []):
                    marked = highlight_in_text(para_text, sf)
                    para_text = para_text  # re-highlight each time from raw
                ctx_paras.append({"label": f"Paragraph {cp.get('paragraph_id', '')[:12]}…", "html": marked})

            return {
                "comparison_type": "side_by_side",
                "left_label": "Variant(s)",
                "left_html": "<br>".join(left_parts),
                "right_label": "Canonical",
                "right_html": (
                    f'<span class="source-entity-name">{canon_html}</span>'
                    f' <span style="font-size:11px;color:var(--text-secondary)">({canon_count} mentions)</span>'
                ),
                "context_paragraphs": ctx_paras,
                "badges": [],
            }

        if issue_class == "duplicate_entity_alias":
            canon = evidence.get("canonical_entity", {})
            alias = evidence.get("alias_entity", {})
            if not canon or not alias:
                return None
            a_html, c_html = char_diff_html(alias.get("name", ""), canon.get("name", ""))
            ctx_paras = []
            for cp in evidence.get("context_paragraphs", [])[:3]:
                para_text = cp.get("clean_text") or cp.get("raw_ocr_text", "")
                marked = para_text
                for sf in cp.get("mention_surface_forms", []):
                    marked = highlight_in_text(para_text, sf)
                ctx_paras.append({"label": f"Paragraph {cp.get('paragraph_id', '')[:12]}…", "html": marked})
            return {
                "comparison_type": "side_by_side",
                "left_label": "Alias",
                "left_html": (
                    f'<span class="source-entity-name">{a_html}</span>'
                    f' <span style="font-size:11px;color:var(--text-secondary)">'
                    f'({evidence.get("alias_mention_count", 0)} mentions)</span>'
                ),
                "right_label": "Canonical",
                "right_html": (
                    f'<span class="source-entity-name">{c_html}</span>'
                    f' <span style="font-size:11px;color:var(--text-secondary)">'
                    f'({evidence.get("canonical_mention_count", 0)} mentions)</span>'
                ),
                "context_paragraphs": ctx_paras,
                "badges": [],
            }

        # -- Junk mention classes: show paragraph with mention highlighted --
        if issue_class in _JUNK_CLASSES:
            mentions = evidence.get("affected_mentions", [])
            if not mentions:
                return None
            # Pick the first mention that has paragraph text
            m = next(
                (m for m in mentions if m.get("paragraph_clean_text") or m.get("paragraph_raw_ocr_text")),
                mentions[0],
            )
            para_text = m.get("paragraph_clean_text") or m.get("paragraph_raw_ocr_text") or ""
            if not para_text:
                return None
            sf = m.get("surface_form") or m.get("normalized_form") or ""
            marked = highlight_in_text(para_text, sf)
            label_map = {
                "header_contamination": "Header / footer region",
                "boilerplate_contamination": "Boilerplate text",
                "short_generic_token": "Short generic token",
                "ocr_garbage_mention": "OCR garbage",
            }
            return {
                "comparison_type": "single_context",
                "left_label": "Paragraph context",
                "left_html": f'<pre>{marked}</pre>',
                "right_label": "",
                "right_html": "",
                "context_paragraphs": [],
                "badges": [{"cls": "badge-amber", "text": label_map.get(issue_class, issue_class)}],
            }

        # -- Builder repair: source sentence + proposed link --
        if issue_class in ("missing_species_focus", "missing_event_location"):
            sentence = evidence.get("source_sentence", "")
            candidate = evidence.get("candidate_species") or evidence.get("candidate_location") or {}
            if not sentence:
                return None
            entity_name = candidate.get("name", "")
            relation = "SPECIES_FOCUS" if issue_class == "missing_species_focus" else "OCCURRED_AT"
            left_html = highlight_in_text(sentence, entity_name) if entity_name else esc(sentence)
            para_text = evidence.get("paragraph_clean_text") or evidence.get("paragraph_raw_ocr_text") or ""
            ctx_paras = []
            if para_text:
                ctx_paras.append({
                    "label": "Full paragraph",
                    "html": highlight_in_text(para_text, entity_name) if entity_name else esc(para_text),
                })
            return {
                "comparison_type": "side_by_side",
                "left_label": "Source sentence",
                "left_html": f'<em>{left_html}</em>',
                "right_label": "Proposed link",
                "right_html": (
                    f'<span class="badge badge-blue">{esc(relation)}</span> '
                    f'<strong>{esc(entity_name)}</strong>'
                    f'<br><span style="font-size:11px;color:var(--text-secondary)">'
                    f'{esc(candidate.get("entity_type", ""))}</span>'
                ),
                "context_paragraphs": ctx_paras,
                "badges": [],
            }

        if issue_class == "method_overtrigger":
            sentence = evidence.get("source_sentence", "")
            method = evidence.get("method_entity", {})
            if not sentence:
                return None
            method_name = method.get("name", "")
            left_html = highlight_in_text(sentence, method_name) if method_name else esc(sentence)
            return {
                "comparison_type": "side_by_side",
                "left_label": "Source sentence",
                "left_html": f'<em>{left_html}</em>',
                "right_label": "Overtriggered link",
                "right_html": (
                    f'<span class="badge badge-red">METHOD_FOCUS</span> '
                    f'<strong>{esc(method_name)}</strong>'
                    f'<br><span style="font-size:11px;color:var(--text-secondary)">'
                    f'Compatibility: {esc(evidence.get("compatibility", ""))}</span>'
                ),
                "context_paragraphs": [],
                "badges": [],
            }

        # -- Sensitivity: PII --
        if issue_class == "pii_exposure":
            redacted = evidence.get("redacted_sentence", "")
            if not redacted:
                return None
            pattern = evidence.get("matched_pattern", "")
            return {
                "comparison_type": "single_context",
                "left_label": "Redacted sentence",
                "left_html": f'<pre>{esc(redacted)}</pre>',
                "right_label": "",
                "right_html": "",
                "context_paragraphs": [],
                "badges": [{"cls": "badge-red", "text": f"PII: {pattern}"}],
            }

        # -- Sensitivity: Indigenous --
        if issue_class == "indigenous_sensitivity":
            sentence = evidence.get("source_sentence", "")
            term = evidence.get("matched_term", "")
            if not sentence and not term:
                return None
            marked = highlight_in_text(sentence, term) if sentence and term else esc(sentence or term)
            nations = evidence.get("nations", [])
            badges = [{"cls": "badge-amber", "text": f"Sensitivity: {evidence.get('sensitivity', '')}"}]
            for n in nations[:5]:
                badges.append({"cls": "badge-outline", "text": n})
            return {
                "comparison_type": "single_context",
                "left_label": "Source sentence",
                "left_html": f'<em>{marked}</em>',
                "right_label": "",
                "right_html": "",
                "context_paragraphs": [],
                "badges": badges,
            }

        # -- Sensitivity: Living person --
        if issue_class == "living_person_reference":
            sentence = evidence.get("source_sentence", "")
            persons = evidence.get("person_names", [])
            if not sentence:
                return None
            marked = esc(sentence)
            for name in persons:
                marked = highlight_in_text(sentence, name)
                sentence = sentence  # keep re-highlighting from raw
            year = evidence.get("most_recent_year", "")
            badges = [{"cls": "badge-red", "text": f"Most recent year: {year}"}]
            for name in persons[:3]:
                badges.append({"cls": "badge-outline", "text": name})
            return {
                "comparison_type": "single_context",
                "left_label": "Source sentence",
                "left_html": f'<em>{marked}</em>',
                "right_label": "",
                "right_html": "",
                "context_paragraphs": [],
                "badges": badges,
            }

        return None

    @app.get("/review", response_class=HTMLResponse, include_in_schema=False)
    def review_page(
        request: Request,
        queue_name: str | None = Query(None),
        user: UserContext = Depends(require_admin),
    ):
        import html as _html
        import json as _json
        from datetime import date as _date

        review_store = state.get("review_store")
        if review_store is None:
            raise HTTPException(status_code=503, detail="Review store not configured.")

        # Load proposals
        proposals = review_store.list_proposals(
            status="queued", queue_name=queue_name, limit=50
        )

        # Build card data for template
        proposal_cards = []
        for p in proposals:
            targets = review_store.get_proposal_targets(p.proposal_id)
            target_summary = _target_summary(targets, limit=3)
            rev = review_store.get_latest_revision(p.proposal_id)
            evidence: dict = {}
            patch_spec: dict = {}
            if rev:
                try:
                    evidence = _json.loads(rev.evidence_snapshot_json)
                except Exception:
                    pass
                try:
                    patch_spec = _json.loads(rev.patch_spec_json)
                except Exception:
                    pass

            is_sensitivity = p.issue_class in _SENSITIVITY_CLASSES
            ic_label = _ISSUE_CLASS_LABELS.get(p.issue_class, p.issue_class)
            review_objects = _review_objects(
                p.issue_class,
                evidence,
                patch_spec,
                targets,
                limit=3,
            )

            # Build description from evidence using key_info_html from review app
            try:
                from gemynd.review.web.app import _key_info_html
                desc = _key_info_html(p.issue_class, evidence)
            except Exception:
                desc = _html.escape(p.issue_class)
            diagnostic = ""
            if is_sensitivity or p.priority_score >= 0.80:
                diagnostic = _review_card_diagnostic_html(
                    p.issue_class,
                    evidence,
                    is_sensitivity=is_sensitivity,
                )

            proposal_cards.append({
                "proposal_id": p.proposal_id,
                "ic_label": ic_label,
                "is_sensitivity": is_sensitivity,
                "badge_cls": "badge-red" if is_sensitivity else "badge-blue",
                "border_color": "var(--accent-red-border)" if is_sensitivity else "var(--border-medium)",
                "priority": p.priority_score,
                "confidence": p.confidence,
                "impact_size": p.impact_size,
                "description": desc,
                "diagnostic": diagnostic,
                "target_summary": target_summary,
                "review_objects": review_objects,
                "needs_bulk_confirm": is_sensitivity or p.impact_size >= 100,
                "actionable": p.status in ("queued", "deferred"),
                "can_reject": p.status in ("queued", "deferred"),
                "can_defer": p.status == "queued",
            })

        # Queue filter counts
        queue_counts = review_store.proposal_counts_by_queue()
        # Filter to queued status only
        queued_queue_counts = {}
        try:
            queued_proposals_all = review_store.list_proposals(status="queued", limit=10000)
            # Count by queue manually isn't efficient; use the store counts
            # The proposal_counts_by_queue doesn't filter by status, so let's use total queued
            ic_counts = review_store.proposal_counts_by_issue_class(status="queued")
            total_queued = sum(v["count"] for v in ic_counts.values())
        except Exception:
            ic_counts = {}
            total_queued = len(proposals)

        # Sensitivity count
        sensitivity_count = sum(
            v["count"] for k, v in ic_counts.items() if k in _SENSITIVITY_CLASSES
        )

        # Build queue filter pills
        queue_filters = []
        for qname, qlabel in _QUEUE_LABELS.items():
            if qname == "sensitivity":
                continue
            qcount = queue_counts.get(qname, 0)
            if qcount > 0:
                queue_filters.append((qname, qlabel, qcount))

        # Detector count (unique queue names with proposals)
        detector_count = len([q for q in queue_counts.values() if q > 0])

        # Today's activity
        today_iso = _date.today().isoformat()
        event_counts = review_store.correction_event_counts(since_iso=today_iso)
        daily_counts = review_store.correction_event_daily_counts(days=14)
        max_daily_total = max((int(d["total"]) for d in daily_counts), default=0)
        review_trend = []
        for day in daily_counts:
            total = int(day["total"])
            review_trend.append({
                "day": day["day"],
                "total": total,
                "height_pct": round(total / max_daily_total * 100) if max_daily_total else 0,
            })
        new_queued_today = review_store.proposal_count_since(today_iso, status="queued")

        return _templates.TemplateResponse("admin_review.html", {
            "request": request,
            "active_page": "review",
            **_user_ctx(user),
            "total_queued": total_queued,
            "detector_count": detector_count,
            "sensitivity_count": sensitivity_count,
            "queue_filters": queue_filters,
            "active_filter": queue_name,
            "proposal_cards": proposal_cards,
            "accepted_today": event_counts.get("accept", 0),
            "rejected_today": event_counts.get("reject", 0),
            "deferred_today": event_counts.get("defer", 0),
            "review_trend": review_trend,
            "new_queued_today": new_queued_today,
        })

    def _review_action_response(request: Request, action: str) -> HTMLResponse:
        """Return empty for queue-page card deletion, or a confirmation for detail page."""
        if request.headers.get("HX-Target", "").startswith("proposal-card-"):
            return HTMLResponse("")  # hx-swap="delete" removes the card
        return HTMLResponse(
            f'<span class="badge badge-green" style="font-size:12px">{action}</span>'
        )

    @app.get("/review/error-types-options", response_class=HTMLResponse, include_in_schema=False)
    def review_error_types_options(
        error_root_cause: str = Query(""),
        user: UserContext = Depends(require_admin),
    ):
        """HTMX endpoint: return <option> elements for the error_type dropdown."""
        import html as _html
        from gemynd.review.models import ERROR_TYPES, ERROR_TYPE_LABELS
        types = sorted(ERROR_TYPES.get(error_root_cause, set()))
        if not types:
            return HTMLResponse('<option value="">-- select root cause first --</option>')
        options = ['<option value="">-- select error type --</option>']
        for t in types:
            label = _html.escape(ERROR_TYPE_LABELS.get(t, t))
            options.append(f'<option value="{_html.escape(t)}">{label}</option>')
        return HTMLResponse("\n".join(options))

    @app.post("/review/proposals/{proposal_id}/accept", include_in_schema=False)
    def review_accept(
        proposal_id: str,
        request: Request,
        user: UserContext = Depends(require_admin),
        error_root_cause: str = Form(""),
        error_type: str = Form(""),
        reviewer_note: str = Form(""),
    ):
        review_store = state.get("review_store")
        if review_store is None:
            raise HTTPException(status_code=503, detail="Review store not configured.")
        from gemynd.review.actions import accept_proposal, ReviewActionError
        try:
            accept_proposal(
                review_store, proposal_id, reviewer=user.identity,
                reviewer_note=reviewer_note,
                error_root_cause=error_root_cause,
                error_type=error_type,
            )
        except ReviewActionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _review_action_response(request, "Accepted")

    @app.post("/review/proposals/{proposal_id}/reject", include_in_schema=False)
    def review_reject(
        proposal_id: str,
        request: Request,
        user: UserContext = Depends(require_admin),
        error_root_cause: str = Form(""),
        error_type: str = Form(""),
        reviewer_note: str = Form(""),
    ):
        review_store = state.get("review_store")
        if review_store is None:
            raise HTTPException(status_code=503, detail="Review store not configured.")
        from gemynd.review.actions import reject_proposal, ReviewActionError
        try:
            reject_proposal(
                review_store, proposal_id, reviewer=user.identity,
                reviewer_note=reviewer_note,
                error_root_cause=error_root_cause,
                error_type=error_type,
            )
        except ReviewActionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _review_action_response(request, "Rejected")

    @app.post("/review/proposals/{proposal_id}/defer", include_in_schema=False)
    def review_defer(
        proposal_id: str,
        request: Request,
        user: UserContext = Depends(require_admin),
        reviewer_note: str = Form(""),
    ):
        review_store = state.get("review_store")
        if review_store is None:
            raise HTTPException(status_code=503, detail="Review store not configured.")
        from gemynd.review.actions import defer_proposal, ReviewActionError
        try:
            defer_proposal(
                review_store, proposal_id, reviewer=user.identity,
                reviewer_note=reviewer_note,
            )
        except ReviewActionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _review_action_response(request, "Deferred")

    @app.get("/api/training-export", include_in_schema=False)
    def training_export(
        status: str | None = Query(None),
        snapshot_id: str | None = Query(None),
        classified_only: bool = Query(False),
        user: UserContext = Depends(require_admin),
    ):
        """Export correction events with error classifications for model training."""
        from fastapi.responses import JSONResponse as _JSONResponse
        review_store = state.get("review_store")
        if review_store is None:
            raise HTTPException(status_code=503, detail="Review store not configured.")
        data = review_store.export_training_data(
            status=status,
            snapshot_id=snapshot_id,
            with_classification_only=classified_only,
        )
        return _JSONResponse(content=data)

    @app.get("/review/proposals/{proposal_id}", response_class=HTMLResponse, include_in_schema=False)
    def review_detail(
        proposal_id: str,
        request: Request,
        user: UserContext = Depends(require_admin),
    ):
        import json as _json

        review_store = state.get("review_store")
        if review_store is None:
            raise HTTPException(status_code=503, detail="Review store not configured.")

        proposal = review_store.get_proposal(proposal_id)
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found.")

        targets = review_store.get_proposal_targets(proposal_id)
        target_summary = _target_summary(targets, limit=25)
        revisions = review_store.get_revisions(proposal_id)
        events = review_store.get_correction_events(proposal_id)

        latest_rev = revisions[-1] if revisions else None
        patch_json = latest_rev.patch_spec_json if latest_rev else "{}"
        evidence_json = latest_rev.evidence_snapshot_json if latest_rev else "{}"

        patch_dict: dict = {}
        try:
            patch_dict = _json.loads(patch_json)
            patch_formatted = _json.dumps(patch_dict, indent=2)
        except Exception:
            patch_formatted = patch_json
        evidence_dict: dict = {}
        try:
            evidence_dict = _json.loads(evidence_json)
            evidence_formatted = _json.dumps(evidence_dict, indent=2)
        except Exception:
            evidence_formatted = evidence_json

        # Key info description
        try:
            from gemynd.review.web.app import _key_info_html
            description = _key_info_html(proposal.issue_class, evidence_dict)
        except Exception:
            description = ""

        ic_label = _ISSUE_CLASS_LABELS.get(proposal.issue_class, proposal.issue_class)

        # Source comparison context (gracefully None for old evidence)
        try:
            source_comparison = _build_source_comparison_context(
                proposal.issue_class, evidence_dict,
            )
        except Exception:
            source_comparison = None

        review_objects = _review_objects(
            proposal.issue_class,
            evidence_dict,
            patch_dict,
            targets,
            limit=12,
        )

        # Error classification defaults for the form
        from gemynd.review.models import (
            ERROR_ROOT_CAUSE_LABELS,
            ERROR_ROOT_CAUSES,
            ERROR_TYPE_LABELS,
            ERROR_TYPES,
            ISSUE_CLASS_DEFAULT_ROOT_CAUSE,
        )
        default_root_cause = ISSUE_CLASS_DEFAULT_ROOT_CAUSE.get(proposal.issue_class, "")

        return _templates.TemplateResponse("admin_review_detail.html", {
            "request": request,
            "active_page": "review",
            **_user_ctx(user),
            "proposal_id": proposal.proposal_id,
            "issue_class": proposal.issue_class,
            "issue_class_label": ic_label,
            "proposal_type": proposal.proposal_type,
            "status": proposal.status,
            "confidence": f"{proposal.confidence:.2f}",
            "priority_score": f"{proposal.priority_score:.2f}",
            "impact_size": proposal.impact_size,
            "description": description,
            "is_active": proposal.status in ("queued", "deferred"),
            "is_sensitivity": proposal.issue_class in _SENSITIVITY_CLASSES,
            "needs_bulk_confirm": proposal.issue_class in _SENSITIVITY_CLASSES or proposal.impact_size >= 100,
            "source_comparison": source_comparison,
            "review_objects": review_objects,
            "targets": target_summary["sample"],
            "target_summary": target_summary,
            "revisions": revisions,
            "events": events,
            "patch_json": patch_formatted,
            "evidence_json": evidence_formatted,
            # Error classification
            "error_root_causes": sorted(ERROR_ROOT_CAUSES),
            "error_root_cause_labels": ERROR_ROOT_CAUSE_LABELS,
            "error_types": {k: sorted(v) for k, v in ERROR_TYPES.items()},
            "error_type_labels": ERROR_TYPE_LABELS,
            "default_root_cause": default_root_cause,
            "default_error_types": sorted(ERROR_TYPES.get(default_root_cause, set())),
        })

    # ------------------------------------------------------------------
    # GET /ingest — admin: ingest page (upload + status)
    # POST /ingest — admin: upload files for processing
    # GET /ingest/{job_id}/status — HTMX polling fragment
    # ------------------------------------------------------------------
    @app.get("/ingest", response_class=HTMLResponse, include_in_schema=False)
    def ingest_page(
        request: Request,
        job_id: str | None = Query(None),
        user: UserContext = Depends(require_admin),
    ):
        ingest_store = state.get("ingest_store")
        ctx: dict[str, Any] = {
            "request": request,
            "active_page": "ingest",
            **_user_ctx(user),
            "active_job": False,
        }
        if ingest_store and job_id:
            job = ingest_store.get_job(job_id)
            if job:
                docs = ingest_store.list_documents(job_id)
                total = job["total_docs"]
                done = job["completed_docs"] + job["failed_docs"]
                pct = int(done / total * 100) if total else 0
                ctx.update({
                    "active_job": True,
                    "job_id": job_id,
                    "job": job,
                    "docs": docs,
                    "total": total,
                    "done": done,
                    "failed": job["failed_docs"],
                    "pct": pct,
                    "is_running": job["status"] == "running",
                    "total_claims": sum((d.get("claims_count") or 0) for d in docs),
                    "total_mentions": sum((d.get("mention_count") or 0) for d in docs),
                })
        return _templates.TemplateResponse("admin_ingest.html", ctx)

    from fastapi import UploadFile as _UploadFile, BackgroundTasks as _BackgroundTasks

    @app.post("/ingest", include_in_schema=False)
    async def create_ingest_job(
        files: list[_UploadFile],
        background_tasks: _BackgroundTasks,
        user: UserContext = Depends(require_admin),
    ):
        ingest_store = state.get("ingest_store")
        if ingest_store is None:
            raise HTTPException(status_code=503, detail="Ingest store not configured.")

        if not files:
            raise HTTPException(status_code=422, detail="No files uploaded.")

        allowed = {".pdf", ".json"}
        bad = [f.filename for f in files if Path(f.filename or "").suffix.lower() not in allowed]
        if bad:
            raise HTTPException(status_code=422, detail=f"Unsupported file type(s): {', '.join(bad)}")

        out_dir_path = Path(os.environ.get("INGEST_OUT_DIR", "out"))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        staging_dir = out_dir_path / ".ingest_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        staged: list[tuple[Path, str]] = []
        for upload in files:
            orig_name = upload.filename or "unknown"
            dest = staging_dir / orig_name
            dest.write_bytes(await upload.read())
            staged.append((dest, orig_name))

        filenames = [orig for _, orig in staged]
        job_id = ingest_store.create_job(str(out_dir_path), filenames)

        import asyncio as _asyncio

        async def _run_ingest_job(
            jid: str,
            staged_files: list[tuple[Path, str]],
            sdir: Path,
        ) -> None:
            from gemynd.ingest.pdf_converter import ConversionError, convert_pdf_to_json
            from gemynd.ingest.pipeline import _process_single_document
            review_out = os.environ.get("REVIEW_OUT_DIR")
            for file_path, orig in staged_files:
                try:
                    if file_path.suffix.lower() == ".pdf":
                        ingest_store.update_document_status(jid, orig, "converting")
                        json_path = await _asyncio.to_thread(convert_pdf_to_json, file_path, sdir)
                    else:
                        json_path = file_path
                    ingest_store.update_document_status(jid, orig, "extracting")
                    result = await _asyncio.to_thread(
                        _process_single_document, str(json_path), str(out_dir_path), review_out
                    )
                    quality = result.get("quality", {})
                    ingest_store.update_document_status(
                        jid, orig, "completed",
                        output_dir=result.get("structure_output"),
                        claims_count=quality.get("claim_count"),
                        mention_count=quality.get("mention_count"),
                    )
                except Exception as exc:
                    _log.error("Ingest failed for %r: %s", orig, exc, exc_info=True)
                    ingest_store.update_document_status(jid, orig, "failed", error_message=str(exc))

        background_tasks.add_task(_run_ingest_job, job_id, staged, staging_dir)
        return _RedirectResponse(url=f"/ingest?job_id={job_id}", status_code=303)

    @app.get("/ingest/{job_id}/status", response_class=HTMLResponse, include_in_schema=False)
    def ingest_status_fragment(
        job_id: str,
        request: Request,
        _user: UserContext = Depends(require_admin),
    ):
        ingest_store = state.get("ingest_store")
        if ingest_store is None:
            raise HTTPException(status_code=503, detail="Ingest store not configured.")
        job = ingest_store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        docs = ingest_store.list_documents(job_id)
        total = job["total_docs"]
        done = job["completed_docs"] + job["failed_docs"]
        pct = int(done / total * 100) if total else 0
        return _templates.TemplateResponse("fragments/ingest_status_fragment.html", {
            "request": request,
            "job_id": job_id,
            "is_running": job["status"] == "running",
            "docs": docs,
            "total": total,
            "done": done,
            "failed": job["failed_docs"],
            "pct": pct,
            "total_claims": sum((d.get("claims_count") or 0) for d in docs),
            "total_mentions": sum((d.get("mention_count") or 0) for d in docs),
        })

    # ------------------------------------------------------------------
    # GET /users — admin: user management page
    # POST /users — admin: create a new user
    # POST /users/{user_id}/deactivate — admin: deactivate user
    # POST /users/{user_id}/activate   — admin: reactivate user
    # ------------------------------------------------------------------
    @app.get("/users", response_class=HTMLResponse, include_in_schema=False)
    def users_page(request: Request, user: UserContext = Depends(require_admin)):
        user_store = state.get("user_store")
        if user_store is None:
            raise HTTPException(status_code=503, detail="User store not configured.")
        users = user_store.list_users()
        return _templates.TemplateResponse("admin_users.html", {
            "request": request,
            "active_page": "users",
            **_user_ctx(user),
            "users": users,
        })

    @app.post("/users", response_class=HTMLResponse, include_in_schema=False)
    def create_user_page(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        role: str = Form("readonly"),
        user: UserContext = Depends(require_admin),
    ):
        user_store = state.get("user_store")
        if user_store is None:
            raise HTTPException(status_code=503, detail="User store not configured.")
        try:
            new_user = user_store.create_user(email.strip().lower(), password, role)
        except ValueError:
            raise HTTPException(status_code=400, detail="Could not create user. Check email and role.")
        if request.headers.get("HX-Request") == "true":
            return _templates.TemplateResponse("fragments/user_row.html", {"request": request, "user": new_user})
        return _RedirectResponse(url="/users", status_code=303)

    @app.post("/users/{user_id}/deactivate", response_class=HTMLResponse, include_in_schema=False)
    def deactivate_user_page(
        user_id: str,
        request: Request,
        admin: UserContext = Depends(require_admin),
    ):
        user_store = state.get("user_store")
        if user_store is None:
            raise HTTPException(status_code=503, detail="User store not configured.")
        if admin.user_id and admin.user_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account.")
        if not user_store.deactivate_user(user_id):
            raise HTTPException(status_code=404, detail="User not found.")
        updated = user_store.get_by_id(user_id)
        return _templates.TemplateResponse("fragments/user_row.html", {"request": request, "user": updated})

    @app.post("/users/{user_id}/activate", response_class=HTMLResponse, include_in_schema=False)
    def activate_user_page(
        user_id: str,
        request: Request,
        _admin: UserContext = Depends(require_admin),
    ):
        user_store = state.get("user_store")
        if user_store is None:
            raise HTTPException(status_code=503, detail="User store not configured.")
        if not user_store.activate_user(user_id):
            raise HTTPException(status_code=404, detail="User not found.")
        updated = user_store.get_by_id(user_id)
        return _templates.TemplateResponse("fragments/user_row.html", {"request": request, "user": updated})

    # ------------------------------------------------------------------
    # /v1/ API versioning
    # Register canonical versioned paths for all JSON API routes.  The
    # unversioned paths above remain active (include_in_schema=False) so
    # the browser UI and existing integrations continue to work unchanged.
    # ------------------------------------------------------------------
    from fastapi import APIRouter as _APIRouter
    v1 = _APIRouter(prefix="/v1")
    v1.add_api_route("/query", query, methods=["POST"], response_model=QueryResponse)
    v1.add_api_route("/query/provenance", query_provenance, methods=["POST"])
    v1.add_api_route("/api/entities/search", entity_search, methods=["GET"])
    v1.add_api_route("/api/entity/{entity_id}/neighborhood", entity_neighborhood, methods=["GET"])
    v1.add_api_route("/documents", _list_documents_json, methods=["GET"])
    v1.add_api_route("/document/{doc_id}", delete_document, methods=["DELETE"])
    v1.add_api_route("/document/{doc_id}/restore", restore_document, methods=["POST"])
    v1.add_api_route("/document/{doc_id}/notes-panel", get_notes_panel, methods=["GET"])
    v1.add_api_route("/document/{doc_id}/notes", post_document_note, methods=["POST"])
    v1.add_api_route("/api/history", api_history, methods=["GET"])
    v1.add_api_route("/api/history/saved", api_history_saved, methods=["GET"])
    v1.add_api_route("/api/token-stats", api_token_stats, methods=["GET"])
    v1.add_api_route("/api/token-stats/history", api_token_stats_history, methods=["GET"])
    app.include_router(v1)

    # Mark the original unversioned paths as deprecated in the schema.
    for _route in app.routes:
        if hasattr(_route, "path") and hasattr(_route, "include_in_schema"):
            if getattr(_route, "path", None) in {
                "/query", "/query/provenance", "/api/entities/search",
                "/api/entity/{entity_id}/neighborhood", "/documents",
                "/document/{doc_id}", "/document/{doc_id}/restore",
            } and not getattr(_route, "deprecated", False):
                _route.deprecated = True  # type: ignore[attr-defined]
                _route.include_in_schema = False  # type: ignore[attr-defined]

    return app
