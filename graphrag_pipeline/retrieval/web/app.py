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
    graphrag query-serve --port 8788
"""
from __future__ import annotations

import dataclasses
import hashlib as _hashlib
import logging
import os
import re
import sqlite3 as _sqlite3
import statistics as _statistics
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
from graphrag_pipeline.auth.dependencies import NeedsLoginException, require_admin, require_archivist_or_admin
from graphrag_pipeline.auth.router import create_auth_router
from ..classifier import classify_query
from ..context_assembler import ProvenanceContextAssembler, _serialise_block
from ..conversation_log import ClaimInteraction, ConversationLogger, LogRecord, make_conversation_id
from graphrag_pipeline.core.graph.cypher import (
    CORPUS_STATS_QUERY,
    COUNT_QUARANTINED_CLAIMS_QUERY,
    SOFT_DELETE_DOCUMENT_QUERY,
    RESTORE_DOCUMENT_QUERY,
    LIST_DOCUMENTS_QUERY,
    STATS_DOC_OVERVIEW_QUERY,
    STATS_DOC_TYPE_QUERY,
    STATS_CLAIM_TYPE_QUERY,
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
from ..entity_gateway import EntityResolutionGateway
from ..executor import Neo4jQueryExecutor
from ..models import RetrievalStats, SynthesisResult
from ..query_builder import CypherQueryBuilder
from ..synthesis import SynthesisEngine


# Allowlist for entity_id URL path parameters.
# Format: lowercase letters/underscores (2-30 chars) + underscore + 8-32 hex chars
# e.g. entity_id_a3f1b2c4d5e6f7a8
_ENTITY_ID_RE = re.compile(r'^[a-z_]{2,30}_[0-9a-f]{8,32}$')

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
    retrieval_stats: dict[str, int] | None = None
    synthesis_available: bool = True  # False when API call failed; provenance blocks still returned
    quarantined_claims_count: int = 0  # claims excluded from this answer due to sensitivity review
    supporting_claims_detail: list[dict[str, Any]] = Field(default_factory=list)
    # Each entry: {claim_id, confidence, epistemic_status, claim_type, doc_id, doc_title}
    user_can_annotate: bool = False  # True when the requesting user has archivist/admin role


# ---------------------------------------------------------------------------
# Collection statistics HTML renderer
# ---------------------------------------------------------------------------

_TEMPLATES_DIR: Path = Path(__file__).parent / "templates"

# Expected claim-type share for a wildlife refuge annual-report corpus.
# Used by gap detection to identify underrepresented topical areas.
_EXPECTED_CLAIM_SHARES: dict[str, float] = {
    "population_estimate": 0.22,
    "species_presence":    0.18,
    "management_action":   0.14,
    "habitat_condition":   0.10,
    "breeding_activity":   0.07,
    "migration_timing":    0.05,
    "weather_observation": 0.04,
}

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

def _compute_temporal_gaps(temporal_rows: list[dict]) -> list[dict]:
    """Return consecutive sparse-year periods that fall between active coverage."""
    if not temporal_rows:
        return []
    year_data = {row["year"]: row.get("doc_count", 0) for row in temporal_rows}
    all_years = sorted(year_data)
    if len(all_years) < 2:
        return []
    min_year, max_year = all_years[0], all_years[-1]
    full = {y: year_data.get(y, 0) for y in range(min_year, max_year + 1)}
    active = [v for v in full.values() if v > 0]
    if not active:
        return []
    median_active = _statistics.median(active)
    gap_threshold = max(1, median_active * 0.2)

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
                for d in range((min_year // 10) * 10, max_year + 10, 10):
                    cnt = decade_map.get(d, 0)
                    temporal_rows.append({"label": f"{d}s", "count": cnt, "is_gap": cnt == 0})
            else:
                for y in range(min_year, max_year + 1):
                    cnt = year_map.get(y, 0)
                    temporal_rows.append({"label": str(y), "count": cnt, "is_gap": cnt == 0})

    safe_total = max(total_claims, 1)
    conf_avg_str = f"{avg_conf:.2f}" if avg_conf is not None else "—"

    return {
        "total_docs": total_docs,
        "total_claims": total_claims,
        "total_entities": total_entities,
        "date_span": date_span,
        "total_pages": total_pages,
        "donor_restricted": donor_restricted,
        "doc_types": doc_types,
        "claim_types": claim_types,
        "entity_types": entity_types,
        "temporal_rows": temporal_rows,
        "high": high,
        "medium": medium,
        "low": low,
        "high_pct": high * 100 // safe_total,
        "medium_pct": medium * 100 // safe_total,
        "low_pct": low * 100 // safe_total,
        "uncertain_epistemic": uncertain_epistemic,
        "conf_avg_str": conf_avg_str,
    }


def _build_gaps_context(
    temporal: list[dict],
    entity_depth: list[dict],
    geo_coverage: list[dict],
    claim_types: list[dict],
    conv_gaps: list[dict],
) -> dict:
    temporal_gaps = _compute_temporal_gaps(temporal)
    year_span = f"{temporal[0]['year']}–{temporal[-1]['year']}" if temporal else "unknown"

    entity_depth_by_type: dict[str, list[dict]] = {}
    for row in entity_depth:
        entity_depth_by_type.setdefault(row["entity_type"], []).append(row)

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

    topic_rows = []
    for claim_type, expected in sorted(_EXPECTED_CLAIM_SHARES.items(), key=lambda x: -x[1]):
        actual = actual_shares.get(claim_type, 0.0)
        actual_pct = round(actual * 100, 1)
        expected_pct = round(expected * 100, 1)
        if actual < expected * 0.25:
            status = "gap"
        elif actual < expected * 0.50:
            status = "thin"
        else:
            status = "ok"
        topic_rows.append({
            "claim_type": claim_type,
            "actual_pct": actual_pct,
            "expected_pct": expected_pct,
            "status": status,
        })

    return {
        "temporal_gaps": temporal_gaps,
        "year_span": year_span,
        "entity_depth_by_type": sorted(entity_depth_by_type.items()),
        "geo_coverage": geo_coverage,
        "show_unclassified_banner": show_unclassified_banner,
        "unclassified_pct": unclassified_pct,
        "topic_rows": topic_rows,
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
        rerun_url = "/?q=" + _urlparse.quote(qtext)
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
        srerun_url = "/?q=" + _urlparse.quote(sqtext)
        saved_ctxs.append({
            "label": slabel,
            "sqtext": sqtext,
            "sqtrun": sqtrun,
            "sbkt": sbkt,
            "syr_str": syr_str,
            "screated_by": screated_by,
            "sts": sts,
            "saved_id": saved_id,
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
        pagination = {
            "has_prev": has_prev,
            "has_next": has_next,
            "prev_url": f"/history?offset={prev_offset}&limit={limit}{base_qs}",
            "next_url": f"/history?offset={next_offset}&limit={limit}{base_qs}",
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
    from graphrag_pipeline.shared.logging_config import setup_logging
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

    from graphrag_pipeline.retrieval.web.rate_limit import TokenBucketLimiter
    _query_limiter = TokenBucketLimiter(
        max_calls=int(os.environ.get("QUERY_RATE_LIMIT_MAX", "20")),
        period_seconds=float(os.environ.get("QUERY_RATE_LIMIT_PERIOD", "60")),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[misc]
        executor = Neo4jQueryExecutor(
            uri=uri,
            user=user,
            password=password,
            database=database,
            trust_mode=trust,
            ca_cert_path=ca_cert,
        )
        executor.ensure_schema()
        corpus_rows = executor.run(CORPUS_STATS_QUERY, {})
        state["corpus_stats"] = corpus_rows[0] if corpus_rows else {"total_paragraphs": 0, "total_documents": 0}
        state["executor"] = executor
        state["query_builder"] = CypherQueryBuilder(executor)
        _ann_db = annotation_db_path or os.environ.get("ANNOTATION_DB")
        _annotation_store = None
        if _ann_db:
            from graphrag_pipeline.ingest.annotation import AnnotationStore
            _annotation_store = AnnotationStore(_ann_db)
            state["annotation_store"] = _annotation_store
        state["assembler"] = ProvenanceContextAssembler(executor, annotation_store=_annotation_store)
        state["gateway"] = EntityResolutionGateway()
        synthesis_ctx: str | None = None
        if domain_dir:
            from ...shared.resource_loader import load_domain_profile
            synthesis_ctx = load_domain_profile(Path(domain_dir)).get("synthesis_context")
        # Token usage tracker
        from graphrag_pipeline.shared.token_tracker import TokenUsageLogger, TokenUsageStore, load_pricing
        _token_pricing = load_pricing()
        _token_db = Path(os.environ.get("TOKEN_USAGE_DB", "data/token_usage.db"))
        _token_logger = TokenUsageLogger(_token_db, _token_pricing)
        state["token_logger"] = _token_logger
        state["token_store"] = TokenUsageStore(_token_db)

        state["synthesis"] = SynthesisEngine(
            api_key=api_key,
            max_tokens=max_tokens,
            synthesis_context=synthesis_ctx,
            timeout=float(os.environ.get("SYNTHESIS_TIMEOUT", "60")),
            token_logger=_token_logger,
        )
        conv_log_path = Path(os.environ.get("CONV_LOG_DB", "data/conversation_log.db"))
        state["conv_logger"] = ConversationLogger(conv_log_path)
        from graphrag_pipeline.retrieval.conversation_log import QueryHistoryStore as _QHSInit
        state["history_store"] = _QHSInit(conv_log_path)
        write_audit_path = os.environ.get("WRITE_AUDIT_DB", "data/write_audit.db")
        from .write_audit_log import WriteAuditLogger
        state["write_audit"] = WriteAuditLogger(write_audit_path)

        # Admin stores — optional, gracefully None if DBs missing.
        # After construction (schema + migrations), reopen with
        # check_same_thread=False so sync endpoints in FastAPI's
        # threadpool can use the connection safely.
        def _reopen_conn(store: Any) -> None:
            """Replace the store's SQLite conn with a thread-safe one."""
            store._conn.close()
            store._conn = _sqlite3.connect(str(store.db_path), check_same_thread=False)
            store._conn.row_factory = _sqlite3.Row
            store._conn.execute("PRAGMA journal_mode=WAL")
            store._conn.execute("PRAGMA foreign_keys=ON")

        _review_db = os.environ.get("REVIEW_DB", "data/review.db")
        try:
            from graphrag_pipeline.review.store import ReviewStore
            _rs = ReviewStore(_review_db)
            _reopen_conn(_rs)
            state["review_store"] = _rs
        except Exception:
            _log.warning("Review store unavailable (REVIEW_DB=%s)", _review_db)
            state["review_store"] = None

        _ingest_db = os.environ.get("INGEST_DB", "data/ingest_jobs.db")
        try:
            from graphrag_pipeline.ingest.store import IngestStore
            state["ingest_store"] = IngestStore(_ingest_db)
        except Exception:
            _log.warning("Ingest store unavailable (INGEST_DB=%s)", _ingest_db)
            state["ingest_store"] = None

        _users_db = os.environ.get("USERS_DB", "data/users.db")
        try:
            from graphrag_pipeline.auth.store import UserStore
            state["user_store"] = UserStore(_users_db)
        except Exception:
            _log.warning("User store unavailable (USERS_DB=%s)", _users_db)
            state["user_store"] = None

        yield
        executor.close()
        if state.get("annotation_store"):
            state["annotation_store"].close()
        if state.get("history_store"):
            state["history_store"].close()
        if state.get("token_store"):
            state["token_store"].close()

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

    from graphrag_pipeline.auth.setup import is_setup_needed

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
        return _templates.TemplateResponse("chat.html", {"request": request, **_user_ctx(user)})

    # ------------------------------------------------------------------
    # POST /query
    # ------------------------------------------------------------------
    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, request: Request, user: UserContext = Depends(require_user)) -> QueryResponse:
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
        intent = classify_query(
            text=req.text,
            year_range=req.year_range,
            entity_hints=req.entity_hints or [],
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
            )

        # Layer 1: entity resolution.
        entity_ctx = gateway.resolve(
            surface_forms=intent.entities,
            entity_hints=req.entity_hints,
        )
        _log.debug("entities extracted: %s", intent.entities)
        _log.debug("resolved: %s", [(r.surface_form, r.entity_id, r.resolution_confidence) for r in entity_ctx.resolved])
        _log.debug("unresolved: %s", entity_ctx.unresolved)

        is_hybrid = intent.bucket == "hybrid"
        analytical_result = None

        # Layer 2A: analytical path.
        if intent.bucket in ("analytical", "hybrid"):
            # Run species trend for the first resolved species entity, if any.
            species_entities = [e for e in entity_ctx.resolved if e.entity_type == "Species"]
            habitat_entities = [e for e in entity_ctx.resolved if e.entity_type == "Habitat"]

            if species_entities:
                analytical_result = builder.species_trend(
                    species_id=species_entities[0].entity_id,
                    year_min=intent.year_min,
                    year_max=intent.year_max,
                    permitted_levels=user.permitted_levels,
                    institution_id=user.institution_id,
                )
            elif habitat_entities:
                analytical_result = builder.habitat_conditions(
                    habitat_id=habitat_entities[0].entity_id,
                    year_min=intent.year_min,
                    year_max=intent.year_max,
                    permitted_levels=user.permitted_levels,
                    institution_id=user.institution_id,
                )

        # Layer 2B: conversational path.
        blocks, context_text = assembler.assemble(
            query_text=req.text,
            entity_context=entity_ctx,
            year_min=intent.year_min,
            year_max=intent.year_max,
            is_hybrid=is_hybrid,
            permitted_levels=user.permitted_levels,
            institution_id=user.institution_id,
        )

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
        )

        # Layer 3: synthesis.
        synthesis_available = True
        try:
            result: SynthesisResult = engine.synthesise(
                query=req.text,
                provenance_blocks=blocks,
                context_text=context_text,
                analytical_result=analytical_result,
                conversation_history=[t.model_dump() for t in req.conversation_history],
            )
        except Exception:
            # API unavailable, rate-limited, or response unparseable.
            # Return retrieved provenance blocks with a fallback answer so the
            # archivist can read source sentences directly without a 500 error.
            synthesis_available = False
            result = SynthesisResult(
                answer=(
                    "AI synthesis is currently unavailable. "
                    "Retrieved source sentences are provided via supporting_claim_ids for direct review."
                ),
                confidence_assessment="",
                supporting_claim_ids=[b.claim_id for b in blocks],
                caveats=["AI synthesis was unavailable at query time. Source provenance blocks were retrieved successfully."],
            )

        conv_logger: ConversationLogger | None = state.get("conv_logger")
        if conv_logger is not None:
            cited_ids = set(result.supporting_claim_ids)
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
            ))

        # Count quarantined claims for this institution to include in response.
        quarantined_claims_count = 0
        try:
            _executor: Neo4jQueryExecutor = state["executor"]
            _qrows = _executor.run(
                COUNT_QUARANTINED_CLAIMS_QUERY,
                {
                    "institution_id": user.institution_id,
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
        blocks_by_id = {b.claim_id: b for b in blocks}
        claims_detail = [
            {
                "claim_id": cid,
                "source_sentence": blocks_by_id[cid].source_sentence if cid in blocks_by_id else "",
                "confidence": blocks_by_id[cid].extraction_confidence if cid in blocks_by_id else None,
                "epistemic_status": blocks_by_id[cid].epistemic_status if cid in blocks_by_id else "unknown",
                "claim_type": blocks_by_id[cid].claim_type if cid in blocks_by_id else "",
                "doc_id": blocks_by_id[cid].doc_id if cid in blocks_by_id else "",
                "doc_title": blocks_by_id[cid].doc_title if cid in blocks_by_id else "",
            }
            for cid in result.supporting_claim_ids
        ]

        return QueryResponse(
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

    # ------------------------------------------------------------------
    # POST /query/provenance
    # ------------------------------------------------------------------
    @app.post("/query/provenance")
    def query_provenance(req: ProvenanceRequest, user: UserContext = Depends(require_user)) -> dict[str, Any]:
        """Return the full raw provenance chain for a known *claim_id*.

        Does not invoke the synthesis engine — useful for citation
        verification and the review web interface.
        """
        assembler: ProvenanceContextAssembler = state["assembler"]
        blocks = assembler.chain_for_claim(
            req.claim_id,
            permitted_levels=user.permitted_levels,
            institution_id=user.institution_id,
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
    def collection_stats(request: Request, user: UserContext = Depends(require_user)):
        executor: Neo4jQueryExecutor = state["executor"]
        params = {"institution_id": user.institution_id, "permitted_levels": user.permitted_levels}
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
    def collection_gaps(request: Request, user: UserContext = Depends(require_admin)):
        executor: Neo4jQueryExecutor = state["executor"]
        params = {"institution_id": user.institution_id, "permitted_levels": user.permitted_levels}
        thin_params = {**params, "thin_threshold": 3, "limit": 50}

        temporal = executor.run(GAP_TEMPORAL_DENSITY_QUERY, params) or []
        entity_depth = executor.run(GAP_ENTITY_DEPTH_QUERY, thin_params) or []
        geo_coverage = executor.run(GAP_GEOGRAPHIC_COVERAGE_QUERY, thin_params) or []
        claim_types = executor.run(STATS_CLAIM_TYPE_QUERY, params) or []

        conv_gaps: list[dict] = []
        conv_log_db = os.environ.get("CONV_LOG_DB", "")
        if conv_log_db and os.path.exists(conv_log_db):
            conv_gaps = _read_query_signal_gaps(conv_log_db)

        ctx = _build_gaps_context(temporal, entity_depth, geo_coverage, claim_types, conv_gaps)
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
        user: UserContext = Depends(require_user),
    ) -> list[dict]:
        if not q.strip():
            return []
        executor: Neo4jQueryExecutor = state["executor"]
        return executor.run(ENTITY_SEARCH_QUERY, {"query": q.strip(), "limit": limit}) or []

    # ------------------------------------------------------------------
    # GET /api/entity/{entity_id}/neighborhood  — graph neighborhood data
    # ------------------------------------------------------------------
    @app.get("/api/entity/{entity_id}/neighborhood")
    def entity_neighborhood(
        entity_id: str,
        limit: int = Query(15, le=30),
        user: UserContext = Depends(require_user),
    ) -> dict[str, Any]:
        if not _ENTITY_ID_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity_id format")
        executor: Neo4jQueryExecutor = state["executor"]
        params = {"institution_id": user.institution_id, "permitted_levels": user.permitted_levels}

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
            "is_center": True,
        }}]
        edges: list[dict] = []
        for n in neighbors:
            nodes.append({"data": {
                "id": n["entity_id"], "label": n["name"],
                "entity_type": n.get("entity_type", ""),
                "claim_count": 0,
                "is_center": False,
            }})
            edges.append({"data": {
                "id": f"{entity_id}__{n['entity_id']}",
                "source": entity_id, "target": n["entity_id"],
                "weight": n["co_occurrence_count"],
                "relationship_types": n.get("relationship_types") or [],
                "sample_sentences": n.get("sample_sentences") or [],
                "sample_claim_ids": n.get("sample_claim_ids") or [],
                "source_label": center["name"],
                "target_label": n["name"],
            }})

        return {"nodes": nodes, "edges": edges, "center": center}

    # ------------------------------------------------------------------
    # GET /documents  — admin: list all documents including soft-deleted
    # ------------------------------------------------------------------
    def _list_documents_json(user: UserContext = Depends(require_admin)) -> dict[str, Any]:
        """JSON list for /v1/documents API."""
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(LIST_DOCUMENTS_QUERY, {"institution_id": user.institution_id})
        return {"institution_id": user.institution_id, "documents": rows}

    @app.get("/documents", response_class=HTMLResponse, include_in_schema=False)
    def list_documents(request: Request, user: UserContext = Depends(require_admin)):
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(LIST_DOCUMENTS_QUERY, {"institution_id": user.institution_id})
        return _templates.TemplateResponse("admin_documents.html", {
            "request": request,
            "active_page": "documents",
            **_user_ctx(user),
            "documents": rows,
            "institution_id": user.institution_id,
        })

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
    from graphrag_pipeline.retrieval.conversation_log import QueryHistoryStore as _QHS

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
    # GET /api/token-stats  — admin: current-period token usage summary
    # ------------------------------------------------------------------
    @app.get("/api/token-stats")
    def api_token_stats(
        user: UserContext = Depends(require_admin),
    ) -> dict[str, Any]:
        from graphrag_pipeline.shared.token_tracker import TokenUsageStore
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
        from graphrag_pipeline.shared.token_tracker import TokenUsageStore
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
        user: UserContext = Depends(require_admin),
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
            "institution_id": user.institution_id,
            "deleted_at": deleted_at,
            "deleted_by": deleted_by,
        })
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id!r} not found in institution {user.institution_id!r}",
            )
        write_audit: Any = state.get("write_audit")
        if write_audit is not None:
            write_audit.log(
                event_type="soft_delete",
                doc_id=doc_id,
                doc_title=rows[0].get("title"),
                institution_id=user.institution_id,
                performed_by=deleted_by,
            )
        if request.headers.get("HX-Request") == "true":
            doc = {"doc_id": doc_id, "title": rows[0].get("title"), "access_level": rows[0].get("access_level"), "deleted_at": deleted_at}
            return _templates.TemplateResponse("fragments/document_row.html", {"request": request, "doc": doc})
        return {"status": "deleted", "doc_id": doc_id, "deleted_at": deleted_at, "deleted_by": deleted_by}

    # ------------------------------------------------------------------
    # POST /document/{doc_id}/restore  — admin: restore a soft-deleted document
    # ------------------------------------------------------------------
    @app.post("/document/{doc_id}/restore")
    def restore_document(
        doc_id: str,
        request: Request,
        user: UserContext = Depends(require_admin),
    ) -> dict[str, Any]:
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(RESTORE_DOCUMENT_QUERY, {
            "doc_id": doc_id,
            "institution_id": user.institution_id,
        })
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id!r} not found in institution {user.institution_id!r}",
            )
        restored_at = datetime.now(timezone.utc).isoformat()
        performed_by = user.identity
        write_audit: Any = state.get("write_audit")
        if write_audit is not None:
            write_audit.log(
                event_type="restore",
                doc_id=doc_id,
                doc_title=rows[0].get("title"),
                institution_id=user.institution_id,
                performed_by=performed_by,
            )
        if request.headers.get("HX-Request") == "true":
            doc = {"doc_id": doc_id, "title": rows[0].get("title"), "access_level": rows[0].get("access_level"), "deleted_at": None}
            return _templates.TemplateResponse("fragments/document_row.html", {"request": request, "doc": doc})
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

    def _build_source_comparison_context(
        issue_class: str, evidence: dict,
    ) -> dict | None:
        """Build source comparison context for the detail template.

        Returns *None* when the evidence lacks the enriched fields needed to
        render a meaningful comparison.
        """
        import html as _html
        from graphrag_pipeline.review.web.diff_utils import char_diff_html, highlight_in_text

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
            rev = review_store.get_latest_revision(p.proposal_id)
            evidence: dict = {}
            if rev:
                try:
                    evidence = _json.loads(rev.evidence_snapshot_json)
                except Exception:
                    pass

            is_sensitivity = p.issue_class in _SENSITIVITY_CLASSES
            ic_label = _ISSUE_CLASS_LABELS.get(p.issue_class, p.issue_class)

            # Build description from evidence using key_info_html from review app
            try:
                from graphrag_pipeline.review.web.app import _key_info_html
                desc = _key_info_html(p.issue_class, evidence)
            except Exception:
                desc = _html.escape(p.issue_class)

            proposal_cards.append({
                "proposal_id": p.proposal_id,
                "ic_label": ic_label,
                "badge_cls": "badge-red" if is_sensitivity else "badge-blue",
                "border_color": "var(--accent-red-border)" if is_sensitivity else "var(--border-medium)",
                "priority": p.priority_score,
                "confidence": p.confidence,
                "description": desc,
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
            qcount = queue_counts.get(qname, 0)
            if qcount > 0:
                queue_filters.append((qname, qlabel, qcount))

        # Detector count (unique queue names with proposals)
        detector_count = len([q for q in queue_counts.values() if q > 0])

        # Today's activity
        today_iso = _date.today().isoformat()
        event_counts = review_store.correction_event_counts(since_iso=today_iso)

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
        from graphrag_pipeline.review.models import ERROR_TYPES, ERROR_TYPE_LABELS
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
        from graphrag_pipeline.review.actions import accept_proposal, ReviewActionError
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
        from graphrag_pipeline.review.actions import reject_proposal, ReviewActionError
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
        from graphrag_pipeline.review.actions import defer_proposal, ReviewActionError
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
        revisions = review_store.get_revisions(proposal_id)
        events = review_store.get_correction_events(proposal_id)

        latest_rev = revisions[-1] if revisions else None
        patch_json = latest_rev.patch_spec_json if latest_rev else "{}"
        evidence_json = latest_rev.evidence_snapshot_json if latest_rev else "{}"

        try:
            patch_formatted = _json.dumps(_json.loads(patch_json), indent=2)
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
            from graphrag_pipeline.review.web.app import _key_info_html
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

        # Error classification defaults for the form
        from graphrag_pipeline.review.models import (
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
            "source_comparison": source_comparison,
            "targets": targets,
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
            from graphrag_pipeline.ingest.pdf_converter import ConversionError, convert_pdf_to_json
            from graphrag_pipeline.ingest.pipeline import _process_single_document
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
