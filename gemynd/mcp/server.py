"""Minimal, stateless MCP server for Kettle's research marketplace MVP.

This adapter intentionally has no payment-provider dependency.  It exposes
the product primitives from the engineering handoff and records a deterministic
usage receipt.  A production deployment can require a payment adapter later
without changing the tool contracts.

Run locally::

    gemynd-mcp

Configuration is environment based:

``KETTLE_MCP_REGISTRY_PATH``
    Corpus registry YAML (defaults to ``data/corpus_registry.yaml``).
``KETTLE_MCP_EVIDENCE_PATH``
    Optional JSON or JSONL evidence fixture for the standalone server.
``KETTLE_MCP_SNAPSHOT_PATH``
    Optional versioned evidence snapshot directory.  The snapshot manifest and
    evidence checksum are verified before any records are served.
``KETTLE_API_BASE_URL``
    Optional running Gemynd retrieval API to use as the evidence backend.
``KETTLE_API_EVIDENCE_PATH``
    Optional path for the evidence endpoint (defaults to ``/v1/query`` for
    compatibility with the current Gemynd API).
``KETTLE_API_TOKEN``
    Optional bearer token forwarded to the retrieval API.
``KETTLE_MCP_API_KEY``
    Optional API key required by the MCP endpoint.
``KETTLE_MCP_REQUIRE_PAYMENT``
    Set to ``true`` to reject paid tools until a payment adapter is attached.
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest
from urllib.request import urlopen

from ..shared.publication_titles import is_lccn, publication_title
from .payments import (
    HTTPPaymentVerifier,
    PaymentError,
    PaymentLedger,
    PaymentVerifier,
    RejectingPaymentVerifier,
)
from .query import matched_terms, query_terms, strong_match_threshold
from .snapshot import EvidenceSnapshot
from .sqlite_snapshot import (
    SUPPORTED_SQLITE_SNAPSHOT_FORMATS,
    SQLiteEvidenceSnapshotBackend,
)


class _JsonLogFormatter(logging.Formatter):
    """Emit one secret-safe JSON object per request/event."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        for key in (
            "request_id",
            "tenant_id",
            "method",
            "tool",
            "status",
            "latency_ms",
            "error_code",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("kettle.mcp")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonLogFormatter())
        logger.addHandler(handler)
    logger.setLevel(os.getenv("KETTLE_MCP_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger


class MCPToolError(RuntimeError):
    """A safe, machine-readable tool failure."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class EvidenceBackend(Protocol):
    def search(
        self, question: str, *, time_range: dict[str, int] | None, limit: int
    ) -> list[dict[str, Any]]:
        """Return normalized or raw evidence records for a question."""


class EmptyEvidenceBackend:
    """Safe default when no retrieval backend has been configured."""

    def search(
        self, question: str, *, time_range: dict[str, int] | None, limit: int
    ) -> list[dict[str, Any]]:
        return []


class JsonEvidenceBackend:
    """Small deterministic JSON/JSONL backend for local development and tests."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records = self._load(path)

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise MCPToolError("backend_unavailable", f"Evidence file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
            rows = payload if isinstance(payload, list) else payload.get("evidence", [])
        except json.JSONDecodeError:
            try:
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise MCPToolError(
                    "backend_invalid", f"Evidence file is not valid JSON or JSONL: {exc}"
                ) from exc
        except OSError as exc:
            raise MCPToolError(
                "backend_unavailable", f"Could not read evidence file: {exc}"
            ) from exc
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise MCPToolError("backend_invalid", "Evidence file must contain an array of objects")
        return [dict(row) for row in rows]

    def search(
        self, question: str, *, time_range: dict[str, int] | None, limit: int
    ) -> list[dict[str, Any]]:
        terms = query_terms(question)
        start = int(time_range["start"]) if time_range and "start" in time_range else None
        end = int(time_range["end"]) if time_range and "end" in time_range else None
        ranked: list[tuple[int, dict[str, Any]]] = []
        for row in self.records:
            score = len(matched_terms(terms, row))
            year = _record_year(row)
            if start is not None and year is not None and year < start:
                continue
            if end is not None and year is not None and year > end:
                continue
            if score:
                ranked.append((score, row))
        ranked.sort(key=lambda item: (-item[0], str(item[1].get("evidence_id", ""))))
        return [dict(row) for _, row in ranked[:limit]]

    def stats(self) -> dict[str, Any]:
        years = [year for year in map(_record_year, self.records) if year is not None]
        institutions = {
            str(_source_field(row, "institution") or row.get("institution_id") or "")
            for row in self.records
        }
        return {
            "record_count": len(self.records),
            "institutions": sorted(item for item in institutions if item),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
        }


class SnapshotEvidenceBackend(JsonEvidenceBackend):
    """Read-only backend for a checksum-verified standalone snapshot."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        snapshot = EvidenceSnapshot(self.path)
        self.manifest = snapshot.manifest
        self.records = snapshot.records


class HTTPQueryBackend:
    """Adapter for a Gemynd endpoint that returns source-linked evidence.

    The endpoint may return an ``answer`` internally, but this adapter never
    forwards that field. MCP callers receive only ``supporting_claims_detail``
    (or an explicit backend error when the endpoint cannot provide evidence).
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = 60.0,
        evidence_path: str = "/v1/query",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.evidence_path = "/" + evidence_path.lstrip("/")

    def search(
        self, question: str, *, time_range: dict[str, int] | None, limit: int
    ) -> list[dict[str, Any]]:
        year_range = None
        if time_range and "start" in time_range and "end" in time_range:
            year_range = [time_range["start"], time_range["end"]]
        payload = {
            "text": question,
            "mode": "auto",
            "evidence_only": True,
            "year_range": year_range,
        }
        request = URLRequest(
            f"{self.base_url}{self.evidence_path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise MCPToolError(
                "retrieval_error",
                f"Gemynd retrieval returned HTTP {exc.code}: {detail}",
                retryable=exc.code >= 500,
            ) from exc
        except URLError as exc:
            raise MCPToolError(
                "retrieval_unavailable",
                f"Could not reach Gemynd retrieval API: {exc.reason}",
                retryable=True,
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise MCPToolError(
                "retrieval_unavailable",
                f"Could not reach Gemynd retrieval API: {exc}",
                retryable=True,
            ) from exc
        details = result.get("supporting_claims_detail") or []
        if details:
            return [dict(item) for item in details[:limit]]
        if result.get("supporting_claim_ids"):
            raise MCPToolError(
                "backend_invalid",
                "Evidence endpoint returned claim IDs without source-linked evidence details",
            )
        return []

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


@dataclass
class UsageMeter:
    """In-memory tenant meter for the MVP; replace with a durable ledger later."""

    daily_cap_usd: float = 5.0
    spent: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def authorize(self, tenant_id: str, price_usd: float) -> None:
        # Tool calls run on worker threads; check-and-spend must be atomic.
        with self._lock:
            current = self.spent.get(tenant_id, 0.0)
            if current + price_usd > self.daily_cap_usd:
                raise MCPToolError(
                    "quota_exceeded",
                    f"Tenant daily research cap of ${self.daily_cap_usd:.2f} would be exceeded",
                )
            self.spent[tenant_id] = round(current + price_usd, 6)


@dataclass
class RequestRateLimiter:
    """Small bounded-window limiter for the stateless HTTP adapter."""

    max_requests: int = 120
    window_seconds: float = 60.0
    _requests: dict[str, list[float]] = field(default_factory=dict)

    def allow(self, tenant_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        timestamps = [stamp for stamp in self._requests.get(tenant_id, []) if stamp > cutoff]
        if len(timestamps) >= self.max_requests:
            self._requests[tenant_id] = timestamps
            return False
        timestamps.append(now)
        self._requests[tenant_id] = timestamps
        # Avoid retaining idle tenant keys indefinitely.
        if len(self._requests) > 10_000:
            self._requests = {key: values for key, values in self._requests.items() if values}
        return True


_QUESTION = {
    "type": "string",
    "minLength": 3,
    "description": (
        "What to look for. Lexical search over OCR'd historical text: lead with names, "
        "places, organizations, and topic words (e.g. 'Jimmy Durkin saloon license'). "
        "Filler words are ignored; every remaining term counts toward the match."
    ),
}
_TIME_RANGE = {
    "type": "object",
    "description": 'Inclusive year bounds, e.g. {"start": 1905, "end": 1912}.',
    "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
    "additionalProperties": False,
}
_DOCUMENT = {
    "type": "string",
    "description": "Case-insensitive substring filter on the source document label.",
}
_LOCATION = {
    "type": "string",
    "description": "Case-insensitive substring filter on the record's place or collection.",
}
_OFFSET = {
    "type": "integer",
    "minimum": 0,
    "maximum": 10000,
    "description": "Pagination offset; use pagination.next_offset from the previous page.",
}
_DEPTH = {
    "type": "string",
    "enum": ["brief", "standard", "deep"],
    "default": "standard",
    "description": "Price tier: brief $0.03, standard $0.06, deep $0.15.",
}
_MAX_OBJECTS = {
    "type": "integer",
    "minimum": 1,
    "maximum": 100,
    "default": 40,
    "description": "Upper bound on evidence objects returned.",
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "coverage_probe",
        "title": "Check archive coverage",
        "description": (
            "Free. Call this first for any historical question about the Pacific Northwest "
            "(Washington, Oregon, North Idaho), especially Spokane-area people, businesses, "
            "labor, civic life, and events. It checks the archive itself and reports "
            "coverage_status (strong / partial / weak / none), how many records matched, "
            "a few sample claims, and the corpus's date span. If status is weak or none, "
            "use another source instead of spending calls here."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": _QUESTION,
                "geography_hint": {
                    "type": "string",
                    "description": "Optional place context not already in the question.",
                },
                "time_range": _TIME_RANGE,
                "document": _DOCUMENT,
                "location": _LOCATION,
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Check archive coverage",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "search_evidence",
        "title": "Search archival evidence",
        "description": (
            "Free. Returns up to 50 source-linked claims extracted from archival records, "
            "each with its date, source document, page, and a source_url for citation. "
            "matched_terms shows which query terms each record contains; records matching "
            "only one of several terms are usually off-topic. Results are evidence, not "
            "answers: read the claims and cite their sources. Page with offset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": _QUESTION,
                "geography_hint": {
                    "type": "string",
                    "description": "Optional place context not already in the question.",
                },
                "time_range": _TIME_RANGE,
                "document": _DOCUMENT,
                "location": _LOCATION,
                "offset": _OFFSET,
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Evidence objects per page.",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Search archival evidence",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "research_subject",
        "title": "Research a subject",
        "description": (
            "Paid ($0.03-$0.15 by depth; free when nothing is found). Gathers a larger "
            "evidence set for one subject and adds a date-ordered timeline, mentioned "
            "subjects, relationships, and a provenance summary. Use after coverage_probe "
            "reports strong coverage. Returns evidence only, never a written answer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": _QUESTION,
                "depth": _DEPTH,
                "max_evidence_objects": _MAX_OBJECTS,
                "time_range": _TIME_RANGE,
                "document": _DOCUMENT,
                "location": _LOCATION,
                "offset": _OFFSET,
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Research a subject",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "build_evidence_packet",
        "title": "Build an evidence packet",
        "description": (
            "Paid ($0.25; free when nothing is found). Everything research_subject returns, "
            "plus a packet_id and provenance bundle for an auditable citation package. "
            "Use only when a durable, citable packet is needed; otherwise prefer "
            "research_subject. Returns evidence only, never a written answer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": _QUESTION,
                "depth": _DEPTH,
                "max_evidence_objects": _MAX_OBJECTS,
                "time_range": _TIME_RANGE,
                "document": _DOCUMENT,
                "location": _LOCATION,
                "offset": _OFFSET,
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Build an evidence packet",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
]

_TOOL_ARGUMENTS = {
    tool["name"]: frozenset(tool["inputSchema"]["properties"]) for tool in TOOL_DEFINITIONS
}
_DEPTH_PRICES = {"brief": 0.03, "standard": 0.06, "deep": 0.15}
_PROBE_SAMPLE_SIZE = 25


def _source_field(record: dict[str, Any], key: str) -> Any:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    return source.get(key) or record.get(key)


def _iso_date(value: Any) -> str | None:
    text = str(value or "").strip()[:10]
    return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else None


def _record_year(record: dict[str, Any]) -> int | None:
    value = record.get("year")
    if value is None:
        value = (
            record.get("doc_date_start")
            or record.get("report_year")
            or record.get("date")
            or _source_field(record, "date_start")
        )
    if value is None and isinstance(record.get("time"), dict):
        value = record["time"].get("start")
    try:
        return int(str(value)[:4]) if value is not None else None
    except (TypeError, ValueError):
        return None


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _safe_int_range(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise MCPToolError(
            "invalid_arguments", "time_range must be an object with integer start/end"
        )
    result: dict[str, int] = {}
    for key in ("start", "end"):
        if key in value and value[key] is not None:
            if isinstance(value[key], bool):
                raise MCPToolError("invalid_arguments", f"time_range.{key} must be an integer")
            try:
                result[key] = int(value[key])
            except (TypeError, ValueError) as exc:
                raise MCPToolError(
                    "invalid_arguments", f"time_range.{key} must be an integer"
                ) from exc
    if "start" in result and "end" in result and result["start"] > result["end"]:
        raise MCPToolError("invalid_arguments", "time_range.start must be <= time_range.end")
    return result or None


def _normalize_evidence(row: dict[str, Any], terms: list[str] | None = None) -> dict[str, Any]:
    claim = str(
        row.get("claim")
        or row.get("source_sentence")
        or row.get("text")
        or row.get("excerpt")
        or ""
    )
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    source = {
        "institution": source.get("institution")
        or row.get("institution")
        or row.get("institution_id")
        or "",
        "collection": source.get("collection") or row.get("collection") or row.get("series") or "",
        "document": (
            source.get("document")
            or row.get("document")
            or row.get("doc_title")
            or row.get("publication_title")
            or row.get("title")
            or row.get("doc_id")
            or ""
        ),
        "page_or_record": (
            source.get("page_or_record")
            or row.get("page_or_record")
            or row.get("page_number")
            or row.get("paragraph_id")
            or ""
        ),
        "source_url": source.get("source_url")
        or row.get("source_url")
        or row.get("source_path")
        or "",
    }
    publication = _publication(row)
    if publication:
        # Agents cite the label they are given; an LCCN is not a citable name.
        source["publication"] = publication
        source["document"] = _display_document(row)
    year = _record_year(row)
    day = _iso_date(_source_field(row, "date_start"))
    if isinstance(row.get("time"), dict):
        time_obj = row["time"]
    elif day is not None:
        end_day = _iso_date(_source_field(row, "date_end"))
        time_obj = {
            "start": day,
            "end": end_day if end_day and end_day != day else None,
            "precision": "day",
        }
    else:
        time_obj = {
            "start": f"{year:04d}-01-01" if year is not None else None,
            "end": None,
            "precision": "year" if year is not None else "unknown",
        }
    evidence_id = str(
        row.get("evidence_id") or _stable_id("ktl_ev", claim, json.dumps(source, sort_keys=True))
    )
    confidence_value = row.get("confidence", row.get("extraction_confidence", 0.0))
    try:
        confidence = float(confidence_value or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    excerpt = str(row.get("excerpt") or "")[:500]
    normalized = {
        "evidence_id": evidence_id,
        "claim": claim,
        "claim_type": str(row.get("claim_type") or "other"),
        "subjects": row.get("subjects") if isinstance(row.get("subjects"), list) else [],
        "time": time_obj,
        "place": row.get("place") if isinstance(row.get("place"), list) else [],
        "confidence": max(0.0, min(1.0, confidence)),
        "attestation": str(row.get("attestation") or row.get("epistemic_status") or "direct"),
        "source": source,
        "provenance_id": str(
            row.get("provenance_id")
            or row.get("claim_id")
            or _stable_id("ktl_barcode", evidence_id)
        ),
        # Only surrounding text that adds context; repeating the claim wastes
        # agent context on every object.
        "excerpt": "" if excerpt.strip() == claim.strip() else excerpt,
        "relationships": row.get("relationships")
        if isinstance(row.get("relationships"), list)
        else [],
    }
    if terms is not None:
        normalized["matched_terms"] = matched_terms(terms, row)
    return normalized


def _supported_row(row: dict[str, Any]) -> bool:
    """Require a stable claim and at least one source identity field."""
    claim = str(
        row.get("claim")
        or row.get("source_sentence")
        or row.get("text")
        or row.get("excerpt")
        or ""
    ).strip()
    if not claim or not (row.get("evidence_id") or row.get("claim_id")):
        return False
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return any(
        str(source.get(key) or row.get(key) or "").strip()
        for key in ("document", "doc_title", "doc_id", "paragraph_id", "source_url", "source_path")
    )


def _publication(row: dict[str, Any]) -> str | None:
    """Readable newspaper title when the collection is a known LCCN."""
    collection = str(_source_field(row, "collection") or row.get("series") or "").strip()
    if not is_lccn(collection):
        return None
    title = publication_title(collection)
    return title if title and title != collection else None


def _display_document(row: dict[str, Any]) -> str:
    document = str(_source_field(row, "document") or "")
    collection = str(_source_field(row, "collection") or row.get("series") or "").strip()
    publication = _publication(row)
    if publication and collection and document.startswith(collection):
        return publication + document[len(collection) :]
    return document


def _matches_filters(row: dict[str, Any], *, document: str | None, location: str | None) -> bool:
    if document:
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        haystack = " ".join(
            str(row.get(key) or source.get(key) or "")
            for key in ("document", "doc_title", "doc_id", "title")
        )
        haystack = f"{haystack} {_publication(row) or ''}"
        if document.casefold() not in haystack.casefold():
            return False
    if location:
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        values: list[str] = [
            str(row.get(key) or source.get(key) or "")
            for key in ("location", "place", "collection")
        ]
        values.extend(str(item) for item in (row.get("places") or []) if item is not None)
        if location.casefold() not in " ".join(values).casefold():
            return False
    return True


_REGION_TERMS = (
    "washington",
    "oregon",
    "north idaho",
    "idaho",
    "spokane",
    "seattle",
    "portland",
    "pacific northwest",
    "pnw",
)

_COVERAGE_SUMMARY_KEYS = (
    "coverage_status",
    "query_terms",
    "evidence_found",
    "strong_matches",
    "geographic_fit",
    "temporal_fit",
    "source_diversity",
    "relevant_corpora",
    "known_gaps",
    "external_search_recommended",
)


def _coverage_summary(coverage: dict[str, Any]) -> dict[str, Any]:
    return {key: coverage[key] for key in _COVERAGE_SUMMARY_KEYS}


def _clean_filter(value: Any) -> str | None:
    return str(value or "").strip() or None


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _bounded_int(args: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = args.get(key)
    if value is None:
        value = default
    if isinstance(value, bool):
        raise MCPToolError("invalid_arguments", f"{key} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MCPToolError("invalid_arguments", f"{key} must be an integer") from exc
    return max(low, min(number, high))


@dataclass
class KettleMCPService:
    backend: EvidenceBackend = field(default_factory=EmptyEvidenceBackend)
    registry_path: Path = Path("data/corpus_registry.yaml")
    meter: UsageMeter = field(default_factory=UsageMeter)
    require_payment: bool = False
    payment_verifier: PaymentVerifier = field(default_factory=RejectingPaymentVerifier)
    payment_ledger: PaymentLedger | None = None

    def __post_init__(self) -> None:
        self._registry = self._load_registry()

    def _load_registry(self) -> list[dict[str, Any]]:
        try:
            import yaml

            payload = (
                yaml.safe_load(self.registry_path.read_text(encoding="utf-8"))
                if self.registry_path.exists()
                else {}
            )
        except (OSError, ImportError, ValueError):
            payload = {}
        rows = payload.get("corpora", []) if isinstance(payload, dict) else []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def coverage_probe(
        self,
        question: str,
        *,
        geography_hint: str | None = None,
        time_range: dict[str, Any] | None = None,
        document: str | None = None,
        location: str | None = None,
        inspect_corpus: bool = True,
    ) -> dict[str, Any]:
        """Report whether the archive actually holds evidence for a question.

        Status is measured against the corpus, not inferred from keywords: a
        record is a strong match when it contains most of the question's
        content terms.  ``inspect_corpus=False`` serves the unauthenticated
        public funnel, which must not perform retrieval or disclose evidence.
        """
        question = self._question(question)
        normalized_range = _safe_int_range(time_range)
        text = f"{question} {geography_hint or ''}".casefold()
        geo_fit = 0.97 if any(term in text for term in _REGION_TERMS) else 0.35
        terms = query_terms(question)
        if not inspect_corpus:
            return self._registry_coverage(text, geo_fit, terms, normalized_range)
        rows: list[dict[str, Any]] = []
        if terms:
            rows, _ = self._search_rows(
                question,
                time_range=normalized_range,
                limit=_PROBE_SAMPLE_SIZE,
                offset=0,
                document=_clean_filter(document),
                location=_clean_filter(location),
            )
        threshold = strong_match_threshold(terms)
        scored = [(len(matched_terms(terms, row)), row) for row in rows]
        strong = [row for hits, row in scored if hits >= threshold]
        corpus = self._corpus_stats()
        relevant = self._relevant_corpora(rows, text)
        if not rows:
            status = "none"
        elif len(strong) >= 3:
            status = "strong"
        elif strong:
            status = "partial"
        else:
            status = "weak"
        gaps: list[str] = []
        if not terms:
            gaps.append(
                "The question has no searchable terms; include names, places, "
                "organizations, or topic words."
            )
        elif not rows:
            gaps.append("No archive records matched the question's key terms.")
        elif not strong:
            gaps.append(
                "Records matched only some key terms (see sample_evidence.matched_terms); "
                "they are likely off-topic."
            )
        elif len(strong) < 3:
            gaps.append(f"Only {len(strong)} record(s) matched most key terms.")
        temporal_fit = self._temporal_fit(normalized_range, corpus)
        if normalized_range and temporal_fit == 0.0:
            gaps.append(
                f"Requested years fall outside the archive span "
                f"{corpus.get('year_min')}-{corpus.get('year_max')}."
            )
        recommended = {
            "strong": "research_subject",
            "partial": "search_evidence",
            "weak": "search_evidence",
            "none": None,
        }[status]
        documents = {str(_source_field(row, "document") or "") for row in strong or rows} - {""}
        return {
            "coverage_status": status,
            "coverage_basis": "archive_search",
            "query_terms": terms,
            "evidence_found": len(rows),
            "strong_matches": len(strong),
            "sample_size": _PROBE_SAMPLE_SIZE,
            "geographic_fit": geo_fit,
            "temporal_fit": temporal_fit,
            "source_diversity": round(min(1.0, len(documents) / 10), 2),
            "relevant_corpora": [str(row.get("corpus_id")) for row in relevant],
            "corpus_registry": [
                {
                    "corpus_id": row.get("corpus_id"),
                    "name": row.get("display_name"),
                    "description": " ".join(str(row.get("description", "")).split()),
                    "rights_mode": row.get("rights_mode", "derived-only"),
                }
                for row in relevant
            ],
            "corpus": corpus,
            "known_gaps": gaps,
            "recommended_tool": recommended,
            "estimated_max_price_usd": _DEPTH_PRICES["standard"] if status == "strong" else 0.0,
            "external_search_recommended": status in {"weak", "none"},
            "time_range": normalized_range,
            "sample_evidence": [
                {
                    "evidence_id": str(row.get("evidence_id") or row.get("claim_id") or ""),
                    "claim": _truncate(str(row.get("claim") or row.get("text") or ""), 240),
                    "date": _iso_date(_source_field(row, "date_start")) or _record_year(row),
                    "document": _display_document(row),
                    "matched_terms": matched_terms(terms, row),
                }
                for _, row in sorted(scored, key=lambda item: -item[0])[:3]
            ],
        }

    def _registry_coverage(
        self,
        text: str,
        geo_fit: float,
        terms: list[str],
        time_range: dict[str, int] | None,
    ) -> dict[str, Any]:
        """Retrieval-free estimate from the corpus registry and region terms."""
        relevant = self._relevant_corpora([], text)
        if not relevant and geo_fit >= 0.8:
            relevant = list(self._registry)
        if not self._registry:
            status = "none"
        elif geo_fit >= 0.8 and relevant:
            status = "strong"
        elif relevant:
            status = "partial"
        else:
            status = "weak"
        return {
            "coverage_status": status,
            "coverage_basis": "registry_estimate",
            "query_terms": terms,
            "geographic_fit": geo_fit,
            "relevant_corpora": [str(row.get("corpus_id")) for row in relevant],
            "corpus_registry": [
                {
                    "corpus_id": row.get("corpus_id"),
                    "name": row.get("display_name"),
                    "description": " ".join(str(row.get("description", "")).split()),
                    "rights_mode": row.get("rights_mode", "derived-only"),
                }
                for row in relevant
            ],
            "known_gaps": []
            if status == "strong"
            else ["Estimated from the corpus registry; the archive was not searched."],
            "recommended_tool": "research_subject" if status == "strong" else "search_evidence",
            "estimated_max_price_usd": _DEPTH_PRICES["standard"] if status == "strong" else 0.0,
            "external_search_recommended": status in {"weak", "none"},
            "time_range": time_range,
        }

    def _corpus_stats(self) -> dict[str, Any]:
        stats = getattr(self.backend, "stats", None)
        if not callable(stats):
            return {}
        try:
            return dict(stats())
        except Exception:  # coverage must not fail because a summary is unavailable
            return {}

    def _relevant_corpora(self, rows: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
        institutions = {str(_source_field(row, "institution") or "") for row in rows}
        matched = [row for row in self._registry if str(row.get("corpus_id")) in institutions]
        if matched:
            return matched
        question_tokens = set(query_terms(text))
        return [
            row
            for row in self._registry
            if question_tokens
            & set(
                query_terms(
                    f"{row.get('corpus_id', '')} {row.get('display_name', '')} "
                    f"{row.get('description', '')}"
                )
            )
        ]

    @staticmethod
    def _temporal_fit(time_range: dict[str, int] | None, corpus: dict[str, Any]) -> float | None:
        low, high = corpus.get("year_min"), corpus.get("year_max")
        if low is None or high is None:
            return None
        if not time_range:
            return 1.0
        start = time_range.get("start", low)
        end = time_range.get("end", high)
        overlap = min(end, high) - max(start, low) + 1
        return round(max(0, overlap) / (end - start + 1), 2)

    def _search_rows(
        self,
        question: str,
        *,
        time_range: dict[str, int] | None,
        limit: int,
        offset: int,
        document: str | None,
        location: str | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        raw = self.backend.search(
            question, time_range=time_range, limit=min(10_000, limit + offset + 1)
        )
        filtered = [
            row
            for row in raw
            if _supported_row(row) and _matches_filters(row, document=document, location=location)
        ]
        page = filtered[offset : offset + limit]
        return page, len(filtered) > offset + limit

    def search_evidence(self, question: str, **kwargs: Any) -> dict[str, Any]:
        question = self._question(question)
        limit = _bounded_int(kwargs, "limit", 10, 1, 50)
        time_range = _safe_int_range(kwargs.get("time_range"))
        offset = _bounded_int(kwargs, "offset", 0, 0, 10_000)
        document = _clean_filter(kwargs.get("document"))
        location = _clean_filter(kwargs.get("location"))
        coverage = self.coverage_probe(
            question,
            geography_hint=kwargs.get("geography_hint"),
            time_range=time_range,
            document=document,
            location=location,
        )
        raw, has_more = self._search_rows(
            question,
            time_range=time_range,
            limit=limit,
            offset=offset,
            document=document,
            location=location,
        )
        evidence = [_normalize_evidence(row, coverage["query_terms"]) for row in raw]
        return {
            "coverage": _coverage_summary(coverage),
            "evidence_objects": evidence,
            "pagination": {
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
                "next_offset": offset + limit if has_more else None,
            },
            "research_receipt": self._receipt(
                "search_evidence", 0.0, evidence, collections=len(coverage["relevant_corpora"])
            ),
        }

    def research_subject(self, question: str, **kwargs: Any) -> dict[str, Any]:
        question = self._question(question)
        depth = str(kwargs.get("depth") or "standard")
        if depth not in _DEPTH_PRICES:
            raise MCPToolError(
                "invalid_arguments", f"depth must be one of: {', '.join(_DEPTH_PRICES)}"
            )
        max_objects = _bounded_int(kwargs, "max_evidence_objects", 40, 1, 100)
        time_range = _safe_int_range(kwargs.get("time_range"))
        offset = _bounded_int(kwargs, "offset", 0, 0, 10_000)
        document = _clean_filter(kwargs.get("document"))
        location = _clean_filter(kwargs.get("location"))
        coverage = self.coverage_probe(
            question, time_range=time_range, document=document, location=location
        )
        raw, _has_more = self._search_rows(
            question,
            time_range=time_range,
            limit=max_objects,
            offset=offset,
            document=document,
            location=location,
        )
        evidence = [_normalize_evidence(row, coverage["query_terms"]) for row in raw]
        price = _DEPTH_PRICES[depth]
        if not evidence:
            price = 0.0
        receipt = self._receipt(
            "research_subject", price, evidence, collections=len(coverage["relevant_corpora"])
        )
        subjects: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in evidence:
            for subject in item["subjects"]:
                if not isinstance(subject, dict):
                    continue
                key = str(subject.get("entity_id") or subject.get("name") or "").casefold()
                if key and key not in seen:
                    seen.add(key)
                    subjects.append(dict(subject))
        # The timeline references evidence_objects instead of repeating them.
        timeline = [
            {"date": item["time"].get("start"), "evidence_id": item["evidence_id"]}
            for item in sorted(
                evidence,
                key=lambda item: (
                    str(item.get("time", {}).get("start") or "9999"),
                    item["evidence_id"],
                ),
            )
        ]
        relationships = [rel for item in evidence for rel in item.get("relationships", [])]
        return {
            "status": "ok" if evidence else "no_result",
            "question": question,
            "resolved_subjects": subjects,
            "timeline": timeline,
            "evidence_objects": evidence,
            "relationships": relationships,
            "disagreements_or_uncertainty": self._find_disagreements(evidence),
            "coverage": _coverage_summary(coverage),
            "provenance_summary": {
                "source_count": len(
                    {
                        item["source"].get("document")
                        for item in evidence
                        if item["source"].get("document")
                    }
                ),
                "evidence_count": len(evidence),
            },
            "research_receipt": receipt,
        }

    def trace_relationship(self, question: str, **kwargs: Any) -> dict[str, Any]:
        raise MCPToolError(
            "unsupported_operation",
            "Arbitrary relationship tracing is disabled; use research_subject for source-linked relationships.",
        )

    def build_evidence_packet(self, question: str, **kwargs: Any) -> dict[str, Any]:
        result = self.research_subject(question, **kwargs)
        evidence = result["evidence_objects"]
        packet_id = _stable_id("ktl_packet", question, [item["evidence_id"] for item in evidence])
        result["packet_id"] = packet_id
        result["conclusion"] = (
            f"Compiled {len(evidence)} source-linked evidence object(s) for the requested proposition."
            if evidence
            else "No evidence packet was produced because the configured corpus returned no relevant evidence."
        )
        # IDs reference evidence_objects so the packet carries each object once.
        result["supporting_evidence"] = [item["evidence_id"] for item in evidence]
        result["contradictory_evidence"] = [
            item["evidence_id"]
            for item in evidence
            if item["attestation"] in {"inferred", "uncertain"}
        ]
        result["provenance_bundle"] = [item["provenance_id"] for item in evidence]
        result["research_receipt"]["operation"] = "build_evidence_packet"
        result["research_receipt"]["price_usd"] = 0.25 if evidence else 0.0
        return result

    def call(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        *,
        tenant_id: str = "local",
        idempotency_key: str | None = None,
        payment_receipt: str | None = None,
    ) -> dict[str, Any]:
        args = arguments or {}
        if not isinstance(args, dict):
            raise MCPToolError("invalid_arguments", "arguments must be an object")
        allowed = _TOOL_ARGUMENTS.get(name)
        if allowed is not None and set(args) - allowed:
            # Reject typos loudly: a silently ignored filter returns misleading evidence.
            raise MCPToolError(
                "invalid_arguments",
                f"Unknown argument(s) for {name}: {', '.join(sorted(set(args) - allowed))}. "
                f"Allowed: {', '.join(sorted(allowed))}",
            )
        try:
            if name == "coverage_probe":
                return self.coverage_probe(**args)
            if name == "search_evidence":
                return self.search_evidence(**args)
            if name == "research_subject":
                result = self.research_subject(**args)
                self._authorize_paid(result, tenant_id, idempotency_key, payment_receipt)
                return result
            if name == "trace_relationship":
                # Retain a stable direct-call response for older clients while
                # keeping the unsupported tool out of tools/list.
                return self.trace_relationship(**args)
            if name == "build_evidence_packet":
                result = self.build_evidence_packet(**args)
                self._authorize_paid(result, tenant_id, idempotency_key, payment_receipt)
                return result
        except TypeError as exc:
            raise MCPToolError("invalid_arguments", str(exc)) from exc
        raise MCPToolError("unknown_tool", f"Unknown tool: {name}")

    def instructions(self) -> str:
        """Server instructions sent at initialize, describing the live corpus."""
        corpus = self._corpus_stats()
        names = [str(row.get("display_name") or row.get("corpus_id")) for row in self._registry]
        served = corpus.get("institutions") or []
        span = (
            f" covering {corpus['year_min']}-{corpus['year_max']}"
            if corpus.get("year_min") is not None
            else ""
        )
        holdings = (
            f"This server holds {corpus.get('record_count', 'an unknown number of')} "
            f"source-linked claims from {', '.join(served) or 'the configured archive'}{span}. "
            if corpus
            else ""
        )
        return (
            "Kettle is an evidence service over digitized Pacific Northwest archives "
            f"({'; '.join(names) or 'regional historical records'}). {holdings}"
            "Workflow: call coverage_probe first (free). If coverage_status is strong or "
            "partial, call search_evidence (free) and cite each claim's source.document and "
            "source.source_url. Use research_subject or build_evidence_packet (paid) only for "
            "strong coverage. If status is weak or none, look elsewhere. Search is lexical over "
            "OCR text: query with names, places, and topic words; try spelling variants for "
            "names. Tools return evidence, never answers; OCR errors are common, so quote "
            "claims cautiously."
        )

    def _authorize_paid(
        self,
        result: dict[str, Any],
        tenant_id: str,
        idempotency_key: str | None,
        payment_receipt: str | None,
    ) -> None:
        receipt = result.get("research_receipt") or {}
        price = float(receipt.get("price_usd") or 0.0)
        if price <= 0:
            return
        if self.require_payment:
            if not idempotency_key or not payment_receipt:
                raise MCPToolError(
                    "payment_required",
                    "A payment authorization requires Idempotency-Key and X-Kettle-Payment-Receipt headers",
                )
            if self.payment_ledger is None:
                raise MCPToolError("payment_unavailable", "Payment ledger is not configured")
            existing = self.payment_ledger.existing(idempotency_key)
            try:
                if existing is None:
                    verification = self.payment_verifier.verify(
                        payment_receipt=payment_receipt,
                        idempotency_key=idempotency_key,
                        tenant_id=tenant_id,
                        operation=str(receipt.get("operation") or ""),
                        price_usd=price,
                    )
                    snapshot_id = getattr(self.backend, "manifest", {}).get("snapshot_id")
                    authorization = self.payment_ledger.record(
                        idempotency_key=idempotency_key,
                        payment_receipt=payment_receipt,
                        tenant_id=tenant_id,
                        operation=str(receipt.get("operation") or ""),
                        price_usd=price,
                        snapshot_id=snapshot_id,
                        provider_reference=verification.provider_reference,
                    )
                else:
                    authorization = self.payment_ledger.record(
                        idempotency_key=idempotency_key,
                        payment_receipt=payment_receipt,
                        tenant_id=tenant_id,
                        operation=str(receipt.get("operation") or ""),
                        price_usd=price,
                        snapshot_id=existing.get("snapshot_id"),
                        provider_reference=str(existing["provider_reference"]),
                    )
            except PaymentError as exc:
                raise MCPToolError(exc.code, exc.message) from exc
            receipt.update(
                {
                    "billing_status": "authorized",
                    "charged": True,
                    "idempotency_key": idempotency_key,
                    "payment_receipt_sha256": authorization["payment_receipt_sha256"],
                    "payment_reference": authorization["provider_reference"],
                    "snapshot_id": authorization.get("snapshot_id"),
                    "idempotent_replay": authorization["replayed"],
                }
            )
            return
        self.meter.authorize(tenant_id, price)
        receipt["billing_status"] = "metered"
        receipt["charged"] = False

    @staticmethod
    def _question(question: Any) -> str:
        if not isinstance(question, str) or len(question.strip()) < 3:
            raise MCPToolError("invalid_arguments", "question must be at least 3 characters")
        return question.strip()

    @staticmethod
    def _find_disagreements(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_claim: dict[str, list[dict[str, Any]]] = {}
        for item in evidence:
            key = re.sub(r"\W+", " ", item["claim"].casefold()).strip()
            if key:
                by_claim.setdefault(key, []).append(item)
        return [
            {"claim_key": key, "evidence_ids": [item["evidence_id"] for item in items]}
            for key, items in by_claim.items()
            if len(items) > 1
        ]

    @staticmethod
    def _receipt(
        operation: str, price_usd: float, evidence: list[dict[str, Any]], *, collections: int
    ) -> dict[str, Any]:
        institutions = {
            item["source"].get("institution")
            for item in evidence
            if item["source"].get("institution")
        }
        return {
            "receipt_id": _stable_id(
                "ktl_receipt",
                operation,
                datetime.now(timezone.utc).date().isoformat(),
                uuid.uuid4(),
            ),
            "operation": operation,
            "kettle_calls": 1,
            "collections_searched": collections,
            "source_institutions": len(institutions),
            "candidate_records_examined": len(evidence),
            "candidate_claims_evaluated": len(evidence),
            "evidence_objects_returned": len(evidence),
            "price_usd": round(price_usd, 2),
            "charged": False,
            "billing_status": "free" if price_usd == 0 else "metered_pending_payment_adapter",
            "benchmark_class": "regional_research_mvp_v1",
        }


def _service_from_env() -> KettleMCPService:
    registry_path = Path(os.getenv("KETTLE_MCP_REGISTRY_PATH", "data/corpus_registry.yaml"))
    evidence_path = os.getenv("KETTLE_MCP_EVIDENCE_PATH")
    snapshot_path = os.getenv("KETTLE_MCP_SNAPSHOT_PATH")
    base_url = os.getenv("KETTLE_API_BASE_URL")
    if base_url:
        backend: EvidenceBackend = HTTPQueryBackend(
            base_url,
            os.getenv("KETTLE_API_TOKEN"),
            evidence_path=os.getenv("KETTLE_API_EVIDENCE_PATH", "/v1/query"),
        )
    elif snapshot_path:
        manifest_path = Path(snapshot_path) / "manifest.json"
        try:
            snapshot_format = json.loads(manifest_path.read_text(encoding="utf-8")).get("format")
        except (OSError, json.JSONDecodeError) as exc:
            raise MCPToolError(
                "backend_invalid", f"Could not read snapshot manifest: {exc}"
            ) from exc
        backend = (
            SQLiteEvidenceSnapshotBackend(Path(snapshot_path))
            if snapshot_format in SUPPORTED_SQLITE_SNAPSHOT_FORMATS
            else SnapshotEvidenceBackend(Path(snapshot_path))
        )
    elif evidence_path:
        backend = JsonEvidenceBackend(Path(evidence_path))
    else:
        backend = EmptyEvidenceBackend()
    cap = float(os.getenv("KETTLE_MCP_DAILY_CAP_USD", "5.0"))
    require_payment = os.getenv("KETTLE_MCP_REQUIRE_PAYMENT", "false").casefold() == "true"
    verifier_url = os.getenv("KETTLE_PAYMENT_VERIFIER_URL", "").strip()
    verifier: PaymentVerifier = (
        HTTPPaymentVerifier(
            verifier_url,
            token=os.getenv("KETTLE_PAYMENT_VERIFIER_TOKEN"),
            timeout=float(os.getenv("KETTLE_PAYMENT_VERIFIER_TIMEOUT", "15")),
        )
        if verifier_url
        else RejectingPaymentVerifier()
    )
    ledger = PaymentLedger(os.getenv("KETTLE_PAYMENT_LEDGER", "data/kettle_payments.db"))
    return KettleMCPService(
        backend=backend,
        registry_path=registry_path,
        meter=UsageMeter(cap),
        require_payment=require_payment,
        payment_verifier=verifier,
        payment_ledger=ledger,
    )


def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _configured_api_keys() -> tuple[str, ...]:
    values = [os.getenv("KETTLE_MCP_API_KEY", "")]
    values.extend(os.getenv("KETTLE_MCP_API_KEYS", "").split(","))
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _valid_api_key(authorization: str, configured: tuple[str, ...]) -> bool:
    if not configured:
        return True
    if not authorization.startswith("Bearer "):
        return False
    supplied = authorization.removeprefix("Bearer ")
    return any(hmac.compare_digest(supplied, expected) for expected in configured)


def _authenticated_tenant(
    authorization: str,
    configured: tuple[str, ...],
    requested_tenant: str,
) -> str | None:
    """Derive a non-spoofable tenant identity for limits and receipts."""
    if not configured:
        return requested_tenant[:128] or "local"
    if not _valid_api_key(authorization, configured):
        return None
    supplied = authorization.removeprefix("Bearer ")
    digest = hashlib.sha256(supplied.encode("utf-8")).hexdigest()[:20]
    return f"api_key_{digest}"


# Tool semantics are identical across these revisions, so echo the client's.
_PROTOCOL_VERSIONS = frozenset({"2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26"})


def _compact_json(value: Any) -> str:
    # Agents read the text block; compact separators save ~10% of their context.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def dispatch_jsonrpc(
    payload: dict[str, Any],
    service: KettleMCPService,
    *,
    tenant_id: str = "local",
    idempotency_key: str | None = None,
    payment_receipt: str | None = None,
) -> dict[str, Any] | None:
    request_id = payload.get("id")
    method = payload.get("method")
    if not isinstance(method, str):
        return _jsonrpc_error(request_id, -32600, "Invalid Request")
    if method == "initialize":
        requested = (payload.get("params") or {}).get("protocolVersion")
        version = requested if requested in _PROTOCOL_VERSIONS else "2025-11-25"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "kettle-research", "version": "0.1.0"},
                "instructions": service.instructions(),
            },
        }
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None
    if method in {"ping", "logging/setLevel"}:
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "server/discover":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2026-07-28",
                "serverInfo": {"name": "kettle-research", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOL_DEFINITIONS}}
    if method != "tools/call":
        return _jsonrpc_error(request_id, -32601, f"Method not found: {method}")
    params = payload.get("params") or {}
    name = params.get("name")
    if not isinstance(name, str):
        return _jsonrpc_error(request_id, -32602, "tools/call requires params.name")
    try:
        result = service.call(
            name,
            params.get("arguments"),
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            payment_receipt=payment_receipt,
        )
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": _compact_json(result)}],
                "structuredContent": result,
            },
        }
    except MCPToolError as exc:
        result = {"error_code": exc.code, "message": exc.message, "retryable": exc.retryable}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": _compact_json(result)}],
                "structuredContent": result,
            },
        }


def create_mcp_app(
    service: KettleMCPService | None = None,
    *,
    x402_settings: Any = None,
    x402_resource_server: Any = None,
):
    """Create the stateless FastAPI MCP endpoint."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:  # pragma: no cover - import guard for base installs
        raise RuntimeError("FastAPI is required. Install with: pip install -e .[mcp]") from exc

    app = FastAPI(title="Kettle Research MCP", docs_url=None, redoc_url=None, openapi_url=None)
    logger = _configure_logging()

    @app.middleware("http")
    async def request_logging(request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        # Do not put a caller-controlled tenant name into trusted audit fields.
        # The authenticated route below derives the real tenant from the key.
        tenant_id = "pending_auth"
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            logger.exception(
                "request_failed",
                extra={
                    "event": "request_failed",
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "method": request.method,
                    "status": 500,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise
        finally:
            logger.info(
                "request_complete",
                extra={
                    "event": "request_complete",
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "method": request.method,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )

    tool_service = service or _service_from_env()
    from .x402 import (
        X402Settings,
        estimate_purchase,
        install_x402_routes,
        public_capabilities,
    )

    payment_settings = x402_settings or X402Settings.from_env()
    expected_keys = _configured_api_keys()
    limiter = RequestRateLimiter(
        max_requests=max(1, int(os.getenv("KETTLE_MCP_RATE_LIMIT", "120"))),
        window_seconds=max(1.0, float(os.getenv("KETTLE_MCP_RATE_WINDOW", "60"))),
    )

    def _backend_metadata() -> dict[str, Any]:
        manifest = getattr(tool_service.backend, "manifest", None)
        if not isinstance(manifest, dict):
            return {
                "backend": type(tool_service.backend).__name__,
                "x402_enabled": payment_settings.enabled,
                "x402_network": payment_settings.network if payment_settings.enabled else None,
            }
        return {
            "backend": type(tool_service.backend).__name__,
            "snapshot_id": manifest.get("snapshot_id"),
            "snapshot_format": manifest.get("format"),
            "record_count": manifest.get("record_count"),
            "x402_enabled": payment_settings.enabled,
            "x402_network": payment_settings.network if payment_settings.enabled else None,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "kettle-research",
            "protocol": ["2025-11-25", "2026-07-28"],
            **_backend_metadata(),
        }

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        metadata = _backend_metadata()
        if isinstance(tool_service.backend, EmptyEvidenceBackend):
            return JSONResponse(status_code=503, content={"status": "not_ready", **metadata})
        return {"status": "ready", **metadata}

    def public_rate_limit(request: Request):
        client_host = request.client.host if request.client else "unknown"
        if limiter.allow(f"public:{client_host}"):
            return None
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": str(int(limiter.window_seconds))},
        )

    @app.get("/public/capabilities")
    async def capabilities(request: Request):
        limited = public_rate_limit(request)
        if limited is not None:
            return limited
        return public_capabilities(
            x402_enabled=payment_settings.enabled,
            network=payment_settings.network,
        )

    @app.post("/public/coverage")
    async def public_coverage(request: Request):
        limited = public_rate_limit(request)
        if limited is not None:
            return limited
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "Invalid JSON body"})
        try:
            if not isinstance(body, dict):
                raise MCPToolError("invalid_arguments", "Request body must be a JSON object")
            allowed = {"question", "geography_hint", "time_range", "document", "location"}
            unknown = set(body) - allowed
            if unknown:
                raise MCPToolError(
                    "invalid_arguments", f"Unknown field(s): {', '.join(sorted(unknown))}"
                )
            return tool_service.coverage_probe(**body, inspect_corpus=False)
        except MCPToolError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "error_code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                },
            )

    @app.post("/public/estimate")
    async def public_estimate(request: Request):
        limited = public_rate_limit(request)
        if limited is not None:
            return limited
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "Invalid JSON body"})
        try:
            estimate = estimate_purchase(body)
            estimate["payment_available"] = payment_settings.enabled
            estimate["payment_network"] = payment_settings.network
            return estimate
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/mcp")
    async def mcp(request: Request):
        rpc_started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        tenant_id = _authenticated_tenant(
            request.headers.get("authorization", ""),
            expected_keys,
            request.headers.get("x-kettle-tenant", "local"),
        )
        if tenant_id is None:
            logger.warning(
                "rpc_rejected",
                extra={
                    "event": "rpc_rejected",
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "method": "http_auth",
                    "status": 401,
                    "latency_ms": round((time.perf_counter() - rpc_started) * 1000, 2),
                },
            )
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        try:
            payload = await request.json()
        except Exception:
            logger.warning(
                "rpc_parse_error",
                extra={
                    "event": "rpc_parse_error",
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "method": "json_parse",
                    "status": 400,
                    "latency_ms": round((time.perf_counter() - rpc_started) * 1000, 2),
                },
            )
            return JSONResponse(
                status_code=400, content=_jsonrpc_error(None, -32700, "Parse error")
            )
        idempotency_key = request.headers.get("idempotency-key")
        payment_receipt = request.headers.get("x-kettle-payment-receipt")
        if not limiter.allow(tenant_id):
            logger.warning(
                "rpc_rate_limited",
                extra={
                    "event": "rpc_rate_limited",
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "method": "rate_limit",
                    "status": 429,
                    "latency_ms": round((time.perf_counter() - rpc_started) * 1000, 2),
                },
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(int(limiter.window_seconds))},
            )
        method = (
            payload[0].get("method")
            if isinstance(payload, list) and payload and isinstance(payload[0], dict)
            else payload.get("method")
            if isinstance(payload, dict)
            else "invalid"
        )
        tool = (
            payload[0].get("params", {}).get("name")
            if isinstance(payload, list) and payload and isinstance(payload[0], dict)
            else payload.get("params", {}).get("name")
            if isinstance(payload, dict)
            else None
        )

        def log_rpc(response: Any, status: int = 200) -> None:
            structured = (
                response.get("result", {}).get("structuredContent", {})
                if isinstance(response, dict)
                else {}
            )
            error_code = structured.get("error_code") if isinstance(structured, dict) else None
            event = "rpc_error" if error_code else "rpc_complete"
            logger.warning(
                event,
                extra={
                    "event": event,
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "method": method,
                    "tool": tool,
                    "status": status,
                    "error_code": error_code,
                    "latency_ms": round((time.perf_counter() - rpc_started) * 1000, 2),
                },
            ) if error_code else logger.info(
                event,
                extra={
                    "event": event,
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "method": method,
                    "tool": tool,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - rpc_started) * 1000, 2),
                },
            )

        if isinstance(payload, list):
            responses = await run_in_threadpool(
                lambda: [
                    dispatch_jsonrpc(
                        item,
                        tool_service,
                        tenant_id=tenant_id,
                        idempotency_key=idempotency_key,
                        payment_receipt=payment_receipt,
                    )
                    for item in payload
                    if isinstance(item, dict)
                ]
            )
            responses = [item for item in responses if item is not None]
            if not responses:
                log_rpc({}, 202)
                return Response(status_code=202)
            log_rpc(responses[0])
            return JSONResponse(content=responses)
        if not isinstance(payload, dict):
            logger.warning(
                "rpc_invalid_request",
                extra={
                    "event": "rpc_invalid_request",
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "method": "invalid",
                    "status": 400,
                    "latency_ms": round((time.perf_counter() - rpc_started) * 1000, 2),
                },
            )
            return JSONResponse(
                status_code=400, content=_jsonrpc_error(None, -32600, "Invalid Request")
            )
        # Retrieval is blocking SQLite/HTTP work; keep it off the event loop so
        # one slow query cannot stall /health, /ready, or other tenants.
        response = await run_in_threadpool(
            lambda: dispatch_jsonrpc(
                payload,
                tool_service,
                tenant_id=tenant_id,
                idempotency_key=idempotency_key,
                payment_receipt=payment_receipt,
            )
        )
        if response is None:
            log_rpc({}, 202)
            return Response(status_code=202)
        log_rpc(response)
        return JSONResponse(content=response)

    # Public payment-gated HTTP resources are deliberately separate from the
    # API-key-protected MCP endpoint. This lets x402/Bazaar buyers discover and
    # pay for fixed-price operations without possessing a Kettle API key.
    install_x402_routes(
        app,
        tool_service,
        payment_settings,
        resource_server=x402_resource_server,
    )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kettle research MCP server")
    parser.add_argument("--host", default=os.getenv("KETTLE_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("KETTLE_MCP_PORT", "8789")))
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("uvicorn is required. Install with: pip install -e .[mcp]") from exc
    uvicorn.run(create_mcp_app(), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
