"""Operator-facing health probes for the gemynd deployment.

Surfaces the state of Neo4j, every registered SQLite database, the claim cache,
and the two daemon-thread writers (conversation log, token usage) in a single
report. Used by the ``gemynd status`` CLI command.

Lives in ``shared/`` because the same checks may be reused by a future ``/status``
HTTP endpoint or scheduled job. Cross-layer probes (Neo4j executor, claim cache)
do their imports inside the function body so that ``shared/`` itself remains
importable from every layer.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .settings import Settings

Status = Literal["ok", "warn", "fail"]

_STATUS_RANK = {"ok": 0, "warn": 1, "fail": 2}
_RANK_STATUS = {0: "ok", 1: "warn", 2: "fail"}


@dataclass(slots=True)
class CheckResult:
    name: str
    status: Status
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StatusReport:
    timestamp: str
    checks: list[CheckResult]

    @property
    def overall(self) -> Status:
        if not self.checks:
            return "ok"
        rank = max(_STATUS_RANK[c.status] for c in self.checks)
        return _RANK_STATUS[rank]  # type: ignore[return-value]

    @property
    def exit_code(self) -> int:
        return _STATUS_RANK[self.overall]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_status(
    settings: Settings,
    *,
    cache_warn_rows: int = 500_000,
) -> StatusReport:
    """Run every probe and return a single :class:`StatusReport`.

    Probes are independent: a failure in one never aborts the others.
    """
    checks: list[CheckResult] = [_check_neo4j(settings)]

    sqlite_targets: list[tuple[str, str]] = [
        ("users", settings.users_db),
        ("review", settings.review_db),
        ("ingest", settings.ingest_db),
        ("conv_log", settings.conv_log_db),
        ("token_usage", settings.token_usage_db),
        ("write_audit", settings.write_audit_db),
    ]
    if settings.annotation_db:
        sqlite_targets.append(("annotation", settings.annotation_db))

    for name, path_str in sqlite_targets:
        checks.append(_check_sqlite_db(name, Path(path_str)))

    checks.append(_check_claim_cache(cache_warn_rows))

    conv_freshness = _check_conversation_freshness(Path(settings.conv_log_db))
    checks.append(conv_freshness)
    checks.append(
        _check_token_usage_freshness(
            Path(settings.token_usage_db),
            conv_last_iso=conv_freshness.metrics.get("last_iso"),
        )
    )

    return StatusReport(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        checks=checks,
    )


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _check_neo4j(settings: Settings) -> CheckResult:
    if not settings.neo4j_uri:
        return CheckResult(
            name="neo4j",
            status="warn",
            detail="NEO4J_URI not configured",
        )
    try:
        from gemynd.retrieval.executor import Neo4jQueryExecutor  # deferred: shared/ -> retrieval/
    except Exception as exc:  # pragma: no cover - retrieval extras not installed
        return CheckResult(
            name="neo4j",
            status="fail",
            detail=f"executor import failed: {exc}",
        )

    executor = None
    try:
        executor = Neo4jQueryExecutor(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            trust_mode=settings.neo4j_trust or "system",
            ca_cert_path=settings.neo4j_ca_cert,
        )
        rows = executor.run("RETURN 1 AS ok", {})
        if not rows or rows[0].get("ok") != 1:
            return CheckResult(
                name="neo4j",
                status="fail",
                detail="probe query returned unexpected result",
                metrics={"uri": settings.neo4j_uri, "database": settings.neo4j_database},
            )
        return CheckResult(
            name="neo4j",
            status="ok",
            detail=f"{settings.neo4j_uri} ({settings.neo4j_database})",
            metrics={"uri": settings.neo4j_uri, "database": settings.neo4j_database},
        )
    except Exception as exc:
        return CheckResult(
            name="neo4j",
            status="fail",
            detail=str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__,
            metrics={"uri": settings.neo4j_uri, "database": settings.neo4j_database},
        )
    finally:
        if executor is not None:
            try:
                executor.close()
            except Exception:
                pass


def _check_sqlite_db(name: str, path: Path) -> CheckResult:
    metrics: dict[str, Any] = {"path": str(path)}

    if not path.exists():
        return CheckResult(
            name=name,
            status="fail",
            detail=f"missing: {path}",
            metrics={**metrics, "exists": False},
        )

    sizes = _sqlite_file_sizes(path)
    metrics.update(sizes)
    metrics["exists"] = True

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return CheckResult(
            name=name,
            status="fail",
            detail=f"unreadable: {exc}",
            metrics=metrics,
        )

    detail = f"{path}  {_human_size(sizes['size_bytes'])}"
    if sizes["wal_bytes"]:
        detail += f"  (+ WAL {_human_size(sizes['wal_bytes'])})"
    return CheckResult(name=name, status="ok", detail=detail, metrics=metrics)


def _check_claim_cache(cache_warn_rows: int) -> CheckResult:
    raw = os.environ.get("CLAIM_CACHE_PATH", "") or "data/claim_cache.db"
    path = Path(raw)
    metrics: dict[str, Any] = {"path": str(path), "warn_threshold_rows": cache_warn_rows}

    if not path.exists():
        return CheckResult(
            name="claim_cache",
            status="warn",
            detail=f"absent: {path} (no LLM extractions cached yet)",
            metrics={**metrics, "exists": False, "row_count": 0},
        )

    sizes = _sqlite_file_sizes(path)
    metrics.update(sizes)
    metrics["exists"] = True

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row_count = conn.execute("SELECT COUNT(*) FROM claim_cache").fetchone()[0]
            last_iso = conn.execute("SELECT MAX(created_at) FROM claim_cache").fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        return CheckResult(
            name="claim_cache",
            status="fail",
            detail=f"unreadable: {exc}",
            metrics=metrics,
        )

    metrics["row_count"] = row_count
    metrics["last_iso"] = last_iso

    if row_count > cache_warn_rows:
        return CheckResult(
            name="claim_cache",
            status="warn",
            detail=(
                f"{path}  {_human_size(sizes['size_bytes'])}  "
                f"({row_count:,} rows — exceeds {cache_warn_rows:,} threshold)"
            ),
            metrics=metrics,
        )
    return CheckResult(
        name="claim_cache",
        status="ok",
        detail=f"{path}  {_human_size(sizes['size_bytes'])}  ({row_count:,} rows)",
        metrics=metrics,
    )


def _check_conversation_freshness(conv_log_path: Path) -> CheckResult:
    metrics: dict[str, Any] = {"path": str(conv_log_path), "last_iso": None}

    if not conv_log_path.exists():
        return CheckResult(
            name="conv_log_freshness",
            status="warn",
            detail="conversation log database absent",
            metrics=metrics,
        )

    try:
        conn = sqlite3.connect(f"file:{conv_log_path}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT MAX(created_at) FROM conversation").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return CheckResult(
            name="conv_log_freshness",
            status="fail",
            detail=f"query failed: {exc}",
            metrics=metrics,
        )

    last_iso = row[0] if row else None
    metrics["last_iso"] = last_iso

    if not last_iso:
        return CheckResult(
            name="conv_log_freshness",
            status="warn",
            detail="no conversations recorded yet",
            metrics=metrics,
        )

    age = _age_text(last_iso)
    return CheckResult(
        name="conv_log_freshness",
        status="ok",
        detail=f"last write {last_iso} ({age})",
        metrics=metrics,
    )


def _check_token_usage_freshness(
    token_usage_path: Path,
    *,
    conv_last_iso: str | None,
) -> CheckResult:
    metrics: dict[str, Any] = {"path": str(token_usage_path), "last_iso": None}

    if not token_usage_path.exists():
        return CheckResult(
            name="token_usage_freshness",
            status="warn",
            detail="token usage database absent",
            metrics=metrics,
        )

    try:
        conn = sqlite3.connect(f"file:{token_usage_path}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT MAX(created_at) FROM token_usage").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return CheckResult(
            name="token_usage_freshness",
            status="fail",
            detail=f"query failed: {exc}",
            metrics=metrics,
        )

    last_iso = row[0] if row else None
    metrics["last_iso"] = last_iso
    metrics["conv_last_iso"] = conv_last_iso

    if not last_iso:
        return CheckResult(
            name="token_usage_freshness",
            status="warn",
            detail="no token usage recorded yet",
            metrics=metrics,
        )

    age = _age_text(last_iso)

    # Indirect heartbeat: every conversation should produce a token record, so
    # if conversations are flowing but token writes have stalled, the daemon
    # writer thread is likely dead.
    if conv_last_iso and last_iso < conv_last_iso:
        return CheckResult(
            name="token_usage_freshness",
            status="warn",
            detail=(
                f"last write {last_iso} ({age}) — stale vs. conv log {conv_last_iso}; "
                "writer thread may be dead"
            ),
            metrics=metrics,
        )

    return CheckResult(
        name="token_usage_freshness",
        status="ok",
        detail=f"last write {last_iso} ({age})",
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_status_text(report: StatusReport) -> str:
    lines: list[str] = [f"gemynd status - {report.timestamp}", ""]

    sqlite_names = {"users", "review", "ingest", "conv_log", "token_usage", "write_audit", "annotation", "claim_cache"}
    by_name = {c.name: c for c in report.checks}

    neo4j = by_name.get("neo4j")
    if neo4j is not None:
        lines.append(f"Neo4j             {_status_label(neo4j.status)}  {neo4j.detail}")

    lines.append("Databases")
    for c in report.checks:
        if c.name in sqlite_names:
            lines.append(f"  {c.name:<13} {_status_label(c.status)}  {c.detail}")

    liveness = [c for c in report.checks if c.name.endswith("_freshness")]
    if liveness:
        lines.append("Liveness")
        label_map = {
            "conv_log_freshness": "conv log",
            "token_usage_freshness": "token log",
        }
        for c in liveness:
            label = label_map.get(c.name, c.name)
            lines.append(f"  {label:<13} {_status_label(c.status)}  {c.detail}")

    lines.append("")
    lines.append(f"Overall: {report.overall.upper()} (exit {report.exit_code})")
    return "\n".join(lines)


def format_status_json(report: StatusReport) -> str:
    import json

    payload = {
        "timestamp": report.timestamp,
        "overall": report.overall,
        "exit_code": report.exit_code,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "detail": c.detail,
                "metrics": c.metrics,
            }
            for c in report.checks
        ],
    }
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sqlite_file_sizes(path: Path) -> dict[str, int]:
    main = path.stat().st_size if path.exists() else 0
    wal = path.with_name(path.name + "-wal")
    shm = path.with_name(path.name + "-shm")
    wal_bytes = wal.stat().st_size if wal.exists() else 0
    shm_bytes = shm.stat().st_size if shm.exists() else 0
    return {"size_bytes": main, "wal_bytes": wal_bytes, "shm_bytes": shm_bytes}


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    value = float(n) / 1024.0
    for unit in units:
        if value < 1024.0:
            return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _age_text(iso: str, *, now: datetime | None = None) -> str:
    try:
        ts = datetime.fromisoformat(iso)
    except ValueError:
        return "unknown age"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    delta = reference - ts
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "in the future"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _status_label(status: Status) -> str:
    return f"{status:<4}"
