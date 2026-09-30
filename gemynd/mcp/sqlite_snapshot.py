"""SQLite/FTS-backed evidence snapshots for large standalone corpora."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .query import query_terms

SQLITE_SNAPSHOT_FORMAT_V1 = "kettle-evidence-sqlite-snapshot-v1"
SQLITE_SNAPSHOT_FORMAT = "kettle-evidence-sqlite-snapshot-v2"
SUPPORTED_SQLITE_SNAPSHOT_FORMATS = frozenset({SQLITE_SNAPSHOT_FORMAT_V1, SQLITE_SNAPSHOT_FORMAT})


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a large artifact without loading it into process memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any, fallback: Any) -> str:
    return json.dumps(value if value is not None else fallback, ensure_ascii=False, sort_keys=True)


def build_sqlite_snapshot(
    rows: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    snapshot_id: str,
    scope: str | None = None,
    batch_size: int = 5000,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize evidence rows into a compact, read-only SQLite/FTS snapshot.

    Version 2 uses an external-content FTS table. SQLite stores the canonical
    claim/provenance row once while FTS stores only its search index, rather
    than keeping a second copy of every claim, excerpt, and source record.
    """
    destination = Path(output_path).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    db_path = destination / "evidence.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE evidence (
            rowid INTEGER PRIMARY KEY,
            evidence_id TEXT NOT NULL UNIQUE,
            claim TEXT NOT NULL,
            excerpt TEXT NOT NULL,
            year INTEGER,
            confidence REAL,
            claim_type TEXT,
            attestation TEXT,
            institution_id TEXT NOT NULL,
            provenance_id TEXT,
            source_json TEXT NOT NULL,
            subjects_json TEXT NOT NULL,
            relationships_json TEXT NOT NULL,
            place_json TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE evidence_fts USING fts5(
            claim, excerpt, source_json, subjects_json, relationships_json, place_json,
            content='evidence', content_rowid='rowid'
        );
        """
    )
    insert_rows: list[tuple[Any, ...]] = []
    institution_counts: dict[str, int] = {}
    count = 0
    try:
        for row in rows:
            evidence_id = str(row.get("evidence_id") or "").strip()
            claim = str(row.get("claim") or "").strip()
            source = row.get("source")
            if not evidence_id or not claim or not isinstance(source, dict):
                continue
            excerpt = str(row.get("excerpt") or claim).strip()
            if excerpt == claim:
                excerpt = ""
            institution_id = str(
                source.get("institution") or row.get("institution_id") or ""
            ).strip()
            insert_rows.append(
                (
                    evidence_id,
                    claim,
                    excerpt,
                    row.get("year"),
                    row.get("confidence"),
                    row.get("claim_type"),
                    row.get("attestation"),
                    institution_id,
                    row.get("provenance_id") or evidence_id,
                    _json(source, {}),
                    _json(row.get("subjects"), []),
                    _json(row.get("relationships"), []),
                    _json(row.get("place"), []),
                )
            )
            if len(insert_rows) >= batch_size:
                connection.executemany(
                    "INSERT OR IGNORE INTO evidence "
                    "(evidence_id, claim, excerpt, year, confidence, claim_type, "
                    "attestation, institution_id, provenance_id, source_json, "
                    "subjects_json, relationships_json, place_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    insert_rows,
                )
                insert_rows.clear()
        if insert_rows:
            connection.executemany(
                "INSERT OR IGNORE INTO evidence "
                "(evidence_id, claim, excerpt, year, confidence, claim_type, "
                "attestation, institution_id, provenance_id, source_json, "
                "subjects_json, relationships_json, place_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                insert_rows,
            )
        connection.execute("CREATE INDEX evidence_year ON evidence(year)")
        connection.execute("CREATE INDEX evidence_institution ON evidence(institution_id)")
        connection.execute("INSERT INTO evidence_fts(evidence_fts) VALUES('rebuild')")
        connection.execute("ANALYZE")
        connection.commit()
        count = int(connection.execute("SELECT count(*) FROM evidence").fetchone()[0])
        institution_counts = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT institution_id, count(*) FROM evidence "
                "WHERE institution_id <> '' GROUP BY institution_id"
            )
        }
        integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if integrity != "ok":
            raise ValueError(f"SQLite evidence snapshot integrity check failed: {integrity}")
    finally:
        connection.close()
    digest = sha256_file(db_path)
    manifest: dict[str, Any] = {
        "format": SQLITE_SNAPSHOT_FORMAT,
        "snapshot_id": snapshot_id,
        "record_count": count,
        "evidence_file": db_path.name,
        "sha256": digest,
        "size_bytes": db_path.stat().st_size,
        "institutions": dict(sorted(institution_counts.items())),
    }
    if scope:
        manifest["scope"] = scope
    if metadata:
        manifest["build"] = dict(metadata)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_sqlite_snapshot(
    path: str | Path, *, verify_integrity: bool = True
) -> tuple[dict[str, Any], Path]:
    """Validate a snapshot manifest, checksum, schema, and row count."""
    snapshot_path = Path(path).resolve()
    try:
        manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"SQLite evidence snapshot manifest is invalid: {exc}") from exc
    if manifest.get("format") not in SUPPORTED_SQLITE_SNAPSHOT_FORMATS:
        raise ValueError("unsupported SQLite evidence snapshot format")
    snapshot_id = manifest.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("SQLite evidence snapshot is missing snapshot_id")
    filename = manifest.get("evidence_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("SQLite evidence snapshot evidence_file must be a filename")
    database = snapshot_path / filename
    if not database.is_file():
        raise ValueError("SQLite evidence snapshot database is missing")
    if sha256_file(database) != manifest.get("sha256"):
        raise ValueError("SQLite evidence snapshot checksum mismatch")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        if verify_integrity:
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            if integrity != "ok":
                raise ValueError(f"SQLite evidence snapshot integrity check failed: {integrity}")
        count = int(connection.execute("SELECT count(*) FROM evidence").fetchone()[0])
        if count != int(manifest.get("record_count", -1)):
            raise ValueError("SQLite evidence snapshot record_count mismatch")
    finally:
        connection.close()
    return manifest, database


class SQLiteEvidenceSnapshotBackend:
    """Read-only FTS search over a checksum-verified SQLite snapshot."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.manifest, database = validate_sqlite_snapshot(self.path)
        self._format = str(self.manifest["format"])
        self.connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        self.connection.execute("PRAGMA query_only=ON")
        self._connection_lock = threading.RLock()
        self._stats: dict[str, Any] | None = None

    def search(
        self,
        question: str,
        *,
        time_range: dict[str, int] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        terms = query_terms(question)
        if not terms:
            return []
        limit = max(1, int(limit))
        quoted = [f'"{term}"' for term in terms]
        # Rows containing every content term rank first; any-term matches only
        # fill the remainder, so a lone common word cannot outrank a full match.
        rows = self._query(" AND ".join(quoted), time_range, limit)
        if len(rows) < limit and len(terms) > 1:
            seen = {row["evidence_id"] for row in rows}
            for row in self._query(" OR ".join(quoted), time_range, limit + len(rows)):
                if row["evidence_id"] not in seen:
                    rows.append(row)
                    if len(rows) >= limit:
                        break
        return rows

    def stats(self) -> dict[str, Any]:
        """Corpus span for coverage reporting; cached because the file is immutable."""
        if self._stats is None:
            with self._connection_lock:
                low, high = self.connection.execute(
                    "SELECT min(year), max(year) FROM evidence"
                ).fetchone()
            self._stats = {
                "record_count": self.manifest.get("record_count"),
                "institutions": sorted(self.manifest.get("institutions") or {}),
                "year_min": low,
                "year_max": high,
            }
        return self._stats

    def _query(
        self, match: str, time_range: dict[str, int] | None, limit: int
    ) -> list[dict[str, Any]]:
        clauses = ["evidence_fts MATCH ?"]
        params: list[Any] = [match]
        if time_range and "start" in time_range:
            clauses.append("(e.year IS NULL OR e.year >= ?)")
            params.append(int(time_range["start"]))
        if time_range and "end" in time_range:
            clauses.append("(e.year IS NULL OR e.year <= ?)")
            params.append(int(time_range["end"]))
        params.append(limit)
        if self._format == SQLITE_SNAPSHOT_FORMAT_V1:
            query = (
                "SELECT e.evidence_id, e.claim, e.excerpt, e.year, e.confidence, "
                "e.claim_type, e.attestation, e.source_json "
                "FROM evidence_fts JOIN evidence e "
                "ON e.evidence_id = evidence_fts.evidence_id "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY bm25(evidence_fts), e.evidence_id LIMIT ?"
            )
            with self._connection_lock:
                rows = self.connection.execute(query, params).fetchall()
            return [
                {
                    "evidence_id": row[0],
                    "claim": row[1],
                    "excerpt": row[2] or row[1],
                    "year": row[3],
                    "confidence": row[4],
                    "claim_type": row[5],
                    "attestation": row[6],
                    "source": json.loads(row[7]),
                }
                for row in rows
            ]
        query = (
            "SELECT e.evidence_id, e.claim, e.excerpt, e.year, e.confidence, "
            "e.claim_type, e.attestation, e.provenance_id, e.source_json, "
            "e.subjects_json, e.relationships_json, e.place_json "
            "FROM evidence_fts JOIN evidence e ON e.rowid = evidence_fts.rowid "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY bm25(evidence_fts), e.evidence_id LIMIT ?"
        )
        with self._connection_lock:
            rows = self.connection.execute(query, params).fetchall()
        return [
            {
                "evidence_id": row[0],
                "claim": row[1],
                "excerpt": row[2] or row[1],
                "year": row[3],
                "confidence": row[4],
                "claim_type": row[5],
                "attestation": row[6],
                "provenance_id": row[7],
                "source": json.loads(row[8]),
                "subjects": json.loads(row[9]),
                "relationships": json.loads(row[10]),
                "place": json.loads(row[11]),
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._connection_lock:
            self.connection.close()
