"""SQLite-backed evaluation case store.

Separate from `gemynd.review.store` — the review proposal schema is locked
around a controlled `(anti_pattern_id, issue_class, proposal_type)` vocab,
and forcing cases into it would either break the invariant or pollute every
existing query. "Wraps and supersedes" is at the workflow layer, not storage.

Upsert is keyed on `case_id` so re-running ingestion with the same
`extraction_run_id` does not double-write rows.

Standalone store — intentionally NOT registered with
`gemynd.shared.database_manager`:

- The eval store has a different lifecycle than the registered web-stores
  (`users`, `review`, `ingest`, …). Those exist for the FastAPI server
  processes; the eval store runs inside the ingest pipeline process and is
  opened per-`run_e2e` call.
- The store is opt-in via `kettle_eval_db_path`. Registering through the
  central manager would require Settings-field plumbing for a feature the
  user might not enable.
- Standalone keeps the dependency-tier rule clean: `kettle_eval` does not
  need to import `shared.database_manager`, only the bare `sqlite3` module.

Connection conventions are mirrored deliberately: WAL mode,
`check_same_thread=False`, `foreign_keys=ON`, `busy_timeout=5000`. If future
v1 work wants to expose the eval queue in the FastAPI review UI, the
migration path is: add `kettle_eval` to `database_manager._MIGRATIONS`,
parameterize `KettleEvalStore.__init__` to accept a managed connection,
leave the standalone path as a fallback.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from gemynd.core.ids import stable_hash

from .models import FAILURE_MODES, SEVERITY_MAX, SEVERITY_MIN, EvaluationCase, ReviewLabel

_log = logging.getLogger(__name__)

# Base tables — must exist before migrations and dependent indexes run.
_BASE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_case (
    case_id TEXT PRIMARY KEY,
    case_family_id TEXT NOT NULL,
    decision_class_id TEXT NOT NULL DEFAULT '',
    case_type TEXT NOT NULL,
    corpus_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    page_id TEXT,
    source_span_id TEXT,
    object_id TEXT NOT NULL,
    extraction_run_id TEXT NOT NULL,

    input_text TEXT,
    pipeline_stage TEXT,
    model_name TEXT,
    prompt_hash TEXT,
    config_hash TEXT,

    predicted_failure_mean REAL NOT NULL DEFAULT 0.0,
    predicted_failure_uncertainty REAL NOT NULL DEFAULT 0.0,
    priority_score REAL NOT NULL DEFAULT 0.0,
    review_status TEXT NOT NULL DEFAULT 'unreviewed',

    risk_features_json TEXT NOT NULL DEFAULT '{}',
    model_output_json TEXT NOT NULL DEFAULT '{}',
    source_image_region_json TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_label (
    label_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES evaluation_case(case_id),
    case_family_id TEXT NOT NULL,
    decision_class_id TEXT NOT NULL DEFAULT '',
    severity INTEGER NOT NULL,
    failure_modes_json TEXT NOT NULL DEFAULT '[]',
    reviewer TEXT NOT NULL DEFAULT 'human',
    corrected_output_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_weight (
    case_type TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    weight REAL NOT NULL,
    label_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (case_type, feature_name)
);
"""

# Indexes — created AFTER any pending column migrations so they can reference
# columns added by `_migrate_schema()` on legacy databases.
_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_eval_case_priority
    ON evaluation_case(priority_score DESC);

CREATE INDEX IF NOT EXISTS idx_eval_case_doc
    ON evaluation_case(document_id);

CREATE INDEX IF NOT EXISTS idx_eval_case_status_priority
    ON evaluation_case(review_status, priority_score DESC);

CREATE INDEX IF NOT EXISTS idx_eval_case_object
    ON evaluation_case(object_id);

CREATE INDEX IF NOT EXISTS idx_eval_case_family
    ON evaluation_case(case_family_id);

CREATE INDEX IF NOT EXISTS idx_eval_case_decision_class
    ON evaluation_case(decision_class_id, priority_score DESC);

CREATE INDEX IF NOT EXISTS idx_review_label_case
    ON review_label(case_id);

CREATE INDEX IF NOT EXISTS idx_review_label_family
    ON review_label(case_family_id);

CREATE INDEX IF NOT EXISTS idx_review_label_class
    ON review_label(decision_class_id);

CREATE INDEX IF NOT EXISTS idx_review_label_severity
    ON review_label(severity);

CREATE INDEX IF NOT EXISTS idx_review_label_created
    ON review_label(created_at);
"""

# Kept as a single concatenated constant for backward-compatible imports
# (database_manager-style registration may reference _SCHEMA_SQL by name).
_SCHEMA_SQL = _BASE_TABLE_SQL + _INDEXES_SQL


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Idempotent column-add for pre-decision_class_id databases.

    SQLite has no IF NOT EXISTS on ADD COLUMN, so we check `PRAGMA
    table_info` and catch the duplicate-column OperationalError as a
    defensive fallback. Must run AFTER the base table exists but BEFORE
    indexes that reference the new column.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_case)")}
    if "decision_class_id" not in cols:
        try:
            conn.execute(
                "ALTER TABLE evaluation_case ADD COLUMN decision_class_id TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # raced with another writer; column already exists


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KettleEvalStore:
    """SQLite store for evaluation cases.

    Mirrors the connection conventions used by gemynd.shared.database_manager:
    WAL mode, check_same_thread=False, foreign_keys=ON, busy_timeout=5000.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Order matters on legacy DBs: base table -> migrate -> indexes.
        self._conn.executescript(_BASE_TABLE_SQL)
        _migrate_schema(self._conn)
        self._conn.executescript(_INDEXES_SQL)
        self._conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            _log.debug("Failed to close kettle_eval connection", exc_info=True)

    def __enter__(self) -> KettleEvalStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def save_cases(self, cases: list[EvaluationCase]) -> int:
        """Idempotent upsert. Returns the number of rows written."""
        if not cases:
            return 0
        now = _now_iso()
        rows: list[tuple] = []
        for case in cases:
            created_at = case.created_at or now
            rows.append((
                case.case_id,
                case.case_family_id,
                case.decision_class_id,
                case.case_type,
                case.corpus_id,
                case.document_id,
                case.page_id,
                case.source_span_id,
                case.object_id,
                case.extraction_run_id,
                case.input_text,
                case.pipeline_stage,
                case.model_name,
                case.prompt_hash,
                case.config_hash,
                case.predicted_failure_mean,
                case.predicted_failure_uncertainty,
                case.priority_score,
                case.review_status,
                json.dumps(case.risk_features, ensure_ascii=True, sort_keys=True),
                json.dumps(case.model_output, ensure_ascii=True, sort_keys=True),
                json.dumps(case.source_image_region) if case.source_image_region is not None else None,
                json.dumps(case.payload, ensure_ascii=True, sort_keys=True),
                created_at,
                now,
            ))
        self._conn.executemany(
            """
            INSERT INTO evaluation_case (
                case_id, case_family_id, decision_class_id,
                case_type, corpus_id, document_id,
                page_id, source_span_id, object_id, extraction_run_id,
                input_text, pipeline_stage, model_name, prompt_hash, config_hash,
                predicted_failure_mean, predicted_failure_uncertainty,
                priority_score, review_status,
                risk_features_json, model_output_json,
                source_image_region_json, payload_json,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?
            )
            ON CONFLICT(case_id) DO UPDATE SET
                case_family_id = excluded.case_family_id,
                decision_class_id = excluded.decision_class_id,
                page_id = excluded.page_id,
                source_span_id = excluded.source_span_id,
                input_text = excluded.input_text,
                pipeline_stage = excluded.pipeline_stage,
                model_name = excluded.model_name,
                prompt_hash = excluded.prompt_hash,
                config_hash = excluded.config_hash,
                predicted_failure_mean = excluded.predicted_failure_mean,
                predicted_failure_uncertainty = excluded.predicted_failure_uncertainty,
                priority_score = excluded.priority_score,
                review_status = excluded.review_status,
                risk_features_json = excluded.risk_features_json,
                model_output_json = excluded.model_output_json,
                source_image_region_json = excluded.source_image_region_json,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_case(self, case_id: str) -> EvaluationCase | None:
        row = self._conn.execute(
            "SELECT * FROM evaluation_case WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_case(row)

    def get_cases_by_family(self, case_family_id: str) -> list[EvaluationCase]:
        rows = self._conn.execute(
            "SELECT * FROM evaluation_case WHERE case_family_id = ? ORDER BY created_at",
            (case_family_id,),
        ).fetchall()
        return [_row_to_case(r) for r in rows]

    def top_k(self, k: int, status: str = "unreviewed") -> list[EvaluationCase]:
        rows = self._conn.execute(
            """
            SELECT * FROM evaluation_case
            WHERE review_status = ?
            ORDER BY priority_score DESC
            LIMIT ?
            """,
            (status, k),
        ).fetchall()
        return [_row_to_case(r) for r in rows]

    def top_k_per_decision_class(
        self, k: int, status: str = "unreviewed",
    ) -> list[tuple[EvaluationCase, int]]:
        """Return up to K (case, cluster_size) pairs, one per decision_class.

        The case picked from each class is the highest-priority row in that
        class. Rows with empty decision_class_id (pre-v0.2 rows that never
        got the column populated) are excluded — they would otherwise lump
        into one giant pseudo-class.

        Returns a list of (case, cluster_size) pairs so the queue UI can show
        "this decision was seen N times in the corpus" alongside the case.
        """
        # SQLite does not guarantee which row a bare aggregate returns, so
        # use a correlated subquery to pick the row with MAX(priority_score)
        # per decision_class_id.
        rows = self._conn.execute(
            """
            SELECT ec.*, sub.cluster_size
            FROM evaluation_case ec
            JOIN (
                SELECT decision_class_id,
                       MAX(priority_score) AS max_priority,
                       COUNT(*) AS cluster_size
                FROM evaluation_case
                WHERE review_status = ?
                  AND decision_class_id != ''
                GROUP BY decision_class_id
            ) sub
              ON ec.decision_class_id = sub.decision_class_id
             AND ec.priority_score = sub.max_priority
            WHERE ec.review_status = ?
              AND ec.decision_class_id != ''
            GROUP BY ec.decision_class_id
            ORDER BY ec.priority_score DESC
            LIMIT ?
            """,
            (status, status, k),
        ).fetchall()
        return [(_row_to_case(r), int(r["cluster_size"])) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM evaluation_case").fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # ReviewLabel methods (v0.3 Level 2 substrate)
    # ------------------------------------------------------------------

    def add_label(self, label: ReviewLabel) -> ReviewLabel:
        """Insert a ReviewLabel and mark the parent case 'reviewed'.

        Validates severity and failure_modes against the controlled vocab.
        If `label.label_id` is empty, assigns a stable_hash-derived id. If
        `label.created_at` is empty, assigns the current UTC iso timestamp.
        Returns the (possibly updated) label so the caller sees the final
        id/timestamp.

        Raises ValueError on bad severity or unknown failure mode.
        """
        if not (SEVERITY_MIN <= label.severity <= SEVERITY_MAX):
            raise ValueError(
                f"severity must be in [{SEVERITY_MIN}, {SEVERITY_MAX}], got {label.severity}"
            )
        unknown = [m for m in label.failure_modes if m not in FAILURE_MODES]
        if unknown:
            raise ValueError(
                f"unknown failure_modes {unknown}; allowed: {sorted(FAILURE_MODES)}"
            )

        created_at = label.created_at or _now_iso()
        label_id = label.label_id or f"label_{stable_hash(label.case_id, label.reviewer, created_at)}"
        label.label_id = label_id
        label.created_at = created_at

        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            cur.execute(
                """
                INSERT INTO review_label (
                    label_id, case_id, case_family_id, decision_class_id,
                    severity, failure_modes_json, reviewer,
                    corrected_output_json, notes, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label_id,
                    label.case_id,
                    label.case_family_id,
                    label.decision_class_id,
                    int(label.severity),
                    json.dumps(sorted(set(label.failure_modes)), ensure_ascii=True),
                    label.reviewer,
                    json.dumps(label.corrected_output, ensure_ascii=True, sort_keys=True),
                    label.notes,
                    float(label.confidence),
                    created_at,
                ),
            )
            cur.execute(
                "UPDATE evaluation_case SET review_status = 'reviewed', updated_at = ? WHERE case_id = ?",
                (created_at, label.case_id),
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return label

    def get_labels_for_case(self, case_id: str) -> list[ReviewLabel]:
        rows = self._conn.execute(
            "SELECT * FROM review_label WHERE case_id = ? ORDER BY created_at",
            (case_id,),
        ).fetchall()
        return [_row_to_label(r) for r in rows]

    def get_labels_since(self, since_iso: str | None = None) -> list[ReviewLabel]:
        if since_iso is None:
            rows = self._conn.execute(
                "SELECT * FROM review_label ORDER BY created_at"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM review_label WHERE created_at >= ? ORDER BY created_at",
                (since_iso,),
            ).fetchall()
        return [_row_to_label(r) for r in rows]

    def label_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM review_label").fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # Feature-weight methods (v0.3 adaptive weights)
    # ------------------------------------------------------------------

    def get_feature_weights(self) -> dict[str, dict[str, float]]:
        """Return persisted weights as {case_type: {feature_name: weight}}.

        Empty dict on a fresh DB. Caller is expected to overlay these on
        top of the module-level defaults in selector.py.
        """
        rows = self._conn.execute(
            "SELECT case_type, feature_name, weight FROM feature_weight"
        ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            out.setdefault(r["case_type"], {})[r["feature_name"]] = float(r["weight"])
        return out

    def upsert_feature_weight(
        self,
        case_type: str,
        feature_name: str,
        weight: float,
        increment_label_count: bool = True,
    ) -> None:
        """Persist or update a learned feature weight.

        `increment_label_count` defaults to True so callers don't have to
        track it. Pass False when seeding a baseline.
        """
        now = _now_iso()
        delta = 1 if increment_label_count else 0
        self._conn.execute(
            """
            INSERT INTO feature_weight (case_type, feature_name, weight, label_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(case_type, feature_name) DO UPDATE SET
                weight = excluded.weight,
                label_count = feature_weight.label_count + ?,
                updated_at = excluded.updated_at
            """,
            (case_type, feature_name, float(weight), delta, now, delta),
        )
        self._conn.commit()

    def reset_feature_weights(self) -> int:
        cur = self._conn.execute("DELETE FROM feature_weight")
        self._conn.commit()
        return cur.rowcount


def _row_to_case(row: sqlite3.Row) -> EvaluationCase:
    region_json = row["source_image_region_json"]
    # decision_class_id is read via dict lookup so pre-migration rows
    # (without the column in their row factory) fall back to "".
    decision_class_id = ""
    try:
        decision_class_id = row["decision_class_id"] or ""
    except (IndexError, KeyError):
        pass
    return EvaluationCase(
        case_id=row["case_id"],
        case_family_id=row["case_family_id"],
        decision_class_id=decision_class_id,
        case_type=row["case_type"],
        corpus_id=row["corpus_id"],
        document_id=row["document_id"],
        page_id=row["page_id"],
        source_span_id=row["source_span_id"],
        object_id=row["object_id"],
        extraction_run_id=row["extraction_run_id"],
        input_text=row["input_text"] or "",
        pipeline_stage=row["pipeline_stage"] or "",
        model_name=row["model_name"] or "",
        prompt_hash=row["prompt_hash"] or "",
        config_hash=row["config_hash"] or "",
        predicted_failure_mean=float(row["predicted_failure_mean"]),
        predicted_failure_uncertainty=float(row["predicted_failure_uncertainty"]),
        priority_score=float(row["priority_score"]),
        review_status=row["review_status"],
        risk_features=json.loads(row["risk_features_json"] or "{}"),
        model_output=json.loads(row["model_output_json"] or "{}"),
        source_image_region=json.loads(region_json) if region_json else None,
        payload=json.loads(row["payload_json"] or "{}"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_label(row: sqlite3.Row) -> ReviewLabel:
    return ReviewLabel(
        label_id=row["label_id"],
        case_id=row["case_id"],
        case_family_id=row["case_family_id"],
        decision_class_id=row["decision_class_id"] or "",
        severity=int(row["severity"]),
        failure_modes=json.loads(row["failure_modes_json"] or "[]"),
        reviewer=row["reviewer"],
        corrected_output=json.loads(row["corrected_output_json"] or "{}"),
        notes=row["notes"] or "",
        confidence=float(row["confidence"]),
        created_at=row["created_at"],
    )
