"""SQLite-backed review store for proposals, revisions, and correction events."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    ISSUE_CLASSES,
    PROPOSAL_STATUSES,
    REVIEW_TIERS,
    AntiPatternClass,
    CorrectionEvent,
    Proposal,
    ProposalRevision,
    ProposalTarget,
    ReviewRun,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS anti_pattern_class (
    anti_pattern_id TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    queue_name      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_run (
    review_run_id           TEXT PRIMARY KEY,
    snapshot_id             TEXT NOT NULL,
    doc_id                  TEXT NOT NULL,
    structure_bundle_path   TEXT NOT NULL,
    semantic_bundle_path    TEXT NOT NULL,
    structure_bundle_sha256 TEXT NOT NULL,
    semantic_bundle_sha256  TEXT NOT NULL,
    snapshot_schema_version TEXT NOT NULL DEFAULT 'v1',
    extraction_run_id       TEXT NOT NULL DEFAULT '',
    created_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposal (
    proposal_id         TEXT PRIMARY KEY,
    review_run_id       TEXT NOT NULL REFERENCES review_run(review_run_id),
    snapshot_id         TEXT NOT NULL,
    anti_pattern_id     TEXT NOT NULL REFERENCES anti_pattern_class(anti_pattern_id),
    issue_class         TEXT NOT NULL,
    proposal_type       TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'queued',
    confidence          REAL NOT NULL DEFAULT 0.0,
    priority_score      REAL NOT NULL DEFAULT 0.0,
    impact_size         INTEGER NOT NULL DEFAULT 0,
    current_revision_id TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    review_tier         TEXT NOT NULL DEFAULT 'needs_review'
);

CREATE TABLE IF NOT EXISTS proposal_target (
    proposal_id       TEXT NOT NULL REFERENCES proposal(proposal_id),
    target_kind       TEXT NOT NULL,
    target_id         TEXT NOT NULL,
    target_role       TEXT NOT NULL,
    exists_in_snapshot INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (proposal_id, target_kind, target_id, target_role)
);

CREATE TABLE IF NOT EXISTS proposal_revision (
    revision_id            TEXT PRIMARY KEY,
    proposal_id            TEXT NOT NULL REFERENCES proposal(proposal_id),
    revision_number        INTEGER NOT NULL,
    revision_kind          TEXT NOT NULL,
    patch_spec_json        TEXT NOT NULL,
    patch_spec_fingerprint TEXT NOT NULL,
    evidence_snapshot_json TEXT NOT NULL DEFAULT '{}',
    live_context_json      TEXT NOT NULL DEFAULT '{}',
    reasoning_summary_json TEXT NOT NULL DEFAULT '{}',
    detector_name          TEXT NOT NULL DEFAULT '',
    detector_version       TEXT NOT NULL DEFAULT '',
    generator_type         TEXT NOT NULL DEFAULT 'heuristic',
    llm_provider           TEXT,
    llm_model              TEXT,
    validator_version      TEXT NOT NULL DEFAULT 'v1',
    validation_state       TEXT NOT NULL DEFAULT 'valid',
    created_by             TEXT NOT NULL DEFAULT 'system',
    created_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS correction_event (
    event_id      TEXT PRIMARY KEY,
    proposal_id   TEXT NOT NULL REFERENCES proposal(proposal_id),
    revision_id   TEXT NOT NULL REFERENCES proposal_revision(revision_id),
    action        TEXT NOT NULL,
    reviewer      TEXT NOT NULL DEFAULT '',
    reviewer_note TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proposal_snapshot ON proposal(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_proposal_status ON proposal(status);
CREATE INDEX IF NOT EXISTS idx_proposal_issue_class ON proposal(issue_class);
CREATE INDEX IF NOT EXISTS idx_proposal_priority ON proposal(priority_score DESC, impact_size DESC, confidence DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_proposal_review_run ON proposal(review_run_id);
CREATE INDEX IF NOT EXISTS idx_revision_proposal ON proposal_revision(proposal_id);
CREATE INDEX IF NOT EXISTS idx_target_proposal ON proposal_target(proposal_id);
CREATE INDEX IF NOT EXISTS idx_correction_proposal ON correction_event(proposal_id);
"""

# Default anti-pattern classes for v1
_DEFAULT_ANTI_PATTERNS: list[dict[str, str]] = [
    {"anti_pattern_id": "ap_ocr_spelling", "name": "OCR Spelling Variant", "description": "Entity duplicates or OCR-corrupted entity names", "queue_name": "ocr_entity"},
    {"anti_pattern_id": "ap_duplicate_alias", "name": "Duplicate Entity Alias", "description": "Entity names that are aliases of the same canonical entity", "queue_name": "ocr_entity"},
    {"anti_pattern_id": "ap_header_contamination", "name": "Header Contamination", "description": "Mentions extracted from page headers or footers", "queue_name": "junk_mention"},
    {"anti_pattern_id": "ap_boilerplate_contamination", "name": "Boilerplate Contamination", "description": "Mentions from boilerplate text (letterhead, form labels)", "queue_name": "junk_mention"},
    {"anti_pattern_id": "ap_short_generic", "name": "Short Generic Token", "description": "Short generic tokens falsely detected as entity mentions", "queue_name": "junk_mention"},
    {"anti_pattern_id": "ap_ocr_garbage", "name": "OCR Garbage Mention", "description": "Mentions that are OCR noise, not real text", "queue_name": "junk_mention"},
    {"anti_pattern_id": "ap_missing_species", "name": "Missing Species Focus", "description": "Claim lacks SPECIES_FOCUS link despite evidence of species context", "queue_name": "builder_repair"},
    {"anti_pattern_id": "ap_missing_location", "name": "Missing Event Location", "description": "Claim lacks OCCURRED_AT link despite evidence of location context", "queue_name": "builder_repair"},
    {"anti_pattern_id": "ap_method_overtrigger", "name": "Method Overtrigger", "description": "METHOD_FOCUS link on a claim type where method focus is weak or forbidden", "queue_name": "builder_repair"},
    # Sensitivity monitor anti-patterns
    {"anti_pattern_id": "ap_pii_exposure", "name": "PII Exposure", "description": "Personally identifiable information detected in extracted claims", "queue_name": "sensitivity"},
    {"anti_pattern_id": "ap_indigenous_sensitivity", "name": "Indigenous Cultural Sensitivity", "description": "Potential Indigenous cultural material requiring community review", "queue_name": "sensitivity"},
    {"anti_pattern_id": "ap_living_person", "name": "Living Person Reference", "description": "Reference to potentially living individual with sensitive information", "queue_name": "sensitivity"},
]

QUEUE_NAMES: frozenset[str] = frozenset(
    d["queue_name"] for d in _DEFAULT_ANTI_PATTERNS if "queue_name" in d
)

ISSUE_CLASS_TO_ANTI_PATTERN: dict[str, str] = {
    "ocr_spelling_variant": "ap_ocr_spelling",
    "duplicate_entity_alias": "ap_duplicate_alias",
    "header_contamination": "ap_header_contamination",
    "boilerplate_contamination": "ap_boilerplate_contamination",
    "short_generic_token": "ap_short_generic",
    "ocr_garbage_mention": "ap_ocr_garbage",
    "missing_species_focus": "ap_missing_species",
    "missing_event_location": "ap_missing_location",
    "method_overtrigger": "ap_method_overtrigger",
    # Sensitivity monitor mappings
    "pii_exposure": "ap_pii_exposure",
    "indigenous_sensitivity": "ap_indigenous_sensitivity",
    "living_person_reference": "ap_living_person",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    """SQLite-backed store for the anti-pattern review subsystem."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._migrate()

    def _create_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        for ap in _DEFAULT_ANTI_PATTERNS:
            self._conn.execute(
                "INSERT OR IGNORE INTO anti_pattern_class (anti_pattern_id, name, description, queue_name) VALUES (?, ?, ?, ?)",
                (ap["anti_pattern_id"], ap["name"], ap["description"], ap["queue_name"]),
            )
        self._conn.commit()

    def _migrate(self) -> None:
        """Apply schema migrations to existing databases."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(proposal_target)")}
        if "reviewer_override" not in cols:
            self._conn.execute(
                "ALTER TABLE proposal_target ADD COLUMN reviewer_override TEXT DEFAULT NULL"
            )
            self._conn.commit()
        proposal_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(proposal)")}
        if "review_tier" not in proposal_cols:
            self._conn.execute(
                "ALTER TABLE proposal ADD COLUMN review_tier TEXT NOT NULL DEFAULT 'needs_review'"
            )
            self._conn.commit()
        ce_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(correction_event)")}
        if "error_root_cause" not in ce_cols:
            self._conn.execute(
                "ALTER TABLE correction_event ADD COLUMN error_root_cause TEXT NOT NULL DEFAULT ''"
            )
            self._conn.execute(
                "ALTER TABLE correction_event ADD COLUMN error_type TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- Review runs ---------------------------------------------------------

    def save_review_run(self, run: ReviewRun) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO review_run
               (review_run_id, snapshot_id, doc_id, structure_bundle_path,
                semantic_bundle_path, structure_bundle_sha256, semantic_bundle_sha256,
                snapshot_schema_version, extraction_run_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.review_run_id, run.snapshot_id, run.doc_id,
                run.structure_bundle_path, run.semantic_bundle_path,
                run.structure_bundle_sha256, run.semantic_bundle_sha256,
                run.snapshot_schema_version, run.extraction_run_id, run.created_at,
            ),
        )
        self._conn.commit()

    def get_review_run(self, review_run_id: str) -> ReviewRun | None:
        row = self._conn.execute(
            "SELECT * FROM review_run WHERE review_run_id = ?", (review_run_id,)
        ).fetchone()
        if not row:
            return None
        return ReviewRun(**dict(row))

    # -- Proposals -----------------------------------------------------------

    def upsert_proposal(self, proposal: Proposal, targets: list[ProposalTarget]) -> None:
        now = _now_iso()
        existing = self._conn.execute(
            "SELECT proposal_id FROM proposal WHERE proposal_id = ?", (proposal.proposal_id,)
        ).fetchone()
        if existing:
            self._conn.execute(
                """UPDATE proposal SET
                   review_run_id=?, snapshot_id=?, anti_pattern_id=?, issue_class=?,
                   proposal_type=?, confidence=?, priority_score=?, impact_size=?,
                   current_revision_id=?, updated_at=?, review_tier=?
                   WHERE proposal_id=?""",
                (
                    proposal.review_run_id, proposal.snapshot_id, proposal.anti_pattern_id,
                    proposal.issue_class, proposal.proposal_type, proposal.confidence,
                    proposal.priority_score, proposal.impact_size,
                    proposal.current_revision_id, now, proposal.review_tier, proposal.proposal_id,
                ),
            )
        else:
            self._conn.execute(
                """INSERT INTO proposal
                   (proposal_id, review_run_id, snapshot_id, anti_pattern_id, issue_class,
                    proposal_type, status, confidence, priority_score, impact_size,
                    current_revision_id, created_at, updated_at, review_tier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    proposal.proposal_id, proposal.review_run_id, proposal.snapshot_id,
                    proposal.anti_pattern_id, proposal.issue_class, proposal.proposal_type,
                    proposal.status, proposal.confidence, proposal.priority_score,
                    proposal.impact_size, proposal.current_revision_id,
                    proposal.created_at or now, now, proposal.review_tier,
                ),
            )
        # Replace targets
        self._conn.execute(
            "DELETE FROM proposal_target WHERE proposal_id = ?", (proposal.proposal_id,)
        )
        for target in targets:
            self._conn.execute(
                """INSERT INTO proposal_target
                   (proposal_id, target_kind, target_id, target_role, exists_in_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    proposal.proposal_id, target.target_kind, target.target_id,
                    target.target_role, int(target.exists_in_snapshot),
                ),
            )
        self._conn.commit()

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        row = self._conn.execute(
            "SELECT * FROM proposal WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if not row:
            return None
        return Proposal(**dict(row))

    def get_proposal_targets(self, proposal_id: str) -> list[ProposalTarget]:
        rows = self._conn.execute(
            "SELECT * FROM proposal_target WHERE proposal_id = ?", (proposal_id,)
        ).fetchall()
        return [
            ProposalTarget(
                proposal_id=r["proposal_id"],
                target_kind=r["target_kind"],
                target_id=r["target_id"],
                target_role=r["target_role"],
                exists_in_snapshot=bool(r["exists_in_snapshot"]),
                reviewer_override=r["reviewer_override"],
            )
            for r in rows
        ]

    def set_target_override(
        self,
        proposal_id: str,
        target_kind: str,
        target_id: str,
        override: str | None,
    ) -> None:
        self._conn.execute(
            """UPDATE proposal_target SET reviewer_override = ?
               WHERE proposal_id = ? AND target_kind = ? AND target_id = ?""",
            (override, proposal_id, target_kind, target_id),
        )
        self._conn.commit()

    def list_proposals(
        self,
        *,
        status: str | None = None,
        issue_class: str | None = None,
        queue_name: str | None = None,
        snapshot_id: str | None = None,
        doc_id: str | None = None,
        detector_name: str | None = None,
        review_tier: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Proposal]:
        if status and status not in PROPOSAL_STATUSES:
            raise ValueError(f"Invalid status: {status!r}")
        if issue_class and issue_class not in ISSUE_CLASSES:
            raise ValueError(f"Invalid issue_class: {issue_class!r}")
        if queue_name and queue_name not in QUEUE_NAMES:
            raise ValueError(f"Invalid queue_name: {queue_name!r}")
        if review_tier and review_tier not in REVIEW_TIERS:
            raise ValueError(f"Invalid review_tier: {review_tier!r}")
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("p.status = ?")
            params.append(status)
        if issue_class:
            clauses.append("p.issue_class = ?")
            params.append(issue_class)
        if queue_name:
            clauses.append("a.queue_name = ?")
            params.append(queue_name)
        if snapshot_id:
            clauses.append("p.snapshot_id = ?")
            params.append(snapshot_id)
        if doc_id:
            clauses.append("r.doc_id = ?")
            params.append(doc_id)
        if review_tier:
            clauses.append("p.review_tier = ?")
            params.append(review_tier)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT p.* FROM proposal p
            JOIN anti_pattern_class a ON p.anti_pattern_id = a.anti_pattern_id
            JOIN review_run r ON p.review_run_id = r.review_run_id
            {where}
            ORDER BY p.priority_score DESC, p.impact_size DESC,
                     p.confidence DESC, p.created_at ASC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [Proposal(**dict(row)) for row in rows]

    def update_proposal_status(self, proposal_id: str, new_status: str) -> None:
        self._conn.execute(
            "UPDATE proposal SET status = ?, updated_at = ? WHERE proposal_id = ?",
            (new_status, _now_iso(), proposal_id),
        )
        self._conn.commit()

    def update_proposal_revision(self, proposal_id: str, revision_id: str) -> None:
        self._conn.execute(
            "UPDATE proposal SET current_revision_id = ?, updated_at = ? WHERE proposal_id = ?",
            (revision_id, _now_iso(), proposal_id),
        )
        self._conn.commit()

    # -- Revisions -----------------------------------------------------------

    def save_revision(self, revision: ProposalRevision) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO proposal_revision
               (revision_id, proposal_id, revision_number, revision_kind,
                patch_spec_json, patch_spec_fingerprint,
                evidence_snapshot_json, live_context_json, reasoning_summary_json,
                detector_name, detector_version, generator_type,
                llm_provider, llm_model, validator_version, validation_state,
                created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                revision.revision_id, revision.proposal_id, revision.revision_number,
                revision.revision_kind, revision.patch_spec_json,
                revision.patch_spec_fingerprint,
                revision.evidence_snapshot_json, revision.live_context_json,
                revision.reasoning_summary_json,
                revision.detector_name, revision.detector_version,
                revision.generator_type, revision.llm_provider, revision.llm_model,
                revision.validator_version, revision.validation_state,
                revision.created_by, revision.created_at or _now_iso(),
            ),
        )
        self._conn.commit()

    def get_revisions(self, proposal_id: str) -> list[ProposalRevision]:
        rows = self._conn.execute(
            "SELECT * FROM proposal_revision WHERE proposal_id = ? ORDER BY revision_number",
            (proposal_id,),
        ).fetchall()
        return [ProposalRevision(**dict(row)) for row in rows]

    def get_latest_revision(self, proposal_id: str) -> ProposalRevision | None:
        row = self._conn.execute(
            "SELECT * FROM proposal_revision WHERE proposal_id = ? ORDER BY revision_number DESC LIMIT 1",
            (proposal_id,),
        ).fetchone()
        if not row:
            return None
        return ProposalRevision(**dict(row))

    def next_revision_number(self, proposal_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(revision_number) AS max_rev FROM proposal_revision WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row and row["max_rev"] is not None:
            return row["max_rev"] + 1
        return 1

    # -- Correction events ---------------------------------------------------

    def save_correction_event(self, event: CorrectionEvent) -> None:
        self._conn.execute(
            """INSERT INTO correction_event
               (event_id, proposal_id, revision_id, action, reviewer, reviewer_note,
                error_root_cause, error_type, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id, event.proposal_id, event.revision_id,
                event.action, event.reviewer, event.reviewer_note,
                event.error_root_cause, event.error_type,
                event.created_at or _now_iso(),
            ),
        )
        self._conn.commit()

    def correction_event_counts(self, since_iso: str | None = None) -> dict[str, int]:
        """Return {action: count} for correction events, optionally since a timestamp.

        Pass an ISO timestamp (e.g. today midnight UTC) to get today-only totals.
        """
        if since_iso:
            rows = self._conn.execute(
                "SELECT action, COUNT(*) AS cnt FROM correction_event "
                "WHERE created_at >= ? GROUP BY action",
                (since_iso,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT action, COUNT(*) AS cnt FROM correction_event GROUP BY action"
            ).fetchall()
        return {r["action"]: r["cnt"] for r in rows}

    def get_correction_events(self, proposal_id: str) -> list[CorrectionEvent]:
        rows = self._conn.execute(
            "SELECT * FROM correction_event WHERE proposal_id = ? ORDER BY created_at",
            (proposal_id,),
        ).fetchall()
        return [CorrectionEvent(**dict(row)) for row in rows]

    # -- Anti-pattern classes ------------------------------------------------

    def get_anti_pattern_classes(self) -> list[AntiPatternClass]:
        rows = self._conn.execute("SELECT * FROM anti_pattern_class").fetchall()
        return [AntiPatternClass(**dict(row)) for row in rows]

    # -- Stats / counts ------------------------------------------------------

    def proposal_counts_by_status(self, snapshot_id: str | None = None) -> dict[str, int]:
        if snapshot_id:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) as cnt FROM proposal WHERE snapshot_id = ? GROUP BY status",
                (snapshot_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) as cnt FROM proposal GROUP BY status"
            ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def proposal_counts_by_issue_class(
        self,
        status: str | None = None,
        snapshot_id: str | None = None,
    ) -> dict[str, dict[str, object]]:
        """Return {issue_class: {count, avg_confidence}} for queued proposals.

        Used by the batch review summary table in the list view.
        """
        clauses: list[str] = []
        params: list[object] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if snapshot_id:
            clauses.append("snapshot_id = ?")
            params.append(snapshot_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT issue_class, COUNT(*) AS cnt, AVG(confidence) AS avg_conf "
            f"FROM proposal {where} GROUP BY issue_class ORDER BY cnt DESC",
            params,
        ).fetchall()
        return {
            r["issue_class"]: {"count": r["cnt"], "avg_confidence": round(r["avg_conf"], 3)}
            for r in rows
        }

    def proposal_counts_by_tier(self, snapshot_id: str | None = None) -> dict[str, int]:
        if snapshot_id:
            rows = self._conn.execute(
                "SELECT review_tier, COUNT(*) as cnt FROM proposal WHERE snapshot_id = ? GROUP BY review_tier",
                (snapshot_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT review_tier, COUNT(*) as cnt FROM proposal GROUP BY review_tier"
            ).fetchall()
        return {r["review_tier"]: r["cnt"] for r in rows}

    def proposal_counts_by_queue(self, snapshot_id: str | None = None) -> dict[str, int]:
        params: list[Any] = []
        where = ""
        if snapshot_id:
            where = "WHERE p.snapshot_id = ?"
            params.append(snapshot_id)
        rows = self._conn.execute(
            f"""SELECT a.queue_name, COUNT(*) as cnt FROM proposal p
                JOIN anti_pattern_class a ON p.anti_pattern_id = a.anti_pattern_id
                {where}
                GROUP BY a.queue_name""",
            params,
        ).fetchall()
        return {r["queue_name"]: r["cnt"] for r in rows}

    # -- Export helpers -------------------------------------------------------

    def export_proposals_json(
        self, *, status: str | None = None, snapshot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        proposals = self.list_proposals(status=status, snapshot_id=snapshot_id, limit=10000)
        result = []
        for p in proposals:
            entry = p.to_dict()
            entry["targets"] = [t.to_dict() for t in self.get_proposal_targets(p.proposal_id)]
            entry["revisions"] = [r.to_dict() for r in self.get_revisions(p.proposal_id)]
            entry["correction_events"] = [e.to_dict() for e in self.get_correction_events(p.proposal_id)]
            result.append(entry)
        return result

    def export_training_data(
        self,
        *,
        status: str | None = None,
        snapshot_id: str | None = None,
        with_classification_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Export correction events paired with evidence for model training.

        Each returned dict contains the proposal's issue_class, the reviewer's
        action and error classification, and the full evidence snapshot.
        """
        import json as _json

        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("p.status = ?")
            params.append(status)
        if snapshot_id:
            clauses.append("p.snapshot_id = ?")
            params.append(snapshot_id)
        if with_classification_only:
            clauses.append("ce.error_root_cause != ''")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        rows = self._conn.execute(
            f"""SELECT
                    ce.event_id,
                    ce.proposal_id,
                    ce.action,
                    ce.reviewer,
                    ce.reviewer_note,
                    ce.error_root_cause,
                    ce.error_type,
                    ce.created_at AS event_created_at,
                    p.issue_class,
                    p.proposal_type,
                    p.confidence,
                    p.priority_score,
                    pr.evidence_snapshot_json,
                    pr.patch_spec_json
                FROM correction_event ce
                JOIN proposal p ON ce.proposal_id = p.proposal_id
                LEFT JOIN proposal_revision pr ON ce.revision_id = pr.revision_id
                {where}
                ORDER BY ce.created_at""",
            params,
        ).fetchall()

        result: list[dict[str, Any]] = []
        for r in rows:
            row = dict(r)
            # Parse JSON blobs into dicts
            try:
                row["evidence"] = _json.loads(row.pop("evidence_snapshot_json", "{}") or "{}")
            except Exception:
                row["evidence"] = {}
            try:
                row["patch_spec"] = _json.loads(row.pop("patch_spec_json", "{}") or "{}")
            except Exception:
                row["patch_spec"] = {}
            result.append(row)
        return result

    def export_accepted_patches(self, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        proposals = self.list_proposals(status="accepted_pending_apply", snapshot_id=snapshot_id, limit=10000)
        patches = []
        for p in proposals:
            rev = self.get_latest_revision(p.proposal_id)
            if not rev:
                continue
            patches.append({
                "proposal_id": p.proposal_id,
                "issue_class": p.issue_class,
                "proposal_type": p.proposal_type,
                "patch_spec": json.loads(rev.patch_spec_json),
                "confidence": p.confidence,
                "priority_score": p.priority_score,
            })
        return patches
