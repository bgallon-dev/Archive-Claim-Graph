"""Background sensitivity monitor.

Runs the same PII and Indigenous cultural sensitivity detection logic as the
ingestion-time detector (sensitivity_monitor.py), but operates against live
claim text fetched from Neo4j in batches rather than against bundle objects.

Intended as a scheduled or on-demand background job via:
    graphrag sensitivity-scan [--institution-id ID] [--output report.json]

Use cases:
  - Scan existing claims after new detection rules are added to the vocabulary
  - Re-scan after OCR corrections that changed claim text
  - Initial scan of a collection ingested before the detector was installed
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..graph.cypher import (
    QUARANTINE_CLAIM_QUERY,
    SENSITIVITY_SCAN_BATCH_QUERY,
)
from ..retrieval.executor import Neo4jQueryExecutor
from .detectors.sensitivity_monitor import (
    _load_config,
    _load_vocabulary,
    detect_indigenous_in_text,
    detect_pii_in_text,
)
from .ids import make_proposal_id, make_review_run_id, make_revision_id, patch_spec_fingerprint
from .models import Proposal, ProposalRevision, ProposalTarget, ReviewRun
from .patch_spec import make_patch_spec
from .store import ReviewStore

_BATCH_SIZE = 1000
_SCAN_RUN_DOC_ID = "background_sensitivity_scan"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SensitivityMonitor:
    """Background sensitivity scanner operating against live Neo4j claim data.

    Parameters
    ----------
    executor:
        Connected Neo4jQueryExecutor for querying and quarantining claims.
    store:
        ReviewStore for persisting quarantine proposals and review runs.
    resources_dir:
        Optional override for the resources directory path. Defaults to the
        graphrag_pipeline/resources/ directory relative to this module.
    """

    def __init__(
        self,
        executor: Neo4jQueryExecutor,
        store: ReviewStore,
        resources_dir: Path | None = None,
    ) -> None:
        self._executor = executor
        self._store = store
        self._config = _load_config()
        self._vocab_entries = _load_vocabulary()

    def run_full_scan(self, institution_id: str | None = None) -> dict[str, Any]:
        """Scan all active claims in the graph for PII and Indigenous sensitivity.

        Fetches claims in batches of 1000, applies pattern detection, auto-quarantines
        high-confidence flags, and persists quarantine proposals to the ReviewStore.

        Living person detection is not performed in background scan mode because
        it requires entity-relationship context that is expensive to reconstruct
        per-claim from the graph.

        Returns
        -------
        dict with keys: scanned, flagged, quarantined, proposals_created
        """
        now = _now_iso()
        scan_id = f"scan_{uuid.uuid4().hex[:16]}"

        # Create a synthetic ReviewRun to anchor background-scan proposals.
        review_run = ReviewRun(
            review_run_id=make_review_run_id(scan_id, now),
            snapshot_id=scan_id,
            doc_id=_SCAN_RUN_DOC_ID,
            structure_bundle_path="",
            semantic_bundle_path="",
            structure_bundle_sha256="",
            semantic_bundle_sha256="",
            snapshot_schema_version="v1",
            extraction_run_id="",
            created_at=now,
        )
        self._store.save_review_run(review_run)

        pii_cfg = self._config.get("pii_detection", {})
        indigenous_cfg = self._config.get("indigenous_sensitivity", {})
        pii_enabled = pii_cfg.get("enabled", True)
        indigenous_enabled = indigenous_cfg.get("enabled", True) and bool(self._vocab_entries)
        pii_threshold = self._config.get("pii_detection", {}).get("auto_quarantine_threshold", 0.85)
        indigenous_threshold = indigenous_cfg.get("auto_quarantine_threshold", 0.90)

        results: dict[str, int] = {
            "scanned": 0,
            "flagged": 0,
            "quarantined": 0,
            "proposals_created": 0,
        }

        offset = 0
        while True:
            batch = self._executor.run(
                SENSITIVITY_SCAN_BATCH_QUERY,
                {"institution_id": institution_id, "offset": offset, "batch_size": _BATCH_SIZE},
            )
            if not batch:
                break

            for row in batch:
                claim_id = row.get("claim_id") or ""
                source_sentence = row.get("source_sentence") or ""
                results["scanned"] += 1

                flags: list[tuple[str, str, float]] = []  # (issue_class, reason_detail, confidence)

                if pii_enabled:
                    for label, confidence in detect_pii_in_text(source_sentence, pii_cfg):
                        flags.append(("pii_exposure", f"pii_exposure:{label}", confidence))

                if indigenous_enabled:
                    for match in detect_indigenous_in_text(source_sentence, self._vocab_entries):
                        flags.append((
                            "indigenous_sensitivity",
                            f"indigenous_sensitivity:{match['category']}",
                            0.90,
                        ))

                if not flags:
                    continue

                results["flagged"] += 1

                for issue_class, reason_detail, confidence in flags:
                    # Determine anti-pattern and threshold.
                    anti_pattern_id = {
                        "pii_exposure": "ap_pii_exposure",
                        "indigenous_sensitivity": "ap_indigenous_sensitivity",
                    }.get(issue_class, "ap_pii_exposure")
                    threshold = pii_threshold if issue_class == "pii_exposure" else indigenous_threshold

                    # Auto-quarantine high-confidence flags directly in the graph.
                    if confidence >= threshold:
                        self._quarantine_claim(claim_id, issue_class)
                        results["quarantined"] += 1

                    # Persist proposal for archivist review.
                    patch = make_patch_spec(
                        "quarantine_claim",
                        claim_id=claim_id,
                        reason=reason_detail,
                    )
                    target_tuples = [("claim", claim_id, "flagged_claim")]
                    proposal_id = make_proposal_id(
                        scan_id, issue_class, "quarantine_claim", target_tuples, patch
                    )
                    proposal = Proposal(
                        proposal_id=proposal_id,
                        review_run_id=review_run.review_run_id,
                        snapshot_id=scan_id,
                        anti_pattern_id=anti_pattern_id,
                        issue_class=issue_class,
                        proposal_type="quarantine_claim",
                        status="queued",
                        confidence=confidence,
                        priority_score=min(1.0, 0.7 * confidence + 0.3),
                        impact_size=1,
                        current_revision_id="",
                        created_at=now,
                        updated_at=now,
                    )
                    targets = [
                        ProposalTarget(
                            proposal_id=proposal_id,
                            target_kind="claim",
                            target_id=claim_id,
                            target_role="flagged_claim",
                            exists_in_snapshot=True,
                        )
                    ]
                    self._store.upsert_proposal(proposal, targets)

                    revision_number = self._store.next_revision_number(proposal_id)
                    revision_id = make_revision_id(proposal_id, revision_number)
                    fp = patch_spec_fingerprint(patch)
                    revision = ProposalRevision(
                        revision_id=revision_id,
                        proposal_id=proposal_id,
                        revision_number=revision_number,
                        revision_kind="generated",
                        patch_spec_json=json.dumps(patch),
                        patch_spec_fingerprint=fp,
                        evidence_snapshot_json=json.dumps({"source_sentence": source_sentence[:200]}),
                        reasoning_summary_json=json.dumps({"summary": reason_detail}),
                        detector_name="sensitivity_monitor_background",
                        detector_version="v1",
                        created_at=now,
                    )
                    self._store.save_revision(revision)
                    self._store.update_proposal_revision(proposal_id, revision_id)
                    results["proposals_created"] += 1

            offset += _BATCH_SIZE

        return results

    def _quarantine_claim(self, claim_id: str, reason: str) -> None:
        """Set quarantine_status='quarantined' on the Claim node in the graph."""
        self._executor.run(
            QUARANTINE_CLAIM_QUERY,
            {
                "claim_id": claim_id,
                "reason": reason,
                "quarantine_timestamp": _now_iso(),
            },
        )
