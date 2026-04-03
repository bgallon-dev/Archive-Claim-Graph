"""Sensitivity gate implementations for the extract/persist pipeline split."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from graphrag_pipeline.core.models import SemanticBundle, StructureBundle
from graphrag_pipeline.ingest.extraction_result import (
    GateResult,
    QuarantineSummary,
)

_log = logging.getLogger(__name__)


class DefaultSensitivityGate:
    """Wraps :func:`sensitivity_monitor.detect` with auto-quarantine logic.

    On detector failure, quarantines ALL active claims as a precaution —
    preserving the fail-safe behavior of the original inline implementation.
    """

    def __call__(
        self,
        structure: StructureBundle,
        semantic: SemanticBundle,
    ) -> GateResult:
        now_ts = datetime.now(timezone.utc).isoformat()
        quarantined_ids: list[str] = []

        try:
            from graphrag_pipeline.review.detectors import sensitivity_monitor as _sm

            proposals = _sm.detect(structure, semantic, snapshot_id="")
            cfg = _sm._load_config()
            threshold = (
                cfg.get("pii_detection", {}).get("auto_quarantine_threshold", 0.85)
            )

            seen: set[str] = set()
            for prop in proposals:
                if prop.confidence >= threshold:
                    for target in prop.targets:
                        if target.target_kind == "claim" and target.target_id not in seen:
                            for claim in semantic.claims:
                                if claim.claim_id == target.target_id:
                                    claim.quarantine_status = "quarantined"
                                    claim.quarantine_reason = prop.issue_class
                                    claim.quarantine_timestamp = now_ts
                                    quarantined_ids.append(claim.claim_id)
                                    seen.add(claim.claim_id)

            return GateResult(
                quarantine_summary=QuarantineSummary(
                    total_claims=len(semantic.claims),
                    quarantined_count=len(quarantined_ids),
                    quarantined_ids=quarantined_ids,
                ),
                semantic=semantic,
            )

        except Exception as exc:
            _log.error(
                "Sensitivity gate failed: %s — quarantining all claims as precaution",
                exc,
                exc_info=True,
            )
            for claim in semantic.claims:
                if claim.quarantine_status == "active":
                    claim.quarantine_status = "quarantined"
                    claim.quarantine_reason = "sensitivity_gate_error"
                    claim.quarantine_timestamp = now_ts
                    quarantined_ids.append(claim.claim_id)

            return GateResult(
                quarantine_summary=QuarantineSummary(
                    total_claims=len(semantic.claims),
                    quarantined_count=len(quarantined_ids),
                    quarantined_ids=quarantined_ids,
                    gate_error=True,
                    gate_error_message=str(exc),
                ),
                semantic=semantic,
            )


class NullSensitivityGate:
    """No-op gate that passes all claims through without quarantine."""

    def __call__(
        self,
        structure: StructureBundle,
        semantic: SemanticBundle,
    ) -> GateResult:
        return GateResult(
            quarantine_summary=QuarantineSummary(
                total_claims=len(semantic.claims),
                quarantined_count=0,
                quarantined_ids=[],
            ),
            semantic=semantic,
        )
