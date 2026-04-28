"""Sensitivity gate implementations for the extract/persist pipeline split."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from gemynd.core.models import SemanticBundle, StructureBundle
from gemynd.ingest.extraction_result import (
    GateResult,
    QuarantineSummary,
)

_log = logging.getLogger(__name__)


def _classify_address_context(sentence: str, cfg_era: dict[str, Any]) -> str:
    """Return 'commercial', 'residential', or 'ambiguous' for an address sentence.

    Residential signals win when both vocabularies match in the same sentence —
    a saloon owner who died at his home above the bar is still a death notice.
    """
    text = (sentence or "").lower()
    commercial_kws = cfg_era.get("address_commercial_keywords") or []
    residential_kws = cfg_era.get("address_residential_keywords") or []
    if any(kw in text for kw in residential_kws):
        return "residential"
    if any(kw in text for kw in commercial_kws):
        return "commercial"
    return "ambiguous"


def _should_relax_pii_proposal(
    prop: Any,
    structure: StructureBundle,
    semantic: SemanticBundle,
    cfg_era: dict[str, Any],
) -> bool:
    """Return True when an address-only PII proposal in a pre-cutoff document
    should be allowed through (commercial-anchored).  All other proposals,
    eras, and patterns fall through to the normal threshold gate.
    """
    if not cfg_era.get("enabled"):
        return False
    if getattr(prop, "issue_class", None) != "pii_exposure":
        return False
    evidence = getattr(prop, "evidence_snapshot", None) or {}
    matched = evidence.get("matched_pattern")
    all_patterns = list(evidence.get("all_patterns") or [])
    if matched != "address" or all_patterns != ["address"]:
        return False
    report_year = getattr(structure.document, "report_year", None)
    if report_year is None or report_year >= cfg_era.get("cutoff_year", 1950):
        return False
    # Find the source sentence for the targeted claim.
    target_claim_id = next(
        (t.target_id for t in (prop.targets or []) if t.target_kind == "claim"),
        None,
    )
    if not target_claim_id:
        return False
    sentence = next(
        (c.source_sentence for c in semantic.claims if c.claim_id == target_claim_id),
        "",
    )
    return _classify_address_context(sentence, cfg_era) == "commercial"


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
            from gemynd.review.detectors import sensitivity_monitor as _sm

            proposals = _sm.detect(structure, semantic, snapshot_id="")
            cfg = _sm._load_config()
            pii_cfg = cfg.get("pii_detection", {})
            threshold = pii_cfg.get("auto_quarantine_threshold", 0.85)
            era_cfg = pii_cfg.get("era_relaxation", {}) or {}

            seen: set[str] = set()
            for prop in proposals:
                if prop.confidence >= threshold:
                    if _should_relax_pii_proposal(prop, structure, semantic, era_cfg):
                        continue
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
