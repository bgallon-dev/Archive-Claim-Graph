"""Sensitivity monitor — fourth detector family.

Detects three categories of sensitive content in ingested claims:

1. PII (structured and contextual): phone numbers, SSNs, email addresses,
   postal addresses, dates of birth.
2. Indigenous cultural sensitivity: terms matching an institution-specific
   vocabulary populated through tribal consultation.
3. Living person references: Person entities linked to claims with dates
   within the last 100 years.

At ingestion time, proposals with confidence >= sensitivity_config.yaml
auto_quarantine_threshold are applied in-memory by the ingestion gate
(pipeline.py) before the claim reaches the graph. The proposals themselves
are stored in the ReviewStore for archivist review.

The background monitor (review/monitor.py) uses _detect_pii_from_text() and
_detect_indigenous_from_text() directly against live claim text fetched from
Neo4j, following the same pattern but operating outside the bundle context.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]

from ...models import ClaimRecord, SemanticBundle, StructureBundle
from ..models import DetectorProposal, ProposalTarget
from ..patch_spec import make_patch_spec

DETECTOR_NAME = "sensitivity_monitor"
DETECTOR_VERSION = "v1"

# Default thresholds — overridden by sensitivity_config.yaml when present.
_DEFAULT_AUTO_QUARANTINE_THRESHOLD = 0.85
LIVING_PERSON_YEAR_THRESHOLD = datetime.now().year - 100

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(
    r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
_SSN_RE = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|"
    r"Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\b",
    re.IGNORECASE,
)
_DATE_OF_BIRTH_RE = re.compile(
    r"\b(?:born|birth(?:date)?|DOB|b\.)\s*:?\s*"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|"
    r"Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

# (pattern, label, base_confidence)
_PII_PATTERN_SPECS: list[tuple[re.Pattern, str, float]] = [
    (_SSN_RE, "ssn", 0.92),
    (_DATE_OF_BIRTH_RE, "date_of_birth", 0.92),
    (_PHONE_RE, "phone", 0.85),
    (_EMAIL_RE, "email", 0.85),
    (_ADDRESS_RE, "address", 0.85),
]


def _redact_sentence(text: str) -> str:
    """Replace all PII pattern matches with [REDACTED] for safe display in evidence."""
    for pattern, label, _ in _PII_PATTERN_SPECS:
        text = pattern.sub(f"[{label.upper()} REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Config and vocabulary loading
# ---------------------------------------------------------------------------

def _resources_dir() -> Path:
    return Path(__file__).parent.parent.parent / "resources"


def _load_config() -> dict[str, Any]:
    cfg_path = _resources_dir() / "sensitivity_config.yaml"
    if _yaml is None or not cfg_path.exists():
        return {
            "pii_detection": {"enabled": True, "auto_quarantine_threshold": 0.85,
                              "patterns": {"phone": True, "ssn": True, "email": True,
                                           "address": True, "date_of_birth": True}},
            "living_person": {"enabled": True, "year_threshold": 100,
                              "auto_quarantine_threshold": 0.70},
            "indigenous_sensitivity": {"enabled": True,
                                       "vocabulary_file": "indigenous_cultural_terms.yaml",
                                       "auto_quarantine_threshold": 0.90,
                                       "require_tribal_consultation_before_clear": True},
        }
    with open(cfg_path, encoding="utf-8") as fh:
        return _yaml.safe_load(fh) or {}


def _load_vocabulary() -> list[dict[str, Any]]:
    """Load flattened list of {terms, sensitivity, nations, category} from the vocabulary file."""
    if _yaml is None:
        return []
    vocab_path = _resources_dir() / "indigenous_cultural_terms.yaml"
    if not vocab_path.exists():
        return []
    with open(vocab_path, encoding="utf-8") as fh:
        raw = _yaml.safe_load(fh) or {}
    entries: list[dict[str, Any]] = []
    for category, items in raw.items():
        if category == "version" or not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            terms = item.get("terms") or []
            if terms:
                entries.append({
                    "terms": [t.lower() for t in terms],
                    "sensitivity": item.get("sensitivity", "medium"),
                    "nations": item.get("nations") or [],
                    "category": category,
                })
    return entries


# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------

def detect_pii_in_text(text: str, config: dict[str, Any]) -> list[tuple[str, float]]:
    """Return list of (pattern_label, confidence) for all PII matches in text."""
    patterns_cfg = config.get("patterns", {})
    results = []
    for pattern, label, confidence in _PII_PATTERN_SPECS:
        if not patterns_cfg.get(label, True):
            continue
        if pattern.search(text):
            results.append((label, confidence))
    return results


def _detect_pii(claims: list[ClaimRecord], config: dict[str, Any]) -> list[DetectorProposal]:
    proposals: list[DetectorProposal] = []
    for claim in claims:
        matches = detect_pii_in_text(claim.source_sentence, config)
        if not matches:
            continue
        # Use highest confidence among matched patterns.
        best_label, best_confidence = max(matches, key=lambda x: x[1])
        redacted = _redact_sentence(claim.source_sentence)
        patch = make_patch_spec(
            "quarantine_claim",
            claim_id=claim.claim_id,
            reason=f"pii_exposure:{best_label}",
        )
        proposal = DetectorProposal(
            anti_pattern_id="ap_pii_exposure",
            issue_class="pii_exposure",
            proposal_type="quarantine_claim",
            confidence=best_confidence,
            targets=[
                ProposalTarget(
                    proposal_id="",
                    target_kind="claim",
                    target_id=claim.claim_id,
                    target_role="flagged_claim",
                    exists_in_snapshot=True,
                )
            ],
            patch_spec=patch,
            evidence_snapshot={
                "matched_pattern": best_label,
                "all_patterns": [lbl for lbl, _ in matches],
                "claim_id": claim.claim_id,
                "redacted_sentence": redacted,
            },
            reasoning_summary={
                "summary": f"PII pattern '{best_label}' detected in claim source sentence.",
            },
            detector_name=DETECTOR_NAME,
            detector_version=DETECTOR_VERSION,
        )
        proposals.append(proposal)
    return proposals


# ---------------------------------------------------------------------------
# Indigenous cultural sensitivity detection
# ---------------------------------------------------------------------------

def detect_indigenous_in_text(
    text: str, vocab_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return list of matched vocabulary entries for text."""
    text_lower = text.lower()
    matches = []
    for entry in vocab_entries:
        for term in entry["terms"]:
            if term in text_lower:
                matches.append({**entry, "matched_term": term})
                break  # one match per entry is sufficient
    return matches


def _detect_indigenous_sensitivity(
    claims: list[ClaimRecord],
    vocab_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[DetectorProposal]:
    if not vocab_entries:
        return []
    require_consultation = config.get("require_tribal_consultation_before_clear", True)
    proposals: list[DetectorProposal] = []
    for claim in claims:
        matches = detect_indigenous_in_text(claim.source_sentence, vocab_entries)
        if not matches:
            continue
        best = max(matches, key=lambda m: (m["sensitivity"] == "high", m["sensitivity"] == "medium"))
        patch = make_patch_spec(
            "quarantine_claim",
            claim_id=claim.claim_id,
            reason=f"indigenous_sensitivity:{best['category']}",
        )
        proposal = DetectorProposal(
            anti_pattern_id="ap_indigenous_sensitivity",
            issue_class="indigenous_sensitivity",
            proposal_type="quarantine_claim",
            confidence=0.90,
            targets=[
                ProposalTarget(
                    proposal_id="",
                    target_kind="claim",
                    target_id=claim.claim_id,
                    target_role="flagged_claim",
                    exists_in_snapshot=True,
                )
            ],
            patch_spec=patch,
            evidence_snapshot={
                "matched_term": best["matched_term"],
                "category": best["category"],
                "sensitivity": best["sensitivity"],
                "nations": best["nations"],
                "claim_id": claim.claim_id,
                "require_tribal_consultation_before_clear": require_consultation,
            },
            reasoning_summary={
                "summary": (
                    f"Term '{best['matched_term']}' matched Indigenous cultural vocabulary "
                    f"category '{best['category']}' (sensitivity: {best['sensitivity']})."
                ),
            },
            detector_name=DETECTOR_NAME,
            detector_version=DETECTOR_VERSION,
        )
        proposals.append(proposal)
    return proposals


# ---------------------------------------------------------------------------
# Living person reference detection
# ---------------------------------------------------------------------------

def _detect_living_persons(
    claims: list[ClaimRecord],
    semantic: SemanticBundle,
    config: dict[str, Any],
) -> list[DetectorProposal]:
    year_threshold = datetime.now().year - config.get("year_threshold", 100)
    confidence = 0.70  # below default auto-quarantine threshold — queued for review

    # Build lookups.
    person_entity_ids: set[str] = {
        e.entity_id for e in semantic.entities if e.entity_type == "Person"
    }
    if not person_entity_ids:
        return []

    # claim_id → set of person entity_ids linked to that claim
    claim_to_persons: dict[str, set[str]] = {}
    for link in semantic.claim_entity_links:
        if link.entity_id in person_entity_ids:
            claim_to_persons.setdefault(link.claim_id, set()).add(link.entity_id)

    if not claim_to_persons:
        return []

    # claim_id → max year (from observations)
    claim_to_max_year: dict[str, int] = {}
    for obs in semantic.observations:
        if obs.year is not None and obs.claim_id in claim_to_persons:
            existing = claim_to_max_year.get(obs.claim_id)
            if existing is None or obs.year > existing:
                claim_to_max_year[obs.claim_id] = obs.year

    # Also check claim_date field directly on ClaimRecord.
    claims_by_id: dict[str, ClaimRecord] = {c.claim_id: c for c in claims}
    for claim_id in claim_to_persons:
        claim = claims_by_id.get(claim_id)
        if claim and claim.claim_date:
            try:
                year = int(claim.claim_date[:4])
                existing = claim_to_max_year.get(claim_id)
                if existing is None or year > existing:
                    claim_to_max_year[claim_id] = year
            except (ValueError, IndexError):
                pass

    # entity_id → name
    entity_names: dict[str, str] = {e.entity_id: e.name for e in semantic.entities}

    proposals: list[DetectorProposal] = []
    for claim in claims:
        if claim.claim_id not in claim_to_persons:
            continue
        max_year = claim_to_max_year.get(claim.claim_id)
        if max_year is None or max_year < year_threshold:
            continue
        person_ids = sorted(claim_to_persons[claim.claim_id])
        person_names = [entity_names.get(pid, pid) for pid in person_ids]
        patch = make_patch_spec(
            "quarantine_claim",
            claim_id=claim.claim_id,
            reason="living_person_reference",
        )
        proposal = DetectorProposal(
            anti_pattern_id="ap_living_person",
            issue_class="living_person_reference",
            proposal_type="quarantine_claim",
            confidence=confidence,
            targets=[
                ProposalTarget(
                    proposal_id="",
                    target_kind="claim",
                    target_id=claim.claim_id,
                    target_role="flagged_claim",
                    exists_in_snapshot=True,
                )
            ],
            patch_spec=patch,
            evidence_snapshot={
                "claim_id": claim.claim_id,
                "person_names": person_names,
                "most_recent_year": max_year,
                "source_sentence": claim.source_sentence,
                "year_threshold": year_threshold,
            },
            reasoning_summary={
                "summary": (
                    f"Claim references Person entity '{', '.join(person_names)}' "
                    f"with most recent associated year {max_year} "
                    f"(within the last {config.get('year_threshold', 100)} years)."
                ),
            },
            detector_name=DETECTOR_NAME,
            detector_version=DETECTOR_VERSION,
        )
        proposals.append(proposal)
    return proposals


# ---------------------------------------------------------------------------
# Main entry point (ingestion-time detector interface)
# ---------------------------------------------------------------------------

def detect(
    structure: StructureBundle,
    semantic: SemanticBundle,
    snapshot_id: str,
) -> list[DetectorProposal]:
    """Scan claims in *semantic* for sensitivity issues.

    Follows the standard detector interface:
      detect(structure, semantic, snapshot_id) -> list[DetectorProposal]

    High-confidence proposals are auto-quarantined by the ingestion gate in
    pipeline.py before the ClaimRecords are written to the graph.
    """
    config = _load_config()
    vocab_entries = _load_vocabulary()
    proposals: list[DetectorProposal] = []

    pii_cfg = config.get("pii_detection", {})
    if pii_cfg.get("enabled", True):
        proposals.extend(_detect_pii(semantic.claims, pii_cfg))

    indigenous_cfg = config.get("indigenous_sensitivity", {})
    if indigenous_cfg.get("enabled", True):
        proposals.extend(
            _detect_indigenous_sensitivity(semantic.claims, vocab_entries, indigenous_cfg)
        )

    living_cfg = config.get("living_person", {})
    if living_cfg.get("enabled", True):
        proposals.extend(_detect_living_persons(semantic.claims, semantic, living_cfg))

    return proposals
