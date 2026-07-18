"""Recall-first unclassified fallback on RuleBasedClaimExtractor.

When enabled, any sentence that no claim-type pattern matched but that still reads
as a genuine assertive sentence becomes a low-confidence ``unclassified_assertion``
candidate for an LLM validator to confirm or reject.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gemynd.core.claim_contract import UNCLASSIFIED_TYPE
from gemynd.core.domain_config import load_domain_config
from gemynd.ingest.extractors.claim_extractor import RuleBasedClaimExtractor

NEWS = Path("gemynd/resources_newspaper")

# Verified against the newspaper domain config:
UNTYPED_VALID = "The old stone bridge was near the mill on the north side of the river."
TYPED = "The convention voted to nominate the candidate for president of the party."
INVALID_SHORT = "A short note."       # < MIN_CHARS
INVALID_AD = "Choicest goods of the season."  # no finite verb


@pytest.fixture(scope="module")
def cfg():
    return load_domain_config(NEWS)


def test_fallback_off_by_default(cfg) -> None:
    ex = RuleBasedClaimExtractor(config=cfg)  # default: no fallback
    assert ex.extract(UNTYPED_VALID) == []


def test_fallback_proposes_unclassified_for_untyped_valid_sentence(cfg) -> None:
    ex = RuleBasedClaimExtractor(config=cfg, unclassified_fallback=True)
    drafts = ex.extract(UNTYPED_VALID)
    assert len(drafts) == 1
    d = drafts[0]
    assert d.claim_type == UNCLASSIFIED_TYPE
    assert d.extraction_source == "rules_unclassified"
    assert d.extraction_confidence == RuleBasedClaimExtractor._UNCLASSIFIED_CONF
    assert "needs_llm_validation" in d.decision_trace
    assert d.evidence_start is not None and d.evidence_end is not None


def test_fallback_does_not_override_pattern_typed_claims(cfg) -> None:
    ex = RuleBasedClaimExtractor(config=cfg, unclassified_fallback=True)
    drafts = ex.extract(TYPED)
    assert len(drafts) == 1
    assert drafts[0].claim_type != UNCLASSIFIED_TYPE           # kept its real type
    assert drafts[0].extraction_source == "rules"


@pytest.mark.parametrize("junk", [INVALID_SHORT, INVALID_AD])
def test_fallback_still_respects_the_validity_gate(cfg, junk: str) -> None:
    ex = RuleBasedClaimExtractor(config=cfg, unclassified_fallback=True)
    assert ex.extract(junk) == []  # invalid sentences never become candidates


def test_fallback_read_from_env(cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_RULES_FALLBACK", "1")
    ex = RuleBasedClaimExtractor(config=cfg)  # flag comes from env
    drafts = ex.extract(UNTYPED_VALID)
    assert len(drafts) == 1 and drafts[0].claim_type == UNCLASSIFIED_TYPE
