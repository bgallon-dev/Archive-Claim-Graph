"""LLM-as-validator: verdict mapping + the HybridClaimExtractor validator role."""
from __future__ import annotations

import pytest

from gemynd.core.claim_contract import UNCLASSIFIED_TYPE
from gemynd.ingest.extractors.claim_extractor import (
    ClaimDraft,
    HybridClaimExtractor,
    apply_verdicts,
)


def _cand(sent: str) -> ClaimDraft:
    return ClaimDraft(
        claim_type=UNCLASSIFIED_TYPE, source_sentence=sent, normalized_sentence=sent.lower(),
        epistemic_status="certain", extraction_confidence=0.30, extraction_source="rules_unclassified",
    )


def test_apply_verdicts_confirms_and_retypes() -> None:
    out = apply_verdicts([_cand("The mayor was elected Tuesday.")],
                         [{"is_claim": True, "claim_type": "election_event", "confidence": 0.9}])
    assert len(out) == 1
    assert out[0].claim_type == "election_event"
    assert out[0].extraction_confidence == 0.9
    assert out[0].extraction_source == "rules_validated"
    assert any("llm_validated:confirmed" in t for t in out[0].decision_trace)


def test_apply_verdicts_drops_rejected() -> None:
    assert apply_verdicts([_cand("Choicest goods of the season.")],
                          [{"is_claim": False, "claim_type": None, "confidence": 1.0}]) == []


def test_apply_verdicts_floors_confidence_and_keeps_type_when_null() -> None:
    out = apply_verdicts([_cand("Some plain assertion here.")],
                         [{"is_claim": True, "claim_type": None, "confidence": 0.1}])
    assert out[0].extraction_confidence == 0.5           # floored above the 0.30 marker
    assert out[0].claim_type == UNCLASSIFIED_TYPE         # kept when verdict gives no type


class _FakeRules:
    def __init__(self, drafts): self._d = drafts
    def extract(self, _text): return list(self._d)


class _FakeValidator:
    def __init__(self, verdicts): self.calls = []; self._v = verdicts
    def validate(self, sentences): self.calls.append(list(sentences)); return self._v


def _typed():
    return ClaimDraft(claim_type="election_event", source_sentence="X won.", normalized_sentence="x won",
                      epistemic_status="certain", extraction_confidence=0.7, extraction_source="rules")


def test_validator_role_validates_only_candidates_and_passes_typed() -> None:
    rules = _FakeRules([_typed(), _cand("The bridge was near the mill.")])
    val = _FakeValidator([{"is_claim": True, "claim_type": "civic_development", "confidence": 0.8}])
    h = HybridClaimExtractor(rules_extractor=rules, validator=val, role="validator")
    out = h.extract("whatever")
    # only the unclassified candidate was sent to the validator
    assert val.calls == [["The bridge was near the mill."]]
    types = {d.claim_type for d in out}
    assert types == {"election_event", "civic_development"}  # typed passed, candidate re-typed
    assert len(out) == 2


def test_validator_role_drops_rejected_candidates() -> None:
    rules = _FakeRules([_cand("An advertisement for cheap goods.")])
    val = _FakeValidator([{"is_claim": False, "claim_type": None, "confidence": 1.0}])
    h = HybridClaimExtractor(rules_extractor=rules, validator=val, role="validator")
    assert h.extract("x") == []


def test_validator_role_fails_open_without_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIM_LLM_BASE_URL", raising=False)  # so try_create_validator returns None
    rules = _FakeRules([_cand("Some claim survives unvalidated.")])
    h = HybridClaimExtractor(rules_extractor=rules, validator=None, role="validator")
    out = h.extract("x")
    assert len(out) == 1 and out[0].extraction_source == "rules_unclassified"
