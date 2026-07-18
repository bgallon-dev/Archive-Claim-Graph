"""LLM-call gating on HybridClaimExtractor — the LLM as a fallback to the rules."""
from __future__ import annotations

import pytest

from gemynd.ingest.extractors.claim_extractor import (
    ClaimDraft,
    HybridClaimExtractor,
    _paragraph_is_low_value,
)

RULE_PROSE = "The refuge recorded nine hundred mallards during the autumn survey season here."
DRY_PROSE = "A perfectly ordinary sentence of prose that the rules did not match at all today."
JUNK = "3000 2500 1700 900 100 500 700 173.5 347.00 175"


def _draft(sent: str) -> ClaimDraft:
    return ClaimDraft(
        claim_type="x", source_sentence=sent, normalized_sentence=sent.lower(),
        epistemic_status="observed", extraction_confidence=0.9,
    )


class _FakeRules:
    def __init__(self, hits: dict[str, list]) -> None:
        self._hits = hits

    def extract(self, text: str) -> list[ClaimDraft]:
        return list(self._hits.get(text, []))


class _RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, text: str) -> list[ClaimDraft]:
        self.calls.append(text)
        return [_draft("llm-" + text[:8])]


def _hybrid(gate, llm):
    rules = _FakeRules({RULE_PROSE: [_draft(RULE_PROSE)]})
    return HybridClaimExtractor(rules_extractor=rules, llm_extractor=llm, gate=gate)


def test_gate_always_calls_llm_on_every_paragraph() -> None:
    llm = _RecordingLLM()
    h = _hybrid("always", llm)
    h.extract(RULE_PROSE)
    h.extract(DRY_PROSE)
    assert llm.calls == [RULE_PROSE, DRY_PROSE]


def test_gate_on_miss_skips_llm_when_rules_hit() -> None:
    llm = _RecordingLLM()
    h = _hybrid("on_miss", llm)
    h.extract(RULE_PROSE)  # rules found a claim -> LLM skipped
    h.extract(DRY_PROSE)   # rules dry -> LLM runs (its fallback role)
    assert llm.calls == [DRY_PROSE]


def test_gate_prefilter_skips_low_value_paragraphs() -> None:
    llm = _RecordingLLM()
    h = _hybrid("prefilter", llm)
    h.extract(DRY_PROSE)  # real prose -> call
    h.extract(JUNK)       # OCR junk -> skip
    assert llm.calls == [DRY_PROSE]


def test_gate_on_miss_prefilter_combines_both() -> None:
    llm = _RecordingLLM()
    h = _hybrid("on_miss_prefilter", llm)
    h.extract(RULE_PROSE)  # rules hit -> skip
    h.extract(JUNK)        # low value -> skip
    h.extract(DRY_PROSE)   # dry + real prose -> call
    assert llm.calls == [DRY_PROSE]


def test_gate_default_is_always_backward_compatible() -> None:
    llm = _RecordingLLM()
    rules = _FakeRules({RULE_PROSE: [_draft(RULE_PROSE)]})
    h = HybridClaimExtractor(rules_extractor=rules, llm_extractor=llm)  # no gate, no env
    h.extract(RULE_PROSE)
    assert llm.calls == [RULE_PROSE]  # LLM still ran despite a rule hit


def test_gate_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_LLM_GATE", "on_miss")
    llm = _RecordingLLM()
    rules = _FakeRules({RULE_PROSE: [_draft(RULE_PROSE)]})
    h = HybridClaimExtractor(rules_extractor=rules, llm_extractor=llm)  # gate from env
    h.extract(RULE_PROSE)
    assert llm.calls == []


def test_low_value_heuristic() -> None:
    assert _paragraph_is_low_value("3000 2500 1700 900 100 500 700 173.5")  # digits
    assert _paragraph_is_low_value("SHORT ALLCAPS MASTHEAD LINE")           # allcaps/short
    assert _paragraph_is_low_value("a b c d e")                             # too few words
    assert not _paragraph_is_low_value(DRY_PROSE)                           # real prose kept
