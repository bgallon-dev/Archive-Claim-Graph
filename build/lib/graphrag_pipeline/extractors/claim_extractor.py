from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ClaimDraft:
    claim_type: str
    raw_sentence: str
    normalized_sentence: str
    certainty: str
    extraction_confidence: float
    evidence_start: int | None = None
    evidence_end: int | None = None
    claim_date: str | None = None
    notes: str = ""


class ClaimExtractor(Protocol):
    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        ...


def _normalize_sentence(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"\s+", " ", lowered)


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    text = text.strip()
    if not text:
        return []
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[^.!?]+[.!?]?", text):
        sentence = match.group(0).strip()
        if sentence:
            spans.append((sentence, match.start(), match.end()))
    return spans


class RuleBasedClaimExtractor:
    _type_patterns: list[tuple[str, re.Pattern[str]]] = [
        ("predator_control", re.compile(r"\b(coyote|predator|trapp?ed|taken)\b", re.IGNORECASE)),
        ("fire_incident", re.compile(r"\b(fire|burned|burnt|suppression)\b", re.IGNORECASE)),
        ("weather_condition", re.compile(r"\b(temperature|rain|degrees|weather|dried up)\b", re.IGNORECASE)),
        ("economic_use", re.compile(r"\b(hay|haying|grazing|revenue|permit|farmers?)\b", re.IGNORECASE)),
        ("development_activity", re.compile(r"\b(wpa|development|construction|project started|planting)\b", re.IGNORECASE)),
        ("public_contact", re.compile(r"\b(meeting|contacts?|club|council|association|public relations)\b", re.IGNORECASE)),
        (
            "wildlife_count",
            re.compile(
                r"\b(mallard|teal|coot|goose|pheasant|quail|partridge|grouse|deer|coyote|geese|duck)\b",
                re.IGNORECASE,
            ),
        ),
        ("management_action", re.compile(r"\b(allowed|restricted|started|managed|control|planted)\b", re.IGNORECASE)),
    ]
    _uncertain_tokens = re.compile(r"\b(about|approx|approximately|around|estimated|reported|possibly|likely)\b", re.IGNORECASE)

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        claims: list[ClaimDraft] = []
        for sentence, start, end in _split_sentences(paragraph_text):
            claim_type = self._detect_type(sentence)
            if not claim_type:
                continue
            normalized = _normalize_sentence(sentence)
            certainty = "uncertain" if self._uncertain_tokens.search(sentence) else "certain"
            confidence = 0.78 if certainty == "certain" else 0.68
            claims.append(
                ClaimDraft(
                    claim_type=claim_type,
                    raw_sentence=sentence,
                    normalized_sentence=normalized,
                    certainty=certainty,
                    extraction_confidence=confidence,
                    evidence_start=start,
                    evidence_end=end,
                )
            )
        return claims

    def _detect_type(self, sentence: str) -> str | None:
        for claim_type, pattern in self._type_patterns:
            if pattern.search(sentence):
                return claim_type
        return None


class ClaimLLMAdapter(Protocol):
    def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
        ...


class NullLLMAdapter:
    def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
        return []


class LLMClaimExtractor:
    def __init__(self, adapter: ClaimLLMAdapter) -> None:
        self._adapter = adapter

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        drafts: list[ClaimDraft] = []
        for row in self._adapter.extract_claims(paragraph_text):
            raw = str(row.get("raw_sentence", "")).strip()
            if not raw:
                continue
            normalized = str(row.get("normalized_sentence", "")).strip() or _normalize_sentence(raw)
            drafts.append(
                ClaimDraft(
                    claim_type=str(row.get("claim_type", "management_action")),
                    raw_sentence=raw,
                    normalized_sentence=normalized,
                    certainty=str(row.get("certainty", "uncertain")),
                    extraction_confidence=float(row.get("extraction_confidence", 0.6)),
                    evidence_start=int(row["evidence_start"]) if row.get("evidence_start") is not None else None,
                    evidence_end=int(row["evidence_end"]) if row.get("evidence_end") is not None else None,
                    claim_date=str(row["claim_date"]) if row.get("claim_date") else None,
                    notes=str(row.get("notes", "")),
                )
            )
        return drafts


class HybridClaimExtractor:
    def __init__(
        self,
        rules_extractor: ClaimExtractor | None = None,
        llm_extractor: ClaimExtractor | None = None,
    ) -> None:
        self._rules = rules_extractor or RuleBasedClaimExtractor()
        self._llm = llm_extractor or LLMClaimExtractor(NullLLMAdapter())

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        merged: dict[str, ClaimDraft] = {}
        for draft in self._rules.extract(paragraph_text):
            merged[draft.normalized_sentence] = draft
        for draft in self._llm.extract(paragraph_text):
            existing = merged.get(draft.normalized_sentence)
            if not existing or draft.extraction_confidence > existing.extraction_confidence:
                merged[draft.normalized_sentence] = draft
        return list(merged.values())
