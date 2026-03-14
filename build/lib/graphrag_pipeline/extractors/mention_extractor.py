from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class MentionDraft:
    surface_form: str
    normalized_form: str
    start_offset: int
    end_offset: int
    confidence: float
    ocr_suspect: bool


class MentionExtractor(Protocol):
    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        ...


class RuleBasedMentionExtractor:
    _proper_noun = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
    _species_terms = [
        "mallard",
        "green-wing teal",
        "teal",
        "coot",
        "canada goose",
        "canada geese",
        "chinese pheasant",
        "valley quail",
        "hungarian partridge",
        "ruffed grouse",
        "white-tailed deer",
        "coyote",
        "wild celery",
    ]
    _ocr_suspects = {"tumbull", "lightening", "emgman", "turnbuli"}

    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        mentions: list[MentionDraft] = []

        for term in self._species_terms:
            for match in re.finditer(rf"\b{re.escape(term)}\b", paragraph_text, flags=re.IGNORECASE):
                surface = match.group(0)
                mentions.append(
                    MentionDraft(
                        surface_form=surface,
                        normalized_form=self._normalize(surface),
                        start_offset=match.start(),
                        end_offset=match.end(),
                        confidence=0.92,
                        ocr_suspect=self._is_ocr_suspect(surface),
                    )
                )

        for match in self._proper_noun.finditer(paragraph_text):
            surface = match.group(0).strip()
            if len(surface) <= 2:
                continue
            mentions.append(
                MentionDraft(
                    surface_form=surface,
                    normalized_form=self._normalize(surface),
                    start_offset=match.start(),
                    end_offset=match.end(),
                    confidence=0.74,
                    ocr_suspect=self._is_ocr_suspect(surface),
                )
            )

        return self._dedupe(mentions)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _is_ocr_suspect(self, text: str) -> bool:
        lowered = self._normalize(text)
        if lowered in self._ocr_suspects:
            return True
        if re.search(r"[0-9]", lowered):
            return True
        if "rn" in lowered and "m" not in lowered:
            return True
        return False

    @staticmethod
    def _dedupe(mentions: list[MentionDraft]) -> list[MentionDraft]:
        seen: set[tuple[str, int, int]] = set()
        out: list[MentionDraft] = []
        for mention in mentions:
            key = (mention.normalized_form, mention.start_offset, mention.end_offset)
            if key in seen:
                continue
            seen.add(key)
            out.append(mention)
        return out
