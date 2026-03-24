from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..resolver import default_seed_entities
from ..resource_loader import load_ocr_corrections

_LOADED_OCR_CORRECTIONS: frozenset[str] = load_ocr_corrections()


@dataclass(slots=True)
class MentionDraft:
    surface_form: str
    normalized_form: str
    start_offset: int
    end_offset: int
    detection_confidence: float
    ocr_flags: list[str]
    entity_type_hints: list[str] = field(default_factory=list)
    detection_source: str = "proper_noun"
    decision_trace: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    fallback_used: bool = False


class MentionExtractor(Protocol):
    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        ...


def _fuzzy_match_seed(text: str, lexicon_hints: dict[str, list[str]], threshold: float = 0.78) -> str | None:
    best_name: str | None = None
    best_ratio = 0.0
    for name in lexicon_hints:
        if abs(len(name) - len(text)) > 3:
            continue
        ratio = difflib.SequenceMatcher(None, text, name).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, name
    return best_name if best_ratio >= threshold else None


def get_ocr_flags(
    surface: str,
    *,
    known_ocr_errors: frozenset[str] | None = None,
    lexicon_hints: dict[str, list[str]] | None = None,
) -> list[str]:
    normalized = re.sub(r"\s+", " ", surface.strip().lower())
    known_errors = known_ocr_errors if known_ocr_errors is not None else _LOADED_OCR_CORRECTIONS
    hints = lexicon_hints if lexicon_hints is not None else RuleBasedMentionExtractor._lexicon_hints

    flags: list[str] = []
    if normalized in known_errors:
        flags.append("ocr_suspect_list")

    if re.search(r"(?<=[a-z])[0-9]|[0-9](?=[a-z])", normalized):
        flags.append("digit_in_token")
        for digit, letter in [("0", "o"), ("1", "l")]:
            candidate = normalized.replace(digit, letter)
            if candidate in hints:
                flags.append(f"near_seed_term:{candidate}")

    if "rn" in normalized:
        candidate = normalized.replace("rn", "m")
        if candidate in hints:
            flags.append(f"near_seed_term:{candidate}")
        elif "m" not in normalized:
            flags.append("rn_m_confusion")

    if "li" in normalized:
        candidate = normalized.replace("li", "h")
        if candidate in hints:
            flags.append(f"near_seed_term:{candidate}")

    if not any(flag.startswith("near_seed_term") for flag in flags):
        best = _fuzzy_match_seed(normalized, hints)
        if best:
            flags.append(f"near_seed_term:{best}")

    return flags


class RuleBasedMentionExtractor:
    _STOPWORDS: frozenset[str] = frozenset({
        "the", "a", "an", "in", "at", "on", "of", "for", "with", "and", "or",
        "but", "to", "from", "by", "about", "during", "near", "after", "before",
        "that", "this", "these", "those", "it", "its",
    })

    # Stage 2: acronyms and initialized names.
    _acronym = re.compile(
        r"\b[A-Z]{2,}\b"
        r"|[A-Z]\.(?:\s*[A-Z]\.)+(?:\s+[A-Z][a-z]+)*"
    )
    _initialized_name = re.compile(
        r"\b(?:Dr|Mr|Mrs|Prof|Col|Lt|Sgt)\.?\s+(?:[A-Z]\.\s*)*[A-Z][a-z]+"
    )

    # Stage 3: titlecase spans (1+ words, min 3 chars).
    _titlecase_span = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b")

    _known_ocr_errors: frozenset[str] = _LOADED_OCR_CORRECTIONS
    _seed_entities = default_seed_entities()

    _lexicon_hints: dict[str, list[str]] = {}
    for _entity in _seed_entities:
        _lexicon_hints.setdefault(_entity.name.lower(), []).append(_entity.entity_type)

    _lexicon_terms: list[str] = sorted(_lexicon_hints, key=len, reverse=True)

    def __init__(self, resources_dir: Path | None = None) -> None:
        if resources_dir is not None:
            self._known_ocr_errors = load_ocr_corrections(resources_dir)
            seed = default_seed_entities(resources_dir)
            hints: dict[str, list[str]] = {}
            for entity in seed:
                hints.setdefault(entity.name.lower(), []).append(entity.entity_type)
            self._lexicon_hints = hints
            self._lexicon_terms = sorted(hints, key=len, reverse=True)

    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        mentions: list[MentionDraft] = []

        for surface, start, end in self._scan_terms(paragraph_text):
            hints = self._lexicon_hints.get(surface.lower(), [])
            ocr_flags = self._get_ocr_flags(surface)
            confidence = min(
                0.92 + (0.05 if any(flag.startswith("near_seed_term") for flag in ocr_flags) else 0.0),
                1.0,
            )
            trace = [f"stage:seed_lexicon", f"entity_type:{','.join(hints) or 'unknown'}", *ocr_flags]
            mentions.append(
                MentionDraft(
                    surface_form=surface,
                    normalized_form=self._normalize(surface),
                    start_offset=start,
                    end_offset=end,
                    detection_confidence=confidence,
                    ocr_flags=ocr_flags,
                    entity_type_hints=hints,
                    detection_source="seed_lexicon",
                    decision_trace=trace,
                    matched_patterns=[f"seed_lexicon:{surface.lower()}"],
                    fallback_used=False,
                )
            )

        for pattern, pattern_name in ((self._acronym, "acronym_re"), (self._initialized_name, "initialized_name_re")):
            stage_name = "acronym" if pattern_name == "acronym_re" else "initialized_name"
            for match in pattern.finditer(paragraph_text):
                surface = match.group(0).strip()
                if len(surface) <= 1:
                    continue
                ocr_flags = self._get_ocr_flags(surface)
                confidence = 0.82 + (0.05 if any(flag.startswith("near_seed_term") for flag in ocr_flags) else 0.0)
                mentions.append(
                    MentionDraft(
                        surface_form=surface,
                        normalized_form=self._normalize(surface),
                        start_offset=match.start(),
                        end_offset=match.end(),
                        detection_confidence=min(confidence, 0.92),
                        ocr_flags=ocr_flags,
                        entity_type_hints=[],
                        detection_source="acronym",
                        decision_trace=[f"stage:{stage_name}", *ocr_flags],
                        matched_patterns=[pattern_name],
                        fallback_used=False,
                    )
                )

        for match in self._titlecase_span.finditer(paragraph_text):
            surface = match.group(0).strip()
            words = surface.split()
            if words[0].lower() in self._STOPWORDS:
                continue
            normalized_surface = self._normalize(surface)
            fuzzy_hit: str | None = None
            if len(words) == 1 and self._is_sentence_initial(paragraph_text, match.start()):
                if normalized_surface not in self._known_ocr_errors:
                    fuzzy_hit = _fuzzy_match_seed(normalized_surface, self._lexicon_hints)
                    if not fuzzy_hit:
                        continue
            ocr_flags = self._get_ocr_flags(surface)
            confidence = 0.74 + (0.05 if any(flag.startswith("near_seed_term") for flag in ocr_flags) else 0.0)
            trace = ["stage:proper_noun"]
            if fuzzy_hit:
                trace.append(f"fuzzy_match:{fuzzy_hit}")
            trace.extend(ocr_flags)
            mentions.append(
                MentionDraft(
                    surface_form=surface,
                    normalized_form=normalized_surface,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    detection_confidence=min(confidence, 0.85),
                    ocr_flags=ocr_flags,
                    entity_type_hints=[],
                    detection_source="proper_noun",
                    decision_trace=trace,
                    matched_patterns=["titlecase_span"],
                    fallback_used=fuzzy_hit is not None,
                )
            )

        return self._dedupe(mentions)

    def _scan_terms(self, paragraph_text: str) -> list[tuple[str, int, int]]:
        matches: list[tuple[str, int, int]] = []
        occupied: list[tuple[int, int]] = []
        for term in self._lexicon_terms:
            for match in re.finditer(rf"\b{re.escape(term)}\b", paragraph_text, flags=re.IGNORECASE):
                span = (match.start(), match.end())
                if any(not (span[1] <= left or span[0] >= right) for left, right in occupied):
                    continue
                occupied.append(span)
                matches.append((match.group(0), match.start(), match.end()))
        matches.sort(key=lambda row: row[1])
        return matches

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _is_sentence_initial(paragraph_text: str, match_start: int) -> bool:
        prefix = paragraph_text[:match_start].rstrip()
        return not prefix or bool(re.search(r"[.!?]$", prefix))

    def _get_ocr_flags(self, surface: str) -> list[str]:
        return get_ocr_flags(
            surface,
            known_ocr_errors=self._known_ocr_errors,
            lexicon_hints=self._lexicon_hints,
        )

    @staticmethod
    def _dedupe(mentions: list[MentionDraft]) -> list[MentionDraft]:
        best: dict[tuple[int, int], MentionDraft] = {}
        for mention in mentions:
            key = (mention.start_offset, mention.end_offset)
            existing = best.get(key)
            if existing is None or mention.detection_confidence > existing.detection_confidence:
                best[key] = mention
        return sorted(best.values(), key=lambda mention: mention.start_offset)
