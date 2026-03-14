from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .claim_extractor import ClaimDraft


@dataclass(slots=True)
class MeasurementDraft:
    name: str
    raw_value: str
    numeric_value: float | None
    unit: str | None
    approximate: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    qualifier: str | None = None


class MeasurementExtractor(Protocol):
    def extract(self, claim: ClaimDraft) -> list[MeasurementDraft]:
        ...


class RuleBasedMeasurementExtractor:
    _approx_re = re.compile(r"\b(about|around|approx|approximately|estimated)\b", re.IGNORECASE)
    _money_re = re.compile(r"(?:(?:USD|usd)\s*)?\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?|USD|usd)?")
    _pattern_specs: list[tuple[re.Pattern[str], str, str | None]] = [
        (re.compile(r"(\d+(?:\.\d+)?)\s*(days?)\b", re.IGNORECASE), "days_above_threshold", "days"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*(inches?|inch)\b", re.IGNORECASE), "rainfall", "inches"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*(acres?|acre)\b", re.IGNORECASE), "acres", "acres"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*(tons?|ton)\b", re.IGNORECASE), "hay_cut", "tons"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*(farmers?|farmer)\b", re.IGNORECASE), "farmers_count", "count"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*(cords?|cord)\b", re.IGNORECASE), "wood_killed_cords", "cords"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*(covies?|covy)\b", re.IGNORECASE), "coveys_count", "coveys"),
    ]
    _individual_context = re.compile(
        r"\b(mallards?|teal|coot|geese?|goose|pheasant|quail|partridge|grouse|deer|coyotes?)\b",
        re.IGNORECASE,
    )
    _generic_number = re.compile(r"\b(\d+(?:\.\d+)?)\b")

    def extract(self, claim: ClaimDraft) -> list[MeasurementDraft]:
        sentence = claim.raw_sentence
        approximate = bool(self._approx_re.search(sentence))
        measurements: list[MeasurementDraft] = []

        for pattern, default_name, default_unit in self._pattern_specs:
            for match in pattern.finditer(sentence):
                number = float(match.group(1))
                name = self._specialize_name(default_name, claim.claim_type, sentence)
                measurements.append(
                    MeasurementDraft(
                        name=name,
                        raw_value=match.group(0),
                        numeric_value=number,
                        unit=default_unit,
                        approximate=approximate,
                    )
                )

        for match in self._money_re.finditer(sentence):
            raw_match = match.group(0)
            if "$" not in raw_match and "usd" not in raw_match.lower() and "dollar" not in raw_match.lower():
                continue
            number = float(match.group(1))
            money_name = "suppression_cost" if claim.claim_type == "fire_incident" else "revenue"
            measurements.append(
                MeasurementDraft(
                    name=money_name,
                    raw_value=raw_match.strip(),
                    numeric_value=number,
                    unit="USD",
                    approximate=approximate,
                )
            )

        if not measurements and claim.claim_type in {"wildlife_count", "predator_control"}:
            if self._individual_context.search(sentence):
                for match in self._generic_number.finditer(sentence):
                    measurements.append(
                        MeasurementDraft(
                            name="individual_count",
                            raw_value=match.group(1),
                            numeric_value=float(match.group(1)),
                            unit="individuals",
                            approximate=approximate,
                        )
                    )

        return self._dedupe(measurements)

    @staticmethod
    def _specialize_name(default_name: str, claim_type: str, sentence: str) -> str:
        lowered = sentence.lower()
        if default_name == "acres":
            if claim_type == "fire_incident":
                return "acres_burned"
            if "cut" in lowered or claim_type == "economic_use":
                return "land_area"
            return "acres"
        return default_name

    @staticmethod
    def _dedupe(measurements: list[MeasurementDraft]) -> list[MeasurementDraft]:
        seen: set[tuple[str, str, float | None, str | None]] = set()
        output: list[MeasurementDraft] = []
        for item in measurements:
            key = (item.name, item.raw_value, item.numeric_value, item.unit)
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output
