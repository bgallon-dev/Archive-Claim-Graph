from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from typing import Protocol

from .claim_extractor import ClaimDraft
from ..resource_loader import load_measurement_units, load_measurement_species

_LOADED_UNITS: dict[str, tuple[str, str]] = load_measurement_units()
_LOADED_SPECIES: dict[str, Any] = load_measurement_species()
_SPECIES_IMMEDIATE_PATTERN = r"^\s*(" + "|".join(_LOADED_SPECIES["immediate_patterns"]) + r")\b"
_SPECIES_CONTEXT_VERBS = (
    r"counted|estimated|population|total|observed|taken|killed|trapped|shot|removed|reported"
)
_SPECIES_CONTEXT_PATTERN = (
    r"\b(" + "|".join(_LOADED_SPECIES["immediate_patterns"]) + r"|" + _SPECIES_CONTEXT_VERBS + r")\b"
)


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
    methodology_note: str | None = None
    target_surface: str | None = None
    target_entity_type_hint: str | None = None
    extraction_source: str = "rules"
    decision_trace: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    fallback_used: bool = False


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
    _individual_context = re.compile(_SPECIES_CONTEXT_PATTERN, re.IGNORECASE)
    _generic_number = re.compile(r"\b(\d+(?:\.\d+)?)\b")
    _page_table_re = re.compile(
        r"\b(pages?|tables?|figures?|fig\.?|sections?|sec\.?|no\.|number|item)\s*$",
        re.IGNORECASE,
    )
    _percent_re = re.compile(r"^\s*(%|percent)", re.IGNORECASE)
    _acre_right_re = re.compile(r"^\s*acres?\b", re.IGNORECASE)
    _count_context_re = re.compile(_SPECIES_CONTEXT_PATTERN, re.IGNORECASE)
    _month_re = re.compile(
        r"\b(jan\.?|feb\.?|mar\.?|apr\.?|may|jun\.?|jul\.?|aug\.?|sep\.?|sept\.?|oct\.?|nov\.?|dec\.?|"
        r"january|february|march|april|june|july|august|september|october|november|december)\s*$",
        re.IGNORECASE,
    )

    _estimated_re = re.compile(r"\b(estimated|approximately|approx|about|around)\b", re.IGNORECASE)
    _counted_re = re.compile(r"\b(counted|tallied|enumerated)\b", re.IGNORECASE)
    _reported_re = re.compile(r"\b(reported|observed|recorded|noted)\b", re.IGNORECASE)
    _surveyed_re = re.compile(r"\b(surveyed|censused|banded)\b", re.IGNORECASE)

    # Local noun attachment helpers
    _serial_ref_re = re.compile(
        r"\b(permit|serial|lot|parcel|tract|block|chapter|exhibit|appendix|ref\.?)\s*$",
        re.IGNORECASE,
    )
    _date_unit_re = re.compile(
        r"^\s*(years?|months?|weeks?|decades?|centuries?|century)\b",
        re.IGNORECASE,
    )
    _immediate_species_re = re.compile(_SPECIES_IMMEDIATE_PATTERN, re.IGNORECASE)

    # Range and comparative patterns
    _range_hyphen_re = re.compile(r"\b(\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(\d+(?:\.\d+)?)\b")
    _range_between_re = re.compile(
        r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE
    )
    _lower_bound_re = re.compile(
        r"\b(?:more\s+than|over|at\s+least|no?\s*fewer\s+than|not\s+fewer\s+than|"
        r"exceeding|above|greater\s+than)\s+(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )
    _upper_bound_re = re.compile(
        r"\b(?:less\s+than|under|up\s+to|at\s+most|no\s+more\s+than|"
        r"not\s+more\s+than|below|fewer\s+than)\s+(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )
    _unit_right_re = re.compile(
        r"^\s*(acres?|inches?|inch|days?|tons?|cords?|covies?)\b", re.IGNORECASE
    )

    # Unit string → (name, unit) used by range extraction to mirror _pattern_specs
    _UNIT_NAME_MAP: dict[str, tuple[str, str]] = _LOADED_UNITS

    _SPECIES_TYPE_HINTS: dict[str, str] = _LOADED_SPECIES["type_hints"]  # type: ignore[assignment]

    def _infer_methodology_note(self, sentence: str, approximate: bool) -> str | None:
        if approximate or self._estimated_re.search(sentence):
            return "estimated"
        if self._counted_re.search(sentence):
            return "counted"
        if self._surveyed_re.search(sentence):
            return "surveyed"
        if self._reported_re.search(sentence):
            return "reported"
        return None

    def _extract_ranges_and_bounds(
        self,
        sentence: str,
        claim_type: str,
        approximate: bool,
        methodology_note: str | None,
    ) -> tuple[list[MeasurementDraft], set[int]]:
        """Extract range and comparative measurements; return them and their consumed char positions."""
        measurements: list[MeasurementDraft] = []
        consumed: set[int] = set()

        _count_claim_types = {
            "wildlife_count", "predator_control",
            "population_estimate", "species_presence", "breeding_activity",
        }

        def _right_context(end: int) -> str:
            return sentence[end:min(len(sentence), end + 25)]

        def _classify_range_right(right: str, claim_type: str) -> tuple[str, str | None, str | None, str | None]:
            """Return (name, unit, target_surface, target_entity_type_hint) for right context of a range."""
            unit_m = self._unit_right_re.match(right)
            if unit_m:
                unit_word = unit_m.group(1).lower()
                key = unit_word.rstrip("s") if not unit_word.endswith("ies") else unit_word
                # normalise: covies → covy lookup
                if unit_word.startswith("cov"):
                    key = "covies"
                name, unit = self._UNIT_NAME_MAP.get(unit_word, self._UNIT_NAME_MAP.get(key, ("measurement", unit_word)))
                name = self._specialize_name(name, claim_type, sentence)
                return name, unit, None, None
            if claim_type in _count_claim_types:
                species_m = self._immediate_species_re.match(right)
                if species_m:
                    surface = species_m.group(0).strip()
                    k = surface.lower().rstrip("s")
                    hint = self._SPECIES_TYPE_HINTS.get(k, "wildlife")
                    return "individual_count", "individuals", surface, hint
            return "individual_count", "individuals", None, None

        def _range_trace(pattern_key: str, extra: str, meth: str | None) -> list[str]:
            t = [extra]
            if meth:
                t.append(f"methodology:{meth}")
            return t

        # Hyphenated ranges: "200-300 ducks"
        for m in self._range_hyphen_re.finditer(sentence):
            lo, hi = float(m.group(1)), float(m.group(2))
            right = _right_context(m.end())
            name, unit, surface, hint = _classify_range_right(right, claim_type)
            raw = m.group(0)
            measurements.append(MeasurementDraft(
                name=name,
                raw_value=raw,
                numeric_value=None,
                unit=unit,
                approximate=True,
                lower_bound=lo,
                upper_bound=hi,
                methodology_note=methodology_note,
                target_surface=surface,
                target_entity_type_hint=hint,
                decision_trace=_range_trace("range_hyphen", f"range_hyphen:{lo}-{hi}", methodology_note),
                matched_patterns=["range_hyphen"],
            ))
            consumed.update(range(m.start(), m.end()))

        # "between X and Y" ranges
        for m in self._range_between_re.finditer(sentence):
            lo, hi = float(m.group(1)), float(m.group(2))
            right = _right_context(m.end())
            name, unit, surface, hint = _classify_range_right(right, claim_type)
            raw = m.group(0)
            measurements.append(MeasurementDraft(
                name=name,
                raw_value=raw,
                numeric_value=None,
                unit=unit,
                approximate=True,
                lower_bound=lo,
                upper_bound=hi,
                methodology_note=methodology_note,
                target_surface=surface,
                target_entity_type_hint=hint,
                decision_trace=_range_trace("range_between", f"range_between:{lo}-{hi}", methodology_note),
                matched_patterns=["range_between"],
            ))
            consumed.update(range(m.start(), m.end()))

        # Lower-bound comparatives: "more than X", "at least X", etc.
        for m in self._lower_bound_re.finditer(sentence):
            if m.start() in consumed:
                continue
            val = float(m.group(1))
            right = _right_context(m.end())
            name, unit, surface, hint = _classify_range_right(right, claim_type)
            measurements.append(MeasurementDraft(
                name=name,
                raw_value=m.group(0),
                numeric_value=None,
                unit=unit,
                approximate=True,
                lower_bound=val,
                upper_bound=None,
                methodology_note=methodology_note,
                target_surface=surface,
                target_entity_type_hint=hint,
                decision_trace=_range_trace("lower_bound", f"lower_bound:{val}", methodology_note),
                matched_patterns=["lower_bound_re"],
            ))
            consumed.update(range(m.start(), m.end()))

        # Upper-bound comparatives: "less than X", "up to X", etc.
        for m in self._upper_bound_re.finditer(sentence):
            if m.start() in consumed:
                continue
            val = float(m.group(1))
            right = _right_context(m.end())
            name, unit, surface, hint = _classify_range_right(right, claim_type)
            measurements.append(MeasurementDraft(
                name=name,
                raw_value=m.group(0),
                numeric_value=None,
                unit=unit,
                approximate=True,
                lower_bound=None,
                upper_bound=val,
                methodology_note=methodology_note,
                target_surface=surface,
                target_entity_type_hint=hint,
                decision_trace=_range_trace("upper_bound", f"upper_bound:{val}", methodology_note),
                matched_patterns=["upper_bound_re"],
            ))
            consumed.update(range(m.start(), m.end()))

        return measurements, consumed

    def extract(self, claim: ClaimDraft) -> list[MeasurementDraft]:
        sentence = claim.source_sentence
        approximate = bool(self._approx_re.search(sentence))
        methodology_note = self._infer_methodology_note(sentence, approximate)
        measurements: list[MeasurementDraft] = []

        # Range and comparative extraction (runs first; marks consumed spans)
        range_measurements, consumed_spans = self._extract_ranges_and_bounds(
            sentence, claim.claim_type, approximate, methodology_note
        )
        measurements.extend(range_measurements)

        for pattern, default_name, default_unit in self._pattern_specs:
            for match in pattern.finditer(sentence):
                if match.start() in consumed_spans:
                    continue
                number = float(match.group(1))
                name = self._specialize_name(default_name, claim.claim_type, sentence)
                spec_trace = [f"pattern_spec:{name}"]
                if methodology_note:
                    spec_trace.append(f"methodology:{methodology_note}")
                measurements.append(
                    MeasurementDraft(
                        name=name,
                        raw_value=match.group(0),
                        numeric_value=number,
                        unit=default_unit,
                        approximate=approximate,
                        methodology_note=methodology_note,
                        decision_trace=spec_trace,
                        matched_patterns=[f"pattern_spec:{default_name}"],
                    )
                )

        for match in self._money_re.finditer(sentence):
            raw_match = match.group(0)
            if "$" not in raw_match and "usd" not in raw_match.lower() and "dollar" not in raw_match.lower():
                continue
            number = float(match.group(1))
            money_name = "suppression_cost" if claim.claim_type == "fire_incident" else "revenue"
            money_trace = [f"money:{money_name}"]
            if methodology_note:
                money_trace.append(f"methodology:{methodology_note}")
            measurements.append(
                MeasurementDraft(
                    name=money_name,
                    raw_value=raw_match.strip(),
                    numeric_value=number,
                    unit="USD",
                    approximate=approximate,
                    methodology_note=methodology_note,
                    decision_trace=money_trace,
                    matched_patterns=["money_re"],
                )
            )

        if not measurements and claim.claim_type in {
            "wildlife_count", "predator_control",
            "population_estimate", "species_presence", "breeding_activity",
        }:
            if self._individual_context.search(sentence):
                for match in self._generic_number.finditer(sentence):
                    if match.start() in consumed_spans:
                        continue
                    value = float(match.group(1))
                    role = self._classify_number(value, match, sentence)
                    if role == "count_candidate":
                        surface, hint = self._resolve_target_noun(match.end(), sentence)
                        fallback_trace = ["generic_number:fallback", "count_context_match"]
                        if methodology_note:
                            fallback_trace.append(f"methodology:{methodology_note}")
                        measurements.append(
                            MeasurementDraft(
                                name="individual_count",
                                raw_value=match.group(1),
                                numeric_value=value,
                                unit="individuals",
                                approximate=approximate,
                                methodology_note=methodology_note,
                                target_surface=surface,
                                target_entity_type_hint=hint,
                                decision_trace=fallback_trace,
                                matched_patterns=["generic_number"],
                                fallback_used=True,
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
    def _resolve_target_noun(match_end: int, sentence: str) -> tuple[str | None, str | None]:
        """Return (surface_text, entity_type_hint) for the species noun immediately right of position."""
        right = sentence[match_end:min(len(sentence), match_end + 25)]
        m = RuleBasedMeasurementExtractor._immediate_species_re.match(right)
        if not m:
            return None, None
        surface = m.group(0).strip()
        key = surface.lower().rstrip("s")
        type_hint = RuleBasedMeasurementExtractor._SPECIES_TYPE_HINTS.get(key, "wildlife")
        return surface, type_hint

    @staticmethod
    def _classify_number(value: float, match: re.Match[str], sentence: str) -> str:
        # Year: 4-digit integer in historical range
        if value == int(value) and 1800 <= value <= 2100:
            return "year_candidate"

        # Tight local noun attachment (±20 chars) — higher precision than wide context
        left_tight = sentence[max(0, match.start() - 20):match.start()]
        right_tight = sentence[match.end():min(len(sentence), match.end() + 20)]
        if RuleBasedMeasurementExtractor._serial_ref_re.search(left_tight):
            return "other_numeric"
        if RuleBasedMeasurementExtractor._date_unit_re.match(right_tight.lstrip()):
            return "other_numeric"
        if RuleBasedMeasurementExtractor._immediate_species_re.match(right_tight.lstrip()):
            return "count_candidate"

        left_ctx = sentence[max(0, match.start() - 30):match.start()]
        right_ctx = sentence[match.end():min(len(sentence), match.end() + 30)]

        # Page / table / figure / section reference
        if RuleBasedMeasurementExtractor._page_table_re.search(left_ctx):
            return "page_table_candidate"

        # Percentage
        if RuleBasedMeasurementExtractor._percent_re.search(right_ctx):
            return "percentage_candidate"

        # Acreage safety net
        if RuleBasedMeasurementExtractor._acre_right_re.search(right_ctx):
            return "acreage_candidate"

        # Day-of-month inside a date phrase (e.g. "July 3")
        left_narrow = sentence[max(0, match.start() - 15):match.start()]
        if RuleBasedMeasurementExtractor._month_re.search(left_narrow):
            return "other_numeric"

        # Count: species or counting words nearby
        left_wide = sentence[max(0, match.start() - 50):match.start()]
        right_wide = sentence[match.end():min(len(sentence), match.end() + 50)]
        context = f"{left_wide} {right_wide}"
        if RuleBasedMeasurementExtractor._count_context_re.search(context):
            return "count_candidate"

        return "other_numeric"

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
