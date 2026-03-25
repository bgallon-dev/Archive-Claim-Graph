from __future__ import annotations

import re
from dataclasses import dataclass

from graphrag_pipeline.core.models import ClaimRecord


@dataclass(slots=True)
class ConceptAssignment:
    claim_id: str
    concept_id: str
    confidence: float
    matched_rule: str


# Each rule is (concept_id, claim_types, regex_pattern, confidence)
# Listed from most specific to most general within each concept
_CONCEPT_RULES: list[tuple[str, frozenset[str], re.Pattern[str], float]] = [

    # ── Outcome concepts (most specific — check first) ──────────────────────
    (
        "concept_nesting_success",
        frozenset({"breeding_activity", "population_estimate",
                   "wildlife_count", "species_presence"}),
        re.compile(
            r"\b(nest(?:ed|ing|s)?|hatch(?:ed|ing)?|fledg(?:ed|ing|ling)?|"
            r"brood|clutch|reared|young|chick|duckling|gosling|cygnet)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "concept_breeding_success",
        frozenset({"breeding_activity", "population_estimate",
                   "wildlife_count", "species_presence"}),
        re.compile(
            r"\b(reared to maturity|breeding success|successful(?:ly)? bred|"
            r"pairs? nested|breeding pair|mating|courting|courtship)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "concept_population_decline",
        frozenset({"population_estimate", "wildlife_count",
                   "species_presence", "species_absence"}),
        re.compile(
            r"\b(declin(?:e|ed|ing)|decreas(?:e|ed|ing)|reduc(?:e|ed|tion)|"
            r"fewer than|lower than|less than.*(?:year|season|period)|"
            r"not(?:ably)? fewer|absent|disappear|none observed|no\s+\w+\s+seen)\b",
            re.IGNORECASE,
        ),
        0.82,
    ),

    # ── Environmental stress concepts ────────────────────────────────────────
    (
        "concept_drought_condition",
        frozenset({"weather_observation", "weather_condition",
                   "habitat_condition", "species_absence"}),
        re.compile(
            r"\b(drought|dried? up|dry(?:ing)?|no rain|lack of water|"
            r"water scarc|dewater|low water|water shortage|"
            r"potholes?\s+(?:dried|empty|dry)|desicat)\b",
            re.IGNORECASE,
        ),
        0.88,
    ),
    (
        "concept_flood_condition",
        frozenset({"weather_observation", "weather_condition",
                   "habitat_condition", "management_action"}),
        re.compile(
            r"\b(flood(?:ed|ing)?|inundat(?:ed|ion)|high water|overflow(?:ed|ing)?|"
            r"water(?:logged| too high)|standing water|excessive.*rain|"
            r"spring flood|runoff)\b",
            re.IGNORECASE,
        ),
        0.88,
    ),
    (
        "concept_temperature_extremes",
        frozenset({"weather_observation", "weather_condition"}),
        re.compile(
            r"\b(temperature|degrees?|heat wave|cold snap|frost|freeze|frozen|"
            r"below zero|extreme(?:ly)?\s+(?:hot|cold|warm|cool)|"
            r"record(?:ed)?\s+(?:high|low)|thermometer)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "concept_precipitation_pattern",
        frozenset({"weather_observation", "weather_condition"}),
        re.compile(
            r"\b(rain(?:fall)?|precip(?:itation)?|snow(?:fall)?|inch(?:es)? of|"
            r"hundredths? of an inch|moisture|wet(?:ter)?|dry(?:er)?|"
            r"annual precipitation|below normal|above normal)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),

    # ── Habitat state concepts ───────────────────────────────────────────────
    (
        "concept_habitat_degradation",
        frozenset({"habitat_condition", "management_action",
                   "fire_incident", "economic_use"}),
        re.compile(
            r"\b(degrad(?:e|ed|ation)|overgraze|overgraz(?:ed|ing)|"
            r"erode|erosion|deplet(?:e|ed|ion)|deteriorat(?:e|ed|ion)|"
            r"damage(?:d)?|destroy(?:ed)?|invasive|drain(?:ed|age)|"
            r"compacted|weed(?:s|ed|ing)?|cheat grass|thistle)\b",
            re.IGNORECASE,
        ),
        0.82,
    ),
    (
        "concept_water_level_change",
        frozenset({"habitat_condition", "weather_observation",
                   "weather_condition", "management_action"}),
        re.compile(
            r"\b(water level|water(?:s)?\s+(?:rose|risen|fell|dropped|raised|"
            r"lowered|receded|high|low)|pool(?:s|ed|ing)|pothole\s+(?:full|empty|"
            r"dry|low|high)|impoundment|water control|dike|dam)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "concept_habitat_condition",
        frozenset({"habitat_condition", "management_action",
                   "species_presence", "species_absence"}),
        re.compile(
            r"\b(habitat|vegetation|cover|marsh|wetland|grassland|upland|"
            r"shrub|brush|canopy|understory|riparian|range condition|"
            r"fair|good|poor|excellent)\b",
            re.IGNORECASE,
        ),
        0.78,
    ),

    # ── Management / restoration concepts ───────────────────────────────────
    (
        "concept_ecological_restoration",
        frozenset({"management_action", "development_activity",
                   "habitat_condition"}),
        re.compile(
            r"\b(restor(?:e|ed|ation|ing)|revegetat(?:e|ed|ion)|"
            r"reintroduc(?:e|ed|tion)|reseed(?:ed|ing)?|native plant(?:ing)?|"
            r"prescribed burn|controlled burn|burn(?:ing)? for habitat|"
            r"moist soil|water management for)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "concept_infrastructure_rehabilitation",
        frozenset({"development_activity", "management_action"}),
        re.compile(
            r"\b(construct(?:ed|ion)|repair(?:ed|ing)?|built|building|"
            r"dike|road|bridge|culvert|pump(?:ing station)?|headgate|"
            r"structure|facility|equipment|WPA|project\s+(?:started|completed)|"
            r"rehabilitat(?:e|ed|ion))\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "concept_habitat_restoration",
        frozenset({"management_action", "development_activity",
                   "habitat_condition"}),
        re.compile(
            r"\b(improv(?:e|ed|ement|ing)|enhanc(?:e|ed|ement|ing)|"
            r"plant(?:ed|ing)|wild celery|food plot|nesting cover|"
            r"water control|level control|managed(?:\s+for)?)\b",
            re.IGNORECASE,
        ),
        0.78,
    ),

    # ── Observational concepts ───────────────────────────────────────────────
    (
        "concept_population_count",
        frozenset({"population_estimate", "wildlife_count",
                   "species_presence", "predator_control"}),
        re.compile(
            r"\b(\d+\s+(?:mallard|teal|coot|goose|geese|duck|pheasant|"
            r"quail|partridge|grouse|deer|coyote|bird|individual)|"
            r"approx(?:imately)?\s+\d+|counted|census|total\s+of\s+\d+|"
            r"estimated\s+\d+|population\s+of)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "concept_survey_result",
        frozenset({"population_estimate", "wildlife_count",
                   "species_presence", "species_absence",
                   "breeding_activity", "migration_timing"}),
        re.compile(
            r"\b(survey(?:ed|ing)?|census|count(?:ed|ing)?|observed|"
            r"recorded|noted|seen|found|detected|aerial|ground count|"
            r"roadside|transect|banding|trap(?:ped|ping)?)\b",
            re.IGNORECASE,
        ),
        0.78,
    ),
    (
        "concept_breeding_activity",
        frozenset({"breeding_activity", "species_presence",
                   "population_estimate", "wildlife_count"}),
        re.compile(
            r"\b(breed(?:ing|ers?)?|nest(?:ing|ers?)?|pair(?:ing|s)?|"
            r"mating|display|territorial|singing|calling|courtship|"
            r"nuptial|spawning)\b",
            re.IGNORECASE,
        ),
        0.80,
    ),

    # ── Temporal / seasonal concept (broadest — check last) ─────────────────
    (
        "concept_seasonal_condition",
        frozenset({"weather_observation", "weather_condition",
                   "habitat_condition", "migration_timing",
                   "breeding_activity", "species_presence"}),
        re.compile(
            r"\b(spring|summer|fall|winter|autumn|seasonal|annual|"
            r"migration|arrival|departure|open season|hunting season|"
            r"early|late\s+(?:spring|summer|fall|winter)|"
            r"first\s+(?:of\s+the\s+)?season)\b",
            re.IGNORECASE,
        ),
        0.75,
    ),
]


def assign_concepts(claim: ClaimRecord) -> list[ConceptAssignment]:
    """Return concept assignments for a claim based on rule matching.

    A claim may match multiple concepts. All matches are returned —
    the caller decides whether to write all or only the top-scoring one.
    Concepts are only assigned when the claim_type is in the allowed
    set for that concept rule, preventing cross-domain noise.
    """
    assignments: list[ConceptAssignment] = []
    sentence = claim.source_sentence or ""

    for concept_id, allowed_types, pattern, confidence in _CONCEPT_RULES:
        if claim.claim_type not in allowed_types:
            continue
        if pattern.search(sentence):
            assignments.append(
                ConceptAssignment(
                    claim_id=claim.claim_id,
                    concept_id=concept_id,
                    confidence=confidence,
                    matched_rule=pattern.pattern[:60],
                )
            )

    return assignments
