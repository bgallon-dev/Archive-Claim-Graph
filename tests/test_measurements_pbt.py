"""Property-based tests for RuleBasedMeasurementExtractor.

Four invariants exercised across the combinatorial space of counts, species,
months, and range bounds — catching regression classes that fixed-input tests
cannot reach:

  1. Species-count: a bare integer immediately right of a known species noun is
     *always* extracted as an ``individual_count`` measurement with
     ``target_surface`` populated.

  2. Month-day exclusion: an integer immediately right of a month name (when
     *not* adjacent to a species noun) is *never* extracted.

  3. Hyphenated range exactness: ``X-Y species`` produces *exactly one* range
     measurement (lower_bound=X, upper_bound=Y); neither endpoint leaks as a
     separate point value.

  4. Consumed-spans guarantee: ``X-Y acres`` produces *exactly one* measurement;
     the pattern-spec for "N acres" cannot fire on either endpoint because both
     positions are already marked consumed.
"""
from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from gemynd.ingest.extractors.claim_extractor import ClaimDraft
from gemynd.ingest.extractors.measurement_extractor import RuleBasedMeasurementExtractor

# ---------------------------------------------------------------------------
# Shared extractor — constructed once so YAML is loaded only once per session.
# ---------------------------------------------------------------------------
_EXT = RuleBasedMeasurementExtractor()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Integers that are never year-like (1800–2100) so _classify_number cannot
# short-circuit to "year_candidate".
_COUNT = st.integers(min_value=1, max_value=999)

# Endpoints for range tests; both stay safely below the year threshold.
_RANGE_LO = st.integers(min_value=1, max_value=499)
_RANGE_HI = st.integers(min_value=1, max_value=499)

# Day-of-month values (also non-year).
_DAY = st.integers(min_value=1, max_value=28)

# All species whose normalised stem lives in measurement_species.yaml type_hints
# and whose plural form appears in the immediate_patterns list.
_SPECIES = st.sampled_from([
    "mallards", "teal", "coots", "geese",
    "pheasants", "deer", "coyotes", "ducks",
    "birds", "cranes", "rabbits", "muskrats",
    "foxes", "raccoons",
])

# Claim types for which the extractor's fallback and range paths are enabled.
_COUNT_TYPES = st.sampled_from([
    "wildlife_count", "population_estimate",
    "species_presence", "predator_control", "breeding_activity",
])

# Month forms recognised by _month_re (trailing whitespace is irrelevant here
# because the sentence template places a space between month and day).
_MONTHS = st.sampled_from([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.",
    "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _draft(sentence: str, claim_type: str = "wildlife_count") -> ClaimDraft:
    return ClaimDraft(
        claim_type=claim_type,
        source_sentence=sentence,
        normalized_sentence=sentence.lower(),
        epistemic_status="uncertain",
        extraction_confidence=0.8,
    )


# ---------------------------------------------------------------------------
# Property 1 — species-count invariant
# A bare integer immediately followed by a species noun must always be
# extracted as an individual_count with target_surface set.
# ---------------------------------------------------------------------------

@given(count=_COUNT, species=_SPECIES, claim_type=_COUNT_TYPES)
@settings(max_examples=300)
def test_species_count_always_extracted(count: int, species: str, claim_type: str) -> None:
    """``N species`` → ``float(N)`` always appears in extracted numeric values."""
    sentence = f"Approximately {count} {species} were observed on the refuge."
    ms = _EXT.extract(_draft(sentence, claim_type))
    values = {m.numeric_value for m in ms}
    assert float(count) in values, (
        f"count={count} species={species!r} claim_type={claim_type!r}\n"
        f"  sentence : {sentence}\n"
        f"  extracted: {sorted(v for v in values if v is not None)}"
    )


@given(count=_COUNT, species=_SPECIES, claim_type=_COUNT_TYPES)
@settings(max_examples=200)
def test_species_count_name_is_individual_count(
    count: int, species: str, claim_type: str
) -> None:
    """The measurement carrying the species count must be named 'individual_count'."""
    sentence = f"Approximately {count} {species} were observed on the refuge."
    ms = _EXT.extract(_draft(sentence, claim_type))
    matching = [m for m in ms if m.numeric_value == float(count)]
    assert matching, f"No measurement for {count} in {sentence!r}"
    assert all(m.name == "individual_count" for m in matching), (
        f"Unexpected names: {[m.name for m in matching]}"
    )


@given(count=_COUNT, species=_SPECIES, claim_type=_COUNT_TYPES)
@settings(max_examples=200)
def test_species_count_target_surface_populated(
    count: int, species: str, claim_type: str
) -> None:
    """target_surface must be set when a species noun immediately follows the count."""
    sentence = f"Approximately {count} {species} were observed on the refuge."
    ms = _EXT.extract(_draft(sentence, claim_type))
    matching = [m for m in ms if m.numeric_value == float(count)]
    assert matching, f"No measurement for {count} in {sentence!r}"
    for m in matching:
        assert m.target_surface is not None, (
            f"target_surface is None for species={species!r} in {sentence!r}"
        )


# ---------------------------------------------------------------------------
# Property 2 — month-day exclusion
# An integer N that immediately follows a month name (and is NOT itself
# adjacent to a species noun) must never appear in the extracted values.
# ---------------------------------------------------------------------------

@given(
    count=_COUNT,
    species=_SPECIES,
    claim_type=_COUNT_TYPES,
    day=_DAY,
    month=_MONTHS,
)
@settings(max_examples=400)
def test_month_day_not_extracted(
    count: int, species: str, claim_type: str, day: int, month: str
) -> None:
    """``{month} {day}`` at sentence-end must not contribute an individual_count."""
    # Avoid ambiguity when both numbers are equal.
    assume(day != count)
    # Sentence structure: species count first (gives fallback a context match),
    # then an isolated date at the tail so the day is never adjacent to a
    # species noun and is always right of a month name.
    sentence = (
        f"Counted {count} {species} on the refuge; "
        f"last survey was {month} {day}."
    )
    ms = _EXT.extract(_draft(sentence, claim_type))
    extracted = {m.numeric_value for m in ms}
    assert float(day) not in extracted, (
        f"month-day {day!r} (after {month!r}) must not be extracted\n"
        f"  sentence : {sentence}\n"
        f"  extracted: {sorted(v for v in extracted if v is not None)}"
    )


# ---------------------------------------------------------------------------
# Property 3 — hyphenated range: exactly one measurement, no endpoint leak
# X-Y species must produce a single range measurement (lower_bound=X,
# upper_bound=Y, approximate=True); neither X nor Y may appear as a separate
# point value.
# ---------------------------------------------------------------------------

@given(lo=_RANGE_LO, hi=_RANGE_HI, species=_SPECIES, claim_type=_COUNT_TYPES)
@settings(max_examples=400)
def test_hyphenated_range_species_single_measurement(
    lo: int, hi: int, species: str, claim_type: str
) -> None:
    """``X-Y species`` → exactly 1 range measurement; no stray point values for X or Y."""
    assume(lo < hi)
    sentence = f"{lo}-{hi} {species} were counted on the refuge this season."
    ms = _EXT.extract(_draft(sentence, claim_type))

    range_ms = [m for m in ms if m.lower_bound is not None and m.upper_bound is not None]
    assert len(range_ms) == 1, (
        f"Expected 1 range measurement, got {len(range_ms)}\n"
        f"  sentence : {sentence}\n"
        f"  measurements: {[(m.name, m.lower_bound, m.upper_bound, m.numeric_value) for m in ms]}"
    )
    assert range_ms[0].lower_bound == float(lo), "lower_bound mismatch"
    assert range_ms[0].upper_bound == float(hi), "upper_bound mismatch"
    assert range_ms[0].name == "individual_count", (
        f"Expected name='individual_count', got {range_ms[0].name!r}"
    )

    # Consumed-span guard: neither endpoint should leak as a standalone value.
    point_values = {m.numeric_value for m in ms if m.numeric_value is not None}
    assert float(lo) not in point_values, (
        f"lo={lo} leaked as point measurement in {sentence!r}"
    )
    assert float(hi) not in point_values, (
        f"hi={hi} leaked as point measurement in {sentence!r}"
    )


@given(lo=_RANGE_LO, hi=_RANGE_HI, species=_SPECIES, claim_type=_COUNT_TYPES)
@settings(max_examples=200)
def test_hyphenated_range_is_approximate(
    lo: int, hi: int, species: str, claim_type: str
) -> None:
    """Range measurements must carry approximate=True regardless of sentence phrasing."""
    assume(lo < hi)
    sentence = f"{lo}-{hi} {species} were counted on the refuge."
    ms = _EXT.extract(_draft(sentence, claim_type))
    range_ms = [m for m in ms if m.lower_bound is not None and m.upper_bound is not None]
    assert range_ms, f"No range measurement found in {sentence!r}"
    for m in range_ms:
        assert m.approximate is True, (
            f"approximate=False on range measurement in {sentence!r}"
        )


# ---------------------------------------------------------------------------
# Property 4 — consumed-spans guarantee (acres)
# X-Y acres → exactly one measurement; the pattern-spec for "N acres" cannot
# fire on either X or Y because both character positions are marked consumed
# by the range extractor before pattern-specs run.
# ---------------------------------------------------------------------------

@given(lo=_RANGE_LO, hi=_RANGE_HI)
@settings(max_examples=400)
def test_hyphenated_acres_range_no_double_count(lo: int, hi: int) -> None:
    """``X-Y acres`` → 1 acres measurement; pattern-spec cannot fire on hi."""
    assume(lo < hi)
    sentence = f"The fire burned {lo}-{hi} acres of marsh."
    ms = _EXT.extract(_draft(sentence, "fire_incident"))

    acres_ms = [m for m in ms if m.unit == "acres"]
    assert len(acres_ms) == 1, (
        f"Expected 1 acres measurement, got {len(acres_ms)}\n"
        f"  sentence : {sentence}\n"
        f"  measurements: {[(m.name, m.lower_bound, m.upper_bound, m.numeric_value, m.unit) for m in ms]}"
    )
    assert acres_ms[0].lower_bound == float(lo)
    assert acres_ms[0].upper_bound == float(hi)

    point_values = {m.numeric_value for m in ms if m.numeric_value is not None}
    assert float(lo) not in point_values, f"lo={lo} leaked as point in {sentence!r}"
    assert float(hi) not in point_values, f"hi={hi} leaked as point in {sentence!r}"


@given(lo=_RANGE_LO, hi=_RANGE_HI)
@settings(max_examples=300)
def test_between_and_acres_range_no_double_count(lo: int, hi: int) -> None:
    """``between X and Y acres`` (verbal form) also consumes spans correctly."""
    assume(lo < hi)
    sentence = f"Between {lo} and {hi} acres were burned in the fire."
    ms = _EXT.extract(_draft(sentence, "fire_incident"))

    acres_ms = [m for m in ms if m.unit == "acres"]
    assert len(acres_ms) == 1, (
        f"Expected 1 acres measurement, got {len(acres_ms)}\n"
        f"  sentence : {sentence}\n"
        f"  measurements: {[(m.name, m.lower_bound, m.upper_bound, m.numeric_value) for m in ms]}"
    )
    assert acres_ms[0].lower_bound == float(lo)
    assert acres_ms[0].upper_bound == float(hi)

    point_values = {m.numeric_value for m in ms if m.numeric_value is not None}
    assert float(lo) not in point_values, f"lo={lo} leaked as point in {sentence!r}"
    assert float(hi) not in point_values, f"hi={hi} leaked as point in {sentence!r}"
