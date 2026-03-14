import re

from graphrag_pipeline.extractors.claim_extractor import ClaimDraft
from graphrag_pipeline.extractors.measurement_extractor import RuleBasedMeasurementExtractor


def test_measurement_normalization_for_fire_claim() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="fire_incident",
        source_sentence="A fire burned 100 acres and killed about 100 cords of wood with suppression cost of $52.50.",
        normalized_sentence="a fire burned 100 acres and killed about 100 cords of wood with suppression cost of $52.50.",
        epistemic_status="uncertain",
        extraction_confidence=0.7,
    )

    measurements = extractor.extract(claim)
    names = {item.name for item in measurements}

    assert "acres_burned" in names
    assert "wood_killed_cords" in names
    assert "suppression_cost" in names
    assert any(item.approximate for item in measurements)


def test_year_excluded_from_wildlife_count() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="In 1938 approximately 900 mallards were counted on the refuge.",
        normalized_sentence="in 1938 approximately 900 mallards were counted on the refuge.",
        epistemic_status="uncertain",
        extraction_confidence=0.7,
    )
    measurements = extractor.extract(claim)
    values = {m.numeric_value for m in measurements}
    assert 900.0 in values
    assert 1938.0 not in values


def test_page_table_numbers_excluded() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="See table 3 for the 250 mallards observed on page 12.",
        normalized_sentence="see table 3 for the 250 mallards observed on page 12.",
        epistemic_status="certain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    values = {m.numeric_value for m in measurements}
    assert 250.0 in values
    assert 3.0 not in values
    assert 12.0 not in values


def test_predator_control_date_numbers_excluded() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="predator_control",
        source_sentence="Predator control continued: coyotes taken monthly July 3, Aug 5, Sept 2, Oct 1, total 11.",
        normalized_sentence="predator control continued: coyotes taken monthly july 3, aug 5, sept 2, oct 1, total 11.",
        epistemic_status="certain",
        extraction_confidence=0.78,
    )
    measurements = extractor.extract(claim)
    values = {m.numeric_value for m in measurements}
    assert 11.0 in values
    assert 3.0 not in values
    assert 5.0 not in values
    assert 2.0 not in values
    assert 1.0 not in values


def test_count_candidate_with_species_context() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="Estimated 500 coot and 100 teal were observed on the refuge.",
        normalized_sentence="estimated 500 coot and 100 teal were observed on the refuge.",
        epistemic_status="uncertain",
        extraction_confidence=0.7,
    )
    measurements = extractor.extract(claim)
    values = {m.numeric_value for m in measurements}
    assert 500.0 in values
    assert 100.0 in values
    assert len(measurements) == 2


def test_percentage_excluded_from_count() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="About 50% of the 200 mallards were juveniles.",
        normalized_sentence="about 50% of the 200 mallards were juveniles.",
        epistemic_status="uncertain",
        extraction_confidence=0.7,
    )
    measurements = extractor.extract(claim)
    values = {m.numeric_value for m in measurements}
    assert 200.0 in values
    assert 50.0 not in values


def test_classify_number_directly() -> None:
    classify = RuleBasedMeasurementExtractor._classify_number
    generic = re.compile(r"\b(\d+(?:\.\d+)?)\b")

    sentence = "In 1938 about 500 geese were observed on page 7."
    matches = list(generic.finditer(sentence))
    # matches: 1938, 500, 7
    assert classify(1938.0, matches[0], sentence) == "year_candidate"
    assert classify(500.0, matches[1], sentence) == "count_candidate"
    assert classify(7.0, matches[2], sentence) == "page_table_candidate"


# ── Change 1: local noun attachment ──────────────────────────────────────────

def test_serial_ref_excluded() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="Permit 47 was issued for the refuge area and 85 mallards were observed.",
        normalized_sentence="permit 47 was issued for the refuge area and 85 mallards were observed.",
        epistemic_status="certain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    values = {m.numeric_value for m in measurements}
    assert 85.0 in values
    assert 47.0 not in values


def test_date_unit_excluded() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="After 3 years of monitoring 200 mallards were counted.",
        normalized_sentence="after 3 years of monitoring 200 mallards were counted.",
        epistemic_status="certain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    values = {m.numeric_value for m in measurements}
    assert 200.0 in values
    assert 3.0 not in values


def test_immediate_species_high_confidence() -> None:
    """Number immediately followed by species noun should be count_candidate."""
    classify = RuleBasedMeasurementExtractor._classify_number
    generic = re.compile(r"\b(\d+(?:\.\d+)?)\b")
    sentence = "We observed 45 mallards on the pond."
    matches = list(generic.finditer(sentence))
    assert classify(45.0, matches[0], sentence) == "count_candidate"


# ── Change 2: target_surface and target_entity_type_hint ─────────────────────

def test_target_surface_populated() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="500 mallards were counted on the refuge.",
        normalized_sentence="500 mallards were counted on the refuge.",
        epistemic_status="certain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    count_m = next(m for m in measurements if m.name == "individual_count")
    assert count_m.target_surface is not None
    assert "mallard" in count_m.target_surface.lower()
    assert count_m.target_entity_type_hint == "waterfowl"


def test_target_surface_predator() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="predator_control",
        source_sentence="12 coyotes were removed from the area.",
        normalized_sentence="12 coyotes were removed from the area.",
        epistemic_status="certain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    count_m = next((m for m in measurements if m.name == "individual_count"), None)
    assert count_m is not None
    assert count_m.target_entity_type_hint == "predator"


def test_both_species_get_target_surface() -> None:
    """Each count measurement should carry the specific species it refers to."""
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="Estimated 500 coot and 100 teal were observed on the refuge.",
        normalized_sentence="estimated 500 coot and 100 teal were observed on the refuge.",
        epistemic_status="uncertain",
        extraction_confidence=0.7,
    )
    measurements = extractor.extract(claim)
    surfaces = {m.target_surface for m in measurements if m.target_surface}
    assert any("coot" in (s or "").lower() for s in surfaces)
    assert any("teal" in (s or "").lower() for s in surfaces)


# ── Change 3: ranges and comparative forms ────────────────────────────────────

def test_hyphenated_range_species() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="Between 200-300 mallards were counted on the refuge.",
        normalized_sentence="between 200-300 mallards were counted on the refuge.",
        epistemic_status="uncertain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    range_m = next((m for m in measurements if m.lower_bound is not None and m.upper_bound is not None), None)
    assert range_m is not None
    assert range_m.lower_bound == 200.0
    assert range_m.upper_bound == 300.0
    assert range_m.name == "individual_count"
    assert range_m.approximate is True


def test_between_range_acres() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="fire_incident",
        source_sentence="Between 40 and 50 acres were burned in the fire.",
        normalized_sentence="between 40 and 50 acres were burned in the fire.",
        epistemic_status="uncertain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    range_m = next((m for m in measurements if m.lower_bound is not None and m.upper_bound is not None), None)
    assert range_m is not None
    assert range_m.lower_bound == 40.0
    assert range_m.upper_bound == 50.0
    assert range_m.name == "acres_burned"
    assert range_m.unit == "acres"


def test_lower_bound_comparative() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="predator_control",
        source_sentence="More than 12 coyotes were removed from the refuge.",
        normalized_sentence="more than 12 coyotes were removed from the refuge.",
        epistemic_status="certain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    bound_m = next((m for m in measurements if m.lower_bound is not None), None)
    assert bound_m is not None
    assert bound_m.lower_bound == 12.0
    assert bound_m.upper_bound is None
    assert bound_m.approximate is True


def test_upper_bound_comparative() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="wildlife_count",
        source_sentence="Up to 300 geese were observed on the lake.",
        normalized_sentence="up to 300 geese were observed on the lake.",
        epistemic_status="uncertain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    bound_m = next((m for m in measurements if m.upper_bound is not None), None)
    assert bound_m is not None
    assert bound_m.upper_bound == 300.0
    assert bound_m.lower_bound is None
    assert bound_m.approximate is True


def test_no_fewer_than_comparative() -> None:
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="population_estimate",
        source_sentence="No fewer than 20 mallards were recorded this season.",
        normalized_sentence="no fewer than 20 mallards were recorded this season.",
        epistemic_status="uncertain",
        extraction_confidence=0.7,
    )
    measurements = extractor.extract(claim)
    bound_m = next((m for m in measurements if m.lower_bound is not None), None)
    assert bound_m is not None
    assert bound_m.lower_bound == 20.0


def test_range_does_not_double_count_acres() -> None:
    """'200-300 acres' should yield one range measurement, not also a stray '300 acres' point."""
    extractor = RuleBasedMeasurementExtractor()
    claim = ClaimDraft(
        claim_type="fire_incident",
        source_sentence="The fire burned 200-300 acres of marsh.",
        normalized_sentence="the fire burned 200-300 acres of marsh.",
        epistemic_status="uncertain",
        extraction_confidence=0.8,
    )
    measurements = extractor.extract(claim)
    acres_measurements = [m for m in measurements if "acres" in (m.unit or "")]
    assert len(acres_measurements) == 1
    assert acres_measurements[0].lower_bound == 200.0
    assert acres_measurements[0].upper_bound == 300.0
