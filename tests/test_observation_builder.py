from gemynd.core.models import (
    ClaimEntityLinkRecord,
    ClaimLocationLinkRecord,
    ClaimPeriodLinkRecord,
    ClaimRecord,
    EntityRecord,
    MeasurementRecord,
)
from gemynd.ingest.derivation_context import build_derivation_contexts
from gemynd.ingest.observation_builder import build_observations


def _make_claim(
    claim_id: str,
    claim_type: str,
    paragraph_id: str = "para_1",
    epistemic_status: str = "certain",
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        run_id="run_1",
        paragraph_id=paragraph_id,
        claim_type=claim_type,
        source_sentence="test sentence",
        normalized_sentence="test sentence",
        certainty=epistemic_status,
        extraction_confidence=0.78,
        claim_date="1956-04-15",
    )


def _make_measurement(measurement_id: str, claim_id: str, name: str = "individual_count") -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=measurement_id,
        claim_id=claim_id,
        run_id="run_1",
        name=name,
        raw_value="3000",
        numeric_value=3000.0,
        unit="individuals",
    )


def _make_entity(entity_id: str, label: str, name: str) -> EntityRecord:
    return EntityRecord(entity_id=entity_id, entity_type=label, name=name, normalized_form=name.lower())


def _run(
    *,
    claims: list[ClaimRecord],
    measurements: list[MeasurementRecord] | None = None,
    claim_entity_links: list[ClaimEntityLinkRecord] | None = None,
    claim_location_links: list[ClaimLocationLinkRecord] | None = None,
    claim_period_links: list[ClaimPeriodLinkRecord] | None = None,
    entity_lookup: dict[str, EntityRecord] | None = None,
    report_year: int | None = 1956,
):
    contexts = build_derivation_contexts(
        claims=claims,
        measurements=measurements or [],
        claim_entity_links=claim_entity_links or [],
        claim_location_links=claim_location_links or [],
        claim_period_links=claim_period_links or [],
        entity_lookup=entity_lookup or {},
        run_id="run_1",
        report_year=report_year,
    )
    return build_observations(contexts, "run_1")


def test_eligible_claim_produces_observation() -> None:
    claim = _make_claim("c1", "population_estimate")
    species = _make_entity("sp1", "Species", "mallard")
    entity_links = [ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="SPECIES_FOCUS")]

    observations, years, _, _ = _run(
        claims=[claim],
        claim_entity_links=entity_links,
        entity_lookup={species.entity_id: species},
    )

    assert len(observations) == 1
    obs = observations[0]
    assert obs.observation_type == "population_count"
    assert obs.species_id == "sp1"
    assert obs.claim_id == "c1"
    assert obs.year_id == "year_1956"
    assert len(years) == 1
    assert years[0].year == 1956


def test_non_eligible_claim_produces_no_observation() -> None:
    claim = _make_claim("c1", "economic_use")
    observations, _, _, _ = _run(claims=[claim])
    assert len(observations) == 0


def test_measurements_linked_to_observation() -> None:
    claim = _make_claim("c1", "population_estimate")
    measurement = _make_measurement("m1", "c1")
    species = _make_entity("sp1", "Species", "mallard")
    entity_links = [ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="SPECIES_FOCUS")]

    observations, _, obs_links, _ = _run(
        claims=[claim],
        measurements=[measurement],
        claim_entity_links=entity_links,
        entity_lookup={species.entity_id: species},
    )

    assert len(obs_links) == 1
    assert obs_links[0].measurement_id == "m1"
    assert obs_links[0].observation_id == observations[0].observation_id


def test_uncertain_claim_sets_is_estimate() -> None:
    claim = _make_claim("c1", "population_estimate", epistemic_status="uncertain")
    observations, _, _, _ = _run(claims=[claim])
    assert observations[0].is_estimate is True


def test_location_entities_assigned() -> None:
    claim = _make_claim("c1", "species_presence")
    refuge = _make_entity("r1", "Refuge", "Turnbull Refuge")
    place = _make_entity("p1", "Place", "Pine Creek")

    observations, _, _, _ = _run(
        claims=[claim],
        claim_location_links=[
            ClaimLocationLinkRecord(claim_id="c1", entity_id="r1"),
            ClaimLocationLinkRecord(claim_id="c1", entity_id="p1"),
        ],
        entity_lookup={refuge.entity_id: refuge, place.entity_id: place},
    )

    assert observations[0].refuge_id == "r1"
    assert observations[0].place_id == "p1"


def test_management_target_species_and_method_focus_feed_observation() -> None:
    claim = _make_claim("c1", "predator_control")
    species = _make_entity("sp1", "Species", "coyote")
    method = _make_entity("sm1", "SurveyMethod", "ground count")
    habitat = _make_entity("h1", "Habitat", "marsh")

    observations, _, _, _ = _run(
        claims=[claim],
        claim_entity_links=[
            ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="MANAGEMENT_TARGET"),
            ClaimEntityLinkRecord(claim_id="c1", entity_id="sm1", relation_type="METHOD_FOCUS"),
            ClaimEntityLinkRecord(claim_id="c1", entity_id="h1", relation_type="HABITAT_FOCUS"),
        ],
        entity_lookup={species.entity_id: species, method.entity_id: method, habitat.entity_id: habitat},
    )

    obs = observations[0]
    assert obs.species_id == "sp1"
    assert obs.survey_method_id == "sm1"
    assert obs.habitat_id == "h1"


def test_year_derived_from_claim_date() -> None:
    claim = _make_claim("c1", "population_estimate")
    claim.claim_date = "1942-06-15"

    observations, years, _, _ = _run(claims=[claim])

    assert observations[0].year_id == "year_1942"
    assert any(y.year == 1942 for y in years)


def test_year_falls_back_to_report_year() -> None:
    claim = _make_claim("c1", "population_estimate")
    claim.claim_date = None

    observations, years, _, _ = _run(claims=[claim], report_year=1938)

    assert observations[0].year_id == "year_1938"
    assert any(y.year == 1938 for y in years)


def test_source_claim_type_stamped() -> None:
    claim = _make_claim("c1", "population_estimate")
    observations, _, _, _ = _run(claims=[claim])
    assert observations[0].source_claim_type == "population_estimate"


def test_year_source_from_claim_date() -> None:
    claim = _make_claim("c1", "population_estimate")
    claim.claim_date = "1942-06-15"
    observations, _, _, _ = _run(claims=[claim])
    obs = observations[0]
    assert obs.year == 1942
    assert obs.year_source == "claim_date"


def test_year_source_from_report_year() -> None:
    claim = _make_claim("c1", "population_estimate")
    claim.claim_date = None
    observations, _, _, _ = _run(claims=[claim], report_year=1955)
    obs = observations[0]
    assert obs.year == 1955
    assert obs.year_source == "document_primary_year"


def test_year_source_unknown_when_no_date_or_report_year() -> None:
    claim = _make_claim("c1", "population_estimate")
    claim.claim_date = None
    observations, _, _, _ = _run(claims=[claim], report_year=None)
    obs = observations[0]
    assert obs.year is None
    assert obs.year_source == "unknown"
