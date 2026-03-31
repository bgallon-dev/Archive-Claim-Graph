from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from graphrag_pipeline.core.claim_contract import (
    CLAIM_TYPE_TO_OBSERVATION_TYPE,
    OBSERVATION_ELIGIBLE_TYPES,
)
from graphrag_pipeline.core.ids import make_observation_id, make_year_id
from graphrag_pipeline.core.models import (
    ClaimEntityLinkRecord,
    ClaimLocationLinkRecord,
    ClaimPeriodLinkRecord,
    ClaimRecord,
    DocumentYearLinkRecord,
    EntityRecord,
    MeasurementRecord,
    ObservationMeasurementLinkRecord,
    ObservationRecord,
    YearRecord,
)
from graphrag_pipeline.ingest.derivation_context import _extract_year

if TYPE_CHECKING:
    from graphrag_pipeline.ingest.derivation_context import DerivationContext


def build_observations(
    claims: list[ClaimRecord],
    measurements: list[MeasurementRecord],
    claim_entity_links: list[ClaimEntityLinkRecord],
    claim_location_links: list[ClaimLocationLinkRecord],
    claim_period_links: list[ClaimPeriodLinkRecord],
    entity_lookup: dict[str, EntityRecord],
    run_id: str,
    report_year: int | None,
    *,
    _contexts: list[DerivationContext] | None = None,
) -> tuple[
    list[ObservationRecord],
    list[YearRecord],
    list[ObservationMeasurementLinkRecord],
    list[DocumentYearLinkRecord],
]:
    if _contexts is not None:
        return _build_observations_from_contexts(_contexts, run_id)

    # ── Backward-compatible single-pass path ─────────────────────────────────
    measurements_by_claim: dict[str, list[MeasurementRecord]] = defaultdict(list)
    for m in measurements:
        measurements_by_claim[m.claim_id].append(m)

    entity_links_by_claim: dict[str, list[tuple[str, EntityRecord]]] = defaultdict(list)
    for link in claim_entity_links:
        entity = entity_lookup.get(link.entity_id)
        if entity:
            entity_links_by_claim[link.claim_id].append((link.relation_type, entity))

    locations_by_claim: dict[str, list[EntityRecord]] = defaultdict(list)
    for link in claim_location_links:
        entity = entity_lookup.get(link.entity_id)
        if entity:
            locations_by_claim[link.claim_id].append(entity)

    periods_by_claim: dict[str, str] = {}
    for link in claim_period_links:
        periods_by_claim[link.claim_id] = link.period_id

    observations: list[ObservationRecord] = []
    year_map: dict[str, YearRecord] = {}
    obs_measurement_links: list[ObservationMeasurementLinkRecord] = []
    obs_counter = 0

    for claim in claims:
        if claim.claim_type not in OBSERVATION_ELIGIBLE_TYPES:
            continue

        obs_counter += 1
        observation_type = CLAIM_TYPE_TO_OBSERVATION_TYPE.get(claim.claim_type, claim.claim_type)
        observation_id = make_observation_id(run_id, claim.claim_id, obs_counter, observation_type)

        # Resolve linked entities by label
        species_id: str | None = None
        refuge_id: str | None = None
        place_id: str | None = None
        habitat_id: str | None = None
        survey_method_id: str | None = None

        for relation_type, entity in entity_links_by_claim.get(claim.claim_id, []):
            if entity.entity_type == "Species" and species_id is None and relation_type in {"SPECIES_FOCUS", "MANAGEMENT_TARGET"}:
                species_id = entity.entity_id
            elif entity.entity_type == "Habitat" and habitat_id is None and relation_type in {"HABITAT_FOCUS"}:
                habitat_id = entity.entity_id
            elif entity.entity_type == "SurveyMethod" and survey_method_id is None and relation_type in {"METHOD_FOCUS"}:
                survey_method_id = entity.entity_id

        for entity in locations_by_claim.get(claim.claim_id, []):
            if entity.entity_type == "Refuge" and refuge_id is None:
                refuge_id = entity.entity_id
            elif entity.entity_type == "Place" and place_id is None:
                place_id = entity.entity_id

        period_id = periods_by_claim.get(claim.claim_id)

        # Derive year and record its provenance
        year_value, year_source = _extract_year(claim.claim_date, report_year)
        year_id: str | None = None
        if year_value is not None:
            year_id = make_year_id(year_value)
            if year_id not in year_map:
                year_map[year_id] = YearRecord(year_id=year_id, year=year_value, year_label=str(year_value))

        observation = ObservationRecord(
            observation_id=observation_id,
            run_id=run_id,
            observation_type=observation_type,
            claim_id=claim.claim_id,
            paragraph_id=claim.paragraph_id,
            species_id=species_id,
            refuge_id=refuge_id,
            place_id=place_id,
            period_id=period_id,
            year_id=year_id,
            habitat_id=habitat_id,
            survey_method_id=survey_method_id,
            confidence=claim.extraction_confidence,
            is_estimate=claim.epistemic_status == "uncertain",
            source_claim_type=claim.claim_type,
            year=year_value,
            year_source=year_source,
        )
        observations.append(observation)

        # Link measurements from this claim to the observation
        for m in measurements_by_claim.get(claim.claim_id, []):
            obs_measurement_links.append(
                ObservationMeasurementLinkRecord(
                    observation_id=observation_id,
                    measurement_id=m.measurement_id,
                )
            )

    years = list(year_map.values())
    return observations, years, obs_measurement_links, []


def _build_observations_from_contexts(
    contexts: list[DerivationContext],
    run_id: str,
) -> tuple[
    list[ObservationRecord],
    list[YearRecord],
    list[ObservationMeasurementLinkRecord],
    list[DocumentYearLinkRecord],
]:
    """Fast path: build observations directly from pre-computed DerivationContext objects."""
    observations: list[ObservationRecord] = []
    year_map: dict[str, YearRecord] = {}
    obs_measurement_links: list[ObservationMeasurementLinkRecord] = []
    obs_counter = 0

    for ctx in contexts:
        if ctx.observation_type is None:
            continue

        obs_counter += 1
        observation_id = make_observation_id(run_id, ctx.claim.claim_id, obs_counter, ctx.observation_type)
        ctx.observation_id = observation_id  # populate for event_builder bridge

        year_id = ctx.year_id
        if year_id is not None and year_id not in year_map:
            year_map[year_id] = YearRecord(
                year_id=year_id,
                year=ctx.year,
                year_label=str(ctx.year),
            )

        observation = ObservationRecord(
            observation_id=observation_id,
            run_id=run_id,
            observation_type=ctx.observation_type,
            claim_id=ctx.claim.claim_id,
            paragraph_id=ctx.claim.paragraph_id,
            species_id=ctx.species_id,
            refuge_id=ctx.refuge_id,
            place_id=ctx.place_id,
            period_id=ctx.period_id,
            year_id=year_id,
            habitat_id=ctx.habitat_id,
            survey_method_id=ctx.survey_method_id,
            confidence=ctx.claim.extraction_confidence,
            is_estimate=ctx.claim.epistemic_status == "uncertain",
            source_claim_type=ctx.claim.claim_type,
            year=ctx.year,
            year_source=ctx.year_source,
        )
        observations.append(observation)

        for measurement_id in ctx.measurement_ids:
            if ctx.measurement_owner == "observation":
                obs_measurement_links.append(
                    ObservationMeasurementLinkRecord(
                        observation_id=observation_id,
                        measurement_id=measurement_id,
                    )
                )

    return observations, list(year_map.values()), obs_measurement_links, []
