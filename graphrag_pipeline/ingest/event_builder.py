from __future__ import annotations

from collections import defaultdict

from graphrag_pipeline.core.claim_contract import (
    CLAIM_TYPE_TO_EVENT_TYPE,
    EVENT_ELIGIBLE_TYPES,
    OBSERVATION_ELIGIBLE_TYPES,
)
from graphrag_pipeline.core.ids import make_event_id, make_year_id
from graphrag_pipeline.core.models import (
    ClaimEntityLinkRecord,
    ClaimLocationLinkRecord,
    ClaimPeriodLinkRecord,
    ClaimRecord,
    EntityRecord,
    EventMeasurementLinkRecord,
    EventObservationLinkRecord,
    EventRecord,
    MeasurementRecord,
    ObservationRecord,
    YearRecord,
)
from .observation_builder import _extract_year


def build_events(
    claims: list[ClaimRecord],
    measurements: list[MeasurementRecord],
    claim_entity_links: list[ClaimEntityLinkRecord],
    claim_location_links: list[ClaimLocationLinkRecord],
    claim_period_links: list[ClaimPeriodLinkRecord],
    entity_lookup: dict[str, EntityRecord],
    observations: list[ObservationRecord],
    run_id: str,
    report_year: int | None,
) -> tuple[
    list[EventRecord],
    list[EventObservationLinkRecord],
    list[EventMeasurementLinkRecord],
    list[YearRecord],
]:
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

    # Map claim_id → observation_id so we can emit EventObservationLinkRecord
    obs_by_claim: dict[str, str] = {obs.claim_id: obs.observation_id for obs in observations}

    events: list[EventRecord] = []
    event_obs_links: list[EventObservationLinkRecord] = []
    event_meas_links: list[EventMeasurementLinkRecord] = []
    year_map: dict[str, YearRecord] = {}
    evt_counter = 0

    for claim in claims:
        if claim.claim_type not in EVENT_ELIGIBLE_TYPES:
            continue

        evt_counter += 1
        event_type = CLAIM_TYPE_TO_EVENT_TYPE[claim.claim_type]
        event_id = make_event_id(run_id, claim.claim_id, evt_counter, event_type)

        # Resolve linked entities by label (same logic as observation_builder)
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

        year_value, year_source = _extract_year(claim.claim_date, report_year)
        year_id: str | None = None
        if year_value is not None:
            year_id = make_year_id(year_value)
            if year_id not in year_map:
                year_map[year_id] = YearRecord(year_id=year_id, year=year_value, year_label=str(year_value))

        events.append(EventRecord(
            event_id=event_id,
            run_id=run_id,
            event_type=event_type,
            claim_id=claim.claim_id,
            paragraph_id=claim.paragraph_id,
            species_id=species_id,
            refuge_id=refuge_id,
            place_id=place_id,
            period_id=period_id,
            year_id=year_id,
            habitat_id=habitat_id,
            survey_method_id=survey_method_id,
            source_claim_type=claim.claim_type,
            year=year_value,
            year_source=year_source,
            confidence=claim.extraction_confidence,
        ))

        # Bridge: Event → Observation (when both exist)
        if claim.claim_id in obs_by_claim:
            event_obs_links.append(EventObservationLinkRecord(
                event_id=event_id,
                observation_id=obs_by_claim[claim.claim_id],
            ))

        # Measurements belong to Observation when one exists; otherwise claim them here
        if claim.claim_type not in OBSERVATION_ELIGIBLE_TYPES:
            for m in measurements_by_claim.get(claim.claim_id, []):
                event_meas_links.append(EventMeasurementLinkRecord(
                    event_id=event_id,
                    measurement_id=m.measurement_id,
                ))

    return events, event_obs_links, event_meas_links, list(year_map.values())
