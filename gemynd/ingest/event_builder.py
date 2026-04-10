from __future__ import annotations

from typing import TYPE_CHECKING

from gemynd.core.ids import make_event_id
from gemynd.core.models import (
    EventMeasurementLinkRecord,
    EventObservationLinkRecord,
    EventRecord,
    YearRecord,
)

if TYPE_CHECKING:
    from gemynd.ingest.derivation_context import DerivationContext


def build_events(
    contexts: list[DerivationContext],
    run_id: str,
) -> tuple[
    list[EventRecord],
    list[EventObservationLinkRecord],
    list[EventMeasurementLinkRecord],
    list[YearRecord],
]:
    """Build events from pre-computed DerivationContext objects.

    When a context has both observation_type and event_type, emits an
    EventObservationLinkRecord bridging the two (requires build_observations
    to have been called first to populate ctx.observation_id).
    """
    events: list[EventRecord] = []
    event_obs_links: list[EventObservationLinkRecord] = []
    event_meas_links: list[EventMeasurementLinkRecord] = []
    year_map: dict[str, YearRecord] = {}
    evt_counter = 0

    for ctx in contexts:
        if ctx.event_type is None:
            continue

        evt_counter += 1
        event_id = make_event_id(run_id, ctx.claim.claim_id, evt_counter, ctx.event_type)

        year_id = ctx.year_id
        if year_id is not None and year_id not in year_map:
            year_map[year_id] = YearRecord(
                year_id=year_id,
                year=ctx.year,
                year_label=str(ctx.year),
            )

        events.append(EventRecord(
            event_id=event_id,
            run_id=run_id,
            event_type=ctx.event_type,
            claim_id=ctx.claim.claim_id,
            paragraph_id=ctx.claim.paragraph_id,
            species_id=ctx.species_id,
            refuge_id=ctx.refuge_id,
            place_id=ctx.place_id,
            period_id=ctx.period_id,
            year_id=year_id,
            habitat_id=ctx.habitat_id,
            survey_method_id=ctx.survey_method_id,
            source_claim_type=ctx.claim.claim_type,
            year=ctx.year,
            year_source=ctx.year_source,
            confidence=ctx.claim.extraction_confidence,
        ))

        if ctx.observation_id is not None:
            event_obs_links.append(EventObservationLinkRecord(
                event_id=event_id,
                observation_id=ctx.observation_id,
            ))

        if ctx.measurement_owner == "event":
            for measurement_id in ctx.measurement_ids:
                event_meas_links.append(EventMeasurementLinkRecord(
                    event_id=event_id,
                    measurement_id=measurement_id,
                ))

    return events, event_obs_links, event_meas_links, list(year_map.values())
