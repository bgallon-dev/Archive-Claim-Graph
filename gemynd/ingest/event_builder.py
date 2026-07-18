from __future__ import annotations

from typing import TYPE_CHECKING

from gemynd.core.ids import make_event_id
from gemynd.core.models import (
    EventMeasurementLinkRecord,
    EventObservationLinkRecord,
    EventRecord,
    YearRecord,
)

from gemynd.ingest.builders import BuilderSpec, build_records

if TYPE_CHECKING:
    from gemynd.ingest.derivation_context import DerivationContext


def _skip_event(ctx: "DerivationContext") -> bool:
    return ctx.event_type is None


def _event_id(run_id: str, ctx: "DerivationContext", counter: int) -> str:
    return make_event_id(run_id, ctx.claim.claim_id, counter, ctx.event_type)


def _make_event_record(
    run_id: str,
    event_id: str,
    ctx: "DerivationContext",
) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        run_id=run_id,
        event_type=ctx.event_type,
        claim_id=ctx.claim.claim_id,
        paragraph_id=ctx.claim.paragraph_id,
        place_id=ctx.place_id,
        period_id=ctx.period_id,
        year_id=ctx.year_id,
        role_entities=dict(ctx.role_entities),
        source_claim_type=ctx.claim.claim_type,
        year=ctx.year,
        year_source=ctx.year_source,
        confidence=ctx.claim.extraction_confidence,
    )


_EVENT_SPEC = BuilderSpec(
    skip_when=_skip_event,
    make_id=_event_id,
    make_record=_make_event_record,
    measurement_owner="event",
)


def build_events(
    contexts: list["DerivationContext"],
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
    result = build_records(contexts, run_id, _EVENT_SPEC)
    event_obs_links = [
        EventObservationLinkRecord(event_id=event_id, observation_id=ctx.observation_id)
        for event_id, ctx in zip(result.record_ids, result.contexts_used)
        if ctx.observation_id is not None
    ]
    event_meas_links = [
        EventMeasurementLinkRecord(event_id=event_id, measurement_id=measurement_id)
        for event_id, measurement_id in result.measurement_links
    ]
    return result.records, event_obs_links, event_meas_links, result.years
