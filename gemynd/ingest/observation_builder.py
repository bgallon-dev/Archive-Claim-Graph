from __future__ import annotations

from typing import TYPE_CHECKING

from gemynd.core.ids import make_observation_id
from gemynd.core.models import (
    DocumentYearLinkRecord,
    ObservationMeasurementLinkRecord,
    ObservationRecord,
    YearRecord,
)

from gemynd.ingest.builders import BuilderSpec, build_records

if TYPE_CHECKING:
    from gemynd.ingest.derivation_context import DerivationContext


def _skip_observation(ctx: "DerivationContext") -> bool:
    return ctx.observation_type is None


def _observation_id(run_id: str, ctx: "DerivationContext", counter: int) -> str:
    return make_observation_id(run_id, ctx.claim.claim_id, counter, ctx.observation_type)


def _make_observation_record(
    run_id: str,
    observation_id: str,
    ctx: "DerivationContext",
) -> ObservationRecord:
    # Sidecar mutation: the event builder reads ctx.observation_id to emit the
    # Event→Observation bridge in build_events. Preserved per Stage 1 plan.
    ctx.observation_id = observation_id
    return ObservationRecord(
        observation_id=observation_id,
        run_id=run_id,
        observation_type=ctx.observation_type,
        claim_id=ctx.claim.claim_id,
        paragraph_id=ctx.claim.paragraph_id,
        place_id=ctx.place_id,
        period_id=ctx.period_id,
        year_id=ctx.year_id,
        role_entities=dict(ctx.role_entities),
        confidence=ctx.claim.extraction_confidence,
        is_estimate=ctx.claim.epistemic_status == "uncertain",
        source_claim_type=ctx.claim.claim_type,
        year=ctx.year,
        year_source=ctx.year_source,
    )


_OBSERVATION_SPEC = BuilderSpec(
    skip_when=_skip_observation,
    make_id=_observation_id,
    make_record=_make_observation_record,
    measurement_owner="observation",
)


def build_observations(
    contexts: list["DerivationContext"],
    run_id: str,
) -> tuple[
    list[ObservationRecord],
    list[YearRecord],
    list[ObservationMeasurementLinkRecord],
    list[DocumentYearLinkRecord],
]:
    """Build observations from pre-computed DerivationContext objects.

    Populates ``ctx.observation_id`` on each context for the event builder's
    Event→Observation bridge.
    """
    result = build_records(contexts, run_id, _OBSERVATION_SPEC)
    obs_measurement_links = [
        ObservationMeasurementLinkRecord(observation_id=oid, measurement_id=mid)
        for oid, mid in result.measurement_links
    ]
    return result.records, result.years, obs_measurement_links, []
