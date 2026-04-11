from __future__ import annotations

from typing import TYPE_CHECKING

from gemynd.core.ids import make_observation_id
from gemynd.core.models import (
    DocumentYearLinkRecord,
    ObservationMeasurementLinkRecord,
    ObservationRecord,
    YearRecord,
)

if TYPE_CHECKING:
    from gemynd.ingest.derivation_context import DerivationContext


def build_observations(
    contexts: list[DerivationContext],
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
    observations: list[ObservationRecord] = []
    year_map: dict[str, YearRecord] = {}
    obs_measurement_links: list[ObservationMeasurementLinkRecord] = []
    obs_counter = 0

    for ctx in contexts:
        if ctx.observation_type is None:
            continue

        obs_counter += 1
        observation_id = make_observation_id(run_id, ctx.claim.claim_id, obs_counter, ctx.observation_type)
        ctx.observation_id = observation_id

        year_id = ctx.year_id
        if year_id is not None and year_id not in year_map:
            year_map[year_id] = YearRecord(
                year_id=year_id,
                year=ctx.year,
                year_label=str(ctx.year),
            )

        observations.append(ObservationRecord(
            observation_id=observation_id,
            run_id=run_id,
            observation_type=ctx.observation_type,
            claim_id=ctx.claim.claim_id,
            paragraph_id=ctx.claim.paragraph_id,
            place_id=ctx.place_id,
            period_id=ctx.period_id,
            year_id=year_id,
            role_entities=dict(ctx.role_entities),
            confidence=ctx.claim.extraction_confidence,
            is_estimate=ctx.claim.epistemic_status == "uncertain",
            source_claim_type=ctx.claim.claim_type,
            year=ctx.year,
            year_source=ctx.year_source,
        ))

        if ctx.measurement_owner == "observation":
            for measurement_id in ctx.measurement_ids:
                obs_measurement_links.append(
                    ObservationMeasurementLinkRecord(
                        observation_id=observation_id,
                        measurement_id=measurement_id,
                    )
                )

    return observations, list(year_map.values()), obs_measurement_links, []
