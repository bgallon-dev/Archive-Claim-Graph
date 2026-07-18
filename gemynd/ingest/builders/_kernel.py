from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from gemynd.core.models import YearRecord

from .year_merge import merge_year

if TYPE_CHECKING:
    from gemynd.ingest.derivation_context import DerivationContext


@dataclass(slots=True, frozen=True)
class BuilderSpec:
    """Strategy object for ``build_records``.

    Holds the per-record-type callbacks that differ between events and
    observations. ``measurement_owner`` is matched against
    ``DerivationContext.measurement_owner`` to gate optional measurement-link
    emission inside the kernel.
    """

    skip_when: Callable[["DerivationContext"], bool]
    make_id: Callable[[str, "DerivationContext", int], str]
    make_record: Callable[[str, str, "DerivationContext"], Any]
    measurement_owner: str  # "event" | "observation"


@dataclass(slots=True)
class BuildResult:
    """Output of ``build_records`` — parallel lists plus structured side outputs.

    ``record_ids`` and ``contexts_used`` are index-aligned with ``records`` so
    the caller can post-process (e.g. emit Event→Observation bridge links from
    the event builder) without re-running the eligibility filter.

    ``measurement_links`` is a list of ``(record_id, measurement_id)`` tuples;
    the caller converts these to the appropriate edge record type.
    """

    records: list[Any]
    record_ids: list[str]
    contexts_used: list["DerivationContext"]
    years: list[YearRecord]
    measurement_links: list[tuple[str, str]]


def build_records(
    contexts: list["DerivationContext"],
    run_id: str,
    spec: BuilderSpec,
) -> BuildResult:
    """Shared iteration over DerivationContexts for event and observation builders.

    Skip-then-increment counter order is preserved exactly: ``counter`` is only
    bumped after ``skip_when`` returns False, matching the original per-builder
    loops at event_builder.py:38-43 and observation_builder.py:36-41. A
    regression in this order would change the trailing integer in record IDs.
    """
    records: list[Any] = []
    record_ids: list[str] = []
    contexts_used: list["DerivationContext"] = []
    year_map: dict[str, YearRecord] = {}
    measurement_links: list[tuple[str, str]] = []
    counter = 0

    for ctx in contexts:
        if spec.skip_when(ctx):
            continue
        counter += 1
        record_id = spec.make_id(run_id, ctx, counter)
        records.append(spec.make_record(run_id, record_id, ctx))
        record_ids.append(record_id)
        contexts_used.append(ctx)
        merge_year(year_map, ctx.year_id, ctx.year)
        if ctx.measurement_owner == spec.measurement_owner:
            for measurement_id in ctx.measurement_ids:
                measurement_links.append((record_id, measurement_id))

    return BuildResult(
        records=records,
        record_ids=record_ids,
        contexts_used=contexts_used,
        years=list(year_map.values()),
        measurement_links=measurement_links,
    )
