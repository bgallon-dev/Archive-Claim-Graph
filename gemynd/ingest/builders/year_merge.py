from __future__ import annotations

from gemynd.core.models import YearRecord


def merge_year(
    year_map: dict[str, YearRecord],
    year_id: str | None,
    year: int | None,
) -> None:
    """Insert a YearRecord into ``year_map`` under first-occurrence-wins semantics.

    No-op when ``year_id`` is None (the context had no derivable year) or when
    ``year_id`` is already present. Matches the inline dedup that previously
    appeared in event_builder and observation_builder verbatim.
    """
    if year_id is None or year_id in year_map:
        return
    year_map[year_id] = YearRecord(
        year_id=year_id,
        year=year,
        year_label=str(year),
    )


def union_years(
    existing: list[YearRecord],
    new: list[YearRecord],
) -> list[YearRecord]:
    """First-occurrence-wins dedup that preserves order.

    Returns a list containing every YearRecord in ``existing`` (same order,
    same instances) followed by every YearRecord in ``new`` whose ``year_id``
    is not already present, in the order they appeared in ``new``. Replaces
    the hand-rolled dedup at the old ``phases.build_events_phase``.
    """
    seen = {y.year_id for y in existing}
    result = list(existing)
    for yr in new:
        if yr.year_id in seen:
            continue
        result.append(yr)
        seen.add(yr.year_id)
    return result
