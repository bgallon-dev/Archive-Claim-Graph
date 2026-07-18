"""Shared builder kernel + year-merge helpers.

The kernel collapses the structurally-twin event/observation builders into a
single iteration that handles counter, ID generation, year_map dedup, and the
optional measurement-link branch. Per-type record construction stays in
``event_builder.py`` / ``observation_builder.py`` because the record shapes
(notably ``is_estimate`` on observations and the Event→Observation bridge)
diverge enough that pushing them into the kernel would just move the
asymmetry rather than eliminate it.

``year_merge.union_years`` is the third dedup site (after the per-builder
year_maps) used in ``phases.build_events_phase`` to merge event-only years
into ``state.years`` without disturbing observation-year ordering.
"""
from __future__ import annotations

from ._kernel import BuilderSpec, BuildResult, build_records
from .year_merge import merge_year, union_years

__all__ = [
    "BuilderSpec",
    "BuildResult",
    "build_records",
    "merge_year",
    "union_years",
]
