"""Layer 2A — Cypher Query Builder (Analytical Path).

Executes parameterised analytical Cypher templates against Neo4j and returns
typed AnalyticalResult objects.  All templates are defined in
gemynd.graph.cypher to keep schema concerns co-located.
"""
from __future__ import annotations

from ..core.graph.cypher import (
    HABITAT_CONDITION_QUERY,
    PROVENANCE_CHAIN_QUERY,
    SPECIES_TREND_QUERY,
)
from .executor import Neo4jQueryExecutor
from .models import AnalyticalResult


class CypherQueryBuilder:
    """High-level query interface over Neo4jQueryExecutor.

    Parameters
    ----------
    executor:
        An initialised Neo4jQueryExecutor.  Injected so tests can supply
        a mock executor without a live Neo4j instance.
    """

    def __init__(
        self, executor: Neo4jQueryExecutor, *, institution_id: str | None = None,
    ) -> None:
        self._executor = executor
        self._institution_id = institution_id

    # ------------------------------------------------------------------
    # Analytical templates
    # ------------------------------------------------------------------

    def species_trend(
        self,
        species_id: str,
        year_min: int | None = None,
        year_max: int | None = None,
        permitted_levels: list[str] | None = None,
        institution_id: str | None = None,
    ) -> AnalyticalResult:
        """Return observation counts per year for *species_id*."""
        rows = self._executor.run(
            SPECIES_TREND_QUERY,
            {
                "species_id": species_id,
                "year_min": year_min,
                "year_max": year_max,
                "permitted_levels": permitted_levels if permitted_levels is not None else ["public"],
                "institution_id": institution_id or self._institution_id or "",
            },
        )
        return AnalyticalResult(
            query_name="species_trend",
            columns=["species", "year", "observation_count", "avg_confidence", "measurements"],
            rows=rows,
        )

    def habitat_conditions(
        self,
        habitat_id: str,
        year_min: int | None = None,
        year_max: int | None = None,
        permitted_levels: list[str] | None = None,
        institution_id: str | None = None,
    ) -> AnalyticalResult:
        """Return observation counts per year/species for *habitat_id*."""
        rows = self._executor.run(
            HABITAT_CONDITION_QUERY,
            {
                "habitat_id": habitat_id,
                "year_min": year_min,
                "year_max": year_max,
                "permitted_levels": permitted_levels if permitted_levels is not None else ["public"],
                "institution_id": institution_id or self._institution_id or "",
            },
        )
        return AnalyticalResult(
            query_name="habitat_conditions",
            columns=["habitat", "year", "species", "observation_count", "avg_confidence", "measurements"],
            rows=rows,
        )

    # ------------------------------------------------------------------
    # Provenance chain (used by both conversational path and /query/provenance)
    # ------------------------------------------------------------------

    def provenance_chain(self, claim_id: str) -> list[dict]:
        """Return the full documentary provenance chain for a single *claim_id*."""
        return self._executor.run(
            PROVENANCE_CHAIN_QUERY,
            {"claim_id": claim_id},
        )
