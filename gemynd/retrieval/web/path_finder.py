"""Mode 1 — entity-to-entity path-finding orchestration.

Cascades through layers of indirection in the claim graph: direct
co-assertion (one claim links A and B) → 1-hop (one intermediate entity)
→ 2-hop (two intermediates). The first non-empty layer wins. When all
layers are empty within the hop cap, returns ``disconnected: True`` plus
the most-connected direct neighbors of each endpoint as a fallback
affordance.

Design notes
------------
- **Cascade-as-API-contract**: the response always carries both
  ``co_assertions`` and ``paths``. The dominant case in primary-source
  research is direct co-assertion, so it gets dedicated semantics rather
  than being treated as a special case of pathfinding.
- **Inverse-frequency weighting**: paths through generic hub entities
  ("the United States", "the press") add little path-information. Each
  intermediate node's weight is ``log(corpus_total / max(degree, 1))``.
- **Degree cap**: paths whose maximum-degree intermediate sits above the
  corpus 95th-percentile degree are dropped — *unless one of the
  endpoints itself is such a hub* (a researcher querying about a hub
  entity can't be denied paths through other hubs).
- **Path-family collapsing**: paths that differ by ≤1 intermediate
  collapse into one family. The highest-weight member is the primary;
  the rest become ``alternates``.
- **Pure orchestration**: this module does no Cypher of its own — it
  receives an executor, a degree map, and parameters, and returns
  structured data. Tests can substitute a stub executor easily.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gemynd.core.graph.cypher import (
    CO_ASSERTING_CLAIMS_QUERY,
    CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY,
    PATHS_1_HOP_QUERY,
    PATHS_2_HOP_QUERY,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CoAssertion:
    """One claim that mentions both endpoints directly."""

    claim_id: str
    sentence: str
    claim_type: str
    confidence: float
    from_relation: str
    to_relation: str
    doc_id: str
    doc_title: str
    doc_date_start: str | None
    doc_report_year: int | None
    institution_id: str | None


@dataclass(slots=True)
class PathEdge:
    """One claim linking two entities along a path. Edges are ordered
    from-endpoint → to-endpoint."""

    claim_id: str
    sentence: str
    confidence: float
    from_relation: str
    to_relation: str
    doc_id: str
    doc_title: str | None
    doc_date_start: str | None


@dataclass(slots=True)
class PathNode:
    """One intermediate entity along a path."""

    entity_id: str
    name: str
    entity_type: str


@dataclass(slots=True)
class Path:
    """A ranked path from endpoint A to endpoint B with K-1 intermediate
    nodes and K edges (claims)."""

    intermediates: list[PathNode]
    edges: list[PathEdge]
    weight: float
    alternates: list["Path"] = field(default_factory=list)


@dataclass(slots=True)
class FallbackNeighbor:
    """A top co-occurring entity for one endpoint, surfaced when the
    path-finder returns disconnected."""

    entity_id: str
    name: str
    entity_type: str
    co_occurrence_count: int


@dataclass(slots=True)
class PathResult:
    """Cascade-shaped result: co-assertions first, multi-hop paths next,
    fallback neighbors when disconnected."""

    co_assertions: list[CoAssertion] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    disconnected: bool = False
    fallback_from: list[FallbackNeighbor] = field(default_factory=list)
    fallback_to: list[FallbackNeighbor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cypher row → typed object adapters
# ---------------------------------------------------------------------------


def _row_to_co_assertion(row: dict[str, Any]) -> CoAssertion:
    return CoAssertion(
        claim_id=row.get("claim_id", ""),
        sentence=row.get("sentence", "") or "",
        claim_type=row.get("claim_type", "") or "",
        confidence=float(row.get("confidence") or 0.0),
        from_relation=row.get("from_relation", "") or "",
        to_relation=row.get("to_relation", "") or "",
        doc_id=row.get("doc_id", "") or "",
        doc_title=row.get("doc_title", "") or "",
        doc_date_start=row.get("doc_date_start"),
        doc_report_year=row.get("doc_report_year"),
        institution_id=row.get("institution_id"),
    )


def _row_to_1_hop_path(row: dict[str, Any]) -> tuple[list[PathNode], list[PathEdge]]:
    nodes = [PathNode(
        entity_id=row.get("i1_id", "") or "",
        name=row.get("i1_name", "") or "",
        entity_type=row.get("i1_type", "") or "",
    )]
    edges = [
        PathEdge(
            claim_id=row.get("e1_claim_id", "") or "",
            sentence=row.get("e1_sentence", "") or "",
            confidence=float(row.get("e1_conf") or 0.0),
            from_relation=row.get("e1_from_rel", "") or "",
            to_relation=row.get("e1_to_rel", "") or "",
            doc_id=row.get("e1_doc_id", "") or "",
            doc_title=row.get("e1_doc_title"),
            doc_date_start=row.get("e1_doc_date"),
        ),
        PathEdge(
            claim_id=row.get("e2_claim_id", "") or "",
            sentence=row.get("e2_sentence", "") or "",
            confidence=float(row.get("e2_conf") or 0.0),
            from_relation=row.get("e2_from_rel", "") or "",
            to_relation=row.get("e2_to_rel", "") or "",
            doc_id=row.get("e2_doc_id", "") or "",
            doc_title=row.get("e2_doc_title"),
            doc_date_start=row.get("e2_doc_date"),
        ),
    ]
    return nodes, edges


def _row_to_2_hop_path(row: dict[str, Any]) -> tuple[list[PathNode], list[PathEdge]]:
    nodes = [
        PathNode(
            entity_id=row.get("i1_id", "") or "",
            name=row.get("i1_name", "") or "",
            entity_type=row.get("i1_type", "") or "",
        ),
        PathNode(
            entity_id=row.get("i2_id", "") or "",
            name=row.get("i2_name", "") or "",
            entity_type=row.get("i2_type", "") or "",
        ),
    ]
    edges = [
        PathEdge(
            claim_id=row.get(f"{prefix}_claim_id", "") or "",
            sentence=row.get(f"{prefix}_sentence", "") or "",
            confidence=float(row.get(f"{prefix}_conf") or 0.0),
            from_relation=row.get(f"{prefix}_from_rel", "") or "",
            to_relation=row.get(f"{prefix}_to_rel", "") or "",
            doc_id=row.get(f"{prefix}_doc_id", "") or "",
            doc_title=row.get(f"{prefix}_doc_title"),
            doc_date_start=row.get(f"{prefix}_doc_date"),
        )
        for prefix in ("e1", "e2", "e3")
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# Weighting + ranking
# ---------------------------------------------------------------------------


def compute_path_weight(
    nodes: list[PathNode],
    edges: list[PathEdge],
    degree: dict[str, int],
    corpus_total_claims: int,
) -> float:
    """Path weight = sum of inverse-frequency contributions of each
    intermediate node × mean edge confidence, clamped at 0.

    Inverse-frequency is ``log(corpus_total / max(degree, 1))``: an
    intermediate that appears in 8000 claims contributes ~0 weight; one
    that appears in 12 contributes substantially. Multiplying by mean
    confidence anchors the score in evidence quality.

    The clamp guards against degenerate inputs — e.g. a degree miscount
    above corpus_total, an empty degree map (every entity defaults to 1
    so total/1 = total > 1 → log > 0, fine, but defensive), or floating
    point edge cases — that could otherwise produce negative weights and
    flip the ranking inverted.

    Empty paths or paths with no intermediates return 0 — direct
    co-assertions don't go through this function.
    """
    if not nodes or not edges:
        return 0.0
    total = max(corpus_total_claims, 1)
    inv_freq = sum(
        math.log(total / max(degree.get(n.entity_id, 1), 1)) for n in nodes
    )
    mean_conf = sum(e.confidence for e in edges) / len(edges)
    return max(0.0, inv_freq * mean_conf)


def filter_by_degree_cap(
    paths: list[Path],
    degree: dict[str, int],
    cap: int,
    endpoints: tuple[str, str] | None = None,
) -> list[Path]:
    """Drop paths whose maximum-degree intermediate sits above *cap*.

    The cap applies even when one of the endpoints is itself a hub: a
    researcher querying about Turnbull NWR shouldn't be denied paths
    through specific intermediates just because Turnbull happens to be
    high-degree, but they also shouldn't get paths through *other*
    hubs (Spokane County, the United States) — those are noise
    regardless of the endpoint's degree.

    Acts as a safety net when in-Cypher hub exclusion already pruned
    the candidates. *endpoints* is accepted for backwards-compatible
    call signatures but no longer affects the filter.
    """
    if cap <= 0:
        return paths
    return [
        p for p in paths
        if all(degree.get(n.entity_id, 0) <= cap for n in p.intermediates)
    ]


def excluded_hub_ids(
    degree: dict[str, int],
    cap: int,
    endpoints: tuple[str, str],
) -> list[str]:
    """List of entity_ids whose degree exceeds *cap*, used to prune the
    Cypher path search. Endpoints are explicitly excluded from the list
    — they're not intermediates anyway, but listing them defensively
    allows endpoints to be hubs without being filtered out of their
    own path queries.
    """
    if cap <= 0 or not degree:
        return []
    from_id, to_id = endpoints
    return [
        eid for eid, deg in degree.items()
        if deg > cap and eid != from_id and eid != to_id
    ]


def degree_cap_threshold(degree: dict[str, int], percentile: float = 0.95) -> int:
    """Return the *percentile*-th degree value across non-zero entities,
    rounded to int. Zero degrees are excluded so empty entities don't
    skew the threshold."""
    if not degree:
        return 0
    nonzero = sorted(d for d in degree.values() if d > 0)
    if not nonzero:
        return 0
    idx = min(int(len(nonzero) * percentile), len(nonzero) - 1)
    return nonzero[idx]


def collapse_path_families(paths: list[Path], max_diff: int = 1) -> list[Path]:
    """Group paths that share the same hop-count and differ by at most
    *max_diff* node substitutions.

    A "substitution" is a position where one path has entity X and the
    other has entity Y. Paths of different hop-counts never merge — a
    1-hop and a 2-hop path are categorically different findings even if
    they share some intermediates.

    The highest-weight member of each family becomes the primary; the
    rest are stored on its ``alternates`` field. Input is assumed sorted
    by weight descending; output preserves that order across primaries.
    """
    families: list[Path] = []
    for candidate in paths:
        cand_ids = {n.entity_id for n in candidate.intermediates}
        merged = False
        for primary in families:
            if len(primary.intermediates) != len(candidate.intermediates):
                continue
            primary_ids = {n.entity_id for n in primary.intermediates}
            substitutions = max(
                len(cand_ids - primary_ids),
                len(primary_ids - cand_ids),
            )
            if substitutions <= max_diff:
                primary.alternates.append(candidate)
                merged = True
                break
        if not merged:
            families.append(candidate)
    return families


# ---------------------------------------------------------------------------
# Cascade orchestration
# ---------------------------------------------------------------------------


# An executor-like callable: given (cypher, params) returns row dicts.
# Lets tests substitute a recording stub.
ExecutorRun = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


def find_co_assertions(
    run: ExecutorRun,
    *,
    from_id: str,
    to_id: str,
    institution_ids: list[str],
    permitted_levels: list[str],
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[CoAssertion]:
    rows = run(CO_ASSERTING_CLAIMS_QUERY, {
        "from_id": from_id,
        "to_id": to_id,
        "institution_ids": institution_ids,
        "permitted_levels": permitted_levels,
        "year_min": year_min,
        "year_max": year_max,
    })
    return [_row_to_co_assertion(r) for r in rows]


def find_multi_hop_paths(
    run: ExecutorRun,
    *,
    from_id: str,
    to_id: str,
    institution_ids: list[str],
    permitted_levels: list[str],
    year_min: int | None = None,
    year_max: int | None = None,
    max_hops: int = 2,
    degree: dict[str, int] | None = None,
    corpus_total_claims: int = 0,
    k: int = 3,
) -> list[Path]:
    """Run hop-1 then hop-2 (when max_hops>=2). Returns at most *k*
    path-families after weighting, degree-cap filtering, and family
    collapsing. The first non-empty hop layer is the only one used —
    deeper hops are not explored if a closer connection exists."""
    degree = degree or {}
    cap = degree_cap_threshold(degree)
    excluded = excluded_hub_ids(degree, cap, (from_id, to_id))
    params = {
        "from_id": from_id, "to_id": to_id,
        "institution_ids": institution_ids,
        "permitted_levels": permitted_levels,
        "year_min": year_min, "year_max": year_max,
        "excluded_intermediate_ids": excluded,
    }

    for hop_count in range(1, max_hops + 1):
        if hop_count == 1:
            rows = run(PATHS_1_HOP_QUERY, params)
            paths = [
                Path(
                    intermediates=ns, edges=es,
                    weight=compute_path_weight(ns, es, degree, corpus_total_claims),
                )
                for r in rows
                for (ns, es) in [_row_to_1_hop_path(r)]
            ]
        elif hop_count == 2:
            rows = run(PATHS_2_HOP_QUERY, params)
            paths = [
                Path(
                    intermediates=ns, edges=es,
                    weight=compute_path_weight(ns, es, degree, corpus_total_claims),
                )
                for r in rows
                for (ns, es) in [_row_to_2_hop_path(r)]
            ]
        else:
            # Deeper hops not implemented in PR 1; cascade returns empty
            # rather than silently expanding scope.
            paths = []

        if not paths:
            continue

        paths = filter_by_degree_cap(paths, degree, cap, (from_id, to_id))
        if not paths:
            continue

        paths.sort(key=lambda p: p.weight, reverse=True)
        families = collapse_path_families(paths, max_diff=1)
        return families[:k]

    return []


def find_fallback_neighbors(
    run: ExecutorRun,
    *,
    entity_id: str,
    institution_ids: list[str],
    permitted_levels: list[str],
    limit: int = 5,
) -> list[FallbackNeighbor]:
    """Top co-occurring neighbors of one endpoint, used when path-finder
    returns no connection. Reuses
    :data:`CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY` rather than introducing
    a new query."""
    rows = run(CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY, {
        "entity_id": entity_id,
        "institution_ids": institution_ids,
        "permitted_levels": permitted_levels,
    })
    out: list[FallbackNeighbor] = []
    for r in rows[:limit]:
        out.append(FallbackNeighbor(
            entity_id=r.get("entity_id", "") or "",
            name=r.get("name", "") or "",
            entity_type=r.get("entity_type", "") or "",
            co_occurrence_count=int(r.get("co_occurrence_count") or 0),
        ))
    return out


def find_paths(
    run: ExecutorRun,
    *,
    from_id: str,
    to_id: str,
    institution_ids: list[str],
    permitted_levels: list[str],
    year_min: int | None = None,
    year_max: int | None = None,
    max_hops: int = 2,
    degree: dict[str, int] | None = None,
    corpus_total_claims: int = 0,
    k: int = 3,
    include_paths_when_co_asserted: bool = False,
) -> PathResult:
    """Cascade entry point. Returns :class:`PathResult` with whichever
    layer of the cascade fired, or ``disconnected=True`` with fallback
    neighbors when none did.

    When *include_paths_when_co_asserted* is True, multi-hop paths are
    also computed even if direct co-assertions exist — useful for
    researchers who want to see indirect connections alongside direct
    ones. Default False (the cascade short-circuits on co-assertions)
    because the direct case is the dominant one.
    """
    co = find_co_assertions(
        run,
        from_id=from_id, to_id=to_id,
        institution_ids=institution_ids,
        permitted_levels=permitted_levels,
        year_min=year_min, year_max=year_max,
    )

    paths: list[Path] = []
    if not co or include_paths_when_co_asserted:
        paths = find_multi_hop_paths(
            run,
            from_id=from_id, to_id=to_id,
            institution_ids=institution_ids,
            permitted_levels=permitted_levels,
            year_min=year_min, year_max=year_max,
            max_hops=max_hops,
            degree=degree, corpus_total_claims=corpus_total_claims,
            k=k,
        )

    if co or paths:
        return PathResult(co_assertions=co, paths=paths)

    # Disconnected — pull fallback neighbors for both endpoints. This is a
    # research lead in itself (turning absence into an affordance) so we
    # always run it before giving up.
    fb_from = find_fallback_neighbors(
        run, entity_id=from_id,
        institution_ids=institution_ids, permitted_levels=permitted_levels,
    )
    fb_to = find_fallback_neighbors(
        run, entity_id=to_id,
        institution_ids=institution_ids, permitted_levels=permitted_levels,
    )
    return PathResult(disconnected=True, fallback_from=fb_from, fallback_to=fb_to)


# ---------------------------------------------------------------------------
# JSON serialization for the API layer
# ---------------------------------------------------------------------------


def path_result_to_dict(result: PathResult) -> dict[str, Any]:
    """Render a :class:`PathResult` as the JSON shape committed in the
    plan's API contract. The frontend reads this directly."""
    def _node(n: PathNode) -> dict[str, Any]:
        return {"entity_id": n.entity_id, "name": n.name, "entity_type": n.entity_type}

    def _edge(e: PathEdge) -> dict[str, Any]:
        return {
            "claim_id": e.claim_id, "sentence": e.sentence,
            "confidence": e.confidence,
            "from_relation": e.from_relation, "to_relation": e.to_relation,
            "doc_id": e.doc_id, "doc_title": e.doc_title,
            "doc_date_start": e.doc_date_start,
        }

    def _path(p: Path) -> dict[str, Any]:
        return {
            "intermediates": [_node(n) for n in p.intermediates],
            "edges": [_edge(e) for e in p.edges],
            "weight": p.weight,
            "alternates": [
                {
                    "intermediates": [_node(n) for n in alt.intermediates],
                    "edges": [_edge(e) for e in alt.edges],
                    "weight": alt.weight,
                }
                for alt in p.alternates
            ],
        }

    def _co(c: CoAssertion) -> dict[str, Any]:
        return {
            "claim_id": c.claim_id, "sentence": c.sentence,
            "claim_type": c.claim_type, "confidence": c.confidence,
            "from_relation": c.from_relation, "to_relation": c.to_relation,
            "doc_id": c.doc_id, "doc_title": c.doc_title,
            "doc_date_start": c.doc_date_start,
            "doc_report_year": c.doc_report_year,
            "institution_id": c.institution_id,
        }

    def _fb(n: FallbackNeighbor) -> dict[str, Any]:
        return {
            "entity_id": n.entity_id, "name": n.name,
            "entity_type": n.entity_type,
            "co_occurrence_count": n.co_occurrence_count,
        }

    return {
        "co_assertions": [_co(c) for c in result.co_assertions],
        "paths": [_path(p) for p in result.paths],
        "disconnected": result.disconnected,
        "fallback_neighbors": (
            None if not result.disconnected
            else {
                "from": [_fb(n) for n in result.fallback_from],
                "to": [_fb(n) for n in result.fallback_to],
            }
        ),
    }
