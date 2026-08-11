"""Query-aware ordering for provenance context blocks.

The context assembler over-fetches claims and then keeps a budgeted prefix,
so candidate order determines what reaches synthesis. Rankers in this module
only reorder existing blocks; every optional path degrades to a simpler local
strategy when unavailable.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from gemynd.retrieval.models import EntityContext, ProvenanceBlock

_log = logging.getLogger(__name__)

_STRUCTURAL_RELS = [
    "HAS_PAGE",
    "HAS_SECTION",
    "HAS_PARAGRAPH",
    "HAS_CLAIM",
    "CONTAINS_MENTION",
    "EVIDENCED_BY",
    "PROCESSED_BY",
    "SOURCED_FROM",
    "REFERS_TO",
    "POSSIBLY_REFERS_TO",
    "NEXT",
]
_SCAFFOLD_LABELS = ["Year", "Period"]
_UNRANKED_DIST = 99

_PROXIMITY_L1 = """\
MATCH (c:Claim)-[r1]-(a)
WHERE c.claim_id IN $claim_ids AND a.entity_id IN $entity_ids
  AND NOT type(r1) IN $excluded_rels
RETURN DISTINCT c.claim_id AS claim_id
"""
_PROXIMITY_L2 = """\
MATCH (c:Claim)-[r1]-(x)-[r2]-(a)
WHERE c.claim_id IN $claim_ids AND a.entity_id IN $entity_ids
  AND NOT type(r1) IN $excluded_rels AND NOT type(r2) IN $excluded_rels
  AND NONE(l IN labels(x) WHERE l IN $excluded_labels)
RETURN DISTINCT c.claim_id AS claim_id
"""
_PROXIMITY_L3 = """\
MATCH (c:Claim)-[r1]-(x)-[r2]-(y)-[r3]-(a)
WHERE c.claim_id IN $claim_ids AND a.entity_id IN $entity_ids
  AND NOT type(r1) IN $excluded_rels AND NOT type(r2) IN $excluded_rels
  AND NOT type(r3) IN $excluded_rels
  AND NONE(l IN labels(x) WHERE l IN $excluded_labels)
  AND NONE(l IN labels(y) WHERE l IN $excluded_labels)
RETURN DISTINCT c.claim_id AS claim_id
"""

_ANCHOR_INTERNAL_IDS = """\
MATCH (a) WHERE a.entity_id IN $entity_ids
RETURN id(a) AS id
"""
_YEAR_RANGE = """\
MATCH (y:Year) WHERE y.year IS NOT NULL
RETURN min(y.year) AS min_year, max(y.year) AS max_year
"""
_CLAIMS_FOR_NODES = """\
MATCH (c:Claim)-[r]-(n)
WHERE c.claim_id IN $claim_ids AND id(n) IN $node_ids
  AND NOT type(r) IN $excluded_rels
RETURN c.claim_id AS claim_id, collect(DISTINCT id(n)) AS node_ids
"""


class ContextRanker(Protocol):
    """Order candidate blocks best-first; truncation keeps a prefix."""

    def rank(
        self,
        blocks: "list[ProvenanceBlock]",
        *,
        entity_context: "EntityContext",
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> "list[ProvenanceBlock]": ...

    def close(self) -> None: ...


def _by_confidence(blocks: "list[ProvenanceBlock]") -> "list[ProvenanceBlock]":
    return sorted(blocks, key=lambda block: block.extraction_confidence, reverse=True)


class ConfidenceRanker:
    """Legacy static order: extraction confidence descending."""

    def rank(self, blocks, *, entity_context, year_min=None, year_max=None):
        return _by_confidence(blocks)

    def close(self) -> None:
        return None


class GraphProximityRanker:
    """Rank by bounded semantic hop distance from resolved query anchors."""

    def __init__(self, executor, max_hops: int = 3, max_claims: int = 500) -> None:
        self._executor = executor
        self._levels = [_PROXIMITY_L1, _PROXIMITY_L2, _PROXIMITY_L3][
            : max(1, min(max_hops, 3))
        ]
        self._max_claims = max_claims

    def rank(self, blocks, *, entity_context, year_min=None, year_max=None):
        anchors = [entity.entity_id for entity in entity_context.resolved if entity.entity_id]
        if not anchors or not blocks:
            return _by_confidence(blocks)

        ordered = _by_confidence(blocks)
        eligible = [
            block.claim_id
            for block in ordered[: self._max_claims]
            if block.claim_id
        ]
        remaining = set(eligible)
        distance: dict[str, int] = {}
        try:
            for hops, template in enumerate(self._levels, start=1):
                if not remaining:
                    break
                rows = self._executor.run(
                    template,
                    {
                        "claim_ids": sorted(remaining),
                        "entity_ids": anchors,
                        "excluded_rels": _STRUCTURAL_RELS,
                        "excluded_labels": _SCAFFOLD_LABELS,
                    },
                )
                for row in rows:
                    claim_id = row.get("claim_id")
                    if claim_id in remaining:
                        distance[claim_id] = hops
                        remaining.discard(claim_id)
        except Exception as exc:
            _log.warning("proximity ranking failed; using confidence order: %s", exc)
            return ordered

        if not distance:
            return ordered
        return sorted(
            ordered,
            key=lambda block: (
                distance.get(block.claim_id, _UNRANKED_DIST),
                -block.extraction_confidence,
            ),
        )

    def close(self) -> None:
        return None


class KGRTemporalRanker:
    """Use the cloned KGR model for anchored temporal queries.

    KGR's ``ranked`` list is the authoritative best-first output. Its exposed
    scores are currently not monotonic on the live task-2 path, so this adapter
    deliberately maps nodes to rank positions and never re-sorts by score.
    """

    def __init__(
        self,
        executor,
        kgr_root: str,
        fallback: ContextRanker,
        *,
        top_k: int = 50,
        neo4j_uri: str = "",
        neo4j_username: str = "",
        neo4j_password: str = "",
    ) -> None:
        self._executor = executor
        self._root = str(Path(kgr_root).expanduser().resolve())
        self._fallback = fallback
        self._top_k = top_k
        self._engine = None  # None = not tried; False = initialization failed
        self._year_range: tuple[float, float] | None = None
        self._neo4j_env = {
            "NEO4J_URI": neo4j_uri,
            # KGR calls this NEO4J_USERNAME; Gemynd calls it NEO4J_USER.
            "NEO4J_USERNAME": neo4j_username,
            "NEO4J_PASSWORD": neo4j_password,
        }

    def _prepare_model_environment(self) -> None:
        for name, value in self._neo4j_env.items():
            if value:
                os.environ[name] = value
        if os.environ.get("PYTHONHASHSEED") != "0":
            _log.warning(
                "KGR requires PYTHONHASHSEED=0 at process startup for deterministic "
                "edge-type encoding"
            )

    def _get_engine(self):
        if self._engine is False:
            return None
        if self._engine is None:
            try:
                self._prepare_model_environment()
                if self._root not in sys.path:
                    sys.path.insert(0, self._root)
                from src.service.inference_engine import KGRRetriever

                self._engine = KGRRetriever()
                _log.info("KGR temporal ranker ready (root=%s)", self._root)
            except Exception as exc:
                _log.warning(
                    "KGR engine unavailable; temporal queries use %s: %s",
                    type(self._fallback).__name__,
                    exc,
                )
                self._engine = False
                return None
        return self._engine

    def _normalized_window(
        self, year_min: int | None, year_max: int | None
    ) -> tuple[float, float] | None:
        if self._year_range is None:
            rows = self._executor.run(_YEAR_RANGE, {})
            if not rows or rows[0].get("min_year") is None:
                return None
            low = float(rows[0]["min_year"])
            high = max(float(rows[0]["max_year"]), low + 1.0)
            self._year_range = (low, high)
        low, high = self._year_range
        span = high - low
        start = 0.0 if year_min is None else (float(year_min) - low) / span
        end = 1.0 if year_max is None else (float(year_max) - low) / span
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        return (start, end) if start <= end else (end, start)

    def rank(self, blocks, *, entity_context, year_min=None, year_max=None):
        def fall_back():
            return self._fallback.rank(
                blocks,
                entity_context=entity_context,
                year_min=year_min,
                year_max=year_max,
            )

        if not blocks or (year_min is None and year_max is None):
            return fall_back()
        anchors = [entity.entity_id for entity in entity_context.resolved if entity.entity_id]
        if not anchors:
            return fall_back()
        engine = self._get_engine()
        if engine is None:
            return fall_back()

        try:
            seed_rows = self._executor.run(
                _ANCHOR_INTERNAL_IDS, {"entity_ids": anchors}
            )
            seeds = [
                int(row["id"])
                for row in seed_rows
                if row.get("id") is not None
            ]
            window = self._normalized_window(year_min, year_max)
            if not seeds or window is None:
                return fall_back()

            result = engine.retrieve(
                task=2,
                seed_ids=seeds,
                temporal_window=window,
                top_k=self._top_k,
            )
            node_rank = {
                int(node_id): rank
                for rank, (node_id, _score) in enumerate(result.ranked)
            }
            if not node_rank:
                return fall_back()

            claim_ids = [block.claim_id for block in blocks if block.claim_id]
            link_rows = self._executor.run(
                _CLAIMS_FOR_NODES,
                {
                    "claim_ids": claim_ids,
                    "node_ids": list(node_rank),
                    "excluded_rels": _STRUCTURAL_RELS,
                },
            )
            claim_rank: dict[str, int] = {}
            for row in link_rows:
                claim_id = row.get("claim_id")
                ranks = [
                    node_rank[int(node_id)]
                    for node_id in (row.get("node_ids") or [])
                    if int(node_id) in node_rank
                ]
                if claim_id and ranks:
                    claim_rank[claim_id] = min(ranks)
        except Exception as exc:
            _log.warning("KGR temporal ranking failed; using fallback: %s", exc)
            return fall_back()

        if not claim_rank:
            return fall_back()
        base = fall_back()
        promoted = [block for block in base if block.claim_id in claim_rank]
        remainder = [block for block in base if block.claim_id not in claim_rank]
        promoted.sort(
            key=lambda block: (
                claim_rank[block.claim_id],
                -block.extraction_confidence,
            )
        )
        _log.debug("KGR temporal rerank promoted %d/%d blocks", len(promoted), len(base))
        return promoted + remainder

    def close(self) -> None:
        engine = self._engine
        self._engine = False
        if engine not in (None, False):
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        self._fallback.close()


def build_context_ranker(
    executor,
    *,
    mode: str = "auto",
    kgr_root: str = "",
    neo4j_uri: str = "",
    neo4j_username: str = "",
    neo4j_password: str = "",
) -> ContextRanker:
    """Build the configured ranker.

    Modes: ``auto`` (proximity plus KGR when configured), ``proximity``,
    ``kgr``, and ``confidence``.
    """
    normalized_mode = (mode or "auto").strip().lower()
    if normalized_mode == "confidence":
        return ConfidenceRanker()

    proximity = GraphProximityRanker(executor)
    if normalized_mode == "proximity":
        return proximity
    if normalized_mode not in {"auto", "kgr"}:
        _log.warning("unknown context ranker %r; using auto", mode)

    root = (kgr_root or "").strip()
    if root and Path(root).expanduser().is_dir():
        return KGRTemporalRanker(
            executor,
            root,
            fallback=proximity,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
        )
    if root:
        _log.warning("KGR_ROOT does not exist: %s; using proximity", root)
    elif normalized_mode == "kgr":
        _log.warning("CONTEXT_RANKER=kgr but KGR_ROOT is empty; using proximity")
    return proximity
