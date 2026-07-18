"""Query-aware ordering for provenance context blocks.

The context assembler over-fetches candidate claims, then truncates to a
count cap and a character budget — so the ORDER of the candidate list
decides what the synthesis model actually sees.  Historically that order
was static ``extraction_confidence``, which is query-independent and
measured at the random-ordering floor for relevance (nDCG@10 0.26 vs 0.23
random on live-graph temporal tasks; see ``out/kgr_ab_eval_2026-07-07/``).

Three strategies, all falling back gracefully to the next-simpler one:

``ConfidenceRanker``
    The legacy static order.  Always available; the terminal fallback.

``GraphProximityRanker``
    Orders by graph distance (1..3 hops over semantic edges) from the
    query's resolved anchor entities, confidence as tie-break.  Beat the
    confidence sort on every task family in the live A/B (nDCG@10
    0.57-0.97 vs 0.00-0.40).  Pure Cypher — no model, no extra deps.

``KGRTemporalRanker``
    For temporally-scoped queries only: reranks via the frozen Kettle
    Graph Reasoner serving engine (validated routed cell, nDCG@10
    0.64-0.68 vs 0.60 anchor-BFS).  Requires the KGR repo on disk
    (``KGR_ROOT``) and its torch dependency; anything missing or failing
    degrades to the wrapped fallback ranker.  The engine's one-time
    whole-graph pull (~20 s) is paid on the first temporal query of the
    process and memoized thereafter.

Rel-type / label exclusions below are ENGINE-level names (emitted by
gemynd's own ingest loader for every domain), not collection-specific
constants — paths through document containment, extraction provenance,
or Year/Period scaffolding hubs would make graph distance meaningless.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from gemynd.retrieval.models import EntityContext, ProvenanceBlock

_log = logging.getLogger(__name__)

# Structural / provenance / temporal-scaffolding edges emitted by the ingest
# engine itself (see ingest/graph/writer.py).  Excluded from proximity paths:
# they connect everything to everything through Document / Year hubs.
_STRUCTURAL_RELS = [
    "HAS_PAGE", "HAS_SECTION", "HAS_PARAGRAPH", "HAS_CLAIM",
    "CONTAINS_MENTION", "EVIDENCED_BY", "PROCESSED_BY", "SOURCED_FROM",
    "REFERS_TO", "POSSIBLY_REFERS_TO", "NEXT",
]
# Engine-level temporal scaffolding labels (core/models.py record types).
_SCAFFOLD_LABELS = ["Year", "Period"]

# Distance assigned to blocks not reached within the hop cap: they sort
# after every ranked block, in confidence order.
_UNRANKED_DIST = 99

# Claim-side level queries: explicit per-hop patterns (rather than a
# variable-length shortestPath) so the planner can prune at each hop and
# hub fan-out can never trigger an exhaustive path search.
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
    """Orders candidate blocks best-first; truncation keeps a prefix."""

    def rank(
        self,
        blocks: "list[ProvenanceBlock]",
        *,
        entity_context: "EntityContext",
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> "list[ProvenanceBlock]": ...


def _by_confidence(blocks: "list[ProvenanceBlock]") -> "list[ProvenanceBlock]":
    return sorted(blocks, key=lambda b: b.extraction_confidence, reverse=True)


class ConfidenceRanker:
    """Legacy static order: extraction_confidence descending."""

    def rank(self, blocks, *, entity_context, year_min=None, year_max=None):
        return _by_confidence(blocks)


class GraphProximityRanker:
    """Rank by hop distance from resolved anchor entities over semantic edges.

    Level-wise BFS (1..``max_hops`` hops, capped at 3) issued as one bounded
    Cypher per level; blocks the anchors cannot reach sort last.  With no
    resolved anchors, or on any executor error, degrades to confidence order.
    """

    def __init__(self, executor, max_hops: int = 3, max_claims: int = 500) -> None:
        self._executor = executor
        self._levels = [_PROXIMITY_L1, _PROXIMITY_L2, _PROXIMITY_L3][:max(1, min(max_hops, 3))]
        self._max_claims = max_claims

    def rank(self, blocks, *, entity_context, year_min=None, year_max=None):
        anchors = [e.entity_id for e in entity_context.resolved if e.entity_id]
        if not anchors or not blocks:
            return _by_confidence(blocks)

        ordered = _by_confidence(blocks)
        # Distance is only computed for the head of the (possibly huge)
        # candidate list; the tail keeps confidence order after it anyway.
        eligible = [b.claim_id for b in ordered[: self._max_claims] if b.claim_id]
        remaining = set(eligible)
        dist: dict[str, int] = {}
        try:
            for hops, template in enumerate(self._levels, start=1):
                if not remaining:
                    break
                rows = self._executor.run(template, {
                    "claim_ids": sorted(remaining),
                    "entity_ids": anchors,
                    "excluded_rels": _STRUCTURAL_RELS,
                    "excluded_labels": _SCAFFOLD_LABELS,
                })
                for r in rows:
                    cid = r.get("claim_id")
                    if cid in remaining:
                        dist[cid] = hops
                        remaining.discard(cid)
        except Exception as exc:
            _log.warning("proximity ranking failed — using confidence order: %s", exc)
            return ordered

        if not dist:
            return ordered
        return sorted(
            ordered,
            key=lambda b: (dist.get(b.claim_id, _UNRANKED_DIST),
                           -b.extraction_confidence),
        )


class KGRTemporalRanker:
    """KGR-routed reranking for temporally-scoped queries.

    Applies only when the query carries year bounds AND at least one anchor
    entity resolved (the engine grows its subgraph from seeds); every other
    query — and every failure mode — delegates to ``fallback``.  Blocks whose
    linked domain nodes the engine scored are promoted (score desc, confidence
    tie-break) ahead of the fallback-ordered remainder, mirroring the
    deployed retrieve-then-rerank overlay validated in the A/B.
    """

    def __init__(self, executor, kgr_root: str, fallback: ContextRanker,
                 top_k: int = 50) -> None:
        self._executor = executor
        self._root = kgr_root
        self._fallback = fallback
        self._top_k = top_k
        self._engine = None          # None = not tried, False = init failed
        self._year_range: tuple[float, float] | None = None

    # -- lazy engine ---------------------------------------------------------

    def _get_engine(self):
        if self._engine is False:
            return None
        if self._engine is None:
            try:
                import sys
                if self._root not in sys.path:
                    sys.path.insert(0, self._root)
                from src.service.inference_engine import KGRRetriever
                self._engine = KGRRetriever()
                _log.info("KGR temporal ranker ready (root=%s)", self._root)
            except Exception as exc:
                _log.warning(
                    "KGR engine unavailable — temporal queries fall back to "
                    "%s: %s", type(self._fallback).__name__, exc)
                self._engine = False
                return None
        return self._engine

    # -- helpers -------------------------------------------------------------

    def _normalized_window(self, year_min, year_max) -> tuple[float, float] | None:
        if self._year_range is None:
            rows = self._executor.run(_YEAR_RANGE, {})
            if not rows or rows[0].get("min_year") is None:
                return None
            lo, hi = float(rows[0]["min_year"]), float(rows[0]["max_year"])
            self._year_range = (lo, max(hi, lo + 1.0))
        lo, hi = self._year_range
        span = hi - lo
        a = 0.0 if year_min is None else (float(year_min) - lo) / span
        b = 1.0 if year_max is None else (float(year_max) - lo) / span
        a, b = max(0.0, min(1.0, a)), max(0.0, min(1.0, b))
        return (a, b) if a <= b else (b, a)

    # -- ranking -------------------------------------------------------------

    def rank(self, blocks, *, entity_context, year_min=None, year_max=None):
        def _fall():
            return self._fallback.rank(
                blocks, entity_context=entity_context,
                year_min=year_min, year_max=year_max)

        if not blocks or (year_min is None and year_max is None):
            return _fall()
        anchors = [e.entity_id for e in entity_context.resolved if e.entity_id]
        if not anchors:
            return _fall()
        engine = self._get_engine()
        if engine is None:
            return _fall()

        try:
            seed_rows = self._executor.run(
                _ANCHOR_INTERNAL_IDS, {"entity_ids": anchors})
            seeds = [int(r["id"]) for r in seed_rows if r.get("id") is not None]
            window = self._normalized_window(year_min, year_max)
            if not seeds or window is None:
                return _fall()

            result = engine.retrieve(
                task=2, seed_ids=seeds, temporal_window=window,
                top_k=self._top_k)
            node_scores = {int(nid): float(s) for nid, s in result.ranked}
            if not node_scores:
                return _fall()

            claim_ids = [b.claim_id for b in blocks if b.claim_id]
            link_rows = self._executor.run(_CLAIMS_FOR_NODES, {
                "claim_ids": claim_ids,
                "node_ids": list(node_scores),
                "excluded_rels": _STRUCTURAL_RELS,
            })
            claim_score: dict[str, float] = {}
            for r in link_rows:
                cid = r.get("claim_id")
                nids = r.get("node_ids") or []
                scores = [node_scores[int(n)] for n in nids if int(n) in node_scores]
                if cid and scores:
                    claim_score[cid] = max(scores)
        except Exception as exc:
            _log.warning("KGR temporal ranking failed — falling back: %s", exc)
            return _fall()

        if not claim_score:
            return _fall()
        base = _fall()
        scored = [b for b in base if b.claim_id in claim_score]
        rest = [b for b in base if b.claim_id not in claim_score]
        scored.sort(key=lambda b: (-claim_score[b.claim_id],
                                   -b.extraction_confidence))
        _log.debug("KGR temporal rerank promoted %d/%d blocks",
                   len(scored), len(base))
        return scored + rest


def build_context_ranker(
    executor,
    *,
    mode: str | None = None,
    kgr_root: str | None = None,
) -> ContextRanker:
    """Build the configured ranker.

    ``mode`` (or env ``CONTEXT_RANKER``): ``auto`` (default) | ``proximity``
    | ``kgr`` | ``confidence``.  ``auto`` = graph proximity, wrapped by the
    KGR temporal reranker when ``kgr_root`` (or env ``KGR_ROOT``) points at
    a kettle-graph-reasoner checkout.  ``kgr`` without a root degrades to
    proximity with a warning.
    """
    mode = (mode or os.environ.get("CONTEXT_RANKER") or "auto").strip().lower()
    kgr_root = (kgr_root or os.environ.get("KGR_ROOT") or "").strip()

    if mode == "confidence":
        return ConfidenceRanker()
    proximity = GraphProximityRanker(executor)
    if mode == "proximity":
        return proximity
    if mode not in ("auto", "kgr"):
        _log.warning("Unknown CONTEXT_RANKER=%r — using 'auto'", mode)
    if kgr_root:
        return KGRTemporalRanker(executor, kgr_root, fallback=proximity)
    if mode == "kgr":
        _log.warning("CONTEXT_RANKER=kgr but no KGR_ROOT set — using proximity")
    return proximity
