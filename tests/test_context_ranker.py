"""Tests for gemynd.retrieval.context_ranker.

The ranker decides which candidate blocks survive the context assembler's
count/char truncation, so ordering IS the behavior under test.  All graph
access goes through a fake executor keyed on query shape.
"""
from __future__ import annotations

import pytest

from gemynd.retrieval.context_ranker import (
    ConfidenceRanker,
    ContextRanker,
    GraphProximityRanker,
    KGRTemporalRanker,
    build_context_ranker,
)
from gemynd.retrieval.models import EntityContext, ProvenanceBlock, ResolvedEntity


def _block(claim_id: str, confidence: float) -> ProvenanceBlock:
    return ProvenanceBlock(
        doc_title="t",
        doc_date_start=None,
        doc_date_end=None,
        page_number=None,
        paragraph_id="p1",
        claim_id=claim_id,
        claim_type="event_claim",
        extraction_confidence=confidence,
        epistemic_status="asserted",
        source_sentence="s",
    )


def _ctx(*entity_ids: str) -> EntityContext:
    return EntityContext(resolved=[
        ResolvedEntity(
            surface_form=e, entity_id=e, entity_type="Place",
            resolution_confidence=0.9, resolution_relation="REFERS_TO",
        )
        for e in entity_ids
    ])


class FakeExecutor:
    """Returns canned rows per proximity level; records calls."""

    def __init__(self, by_level=None, fail=False):
        # by_level: {1: [claim_ids], 2: [...], 3: [...]}
        self.by_level = by_level or {}
        self.fail = fail
        self.calls: list[str] = []

    def run(self, cypher: str, params=None):
        self.calls.append(cypher)
        if self.fail:
            raise RuntimeError("boom")
        level = 1 + cypher.count("]-(x)") + cypher.count("]-(y)")
        cids = set((params or {}).get("claim_ids") or [])
        return [
            {"claim_id": c}
            for c in self.by_level.get(level, [])
            if c in cids
        ]


class TestConfidenceRanker:
    def test_orders_by_confidence_desc(self):
        blocks = [_block("a", 0.2), _block("b", 0.9), _block("c", 0.5)]
        out = ConfidenceRanker().rank(blocks, entity_context=_ctx())
        assert [b.claim_id for b in out] == ["b", "c", "a"]


class TestGraphProximityRanker:
    def test_distance_then_confidence(self):
        # far (unranked) has the highest confidence; near beats it anyway.
        blocks = [_block("far", 0.99), _block("hop2", 0.5), _block("hop1", 0.1)]
        ex = FakeExecutor(by_level={1: ["hop1"], 2: ["hop2"]})
        out = GraphProximityRanker(ex).rank(
            blocks, entity_context=_ctx("ent_anchor"))
        assert [b.claim_id for b in out] == ["hop1", "hop2", "far"]

    def test_confidence_breaks_ties_within_a_hop(self):
        blocks = [_block("lo", 0.3), _block("hi", 0.8)]
        ex = FakeExecutor(by_level={1: ["lo", "hi"]})
        out = GraphProximityRanker(ex).rank(
            blocks, entity_context=_ctx("ent_anchor"))
        assert [b.claim_id for b in out] == ["hi", "lo"]

    def test_no_resolved_anchors_falls_back_to_confidence(self):
        blocks = [_block("a", 0.2), _block("b", 0.9)]
        ex = FakeExecutor()
        out = GraphProximityRanker(ex).rank(blocks, entity_context=_ctx())
        assert [b.claim_id for b in out] == ["b", "a"]
        assert ex.calls == []  # no graph round-trip without anchors

    def test_executor_error_falls_back_to_confidence(self):
        blocks = [_block("a", 0.2), _block("b", 0.9)]
        ex = FakeExecutor(fail=True)
        out = GraphProximityRanker(ex).rank(
            blocks, entity_context=_ctx("ent_anchor"))
        assert [b.claim_id for b in out] == ["b", "a"]

    def test_stops_querying_once_all_ranked(self):
        blocks = [_block("a", 0.2)]
        ex = FakeExecutor(by_level={1: ["a"]})
        GraphProximityRanker(ex).rank(blocks, entity_context=_ctx("e"))
        assert len(ex.calls) == 1  # L2/L3 never issued


class TestKGRTemporalRanker:
    def test_non_temporal_query_delegates_to_fallback(self):
        blocks = [_block("a", 0.2), _block("b", 0.9)]
        ranker = KGRTemporalRanker(
            FakeExecutor(), kgr_root="C:/nonexistent",
            fallback=ConfidenceRanker())
        out = ranker.rank(blocks, entity_context=_ctx("e"))
        assert [b.claim_id for b in out] == ["b", "a"]
        assert ranker._engine is None  # engine never even constructed

    def test_missing_engine_falls_back(self):
        blocks = [_block("a", 0.2), _block("b", 0.9)]
        ranker = KGRTemporalRanker(
            FakeExecutor(), kgr_root="C:/nonexistent/kgr",
            fallback=ConfidenceRanker())
        out = ranker.rank(
            blocks, entity_context=_ctx("e"), year_min=1900, year_max=1910)
        assert [b.claim_id for b in out] == ["b", "a"]
        assert ranker._engine is False  # init tried once, disabled thereafter

    def test_scored_blocks_promoted_above_fallback_order(self):
        # Stub the engine + executor plumbing: node 7 scores high and links
        # to claim "a"; claim "b" is unscored despite higher confidence.
        class _Result:
            ranked = [(7, -1.5)]

        class _Engine:
            def retrieve(self, **kw):
                return _Result()

        class _Ex(FakeExecutor):
            def run(self, cypher, params=None):
                if "id(a) AS id" in cypher:
                    return [{"id": 7}]
                if "min(y.year)" in cypher:
                    return [{"min_year": 1880, "max_year": 1950}]
                if "collect(DISTINCT id(n))" in cypher:
                    return [{"claim_id": "a", "node_ids": [7]}]
                return []

        ranker = KGRTemporalRanker(
            _Ex(), kgr_root="unused", fallback=ConfidenceRanker())
        ranker._engine = _Engine()
        blocks = [_block("a", 0.2), _block("b", 0.9)]
        out = ranker.rank(
            blocks, entity_context=_ctx("e"), year_min=1900, year_max=1910)
        assert [b.claim_id for b in out] == ["a", "b"]


class TestFactory:
    def test_modes(self, monkeypatch):
        monkeypatch.delenv("CONTEXT_RANKER", raising=False)
        monkeypatch.delenv("KGR_ROOT", raising=False)
        ex = FakeExecutor()
        assert isinstance(
            build_context_ranker(ex, mode="confidence"), ConfidenceRanker)
        assert isinstance(
            build_context_ranker(ex, mode="proximity"), GraphProximityRanker)
        assert isinstance(
            build_context_ranker(ex, mode="auto"), GraphProximityRanker)
        kgr = build_context_ranker(ex, mode="auto", kgr_root="C:/kgr")
        assert isinstance(kgr, KGRTemporalRanker)
        assert isinstance(kgr._fallback, GraphProximityRanker)
        # kgr requested but unconfigured -> proximity, not a crash
        assert isinstance(
            build_context_ranker(ex, mode="kgr"), GraphProximityRanker)

    def test_satisfies_protocol(self):
        ranker: ContextRanker = build_context_ranker(FakeExecutor())
        out = ranker.rank([_block("a", 0.5)], entity_context=_ctx())
        assert [b.claim_id for b in out] == ["a"]


class TestAssemblerIntegration:
    def test_assembler_uses_injected_ranker(self):
        """assemble() must order its output by the injected ranker."""
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler

        rows = [
            {"c": {"claim_id": f"claim_{i}", "claim_type": "event_claim",
                   "extraction_confidence": conf,
                   "source_sentence": "Meeting occurred in the city that year.",
                   "epistemic_status": "asserted"},
             "doc": {"title": "Doc", "doc_id": "d1"},
             "pg": {"page_number": 1}, "para": {"paragraph_id": "p1"}}
            for i, conf in enumerate([0.3, 0.9, 0.6])
        ]

        class _Ex:
            def run(self, cypher, params=None):
                return list(rows)

        class _Reverse:
            def rank(self, blocks, *, entity_context, year_min=None, year_max=None):
                return sorted(blocks, key=lambda b: b.extraction_confidence)

        assembler = ProvenanceContextAssembler(
            _Ex(), context_ranker=_Reverse())
        blocks, _ = assembler.assemble("what happened", _ctx())
        confs = [b.extraction_confidence for b in blocks]
        assert confs == sorted(confs)  # ranker order survived the caps

    def test_assembler_default_is_confidence(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler

        assembler = ProvenanceContextAssembler.__new__(ProvenanceContextAssembler)
        # Constructor default: no ranker injected -> legacy confidence order.
        assembler.__init__(FakeExecutor())
        assert isinstance(assembler._context_ranker, ConfidenceRanker)
