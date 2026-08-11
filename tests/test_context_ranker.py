"""Tests for query-aware provenance context ordering."""
from __future__ import annotations

from types import SimpleNamespace

from gemynd.retrieval.context_ranker import (
    ConfidenceRanker,
    GraphProximityRanker,
    KGRTemporalRanker,
    build_context_ranker,
)
from gemynd.retrieval.models import EntityContext, ProvenanceBlock, ResolvedEntity


def _block(claim_id: str, confidence: float) -> ProvenanceBlock:
    return ProvenanceBlock(
        doc_title="Document",
        doc_date_start=None,
        doc_date_end=None,
        page_number=None,
        paragraph_id="paragraph",
        claim_id=claim_id,
        claim_type="event_claim",
        extraction_confidence=confidence,
        epistemic_status="asserted",
        source_sentence="A documented event occurred.",
    )


def _context(*entity_ids: str) -> EntityContext:
    return EntityContext(
        resolved=[
            ResolvedEntity(
                surface_form=entity_id,
                entity_id=entity_id,
                entity_type="Place",
                resolution_confidence=0.95,
                resolution_relation="REFERS_TO",
            )
            for entity_id in entity_ids
        ]
    )


class ProximityExecutor:
    def __init__(self, by_level=None, fail: bool = False) -> None:
        self.by_level = by_level or {}
        self.fail = fail
        self.calls: list[str] = []

    def run(self, cypher: str, params=None):
        self.calls.append(cypher)
        if self.fail:
            raise RuntimeError("boom")
        level = 1 + cypher.count("]-(x)") + cypher.count("]-(y)")
        eligible = set((params or {}).get("claim_ids") or [])
        return [
            {"claim_id": claim_id}
            for claim_id in self.by_level.get(level, [])
            if claim_id in eligible
        ]


def test_confidence_ranker_orders_descending():
    blocks = [_block("low", 0.2), _block("high", 0.9)]
    result = ConfidenceRanker().rank(blocks, entity_context=_context())
    assert [block.claim_id for block in result] == ["high", "low"]


def test_proximity_ranker_promotes_nearer_claims():
    blocks = [
        _block("unranked", 0.99),
        _block("hop2", 0.5),
        _block("hop1", 0.1),
    ]
    executor = ProximityExecutor(by_level={1: ["hop1"], 2: ["hop2"]})
    result = GraphProximityRanker(executor).rank(
        blocks, entity_context=_context("anchor")
    )
    assert [block.claim_id for block in result] == ["hop1", "hop2", "unranked"]


def test_proximity_ranker_falls_back_on_executor_error():
    blocks = [_block("low", 0.2), _block("high", 0.9)]
    result = GraphProximityRanker(ProximityExecutor(fail=True)).rank(
        blocks, entity_context=_context("anchor")
    )
    assert [block.claim_id for block in result] == ["high", "low"]


def test_kgr_non_temporal_query_does_not_load_engine():
    ranker = KGRTemporalRanker(
        ProximityExecutor(),
        kgr_root="unused",
        fallback=ConfidenceRanker(),
    )
    blocks = [_block("low", 0.2), _block("high", 0.9)]
    result = ranker.rank(blocks, entity_context=_context("anchor"))
    assert [block.claim_id for block in result] == ["high", "low"]
    assert ranker._engine is None


def test_kgr_preserves_model_rank_positions_not_nonmonotonic_scores():
    class Executor:
        def run(self, cypher: str, params=None):
            if "id(a) AS id" in cypher:
                return [{"id": 100}]
            if "min(y.year)" in cypher:
                return [{"min_year": 1900, "max_year": 2000}]
            if "collect(DISTINCT id(n))" in cypher:
                return [
                    {"claim_id": "first", "node_ids": [7]},
                    {"claim_id": "second", "node_ids": [8]},
                ]
            return []

    class Engine:
        def retrieve(self, **kwargs):
            # Scores intentionally contradict the authoritative best-first list.
            return SimpleNamespace(ranked=[(7, -10.0), (8, 100.0)])

    ranker = KGRTemporalRanker(
        Executor(),
        kgr_root="unused",
        fallback=ConfidenceRanker(),
    )
    ranker._engine = Engine()
    blocks = [_block("first", 0.1), _block("second", 0.99)]

    result = ranker.rank(
        blocks,
        entity_context=_context("anchor"),
        year_min=1930,
        year_max=1940,
    )

    assert [block.claim_id for block in result] == ["first", "second"]


def test_kgr_bridges_gemynd_neo4j_username(monkeypatch):
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    ranker = KGRTemporalRanker(
        ProximityExecutor(),
        kgr_root="unused",
        fallback=ConfidenceRanker(),
        neo4j_uri="bolt://database:7687",
        neo4j_username="gemynd_reader",
        neo4j_password="secret",
    )

    ranker._prepare_model_environment()

    assert ranker._neo4j_env["NEO4J_USERNAME"] == "gemynd_reader"
    assert __import__("os").environ["NEO4J_USERNAME"] == "gemynd_reader"


def test_factory_enables_kgr_for_existing_root(tmp_path):
    ranker = build_context_ranker(
        ProximityExecutor(),
        mode="auto",
        kgr_root=str(tmp_path),
    )
    assert isinstance(ranker, KGRTemporalRanker)


def test_settings_load_context_ranker_configuration(monkeypatch, tmp_path):
    from gemynd.shared.settings import Settings

    monkeypatch.setenv("CONTEXT_RANKER", "kgr")
    monkeypatch.setenv("KGR_ROOT", str(tmp_path))

    settings = Settings.from_env()

    assert settings.context_ranker == "kgr"
    assert settings.kgr_root == str(tmp_path)


def test_kgr_close_releases_engine():
    class Engine:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    engine = Engine()
    ranker = KGRTemporalRanker(
        ProximityExecutor(),
        kgr_root="unused",
        fallback=ConfidenceRanker(),
    )
    ranker._engine = engine

    ranker.close()

    assert engine.closed is True
    assert ranker._engine is False


def test_context_assembler_uses_and_closes_injected_ranker():
    from gemynd.retrieval.context_assembler import ProvenanceContextAssembler

    rows = [
        {
            "c": {
                "claim_id": f"claim-{index}",
                "claim_type": "event_claim",
                "extraction_confidence": confidence,
                "source_sentence": "A documented event occurred.",
                "certainty": "asserted",
            },
            "d": {"title": "Document", "institution_id": "turnbull"},
            "pg": {"page_number": 1},
            "para": {"paragraph_id": f"paragraph-{index}"},
            "obs": {},
            "sp": {},
            "y": {},
            "measurements": [],
        }
        for index, confidence in enumerate([0.3, 0.9, 0.6])
    ]

    class Executor:
        def run(self, cypher: str, params=None):
            return list(rows)

    class ReverseRanker:
        def __init__(self) -> None:
            self.closed = False

        def rank(self, blocks, *, entity_context, year_min=None, year_max=None):
            return sorted(blocks, key=lambda block: block.extraction_confidence)

        def close(self) -> None:
            self.closed = True

    ranker = ReverseRanker()
    assembler = ProvenanceContextAssembler(Executor(), context_ranker=ranker)
    blocks, _ = assembler.assemble("what happened", EntityContext())
    assembler.close()

    assert [block.extraction_confidence for block in blocks] == [0.3, 0.6, 0.9]
    assert ranker.closed is True
