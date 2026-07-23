"""Deep-research orchestrator — logic tests with a fake backend.

No Neo4j, no Anthropic: a FakeBackend returns canned blocks per angle so we can
assert the merge/dedup/coverage-accounting behaviour deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from gemynd.retrieval.deep_research import (
    DeepResearchOrchestrator,
    HeuristicPlanner,
    PermissionScope,
    QueryProbe,
    ResearchAngle,
)


@dataclass
class FakeBlock:
    claim_id: str
    extraction_confidence: float = 0.5


@dataclass
class FakeSynthResult:
    answer: str


class FakeBackend:
    """Returns a fixed block set per angle *label substring*, so tests control
    exactly which claims each angle 'retrieves'."""

    def __init__(self, by_angle_substr: dict[str, list[FakeBlock]], probe: QueryProbe):
        self._by = by_angle_substr
        self._probe = probe
        self.synthesise_calls: list[list[FakeBlock]] = []
        # Records (angle_query, allow_llm_cypher) for each gather, so tests can
        # assert the LLM-Cypher cost control.
        self.gather_calls: list[tuple[str, bool]] = []

    def probe(self, query: str) -> QueryProbe:
        return self._probe

    def gather(self, angle_query, scope, *, allow_llm_cypher=True):
        self.gather_calls.append((angle_query, allow_llm_cypher))
        # The entity angle text ("... — specifically regarding Turnbull")
        # contains the original query as a substring, so a first-match lookup
        # is ambiguous. Pick the LONGEST matching key = the most specific angle.
        best_substr = None
        for substr in self._by:
            if substr in angle_query and (best_substr is None or len(substr) > len(best_substr)):
                best_substr = substr
        return list(self._by[best_substr]) if best_substr is not None else []

    def synthesise(self, query, blocks, context_text, scope, conversation_history):
        self.synthesise_calls.append(list(blocks))
        return FakeSynthResult(answer=f"answer over {len(blocks)} claims")


SCOPE = PermissionScope(permitted_levels=["public"], institution_ids=["turnbull"])


def test_heuristic_planner_emits_original_plus_entities_plus_broadened():
    probe = QueryProbe(entities=["Turnbull Refuge", "mallard"], year_min=1930, year_max=1939)
    angles = HeuristicPlanner().plan("what happened?", probe, max_angles=10)
    labels = [a.label for a in angles]
    assert labels[0] == "original"
    assert "entity:Turnbull Refuge" in labels
    assert "entity:mallard" in labels
    assert "broadened" in labels
    # every angle text is distinct, or the passes would be redundant
    assert len({a.query_text for a in angles}) == len(angles)


def test_max_angles_caps_the_fan_out():
    probe = QueryProbe(entities=["a", "b", "c", "d", "e"])
    angles = HeuristicPlanner().plan("q", probe, max_angles=3)
    assert len(angles) == 3
    assert angles[0].label == "original"  # the original is never dropped


def test_merge_dedups_by_claim_id_and_unions_the_superset():
    # original finds c1,c2 ; the entity angle finds c2,c3 (c2 overlaps, c3 is new)
    backend = FakeBackend(
        {
            "regarding Turnbull": [FakeBlock("c2"), FakeBlock("c3")],
            # the bare original query matches neither substring below it, so put
            # the original's blocks under a substring unique to it:
            "what happened": [FakeBlock("c1"), FakeBlock("c2")],
        },
        probe=QueryProbe(entities=["Turnbull"]),
    )
    orch = DeepResearchOrchestrator(backend, max_angles=2)
    res = orch.research("what happened", SCOPE, synthesise=False)

    ids = sorted(b.claim_id for b in res.merged_blocks)
    assert ids == ["c1", "c2", "c3"]  # union, deduped
    assert res.unique_claims == 3


def test_coverage_gain_counts_only_claims_found_past_the_first_angle():
    backend = FakeBackend(
        {
            "what happened": [FakeBlock("c1"), FakeBlock("c2")],
            "regarding Turnbull": [FakeBlock("c2"), FakeBlock("c3"), FakeBlock("c4")],
        },
        probe=QueryProbe(entities=["Turnbull"]),
    )
    orch = DeepResearchOrchestrator(backend, max_angles=2)
    res = orch.research("what happened", SCOPE, synthesise=False)

    assert res.new_claims_by_angle["original"] == 2          # c1, c2
    assert res.new_claims_by_angle["entity:Turnbull"] == 2   # c3, c4 (c2 already seen)
    assert res.coverage_gain == 2  # the two claims a single pass would have missed


def test_higher_confidence_instance_wins_on_overlap():
    backend = FakeBackend(
        {
            "what happened": [FakeBlock("c1", extraction_confidence=0.4)],
            "regarding Turnbull": [FakeBlock("c1", extraction_confidence=0.9)],
        },
        probe=QueryProbe(entities=["Turnbull"]),
    )
    orch = DeepResearchOrchestrator(backend, max_angles=2)
    res = orch.research("what happened", SCOPE, synthesise=False)
    assert res.unique_claims == 1
    assert res.merged_blocks[0].extraction_confidence == 0.9


def test_synthesis_runs_over_the_merged_union():
    backend = FakeBackend(
        {
            "what happened": [FakeBlock("c1")],
            "regarding Turnbull": [FakeBlock("c2")],
        },
        probe=QueryProbe(entities=["Turnbull"]),
    )
    orch = DeepResearchOrchestrator(
        backend, max_angles=2, serialise_fn=lambda b: b.claim_id
    )
    res = orch.research("what happened", SCOPE, synthesise=True)
    assert len(backend.synthesise_calls) == 1
    assert len(backend.synthesise_calls[0]) == 2   # both claims, not one pass
    assert res.synthesis is not None
    assert "c1" in res.context_text and "c2" in res.context_text


def test_synthesise_false_skips_the_model_call():
    backend = FakeBackend({"what happened": [FakeBlock("c1")]}, QueryProbe())
    orch = DeepResearchOrchestrator(backend, max_angles=1)
    res = orch.research("what happened", SCOPE, synthesise=False)
    assert backend.synthesise_calls == []
    assert res.synthesis is None


def test_one_failing_angle_does_not_sink_the_run():
    class FlakyBackend(FakeBackend):
        def gather(self, angle_query, scope, *, allow_llm_cypher=True):
            if "regarding" in angle_query:
                raise RuntimeError("simulated Neo4j hiccup on one angle")
            return [FakeBlock("c1")]

    backend = FlakyBackend({}, QueryProbe(entities=["Turnbull"]))
    orch = DeepResearchOrchestrator(backend, max_angles=2)
    res = orch.research("what happened", SCOPE, synthesise=False)
    # original still contributed; the failed angle recorded zero, no exception
    assert res.unique_claims == 1
    assert res.new_claims_by_angle["entity:Turnbull"] == 0


def test_build_backend_routes_through_component_signatures():
    """Pin the real method names/kwargs build_backend calls, so the wiring
    breaks loudly here if gateway/assembler/engine signatures drift."""
    from gemynd.retrieval.deep_research import build_backend

    calls = {"resolve": 0, "assemble": 0, "synthesise": 0}

    class Intent:
        entities = ["Turnbull"]
        year_min = 1930
        year_max = 1939
        bucket = "conversational"

    class Resolved:
        surface_form = "Turnbull"
        entity_id = "ent_1"

    class ECtx:
        resolved = [Resolved()]

    class Gateway:
        def resolve(self, *, surface_forms, entity_hints, inferred_claim_types):
            calls["resolve"] += 1
            assert surface_forms == ["Turnbull"]
            return ECtx()

    class Assembler:
        def assemble(self, *, query_text, entity_context, year_min, year_max,
                     is_hybrid, permitted_levels, institution_ids, intent,
                     allow_llm_cypher=True):
            calls["assemble"] += 1
            assert permitted_levels == ["public"]
            assert institution_ids == ["turnbull"]
            return ([FakeBlock("c1")], "ctx")

    class Engine:
        def synthesise(self, *, query, provenance_blocks, context_text,
                       conversation_history, searched_institution_ids):
            calls["synthesise"] += 1
            assert searched_institution_ids == ["turnbull"]
            return FakeSynthResult("ok")

    backend = build_backend(
        gateway=Gateway(),
        assembler=Assembler(),
        engine=Engine(),
        classify_fn=lambda q: Intent(),
        infer_claim_types_fn=lambda q: ["observation"],
    )
    orch = DeepResearchOrchestrator(backend, max_angles=1, serialise_fn=lambda b: b.claim_id)
    res = orch.research("what happened", SCOPE, synthesise=True)
    assert calls["resolve"] >= 1     # probe + gather each resolve
    assert calls["assemble"] == 1    # one angle
    assert calls["synthesise"] == 1
    assert res.synthesis.answer == "ok"


def test_char_budget_fn_is_applied_to_the_union():
    backend = FakeBackend(
        {"what happened": [FakeBlock(f"c{i}") for i in range(10)]},
        QueryProbe(),
    )
    # Cap to the 3 highest-ranked blocks.
    orch = DeepResearchOrchestrator(
        backend, max_angles=1, char_budget_fn=lambda blocks: blocks[:3]
    )
    res = orch.research("what happened", SCOPE, synthesise=False)
    assert len(res.merged_blocks) == 3


# --- cost controls ---------------------------------------------------------
def test_llm_cypher_allowed_only_on_the_original_angle_by_default():
    backend = FakeBackend(
        {"what happened": [FakeBlock("c1")], "regarding Turnbull": [FakeBlock("c2")]},
        QueryProbe(entities=["Turnbull"]),
    )
    orch = DeepResearchOrchestrator(backend, max_angles=3)
    orch.research("what happened", SCOPE, synthesise=False)
    flags = {q: allow for q, allow in backend.gather_calls}
    # exactly one gather may spend an LLM-Cypher call: the original
    assert sum(1 for _, allow in backend.gather_calls if allow) == 1
    assert flags["what happened"] is True
    assert all(
        allow is False for q, allow in backend.gather_calls if q != "what happened"
    )


def test_operator_can_opt_all_angles_into_llm_cypher():
    backend = FakeBackend(
        {"what happened": [FakeBlock("c1")], "regarding Turnbull": [FakeBlock("c2")]},
        QueryProbe(entities=["Turnbull"]),
    )
    orch = DeepResearchOrchestrator(
        backend, max_angles=3, llm_cypher_on_reformulations=True
    )
    orch.research("what happened", SCOPE, synthesise=False)
    assert all(allow is True for _, allow in backend.gather_calls)


def test_union_is_ranked_centrality_first_so_a_tight_budget_keeps_corroborated():
    # c2 is surfaced by BOTH angles; c1 and c3 by one each. A budget of 1 must
    # keep c2 (the multiply-corroborated claim), not a single-angle claim.
    backend = FakeBackend(
        {
            "what happened": [FakeBlock("c1", 0.99), FakeBlock("c2", 0.5)],
            "regarding Turnbull": [FakeBlock("c2", 0.5), FakeBlock("c3", 0.99)],
        },
        QueryProbe(entities=["Turnbull"]),
    )
    orch = DeepResearchOrchestrator(
        backend, max_angles=2,
        char_budget_fn=lambda blocks: blocks[:1],  # brutal 1-block synthesis cap
        serialise_fn=lambda b: b.claim_id,
    )
    res = orch.research("what happened", SCOPE, synthesise=False)
    # Despite c1/c3 having higher confidence, c2 wins on centrality (2 angles).
    assert [b.claim_id for b in res.merged_blocks] == ["c2"]
