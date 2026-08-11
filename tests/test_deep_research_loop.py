"""Deterministic state-machine tests for bounded deep research."""
from __future__ import annotations

from dataclasses import dataclass

from gemynd.retrieval.deep_research import (
    DeepResearchOrchestrator,
    EvidenceGapAssessment,
    ModelEvidenceGapAdviser,
    PermissionScope,
    QueryProbe,
    ResearchAngle,
    StopReason,
    build_backend,
)


@dataclass
class Block:
    claim_id: str
    extraction_confidence: float = 0.5
    claim_type: str = "observation"
    doc_title: str = "Paper"
    doc_date_start: str | None = "1910-01-01"
    source_sentence: str = "Evidence."
    doc_id: str = "doc"
    paragraph_id: str = "p1"


class StaticPlanner:
    def __init__(self, angles: list[ResearchAngle]):
        self.angles = angles

    def plan(self, query, probe, max_angles):
        return list(self.angles)


class SequenceBackend:
    def __init__(self, responses, *, synthesis_error=False):
        self.responses = list(responses)
        self.calls = []
        self.synthesis_error = synthesis_error

    def probe(self, query):
        return QueryProbe()

    def gather(self, query, scope, *, allow_llm_cypher=True):
        self.calls.append((query, allow_llm_cypher))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return list(response)

    def synthesise(self, query, blocks, context_text, scope, conversation_history):
        if self.synthesis_error:
            raise RuntimeError("provider secret must not escape")
        return object()


class Adviser:
    def __init__(self, *assessments):
        self.assessments = list(assessments)
        self.snapshots = []

    def advise(self, snapshot):
        self.snapshots.append(snapshot)
        response = self.assessments.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


SCOPE = PermissionScope(["public"], ["spokane"])


def _angles(count=4):
    return [
        ResearchAngle("original", "q", "original"),
        *[
            ResearchAngle(f"fallback-{i}", f"fallback query {i}", "fallback")
            for i in range(1, count)
        ],
    ]


def test_two_successful_zero_gain_searches_stop_before_remaining_candidate():
    backend = SequenceBackend([[Block("c1")], [Block("c1")], [Block("c1")], [Block("c2")]])
    result = DeepResearchOrchestrator(
        backend, planner=StaticPlanner(_angles()), max_angles=4
    ).research("q", SCOPE, synthesise=False)

    assert result.stop_reason is StopReason.NO_NEW_EVIDENCE
    assert len(result.attempts) == 3
    assert len(backend.responses) == 1


def test_positive_gain_resets_zero_gain_streak():
    backend = SequenceBackend([
        [Block("c1")], [Block("c1")], [Block("c2")], [Block("c2")]
    ])
    result = DeepResearchOrchestrator(
        backend, planner=StaticPlanner(_angles()), max_angles=4
    ).research("q", SCOPE, synthesise=False)

    assert result.stop_reason is StopReason.ANGLE_BUDGET_EXHAUSTED
    assert [attempt.new_claims for attempt in result.attempts] == [1, 0, 1, 0]


def test_failures_consume_failure_budget_but_not_no_gain_streak():
    backend = SequenceBackend([[Block("c1")], RuntimeError("one"), RuntimeError("two")])
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner(_angles()),
        max_angles=4,
        max_failures=2,
    ).research("q", SCOPE, synthesise=False)

    assert result.stop_reason is StopReason.FAILURE_BUDGET_EXHAUSTED
    assert [attempt.status for attempt in result.attempts] == ["succeeded", "failed", "failed"]
    assert all(attempt.error_code in (None, "gather_failed") for attempt in result.attempts)


def test_angle_budget_precedes_failure_budget_when_both_reached():
    backend = SequenceBackend([RuntimeError("one"), RuntimeError("two")])
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner(_angles(2)),
        max_angles=2,
        max_failures=2,
    ).research("q", SCOPE, synthesise=False)

    assert result.stop_reason is StopReason.ANGLE_BUDGET_EXHAUSTED


def test_empty_fallback_queue_stops_no_candidate_gaps():
    backend = SequenceBackend([[Block("c1")]])
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner([ResearchAngle("original", "q", "original")]),
    ).research("q", SCOPE, synthesise=False)
    assert result.stop_reason is StopReason.NO_CANDIDATE_GAPS


def test_model_angle_is_prioritized_over_heuristic_fallback():
    adviser = Adviser(EvidenceGapAssessment(proposed_angles=(
        ResearchAngle("named-officeholder", "mayor surname Spokane 1910", "find the name"),
    )))
    backend = SequenceBackend([[Block("c1")], [Block("c2")]])
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner(_angles()),
        adviser=adviser,
        max_angles=2,
    ).research("q", SCOPE, synthesise=False)

    assert backend.calls[1][0] == "mayor surname Spokane 1910"
    assert result.angles[1].label == "named-officeholder"


def test_adviser_failure_disables_it_and_uses_fallback_without_leaking_error():
    adviser = Adviser(RuntimeError("api-key-super-secret"))
    backend = SequenceBackend([[Block("c1")], [Block("c2")]])
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner(_angles()),
        adviser=adviser,
        max_angles=2,
    ).research("q", SCOPE, synthesise=False)

    assert result.adviser_failed is True
    assert len(adviser.snapshots) == 1
    assert result.attempts[1].label == "fallback-1"
    assert "api-key" not in repr(result)


def test_false_complete_and_unknown_fields_are_rejected_by_model_adviser():
    adviser = ModelEvidenceGapAdviser(lambda **_: {
        "complete": True, "gaps": [], "known": [], "missing": [], "conflicts": []
    })
    backend = SequenceBackend([[Block("c1")], [Block("c2")]])
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner(_angles()),
        adviser=adviser,
        max_angles=2,
    ).research("q", SCOPE, synthesise=False)

    assert result.adviser_failed is True
    assert result.attempts[1].label == "fallback-1"


def test_duplicate_and_overlong_proposals_cannot_escape_validation():
    adviser = Adviser(EvidenceGapAssessment(proposed_angles=(
        ResearchAngle("duplicate", "q", "replace original"),
        ResearchAngle("too-long", "x" * 501, "over budget"),
    )))
    backend = SequenceBackend([[Block("c1")], [Block("c2")]])
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner(_angles()),
        adviser=adviser,
        max_angles=2,
    ).research("q", SCOPE, synthesise=False)
    assert result.attempts[1].label == "fallback-1"


def test_only_original_attempt_may_use_llm_cypher_in_dynamic_loop():
    adviser = Adviser(EvidenceGapAssessment(proposed_angles=(
        ResearchAngle("gap", "new query", "gap"),
    )))
    backend = SequenceBackend([[Block("c1")], [Block("c2")]])
    DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner(_angles()),
        adviser=adviser,
        max_angles=2,
    ).research("q", SCOPE, synthesise=False)
    assert [allow for _, allow in backend.calls] == [True, False]


def test_synthesis_failure_returns_partial_research_result():
    backend = SequenceBackend([[Block("c1")]], synthesis_error=True)
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner([ResearchAngle("original", "q", "original")]),
    ).research("q", SCOPE)
    assert result.synthesis is None
    assert result.synthesis_error == "synthesis_failed"
    assert [block.claim_id for block in result.merged_blocks] == ["c1"]


def test_fallback_dedup_key_and_equal_score_order_are_stable():
    first = Block("", paragraph_id="p2", source_sentence="second")
    second = Block("", paragraph_id="p1", source_sentence="first")
    orch = DeepResearchOrchestrator(SequenceBackend([]))
    assert orch._dedup_key(first) == orch._dedup_key(first)

    outputs = []
    for blocks in ([first, second], [second, first]):
        backend = SequenceBackend([blocks])
        result = DeepResearchOrchestrator(
            backend,
            planner=StaticPlanner([ResearchAngle("original", "q", "original")]),
        ).research("q", SCOPE, synthesise=False)
        outputs.append([block.source_sentence for block in result.merged_blocks])
    assert outputs[0] == outputs[1]


def test_valid_model_assessment_is_strictly_parsed():
    adviser = ModelEvidenceGapAdviser(lambda **_: ({
        "gaps": [{"label": "office", "query_text": "mayoral election", "rationale": "tenure"}],
        "known": ["the office existed"],
        "missing": ["the officeholder's name"],
        "conflicts": [],
    }, "opencode"))
    backend = SequenceBackend([[Block("c1")], [Block("c2")]])
    result = DeepResearchOrchestrator(
        backend,
        planner=StaticPlanner(_angles()),
        adviser=adviser,
        max_angles=2,
    ).research("q", SCOPE, synthesise=False)
    assert result.assessment is not None
    assert result.assessment.provider_used == "opencode"
    assert result.assessment.missing == ("the officeholder's name",)


def test_request_bound_backend_propagates_scope_hints_and_filters():
    calls = {}

    class Intent:
        entities = ["Spokane"]
        year_min = 1909
        year_max = 1910
        bucket = "conversational"

    class Context:
        resolved = []

    class Gateway:
        def resolve(self, *, surface_forms, entity_hints, inferred_claim_types):
            calls["entity_hints"] = entity_hints
            return Context()

    class Assembler:
        def assemble(self, **kwargs):
            calls["assemble"] = kwargs
            return [Block("c1")], "context"

    synthesis_kwargs = {}

    def synthesise_fn(**kwargs):
        synthesis_kwargs.update(kwargs)
        return "synthesised"

    backend = build_backend(
        gateway=Gateway(),
        assembler=Assembler(),
        engine=object(),
        classify_fn=lambda _: Intent(),
        infer_claim_types_fn=lambda _: ["observation"],
        entity_hints=["Spokane City"],
        doc_types=["newspaper"],
        series=["Spokesman-Review"],
        synthesise_fn=synthesise_fn,
    )
    backend.probe("q")
    blocks = backend.gather("q", SCOPE, allow_llm_cypher=False)
    answer = backend.synthesise("q", blocks, "context", SCOPE, [{"role": "user", "content": "prior"}])

    assert calls["entity_hints"] == ["Spokane City"]
    assert calls["assemble"]["permitted_levels"] == ["public"]
    assert calls["assemble"]["institution_ids"] == ["spokane"]
    assert calls["assemble"]["doc_types"] == ["newspaper"]
    assert calls["assemble"]["series"] == ["Spokesman-Review"]
    assert calls["assemble"]["allow_llm_cypher"] is False
    assert synthesis_kwargs["conversation_history"][0]["content"] == "prior"
    assert answer == "synthesised"
