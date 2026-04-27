"""Integration tests for the text-to-Cypher LLM fallback (plan B1+B5+B6).

These tests stub both the executor and the generator so the wiring can be
verified end-to-end without a live Neo4j or a live Anthropic call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
from gemynd.retrieval.cypher_validator import CypherSchema
from gemynd.retrieval.llm_cypher import (
    InMemoryCypherCache,
    LLMCypherGenerator,
)
from gemynd.retrieval.models import (
    EntityContext,
    QueryIntent,
)


# A row shape that satisfies _row_to_block.
def _make_row(claim_id: str = "c1") -> dict:
    return {
        "c": {
            "claim_id": claim_id,
            "claim_type": "weather_observation",
            "extraction_confidence": 0.9,
            "certainty": "observed",
            "source_sentence": "It rained on Tuesday.",
        },
        "d": {
            "title": "Test Doc",
            "doc_id": "doc-1",
            "access_level": "public",
        },
        "pg": {"page_number": 1},
        "para": {"paragraph_id": "para-1"},
        "obs": {},
        "sp": {},
        "y": {},
        "measurements": [],
    }


_LLM_CYPHER = """
MATCH (e:Entity {entity_id: $entity_id})<-[:REFERS_TO]-(c:Claim)
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
MATCH (pg:Page)-[:HAS_PARAGRAPH]->(para)
MATCH (d:Document)-[:HAS_PAGE]->(pg)
WHERE d.institution_id IN $institution_ids
  AND d.access_level   IN $permitted_levels
RETURN c, d, pg, para
LIMIT $limit
"""


class _StubExecutor:
    """Returns rows only when handed the LLM-generated cypher.

    Every other query (the rule-based templates) returns the empty list, which
    forces the assembler down the relaxation chain and eventually into the
    LLM fallback.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        self.calls.append((cypher, dict(params or {})))
        if cypher.strip().startswith("MATCH (e:Entity"):
            return [_make_row("c-from-llm")]
        return []


@dataclass
class _FakeLLMGenerator:
    """Returns a fixed cypher; never calls Anthropic."""

    cypher: str | None

    def generate(self, query_text: str, intent: QueryIntent | None = None) -> str | None:
        return self.cypher


def test_llm_fallback_promotes_path_when_rules_return_empty():
    executor = _StubExecutor()
    assembler = ProvenanceContextAssembler(
        executor=executor,
        institution_ids=["test_inst"],
        llm_cypher_generator=_FakeLLMGenerator(cypher=_LLM_CYPHER),
    )
    intent = QueryIntent(
        bucket="conversational",
        classifier_confidence=0.4,  # below ceiling — fallback eligible
        entities=[],
    )
    blocks, ctx = assembler.assemble(
        query_text="something obscure no template can answer",
        entity_context=EntityContext(),
        permitted_levels=["public"],
        intent=intent,
    )
    assert blocks, "expected the LLM-cypher row to flow through to a block"
    assert blocks[0].claim_id == "c-from-llm"
    assert assembler._last_retrieval_path == "llm_cypher"
    # The trace's final attempt is the LLM rung.
    labels = [a.label for a in assembler._last_plan_trace.attempts]
    assert "llm_cypher" in labels


def test_llm_fallback_skipped_when_classifier_confident():
    """High-confidence rule-based zero-result answers must NOT trip the LLM."""
    executor = _StubExecutor()
    assembler = ProvenanceContextAssembler(
        executor=executor,
        institution_ids=["test_inst"],
        llm_cypher_generator=_FakeLLMGenerator(cypher=_LLM_CYPHER),
    )
    intent = QueryIntent(
        bucket="conversational",
        classifier_confidence=0.95,  # above ceiling — confident zero result
        entities=[],
    )
    blocks, _ctx = assembler.assemble(
        query_text="some confident question",
        entity_context=EntityContext(),
        permitted_levels=["public"],
        intent=intent,
    )
    assert blocks == []
    assert assembler._last_retrieval_path == "empty"
    labels = [a.label for a in assembler._last_plan_trace.attempts]
    assert "llm_cypher" not in labels


def test_llm_fallback_skipped_when_rule_path_returned_rows():
    class _RuleHits(_StubExecutor):
        def run(self, cypher, params=None):
            self.calls.append((cypher, dict(params or {})))
            return [_make_row("c-from-rules")]

    executor = _RuleHits()
    fake_gen = _FakeLLMGenerator(cypher=_LLM_CYPHER)
    assembler = ProvenanceContextAssembler(
        executor=executor,
        institution_ids=["test_inst"],
        llm_cypher_generator=fake_gen,
    )
    intent = QueryIntent(
        bucket="conversational",
        classifier_confidence=0.3,
        entities=[],
    )
    blocks, _ = assembler.assemble(
        query_text="anything",
        entity_context=EntityContext(),
        permitted_levels=["public"],
        intent=intent,
    )
    assert blocks
    assert blocks[0].claim_id == "c-from-rules"
    assert assembler._last_retrieval_path != "llm_cypher"


def test_llm_generator_validates_output_before_returning():
    """A generator whose model produces forbidden cypher must yield None."""

    @dataclass
    class _FakeMessage:
        content: list

    @dataclass
    class _FakeBlock:
        text: str

    class _FakeMessages:
        def __init__(self, payload: str):
            self._payload = payload

        def create(self, **_kw) -> _FakeMessage:
            return _FakeMessage(content=[_FakeBlock(text=self._payload)])

    class _FakeClient:
        def __init__(self, payload: str):
            self.messages = _FakeMessages(payload)

    schema = CypherSchema.from_domain(
        domain_labels=frozenset({"Refuge"}),
        domain_edges=frozenset({"REFERS_TO"}),
        anchor_relations=frozenset({"ABOUT_REFUGE"}),
    )
    # A query missing $institution_ids — validator must reject it.
    bad_cypher = (
        "MATCH (c:Claim), (d:Document), (pg:Page), (para:Paragraph) "
        "WHERE d.access_level IN $permitted_levels "
        "RETURN c, d, pg, para LIMIT $limit"
    )
    gen = LLMCypherGenerator(
        client=_FakeClient(bad_cypher),
        domain_config=_StubDomain(),
        schema=schema,
    )
    assert gen.generate("anything") is None


def test_llm_generator_returns_validated_cypher():
    @dataclass
    class _FakeMessage:
        content: list

    @dataclass
    class _FakeBlock:
        text: str

    class _FakeMessages:
        def __init__(self, payload: str):
            self._payload = payload
            self.calls = 0

        def create(self, **_kw) -> _FakeMessage:
            self.calls += 1
            return _FakeMessage(content=[_FakeBlock(text=self._payload)])

    class _FakeClient:
        def __init__(self, payload: str):
            self.messages = _FakeMessages(payload)

    schema = CypherSchema.from_domain(
        domain_labels=frozenset({"Refuge"}),
        domain_edges=frozenset({"REFERS_TO"}),
        anchor_relations=frozenset({"ABOUT_REFUGE"}),
    )
    cache = InMemoryCypherCache()
    client = _FakeClient(_LLM_CYPHER)
    gen = LLMCypherGenerator(
        client=client,
        domain_config=_StubDomain(),
        schema=schema,
        cache=cache,
    )
    out1 = gen.generate("first call")
    assert out1 and "RETURN c, d, pg, para" in out1
    # Second call with the same query text must come from cache, not the LLM.
    out2 = gen.generate("first call")
    assert out2 == out1
    assert client.messages.calls == 1


def test_llm_generator_returns_none_when_client_missing():
    schema = CypherSchema.from_domain(
        domain_labels=frozenset(),
        domain_edges=frozenset(),
    )
    gen = LLMCypherGenerator(client=None, domain_config=_StubDomain(), schema=schema)
    assert gen.generate("anything") is None


# ---------------------------------------------------------------------------
# Stub domain config — minimal surface for the generator's prompt builder.
# ---------------------------------------------------------------------------


class _StubDomain:
    allowed_claim_types = frozenset({"weather_observation", "species_presence"})

    @property
    def members(self):
        return [self]

    anchor_relation = "ABOUT_REFUGE"
