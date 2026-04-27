"""Unit tests for the LLM-Cypher safety validator (plan B3).

The validator is the only thing standing between LLM-generated Cypher and a
live Neo4j executor, so every rule it enforces is exercised here.
"""
from __future__ import annotations

import pytest

from gemynd.retrieval.cypher_validator import (
    CypherSchema,
    validate_cypher,
)


# A schema that mirrors what the wildlife corpus would inject at runtime.
# Domain labels/edges are deliberately small so the allowlist behavior is
# easy to reason about in test fixtures.
@pytest.fixture()
def schema() -> CypherSchema:
    return CypherSchema.from_domain(
        domain_labels=frozenset({"Refuge", "Place"}),
        domain_edges=frozenset({"REFERS_TO", "POSSIBLY_REFERS_TO"}),
        anchor_relations=frozenset({"ABOUT_REFUGE"}),
    )


# A complete shape-correct query — every other test mutates one thing about
# this baseline so failures localise cleanly to the rule under test.
_GOOD_CYPHER = """
MATCH (e:Entity {entity_id: $entity_id})<-[:REFERS_TO]-(c:Claim)
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
MATCH (pg:Page)-[:HAS_PARAGRAPH]->(para)
MATCH (d:Document)-[:HAS_PAGE]->(pg)
MATCH (d)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, c, para, pg, max(r.run_timestamp) AS latest_run
WHERE d.institution_id IN $institution_ids
  AND d.access_level   IN $permitted_levels
RETURN c, d, pg, para
LIMIT $limit
"""


def test_accepts_shape_correct_query(schema):
    result = validate_cypher(_GOOD_CYPHER, schema)
    assert result.ok, result.violations


@pytest.mark.parametrize("verb", ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"])
def test_rejects_mutation_keywords(schema, verb):
    bad = _GOOD_CYPHER.replace("RETURN c, d, pg, para", f"{verb} (n:Tmp) RETURN c, d, pg, para")
    result = validate_cypher(bad, schema)
    assert not result.ok
    assert any(verb in v for v in result.violations)


def test_rejects_unknown_label(schema):
    bad = _GOOD_CYPHER.replace("(c:Claim)", "(c:Claim), (z:NonsenseLabel)")
    result = validate_cypher(bad, schema)
    assert not result.ok
    assert any("NonsenseLabel" in v for v in result.violations)


def test_rejects_unknown_relationship(schema):
    bad = _GOOD_CYPHER.replace("[:REFERS_TO]", "[:NOT_A_REL]")
    result = validate_cypher(bad, schema)
    assert not result.ok
    assert any("NOT_A_REL" in v for v in result.violations)


def test_rejects_missing_institution_ids(schema):
    bad = _GOOD_CYPHER.replace("d.institution_id IN $institution_ids", "d.institution_id IN ['x']")
    result = validate_cypher(bad, schema)
    assert not result.ok
    assert any("institution_ids" in v for v in result.violations)


def test_rejects_missing_permitted_levels(schema):
    bad = _GOOD_CYPHER.replace("d.access_level   IN $permitted_levels", "d.access_level = 'public'")
    result = validate_cypher(bad, schema)
    assert not result.ok
    assert any("permitted_levels" in v for v in result.violations)


def test_rejects_non_conforming_return(schema):
    # Drops the required `pg` and `para` aliases.
    bad = _GOOD_CYPHER.replace("RETURN c, d, pg, para", "RETURN c, d")
    result = validate_cypher(bad, schema)
    assert not result.ok
    assert any("RETURN missing required aliases" in v for v in result.violations)


def test_identifier_with_create_substring_does_not_trip(schema):
    """`created_at` must not collide with the `CREATE` keyword scan."""
    body = _GOOD_CYPHER.replace(
        "RETURN c, d, pg, para",
        "WITH c, d, pg, para, c.created_at AS created_at RETURN c, d, pg, para",
    )
    result = validate_cypher(body, schema)
    assert result.ok, result.violations


def test_string_literal_with_keyword_does_not_trip(schema):
    """A string literal containing 'MERGE' must not be tokenized as a keyword."""
    body = _GOOD_CYPHER.replace(
        "RETURN c, d, pg, para",
        "WITH c, d, pg, para WHERE c.note <> 'this is a MERGE comment' RETURN c, d, pg, para",
    )
    result = validate_cypher(body, schema)
    assert result.ok, result.violations


def test_call_subquery_is_allowed(schema):
    body = """
    CALL {
      WITH 1 AS one
      MATCH (e:Entity {entity_id: $entity_id})<-[:REFERS_TO]-(c:Claim)
      RETURN c
    }
    MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
    MATCH (pg:Page)-[:HAS_PARAGRAPH]->(para)
    MATCH (d:Document)-[:HAS_PAGE]->(pg)
    WHERE d.institution_id IN $institution_ids AND d.access_level IN $permitted_levels
    RETURN c, d, pg, para
    """
    result = validate_cypher(body, schema)
    assert result.ok, result.violations


def test_call_apoc_is_rejected(schema):
    bad = """
    CALL apoc.path.expand(NULL, '', '', 0, 1) YIELD path
    MATCH (c:Claim), (d:Document), (pg:Page), (para:Paragraph)
    WHERE d.institution_id IN $institution_ids AND d.access_level IN $permitted_levels
    RETURN c, d, pg, para
    """
    result = validate_cypher(bad, schema)
    assert not result.ok
    assert any("apoc" in v for v in result.violations)


def test_explain_guard_rejects_all_nodes_scan(schema):
    """The EXPLAIN cost guard rejects plans containing AllNodesScan."""

    class _StubExecutor:
        def run(self, cypher: str, params: dict | None = None) -> list[dict]:
            return [{"plan": "+ProduceResults\n+AllNodesScan(c)"}]

    result = validate_cypher(
        _GOOD_CYPHER, schema, executor=_StubExecutor(), require_explain=True,
    )
    assert not result.ok
    assert any("AllNodesScan" in v for v in result.violations)


def test_explain_executor_required_when_required(schema):
    result = validate_cypher(_GOOD_CYPHER, schema, executor=None, require_explain=True)
    assert not result.ok
    assert any("executor" in v for v in result.violations)


def test_empty_cypher_rejected(schema):
    result = validate_cypher("", schema)
    assert not result.ok
    assert "empty" in result.reason
