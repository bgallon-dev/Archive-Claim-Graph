"""Adversarial tests for domain-agnosticism of the Tier-1 location Cypher.

`LOCATIONS_FOR_ENTITY_QUERY` previously hardcoded `OCCURRED_AT` and
`p:Place OR p:Refuge`. It is now built by `build_locations_for_entity_query`
from the domain's location relation + entity labels, with the Turnbull values
as the default. Runtime correctness (that these queries return the right rows)
was verified separately against a live Neo4j composite graph; these tests pin
the string-construction contract and the assembler seam.
"""
from __future__ import annotations

from gemynd.core.graph.cypher import (
    LOCATIONS_FOR_ENTITY_QUERY,
    build_locations_for_entity_query,
)


class TestLocationsQueryParity:
    def test_default_build_is_byte_identical_to_constant(self):
        assert build_locations_for_entity_query() == LOCATIONS_FOR_ENTITY_QUERY

    def test_default_uses_turnbull_vocab(self):
        q = build_locations_for_entity_query()
        assert "[:OCCURRED_AT]->" in q
        assert "p:Place OR p:Refuge" in q

    def test_default_is_finalized(self):
        # access-control tokens must be expanded, not left as raw placeholders
        assert "{_CLAIM_ACTIVE}" not in LOCATIONS_FOR_ENTITY_QUERY
        assert "{_DOC_ACCESS}" not in LOCATIONS_FOR_ENTITY_QUERY


class TestLocationsQueryAgnostic:
    def test_custom_relation_and_labels(self):
        q = build_locations_for_entity_query(location_relation="RECORDED_IN", location_labels=("City",))
        assert "[:RECORDED_IN]->" in q
        assert "p:City" in q
        # no Turnbull leak
        assert "OCCURRED_AT" not in q
        assert "Refuge" not in q

    def test_single_label_has_no_or(self):
        q = build_locations_for_entity_query(location_relation="OCCURRED_AT", location_labels=("Place",))
        assert "p:Place" in q
        assert " OR " not in q.split("RETURN")[0].split("WHERE")[-1]  # label predicate has no OR

    def test_multiple_labels_join_with_or(self):
        q = build_locations_for_entity_query(location_relation="X", location_labels=("A", "B", "C"))
        assert "p:A OR p:B OR p:C" in q

    def test_newspaper_location_labels_drop_refuge(self):
        # A Place-only domain must not query the wildlife Refuge label.
        q = build_locations_for_entity_query(location_relation="OCCURRED_AT", location_labels=("Place",))
        assert "Refuge" not in q
        assert q != LOCATIONS_FOR_ENTITY_QUERY


class TestAssemblerLocationsSeam:
    def _assembler(self, **kw):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        # executor is only stored, not used, at construction time
        return ProvenanceContextAssembler(executor=object(), **kw)

    def test_default_uses_module_constant(self):
        a = self._assembler()
        assert a._locations_query == LOCATIONS_FOR_ENTITY_QUERY

    def test_custom_query_is_used(self):
        custom = build_locations_for_entity_query(location_relation="RECORDED_IN", location_labels=("City",))
        a = self._assembler(locations_query=custom)
        assert a._locations_query == custom
        assert a._locations_query != LOCATIONS_FOR_ENTITY_QUERY
