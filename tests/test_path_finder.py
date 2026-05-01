"""Unit tests for the Mode 1 path-finder orchestration.

The cypher queries themselves are exercised via integration tests (not
covered here); this module tests the pure-Python orchestration logic:
cascade ordering, weighting, degree-cap with hub exception, family
collapsing, disconnected fallback, temporal-filter parameter
propagation.

A recording stub executor stands in for ``Neo4jQueryExecutor.run``: it
returns canned rows for specific cypher constants and records every
call so tests can assert on what queries fired with which params.
"""
from __future__ import annotations

from typing import Any

import pytest

from gemynd.core.graph.cypher import (
    CO_ASSERTING_CLAIMS_QUERY,
    CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY,
    PATHS_1_HOP_QUERY,
    PATHS_2_HOP_QUERY,
)
from gemynd.retrieval.web.path_finder import (
    CoAssertion,
    Path,
    PathEdge,
    PathNode,
    collapse_path_families,
    compute_path_weight,
    degree_cap_threshold,
    excluded_hub_ids,
    filter_by_degree_cap,
    find_paths,
    path_result_to_dict,
)


# ---------------------------------------------------------------------------
# Recording stub executor
# ---------------------------------------------------------------------------


class StubExecutor:
    """Returns canned rows keyed by cypher template identity, records calls."""

    def __init__(self, canned: dict[str, list[dict[str, Any]]]) -> None:
        self.canned = canned
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append((cypher, dict(params)))
        return list(self.canned.get(cypher, []))


# ---------------------------------------------------------------------------
# Cascade ordering
# ---------------------------------------------------------------------------


class TestCascadeOrdering:
    def test_co_assertions_short_circuit_multi_hop(self):
        stub = StubExecutor({
            CO_ASSERTING_CLAIMS_QUERY: [{
                "claim_id": "c1", "sentence": "A and B share this",
                "claim_type": "subject_of_claim", "confidence": 0.91,
                "from_relation": "SUBJECT_OF_CLAIM", "to_relation": "TOPIC_OF_CLAIM",
                "doc_id": "d1", "doc_title": "Doc 1", "doc_date_start": "1909-03-15",
                "doc_report_year": 1909, "institution_id": "spokane_newspaper",
            }],
            # If the cascade respects co-assertion-first, hop queries should
            # never fire at all.
            PATHS_1_HOP_QUERY: [],
            PATHS_2_HOP_QUERY: [],
        })
        result = find_paths(
            stub.run,
            from_id="ent_a", to_id="ent_b",
            institution_ids=["spokane_newspaper"],
            permitted_levels=["public"],
        )
        assert len(result.co_assertions) == 1
        assert result.paths == []
        assert not result.disconnected
        # Only the co-assertion query ran.
        assert [c[0] for c in stub.calls] == [CO_ASSERTING_CLAIMS_QUERY]

    def test_include_paths_when_co_asserted_runs_both(self):
        stub = StubExecutor({
            CO_ASSERTING_CLAIMS_QUERY: [{
                "claim_id": "c1", "sentence": "direct", "claim_type": "x",
                "confidence": 0.9, "from_relation": "X", "to_relation": "Y",
                "doc_id": "d1", "doc_title": "t", "doc_date_start": None,
                "doc_report_year": None, "institution_id": "x",
            }],
            PATHS_1_HOP_QUERY: [],
            PATHS_2_HOP_QUERY: [],
        })
        result = find_paths(
            stub.run,
            from_id="ent_a", to_id="ent_b",
            institution_ids=["x"], permitted_levels=["public"],
            include_paths_when_co_asserted=True,
        )
        assert len(result.co_assertions) == 1
        # Both hop queries also fired even though co-assertions exist.
        called = [c[0] for c in stub.calls]
        assert PATHS_1_HOP_QUERY in called

    def test_falls_through_to_1_hop_when_no_co_assertion(self):
        stub = StubExecutor({
            CO_ASSERTING_CLAIMS_QUERY: [],
            PATHS_1_HOP_QUERY: [_make_1_hop_row("ent_i", "Intermediate", "Person")],
            PATHS_2_HOP_QUERY: [],
        })
        result = find_paths(
            stub.run,
            from_id="ent_a", to_id="ent_b",
            institution_ids=["x"], permitted_levels=["public"],
        )
        assert result.co_assertions == []
        assert len(result.paths) == 1
        # 1-hop fired; 2-hop should NOT fire because 1-hop returned rows.
        called = [c[0] for c in stub.calls]
        assert PATHS_2_HOP_QUERY not in called

    def test_falls_through_to_2_hop_when_1_hop_empty(self):
        stub = StubExecutor({
            CO_ASSERTING_CLAIMS_QUERY: [],
            PATHS_1_HOP_QUERY: [],
            PATHS_2_HOP_QUERY: [_make_2_hop_row("i1", "I1", "Person", "i2", "I2", "Place")],
        })
        result = find_paths(
            stub.run,
            from_id="ent_a", to_id="ent_b",
            institution_ids=["x"], permitted_levels=["public"],
            max_hops=2,
        )
        assert len(result.paths) == 1
        assert len(result.paths[0].intermediates) == 2

    def test_disconnected_with_fallback_neighbors(self):
        stub = StubExecutor({
            CO_ASSERTING_CLAIMS_QUERY: [],
            PATHS_1_HOP_QUERY: [],
            PATHS_2_HOP_QUERY: [],
            CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY: [
                {"entity_id": "neighbor_1", "name": "N1", "entity_type": "Person",
                 "co_occurrence_count": 12},
                {"entity_id": "neighbor_2", "name": "N2", "entity_type": "Place",
                 "co_occurrence_count": 7},
            ],
        })
        result = find_paths(
            stub.run,
            from_id="ent_a", to_id="ent_b",
            institution_ids=["x"], permitted_levels=["public"],
        )
        assert result.disconnected is True
        assert len(result.fallback_from) == 2
        assert len(result.fallback_to) == 2
        assert result.fallback_from[0].entity_id == "neighbor_1"


# ---------------------------------------------------------------------------
# Weighting + degree cap
# ---------------------------------------------------------------------------


class TestWeighting:
    def test_inverse_frequency_favors_specific_intermediates(self):
        # Two paths, identical edges, different intermediates.
        # I_specific has degree 12; I_generic has degree 8000.
        edges = [
            PathEdge("c1", "s1", 0.9, "X", "Y", "d1", None, None),
            PathEdge("c2", "s2", 0.9, "X", "Y", "d2", None, None),
        ]
        nodes_specific = [PathNode("ent_specific", "S", "Person")]
        nodes_generic = [PathNode("ent_generic", "G", "Place")]
        degree = {"ent_specific": 12, "ent_generic": 8000}
        w_specific = compute_path_weight(nodes_specific, edges, degree, corpus_total_claims=10000)
        w_generic = compute_path_weight(nodes_generic, edges, degree, corpus_total_claims=10000)
        assert w_specific > w_generic

    def test_zero_weight_for_empty_path(self):
        assert compute_path_weight([], [], {}, 1000) == 0.0


class TestDegreeCap:
    def test_filters_paths_through_high_degree_intermediates(self):
        path_through_hub = Path(
            intermediates=[PathNode("hub", "H", "Place")],
            edges=[],
            weight=1.0,
        )
        path_through_specific = Path(
            intermediates=[PathNode("specific", "S", "Person")],
            edges=[],
            weight=1.0,
        )
        degree = {"hub": 5000, "specific": 12, "ep_a": 5, "ep_b": 5}
        # Cap at 95th percentile across nonzero values: with only 4 entities,
        # the cap is the highest value; using 1000 as an explicit cap here.
        kept = filter_by_degree_cap(
            [path_through_hub, path_through_specific],
            degree, cap=1000,
            endpoints=("ep_a", "ep_b"),
        )
        assert len(kept) == 1
        assert kept[0].intermediates[0].entity_id == "specific"

    def test_hub_intermediates_filtered_even_when_endpoint_is_hub(self):
        # Earlier design admitted hub intermediates when one endpoint was
        # itself a hub, on the theory that researchers querying about hub
        # entities couldn't be denied paths through other hubs. In practice
        # that admitted noise (Spokane County, the United States) for any
        # query about a corpus-namesake entity. The filter now applies
        # uniformly — endpoints are still allowed to be hubs (they're
        # excluded from the intermediate set by other clauses), but other
        # hub entities are never useful intermediates.
        path = Path(
            intermediates=[PathNode("other_hub", "OH", "Place")],
            edges=[],
            weight=1.0,
        )
        degree = {"other_hub": 5000, "ep_a": 9000, "ep_b": 12}
        kept = filter_by_degree_cap([path], degree, cap=1000, endpoints=("ep_a", "ep_b"))
        assert len(kept) == 0

    def test_threshold_excludes_zero_degrees(self):
        # Many entities with degree 0 should not pull down the threshold.
        degree = {f"e{i}": 0 for i in range(1000)}
        degree.update({"a": 10, "b": 100, "c": 1000})
        threshold = degree_cap_threshold(degree, percentile=0.95)
        # Threshold is computed only over nonzero values [10, 100, 1000].
        assert threshold in (100, 1000)

    def test_threshold_handles_empty_input(self):
        assert degree_cap_threshold({}) == 0
        assert degree_cap_threshold({"a": 0, "b": 0}) == 0


class TestExcludedHubIds:
    def test_returns_above_cap_excluding_endpoints(self):
        degree = {"hub1": 5000, "hub2": 4000, "ep_a": 9000, "specific": 12, "ep_b": 6}
        excluded = excluded_hub_ids(degree, cap=1000, endpoints=("ep_a", "ep_b"))
        # Both hubs included; endpoints (even hub-degree ep_a) excluded.
        assert set(excluded) == {"hub1", "hub2"}

    def test_empty_when_cap_zero_or_degree_empty(self):
        # Degree map empty (cache failed to load): refuse to filter
        # everything as a hub, which would block all paths.
        assert excluded_hub_ids({}, cap=100, endpoints=("a", "b")) == []
        assert excluded_hub_ids({"a": 1}, cap=0, endpoints=("a", "b")) == []

    def test_passes_excluded_to_cypher_params(self):
        # End-to-end check that the pre-filter list gets threaded into the
        # query params, so Cypher can prune at the database level. Build a
        # degree distribution where the 95th percentile is unambiguously
        # below the hub: 20 low-degree entities + one outlier.
        from gemynd.core.graph.cypher import PATHS_1_HOP_QUERY
        degree = {f"low_{i}": 5 for i in range(20)}
        degree["hub1"] = 5000
        degree["ep_a"] = 10
        degree["ep_b"] = 10
        stub = StubExecutor({CO_ASSERTING_CLAIMS_QUERY: [], PATHS_1_HOP_QUERY: []})
        find_paths(
            stub.run,
            from_id="ep_a", to_id="ep_b",
            institution_ids=["x"], permitted_levels=["public"],
            degree=degree, corpus_total_claims=10000, max_hops=1,
        )
        hop_calls = [c for c in stub.calls if c[0] is PATHS_1_HOP_QUERY]
        assert hop_calls, "1-hop query did not fire"
        assert "excluded_intermediate_ids" in hop_calls[0][1]
        assert "hub1" in hop_calls[0][1]["excluded_intermediate_ids"]
        # Endpoints stay out of the excluded list even if their degree is
        # high enough to cross the cap.
        assert "ep_a" not in hop_calls[0][1]["excluded_intermediate_ids"]
        assert "ep_b" not in hop_calls[0][1]["excluded_intermediate_ids"]


# ---------------------------------------------------------------------------
# Path-family collapsing
# ---------------------------------------------------------------------------


class TestFamilyCollapsing:
    def test_paths_differing_by_one_node_collapse(self):
        # Both paths share intermediate "shared"; differ on one node only.
        p1 = Path(
            intermediates=[PathNode("shared", "S", "Person"),
                           PathNode("alt_a", "A", "Place")],
            edges=[], weight=10.0,
        )
        p2 = Path(
            intermediates=[PathNode("shared", "S", "Person"),
                           PathNode("alt_b", "B", "Place")],
            edges=[], weight=8.0,
        )
        families = collapse_path_families([p1, p2], max_diff=1)
        assert len(families) == 1
        assert families[0] is p1
        assert families[0].alternates == [p2]

    def test_paths_differing_by_two_nodes_stay_separate(self):
        p1 = Path(intermediates=[PathNode("a", "A", "X"), PathNode("b", "B", "Y")],
                  edges=[], weight=10.0)
        p2 = Path(intermediates=[PathNode("c", "C", "X"), PathNode("d", "D", "Y")],
                  edges=[], weight=8.0)
        families = collapse_path_families([p1, p2], max_diff=1)
        assert len(families) == 2

    def test_highest_weight_path_becomes_primary(self):
        # Input is sorted descending by weight by the caller; collapse
        # respects that order, so the first match in each family wins.
        p_high = Path(intermediates=[PathNode("x", "X", "Y")], edges=[], weight=10.0)
        p_low = Path(intermediates=[PathNode("x", "X", "Y")], edges=[], weight=2.0)
        families = collapse_path_families([p_high, p_low])
        assert families[0] is p_high
        assert families[0].alternates == [p_low]


# ---------------------------------------------------------------------------
# Temporal filter propagation
# ---------------------------------------------------------------------------


class TestTemporalFilter:
    def test_year_min_max_propagate_to_co_assertion_query(self):
        stub = StubExecutor({CO_ASSERTING_CLAIMS_QUERY: [], PATHS_1_HOP_QUERY: [], PATHS_2_HOP_QUERY: []})
        find_paths(
            stub.run,
            from_id="a", to_id="b",
            institution_ids=["x"], permitted_levels=["public"],
            year_min=1900, year_max=1920,
        )
        co_call = next(c for c in stub.calls if c[0] is CO_ASSERTING_CLAIMS_QUERY)
        assert co_call[1]["year_min"] == 1900
        assert co_call[1]["year_max"] == 1920

    def test_year_filter_propagates_to_hop_queries(self):
        stub = StubExecutor({CO_ASSERTING_CLAIMS_QUERY: [], PATHS_1_HOP_QUERY: [], PATHS_2_HOP_QUERY: []})
        find_paths(
            stub.run,
            from_id="a", to_id="b",
            institution_ids=["x"], permitted_levels=["public"],
            year_min=1909, year_max=1909,
        )
        for c in stub.calls:
            if c[0] in (CO_ASSERTING_CLAIMS_QUERY, PATHS_1_HOP_QUERY, PATHS_2_HOP_QUERY):
                assert c[1]["year_min"] == 1909
                assert c[1]["year_max"] == 1909


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_co_assertion_only_result(self):
        from gemynd.retrieval.web.path_finder import PathResult
        result = PathResult(co_assertions=[CoAssertion(
            claim_id="c1", sentence="s", claim_type="t", confidence=0.9,
            from_relation="X", to_relation="Y",
            doc_id="d1", doc_title="t", doc_date_start="1909",
            doc_report_year=1909, institution_id="x",
        )])
        d = path_result_to_dict(result)
        assert d["disconnected"] is False
        assert d["fallback_neighbors"] is None
        assert len(d["co_assertions"]) == 1
        assert d["co_assertions"][0]["claim_id"] == "c1"
        assert d["paths"] == []

    def test_disconnected_result_includes_neighbors(self):
        from gemynd.retrieval.web.path_finder import FallbackNeighbor, PathResult
        result = PathResult(
            disconnected=True,
            fallback_from=[FallbackNeighbor("n1", "N1", "Person", 5)],
            fallback_to=[FallbackNeighbor("n2", "N2", "Place", 3)],
        )
        d = path_result_to_dict(result)
        assert d["disconnected"] is True
        assert d["fallback_neighbors"]["from"][0]["entity_id"] == "n1"
        assert d["fallback_neighbors"]["to"][0]["entity_id"] == "n2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1_hop_row(i_id: str, i_name: str, i_type: str) -> dict[str, Any]:
    return {
        "i1_id": i_id, "i1_name": i_name, "i1_type": i_type,
        "e1_claim_id": "c1", "e1_sentence": "s1", "e1_conf": 0.9,
        "e1_from_rel": "X", "e1_to_rel": "Y",
        "e1_doc_id": "d1", "e1_doc_title": "t1", "e1_doc_date": "1909",
        "e2_claim_id": "c2", "e2_sentence": "s2", "e2_conf": 0.85,
        "e2_from_rel": "X", "e2_to_rel": "Y",
        "e2_doc_id": "d2", "e2_doc_title": "t2", "e2_doc_date": "1910",
    }


def _make_2_hop_row(
    i1_id: str, i1_name: str, i1_type: str,
    i2_id: str, i2_name: str, i2_type: str,
) -> dict[str, Any]:
    return {
        "i1_id": i1_id, "i1_name": i1_name, "i1_type": i1_type,
        "i2_id": i2_id, "i2_name": i2_name, "i2_type": i2_type,
        "e1_claim_id": "c1", "e1_sentence": "s1", "e1_conf": 0.9,
        "e1_from_rel": "X", "e1_to_rel": "Y",
        "e1_doc_id": "d1", "e1_doc_title": "t1", "e1_doc_date": "1909",
        "e2_claim_id": "c2", "e2_sentence": "s2", "e2_conf": 0.85,
        "e2_from_rel": "X", "e2_to_rel": "Y",
        "e2_doc_id": "d2", "e2_doc_title": "t2", "e2_doc_date": "1910",
        "e3_claim_id": "c3", "e3_sentence": "s3", "e3_conf": 0.8,
        "e3_from_rel": "X", "e3_to_rel": "Y",
        "e3_doc_id": "d3", "e3_doc_title": "t3", "e3_doc_date": "1911",
    }
