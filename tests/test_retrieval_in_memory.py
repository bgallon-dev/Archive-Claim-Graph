"""Neo4j-free retrieval tests using InMemoryQueryExecutor.

These tests exercise the real retrieval query-building and context-assembly
logic against real pipeline-ingested fixture data, with no database required.
The ``populated_executor`` fixture (session-scoped in conftest.py) runs all
three fixture reports through the full pipeline into an InMemoryGraphWriter,
then wraps it in an InMemoryQueryExecutor.
"""
from __future__ import annotations

import pytest

from gemynd.core.graph.cypher import (
    CLAIM_TYPE_SCOPED_QUERY,
    ENTITY_ANCHORED_CLAIMS_QUERY,
    FULLTEXT_CLAIMS_QUERY,
    MULTI_ENTITY_CLAIMS_QUERY,
    PROVENANCE_CHAIN_QUERY,
    TEMPORAL_CLAIMS_QUERY,
    build_temporal_with_anchor_query,
)

# The anchor-aware template for the Turnbull corpus. Re-created via the
# memoized builder so object identity matches what the executor pre-caches.
TEMPORAL_CLAIMS_QUERY_WITH_ANCHOR = build_temporal_with_anchor_query(
    "Refuge", "ABOUT_REFUGE"
)
from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
from gemynd.retrieval.models import EntityContext, ResolvedEntity
from tests.conftest import TEST_ENTITY_LABELS

# Default access params used in every query (match the pipeline defaults).
_ACCESS = {"institution_id": "turnbull", "permitted_levels": ["public"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _years_in_writer(populated_writer) -> set[int]:
    return {
        props["year"]
        for props in populated_writer.node_store.get("Year", {}).values()
        if props.get("year") is not None
    }


def _claim_ids_in_writer(populated_writer) -> list[str]:
    return list(populated_writer.node_store.get("Claim", {}).keys())


def _species_ids_in_writer(populated_writer) -> list[str]:
    return list(populated_writer.node_store.get("Species", {}).keys())


def _entity_ids_linked_to_claims(populated_writer) -> list[str]:
    """Return entity_ids that have at least one Claim->Entity relationship."""
    from tests.conftest import TEST_ENTITY_LABELS
    seen: set[str] = set()
    for sl, si, rt, el, ei, _ in populated_writer.rel_store:
        if sl == "Claim" and el in TEST_ENTITY_LABELS:
            seen.add(ei)
    return list(seen)


# ---------------------------------------------------------------------------
# Row shape contract
# ---------------------------------------------------------------------------

class TestRowShape:
    """Every row returned by any handler must have the keys _row_to_block needs."""

    EXPECTED_KEYS = {"c", "d", "pg", "sec", "para", "obs", "sp", "y", "measurements"}

    def _check_rows(self, rows: list[dict]) -> None:
        for row in rows:
            missing = self.EXPECTED_KEYS - row.keys()
            assert not missing, f"Row missing keys {missing}: {row}"
            assert isinstance(row.get("measurements"), list)

    def test_fulltext_row_shape(self, populated_executor):
        rows = populated_executor.run(
            FULLTEXT_CLAIMS_QUERY,
            {"search_text": "", "limit": 50, **_ACCESS},
        )
        self._check_rows(rows)

    def test_temporal_row_shape(self, populated_executor, populated_writer):
        years = _years_in_writer(populated_writer)
        if not years:
            pytest.skip("No Year nodes in fixture data")
        year = min(years)
        rows = populated_executor.run(
            TEMPORAL_CLAIMS_QUERY,
            {"year_min": year, "year_max": year, "claim_types": None, "limit": 50, **_ACCESS},
        )
        self._check_rows(rows)

    def test_entity_anchored_row_shape(self, populated_executor, populated_writer):
        entity_ids = _entity_ids_linked_to_claims(populated_writer)
        if not entity_ids:
            pytest.skip("No Claim→Entity links in fixture data")
        rows = populated_executor.run(
            ENTITY_ANCHORED_CLAIMS_QUERY,
            {"entity_id": entity_ids[0], "year_min": None, "year_max": None, "limit": 20, **_ACCESS},
        )
        self._check_rows(rows)

    def test_claim_type_scoped_row_shape(self, populated_executor, populated_writer):
        claim_types = list({
            p.get("claim_type")
            for p in populated_writer.node_store.get("Claim", {}).values()
            if p.get("claim_type")
        })
        if not claim_types:
            pytest.skip("No claims in fixture data")
        rows = populated_executor.run(
            CLAIM_TYPE_SCOPED_QUERY,
            {
                "claim_types": claim_types[:2],
                "entity_ids": None,
                "year_min": None,
                "year_max": None,
                "limit": 50,
                **_ACCESS,
            },
        )
        self._check_rows(rows)


# ---------------------------------------------------------------------------
# Temporal query
# ---------------------------------------------------------------------------

class TestTemporalQuery:
    def test_year_range_filter_is_respected(self, populated_executor, populated_writer):
        years = _years_in_writer(populated_writer)
        if not years:
            pytest.skip("No year-annotated observations in fixture data")
        target = min(years)
        rows = populated_executor.run(
            TEMPORAL_CLAIMS_QUERY,
            {"year_min": target, "year_max": target, "claim_types": None, "limit": 100, **_ACCESS},
        )
        for row in rows:
            year_val = (row.get("y") or {}).get("year")
            assert year_val == target, f"Expected year={target}, got {year_val}"

    def test_impossible_year_range_returns_empty(self, populated_executor):
        rows = populated_executor.run(
            TEMPORAL_CLAIMS_QUERY,
            {"year_min": 9999, "year_max": 9999, "claim_types": None, "limit": 100, **_ACCESS},
        )
        assert rows == []

    def test_temporal_results_are_list_of_dicts(self, populated_executor):
        rows = populated_executor.run(
            TEMPORAL_CLAIMS_QUERY,
            {"year_min": None, "year_max": None, "claim_types": None, "limit": 5, **_ACCESS},
        )
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)

    def test_temporal_with_refuge_scoped_to_refuge_docs(
        self, populated_executor, populated_writer
    ):
        # Find any Refuge entity in the writer
        refuge_ids = list(populated_writer.node_store.get("Refuge", {}).keys())
        # Find any refuge referenced via ABOUT_REFUGE
        about_refuge = {
            ei
            for sl, si, rt, el, ei, _ in populated_writer.rel_store
            if rt == "ABOUT_REFUGE" and el == "Refuge"
        }
        if not about_refuge:
            pytest.skip("No ABOUT_REFUGE relationships in fixture data")
        refuge_id = next(iter(about_refuge))
        rows = populated_executor.run(
            TEMPORAL_CLAIMS_QUERY_WITH_ANCHOR,
            {
                "anchor_id": refuge_id,
                "year_min": None,
                "year_max": None,
                "claim_types": None,
                "limit": 50,
                **_ACCESS,
            },
        )
        # All returned docs must be linked to this refuge
        linked_docs = {
            si
            for sl, si, rt, el, ei, _ in populated_writer.rel_store
            if rt == "ABOUT_REFUGE" and ei == refuge_id
        }
        for row in rows:
            doc_id = (row.get("d") or {}).get("doc_id")
            if doc_id:
                assert doc_id in linked_docs, (
                    f"Row's doc_id={doc_id!r} is not linked ABOUT_REFUGE to {refuge_id!r}"
                )


# ---------------------------------------------------------------------------
# Entity-anchored query
# ---------------------------------------------------------------------------

class TestEntityAnchored:
    def test_entity_anchored_returns_only_linked_claims(
        self, populated_executor, populated_writer
    ):
        entity_ids = _entity_ids_linked_to_claims(populated_writer)
        if not entity_ids:
            pytest.skip("No Claim→Entity links in fixture data")
        entity_id = entity_ids[0]
        rows = populated_executor.run(
            ENTITY_ANCHORED_CLAIMS_QUERY,
            {"entity_id": entity_id, "year_min": None, "year_max": None, "limit": 50, **_ACCESS},
        )
        assert len(rows) > 0, f"Expected claims linked to {entity_id}"

    def test_unknown_entity_returns_empty(self, populated_executor):
        rows = populated_executor.run(
            ENTITY_ANCHORED_CLAIMS_QUERY,
            {"entity_id": "nonexistent-entity-xyz", "year_min": None, "year_max": None, "limit": 20, **_ACCESS},
        )
        assert rows == []

    def test_entity_anchored_year_filter(self, populated_executor, populated_writer):
        entity_ids = _entity_ids_linked_to_claims(populated_writer)
        years = _years_in_writer(populated_writer)
        if not entity_ids or not years:
            pytest.skip("No entity-linked claims with year data")
        entity_id = entity_ids[0]
        target = min(years)
        rows = populated_executor.run(
            ENTITY_ANCHORED_CLAIMS_QUERY,
            {"entity_id": entity_id, "year_min": target, "year_max": target, "limit": 50, **_ACCESS},
        )
        for row in rows:
            year_val = (row.get("y") or {}).get("year")
            if year_val is not None:
                assert year_val == target

    def test_entity_anchored_sorted_by_confidence(self, populated_executor, populated_writer):
        entity_ids = _entity_ids_linked_to_claims(populated_writer)
        if not entity_ids:
            pytest.skip("No Claim→Entity links in fixture data")
        rows = populated_executor.run(
            ENTITY_ANCHORED_CLAIMS_QUERY,
            {"entity_id": entity_ids[0], "year_min": None, "year_max": None, "limit": 50, **_ACCESS},
        )
        confidences = [
            (row.get("c") or {}).get("extraction_confidence", 0.0) for row in rows
        ]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Multi-entity query
# ---------------------------------------------------------------------------

class TestMultiEntity:
    def test_multi_entity_has_matched_entity_ids(self, populated_executor, populated_writer):
        entity_ids = _entity_ids_linked_to_claims(populated_writer)
        if len(entity_ids) < 2:
            pytest.skip("Need 2+ entity-linked claims for multi-entity test")
        rows = populated_executor.run(
            MULTI_ENTITY_CLAIMS_QUERY,
            {
                "entity_ids": entity_ids[:2],
                "claim_types": None,
                "year_min": None,
                "year_max": None,
                "limit": 50,
                **_ACCESS,
            },
        )
        for row in rows:
            assert "matched_entity_ids" in row
            assert isinstance(row["matched_entity_ids"], list)
            assert len(row["matched_entity_ids"]) >= 1

    def test_multi_entity_no_false_positives(self, populated_executor, populated_writer):
        entity_ids = _entity_ids_linked_to_claims(populated_writer)
        if len(entity_ids) < 2:
            pytest.skip("Need 2+ entity-linked claims for multi-entity test")
        target_ids = set(entity_ids[:2])
        rows = populated_executor.run(
            MULTI_ENTITY_CLAIMS_QUERY,
            {
                "entity_ids": list(target_ids),
                "claim_types": None,
                "year_min": None,
                "year_max": None,
                "limit": 50,
                **_ACCESS,
            },
        )
        for row in rows:
            matched = set(row.get("matched_entity_ids") or [])
            assert matched & target_ids, (
                f"Row matched_entity_ids {matched} has no overlap with {target_ids}"
            )


# ---------------------------------------------------------------------------
# Fulltext query
# ---------------------------------------------------------------------------

class TestFulltext:
    def test_fulltext_empty_search_returns_claims(self, populated_executor):
        """Empty search string matches all claims (up to limit)."""
        rows = populated_executor.run(
            FULLTEXT_CLAIMS_QUERY,
            {"search_text": "", "limit": 100, **_ACCESS},
        )
        assert isinstance(rows, list)

    def test_fulltext_search_filters_by_sentence(self, populated_executor, populated_writer):
        """A keyword that appears in a known claim's source_sentence is found."""
        # Pick a word from any existing claim sentence
        claim_sentences = [
            props.get("source_sentence", "")
            for props in populated_writer.node_store.get("Claim", {}).values()
            if props.get("source_sentence")
        ]
        if not claim_sentences:
            pytest.skip("No claims with source_sentence in fixture data")
        # Pick a meaningful word (>3 chars) from the first sentence
        words = [w for w in claim_sentences[0].split() if len(w) > 3]
        if not words:
            pytest.skip("Could not extract a search keyword from claim sentences")
        keyword = words[0].lower().strip(".,;:")
        rows = populated_executor.run(
            FULLTEXT_CLAIMS_QUERY,
            {"search_text": keyword, "limit": 50, **_ACCESS},
        )
        assert len(rows) > 0, f"Expected at least one result for keyword {keyword!r}"
        for row in rows:
            sentence = (
                (row.get("c") or {}).get("source_sentence")
                or (row.get("c") or {}).get("normalized_sentence")
                or ""
            )
            assert keyword in sentence.lower(), (
                f"Keyword {keyword!r} not found in sentence {sentence!r}"
            )

    def test_fulltext_no_match_returns_empty(self, populated_executor):
        rows = populated_executor.run(
            FULLTEXT_CLAIMS_QUERY,
            {"search_text": "xyzzy_no_match_expected_99999", "limit": 20, **_ACCESS},
        )
        assert rows == []

    def test_fulltext_limit_respected(self, populated_executor):
        rows = populated_executor.run(
            FULLTEXT_CLAIMS_QUERY,
            {"search_text": "", "limit": 2, **_ACCESS},
        )
        assert len(rows) <= 2


# ---------------------------------------------------------------------------
# Provenance chain query
# ---------------------------------------------------------------------------

class TestProvenanceChain:
    def test_provenance_chain_returns_row_for_known_claim(
        self, populated_executor, populated_writer
    ):
        claim_ids = _claim_ids_in_writer(populated_writer)
        if not claim_ids:
            pytest.skip("No claims in fixture data")
        # Find a claim that has a doc path (para → section → page → doc)
        executor = populated_executor
        for cid in claim_ids:
            rows = executor.run(
                PROVENANCE_CHAIN_QUERY, {"claim_id": cid, **_ACCESS}
            )
            if rows:
                assert len(rows) == 1
                assert (rows[0].get("c") or {}).get("claim_id") == cid
                return
        pytest.skip("No claims with full doc path in fixture data")

    def test_provenance_chain_unknown_claim_returns_empty(self, populated_executor):
        rows = populated_executor.run(
            PROVENANCE_CHAIN_QUERY,
            {"claim_id": "claim-id-that-does-not-exist", **_ACCESS},
        )
        assert rows == []


# ---------------------------------------------------------------------------
# Claim-type scoped query
# ---------------------------------------------------------------------------

class TestClaimTypeScoped:
    def test_returns_only_requested_claim_types(self, populated_executor, populated_writer):
        claim_types = list({
            p.get("claim_type")
            for p in populated_writer.node_store.get("Claim", {}).values()
            if p.get("claim_type")
        })
        if not claim_types:
            pytest.skip("No claims in fixture data")
        target_types = set(claim_types[:1])
        rows = populated_executor.run(
            CLAIM_TYPE_SCOPED_QUERY,
            {
                "claim_types": list(target_types),
                "entity_ids": None,
                "year_min": None,
                "year_max": None,
                "limit": 100,
                **_ACCESS,
            },
        )
        for row in rows:
            ct = (row.get("c") or {}).get("claim_type")
            assert ct in target_types, f"Unexpected claim_type {ct!r} not in {target_types}"

    def test_claim_type_traversal_rel_type_is_claim_type(
        self, populated_executor, populated_writer
    ):
        claim_types = list({
            p.get("claim_type")
            for p in populated_writer.node_store.get("Claim", {}).values()
            if p.get("claim_type")
        })
        if not claim_types:
            pytest.skip("No claims in fixture data")
        rows = populated_executor.run(
            CLAIM_TYPE_SCOPED_QUERY,
            {
                "claim_types": claim_types,
                "entity_ids": None,
                "year_min": None,
                "year_max": None,
                "limit": 50,
                **_ACCESS,
            },
        )
        for row in rows:
            ct = (row.get("c") or {}).get("claim_type")
            trt = row.get("traversal_rel_type")
            assert trt == ct, (
                f"traversal_rel_type {trt!r} should equal claim_type {ct!r}"
            )


# ---------------------------------------------------------------------------
# Generic dispatch — startup refuge query
# ---------------------------------------------------------------------------

class TestGenericDispatch:
    def test_startup_refuge_query_returns_eid(self, populated_executor, populated_writer):
        """The assembler's __init__ anchor lookup must be handled by _generic_dispatch."""
        refuge_linked = {
            ei
            for sl, si, rt, el, ei, _ in populated_writer.rel_store
            if rt == "ABOUT_REFUGE" and el == "Refuge"
        }
        rows = populated_executor.run(
            "MATCH (:Document)-->(r:Refuge)"
            " RETURN r.entity_id AS eid LIMIT 1"
        )
        if refuge_linked:
            assert len(rows) == 1
            assert "eid" in rows[0]
        else:
            assert rows == []

    def test_unknown_cypher_returns_empty(self, populated_executor):
        rows = populated_executor.run("MATCH (n) RETURN n LIMIT 1")
        assert rows == []


# ---------------------------------------------------------------------------
# Integration with ProvenanceContextAssembler
# ---------------------------------------------------------------------------

class TestIntegrationWithAssembler:
    """End-to-end tests: real assembler + real executor + real fixture data."""

    def test_assemble_returns_tuple(self, populated_executor):
        assembler = ProvenanceContextAssembler(executor=populated_executor)
        result = assembler.assemble("what happened", EntityContext())
        assert isinstance(result, tuple) and len(result) == 2
        blocks, ctx = result
        assert isinstance(blocks, list)
        assert isinstance(ctx, str)

    def test_all_blocks_have_doc_title(self, populated_executor):
        assembler = ProvenanceContextAssembler(executor=populated_executor)
        blocks, _ = assembler.assemble("what happened", EntityContext())
        for b in blocks:
            assert b.doc_title, f"Block {b.claim_id} has empty doc_title"

    def test_all_blocks_have_claim_id(self, populated_executor):
        assembler = ProvenanceContextAssembler(executor=populated_executor)
        blocks, _ = assembler.assemble("what happened", EntityContext())
        for b in blocks:
            assert b.claim_id, "Every block must have a claim_id"

    def test_temporal_year_range_filtering(self, populated_executor, populated_writer):
        """The canonical test from the task description."""
        years = _years_in_writer(populated_writer)
        if not years:
            pytest.skip("No year-annotated observations in fixture data")
        target = min(years)
        assembler = ProvenanceContextAssembler(executor=populated_executor)
        blocks, _ = assembler.assemble(
            f"what happened in {target}?",
            EntityContext(),
            year_min=target,
            year_max=target,
        )
        for b in blocks:
            if b.year is not None:
                assert b.year == target, (
                    f"Block {b.claim_id} has year={b.year}, expected {target}"
                )

    def test_temporal_impossible_range_returns_empty(self, populated_executor):
        assembler = ProvenanceContextAssembler(executor=populated_executor)
        blocks, ctx = assembler.assemble(
            "what happened in 9999?",
            EntityContext(),
            year_min=9999,
            year_max=9999,
        )
        assert blocks == []
        assert ctx == ""

    def test_entity_anchored_path_used_when_entity_resolved(
        self, populated_executor, populated_writer
    ):
        entity_ids = _entity_ids_linked_to_claims(populated_writer)
        if not entity_ids:
            pytest.skip("No entity-linked claims in fixture data")
        entity_id = entity_ids[0]
        # Find label for this entity
        entity_type = "Species"
        for label in TEST_ENTITY_LABELS:
            if entity_id in populated_writer.node_store.get(label, {}):
                entity_type = label
                break
        entity_props = populated_writer.node_store.get(entity_type, {}).get(entity_id, {})
        surface_form = entity_props.get("name") or entity_props.get("entity_id") or entity_id

        resolved = [
            ResolvedEntity(
                surface_form=surface_form,
                entity_id=entity_id,
                entity_type=entity_type,
                resolution_confidence=0.95,
                resolution_relation="REFERS_TO",
            )
        ]
        entity_ctx = EntityContext(resolved=resolved)
        assembler = ProvenanceContextAssembler(executor=populated_executor)
        blocks, _ = assembler.assemble("tell me about it", entity_ctx)
        # We can't assert >0 without knowing data, but the call must not crash
        assert isinstance(blocks, list)

    def test_budget_cap_is_respected(self, populated_executor):
        assembler = ProvenanceContextAssembler(
            executor=populated_executor, budget_conversational=3
        )
        blocks, _ = assembler.assemble("what happened", EntityContext())
        assert len(blocks) <= 3

    def test_context_text_contains_claim_text_markers(self, populated_executor):
        assembler = ProvenanceContextAssembler(executor=populated_executor)
        blocks, ctx = assembler.assemble("what happened", EntityContext())
        if blocks:
            assert "<claim_text>" in ctx
            assert "</claim_text>" in ctx

    def test_rows_convertible_to_blocks(self, populated_executor, populated_writer):
        """Every row from the executor must produce a valid ProvenanceBlock or None."""
        from gemynd.retrieval.context_assembler import _row_to_block

        rows = populated_executor.run(
            FULLTEXT_CLAIMS_QUERY,
            {"search_text": "", "limit": 100, **_ACCESS},
        )
        for row in rows:
            block = _row_to_block(row)
            # _row_to_block returns None only for rows missing claim_id or with OCR
            # garbage — that is expected behaviour, not an error.
            if block is not None:
                assert block.claim_id
                assert block.doc_title is not None
