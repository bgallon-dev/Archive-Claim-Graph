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
_ACCESS = {"institution_ids": ["turnbull"], "permitted_levels": ["public"]}


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

    def test_temporal_impossible_range_triggers_relaxation(self, populated_executor):
        """An impossible year range exhausts the temporal rung; the router's
        relaxation chain then drops the year filter (and the claim_type
        filter) so downstream attempts can recover claims. The trace should
        record that we did NOT exit on the primary temporal attempt.
        """
        assembler = ProvenanceContextAssembler(executor=populated_executor)
        blocks, ctx = assembler.assemble(
            "what happened in 9999?",
            EntityContext(),
            year_min=9999,
            year_max=9999,
        )
        trace = assembler._last_plan_trace
        assert trace is not None
        # At least the primary temporal attempt must have been recorded.
        assert any("temporal" in a.label for a in trace.attempts)
        # The primary attempt with the impossible year range must return 0 rows.
        primary = trace.attempts[0]
        assert primary.row_count == 0
        # If any rows came back, they came from a relaxation rung whose label
        # differs from the primary.
        if blocks:
            assert trace.selected_label != primary.label

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


# ---------------------------------------------------------------------------
# Per-corpus stratification (cross-corpus equity in the candidate pool)
# ---------------------------------------------------------------------------

def _build_two_corpus_writer(
    *,
    n_per_corpus: int = 30,
    high_conf: float = 0.90,
    low_conf: float = 0.70,
):
    """Construct a minimal InMemoryGraphWriter with two corpora sharing one anchor entity.

    Built by directly populating node_store / rel_store rather than running the full
    ingestion pipeline, so the test isolates the per-corpus retrieval behaviour from
    extraction/derivation logic.
    """
    from gemynd.ingest.graph import InMemoryGraphWriter

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)

    writer.node_store["Place"]["place_anchor"] = {
        "entity_id": "place_anchor",
        "name": "Anchor",
    }

    def _add_corpus(institution_id: str, n_claims: int, base_conf: float) -> None:
        doc_id = f"doc_{institution_id}"
        run_id = f"run_{institution_id}"
        page_id = f"page_{institution_id}"
        sec_id = f"sec_{institution_id}"
        para_id = f"para_{institution_id}"

        writer.node_store["Document"][doc_id] = {
            "doc_id": doc_id,
            "title": f"Doc {institution_id}",
            "institution_id": institution_id,
            "access_level": "public",
            "deleted_at": None,
            "quarantine_status": None,
        }
        writer.node_store["ExtractionRun"][run_id] = {
            "run_id": run_id,
            "run_timestamp": "2024-01-01T00:00:00",
        }
        writer.rel_store[("Document", doc_id, "PROCESSED_BY", "ExtractionRun", run_id, None)] = {
            "run_id": run_id,
            "run_timestamp": "2024-01-01T00:00:00",
        }
        writer.node_store["Page"][page_id] = {"page_id": page_id, "page_number": 1}
        writer.node_store["Section"][sec_id] = {"section_id": sec_id}
        writer.node_store["Paragraph"][para_id] = {"paragraph_id": para_id}
        writer.rel_store[("Document", doc_id, "HAS_PAGE", "Page", page_id, None)] = {}
        writer.rel_store[("Page", page_id, "HAS_SECTION", "Section", sec_id, None)] = {}
        writer.rel_store[("Section", sec_id, "HAS_PARAGRAPH", "Paragraph", para_id, None)] = {}

        for i in range(n_claims):
            cid = f"claim_{institution_id}_{i:03d}"
            writer.node_store["Claim"][cid] = {
                "claim_id": cid,
                "claim_type": "test_type",
                "extraction_confidence": base_conf - i * 0.001,
                "source_sentence": f"test sentence {i}",
                "run_id": run_id,
                "quarantine_status": None,
                "certainty": "observed",
            }
            writer.rel_store[("Paragraph", para_id, "HAS_CLAIM", "Claim", cid, None)] = {
                "run_id": run_id,
            }
            writer.rel_store[("Claim", cid, "ABOUT", "Place", "place_anchor", None)] = {}

    _add_corpus("corp_high", n_per_corpus, high_conf)
    _add_corpus("corp_low", n_per_corpus, low_conf)
    return writer


class TestEntityAnchoredPerCorpusStratification:
    """The Spokane/Turnbull regression: validates that the per-corpus LIMIT
    rewrite of ENTITY_ANCHORED_CLAIMS_QUERY actually returns both corpora
    when one has uniformly higher confidence than the other.
    """

    def test_both_corpora_represented_when_confidence_distributions_differ(self):
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor

        writer = _build_two_corpus_writer(n_per_corpus=30, high_conf=0.90, low_conf=0.70)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)

        rows = executor.run(
            ENTITY_ANCHORED_CLAIMS_QUERY,
            {
                "entity_id": "place_anchor",
                "year_min": None,
                "year_max": None,
                "limit": 20,
                "institution_ids": ["corp_high", "corp_low"],
                "permitted_levels": ["public"],
            },
        )
        high_count = sum(1 for r in rows if r["d"]["institution_id"] == "corp_high")
        low_count = sum(1 for r in rows if r["d"]["institution_id"] == "corp_low")

        # Pre-fix behaviour: top 20 by global confidence → high_count=20, low_count=0
        # Post-fix behaviour: per-corpus LIMIT → high_count=20, low_count=20
        assert high_count == 20, f"corp_high underrepresented: {high_count}"
        assert low_count == 20, f"corp_low excluded by global confidence sort: {low_count}"
        assert len(rows) == 40

    def test_single_corpus_preserves_original_limit(self):
        """Single-corpus queries must behave identically to the pre-fix LIMIT $limit."""
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor

        writer = _build_two_corpus_writer(n_per_corpus=30, high_conf=0.90, low_conf=0.70)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)

        rows = executor.run(
            ENTITY_ANCHORED_CLAIMS_QUERY,
            {
                "entity_id": "place_anchor",
                "year_min": None,
                "year_max": None,
                "limit": 20,
                "institution_ids": ["corp_high"],  # single corpus only
                "permitted_levels": ["public"],
            },
        )
        assert len(rows) == 20
        assert all(r["d"]["institution_id"] == "corp_high" for r in rows)

    def test_corpus_with_no_matches_does_not_break_query(self):
        """Querying with multiple institution_ids but only one having data should still return results."""
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor

        writer = _build_two_corpus_writer(n_per_corpus=30, high_conf=0.90, low_conf=0.70)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)

        rows = executor.run(
            ENTITY_ANCHORED_CLAIMS_QUERY,
            {
                "entity_id": "place_anchor",
                "year_min": None,
                "year_max": None,
                "limit": 20,
                "institution_ids": ["corp_high", "ghost_corp"],  # ghost has no docs
                "permitted_levels": ["public"],
            },
        )
        assert len(rows) == 20
        assert all(r["d"]["institution_id"] == "corp_high" for r in rows)


# ---------------------------------------------------------------------------
# Tier 1 — count_claims_for_entity end-to-end against in-memory graph
# ---------------------------------------------------------------------------

class TestTier1CountClaimsForEntity:
    """Validates the full Tier 1 path: assembler.tier_1_lookup() → InMemory
    executor → COUNT_CLAIMS_FOR_ENTITY_QUERY handler → integer answer."""

    def test_count_aggregates_across_corpora(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = _build_two_corpus_writer(n_per_corpus=30, high_conf=0.85, low_conf=0.75)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(
            executor=executor,
            institution_ids=["corp_high", "corp_low"],
        )
        intent = Tier1Intent(
            operation="count_claims_for_entity",
            entity_id="place_anchor",
            entity_name="Anchor",
            entity_type="Place",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high", "corp_low"],
            permitted_levels=["public"],
        )
        # 30 + 30 active claims linked to the shared anchor entity.
        assert result.count == 60
        assert result.entity_name == "Anchor"
        assert result.operation == "count_claims_for_entity"
        assert "60 claims" in result.answer_text

    def test_count_respects_institution_scope(self):
        """Querying with only one institution_id must exclude the other corpus."""
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = _build_two_corpus_writer(n_per_corpus=30, high_conf=0.85, low_conf=0.75)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(
            executor=executor,
            institution_ids=["corp_high"],
        )
        intent = Tier1Intent(
            operation="count_claims_for_entity",
            entity_id="place_anchor",
            entity_name="Anchor",
            entity_type="Place",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high"],
            permitted_levels=["public"],
        )
        assert result.count == 30  # only the high-corpus rows

    def test_count_returns_zero_for_unknown_entity(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = _build_two_corpus_writer(n_per_corpus=10, high_conf=0.80, low_conf=0.70)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(
            executor=executor,
            institution_ids=["corp_high", "corp_low"],
        )
        intent = Tier1Intent(
            operation="count_claims_for_entity",
            entity_id="ghost_entity",
            entity_name="Ghost",
            entity_type="Place",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high", "corp_low"],
            permitted_levels=["public"],
        )
        assert result.count == 0
        assert "0 claims" in result.answer_text

    def test_singular_in_answer(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = _build_two_corpus_writer(n_per_corpus=1, high_conf=0.80, low_conf=0.70)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)
        intent = Tier1Intent(
            operation="count_claims_for_entity",
            entity_id="place_anchor",
            entity_name="Anchor",
            entity_type="Place",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high"],  # only one corpus → 1 claim
            permitted_levels=["public"],
        )
        assert result.count == 1
        # No trailing "s" when count == 1
        assert " 1 claim " in result.answer_text or result.answer_text.endswith(" 1 claim.")

    def test_unknown_operation_raises(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = _build_two_corpus_writer(n_per_corpus=1, high_conf=0.80, low_conf=0.70)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)
        # Forge an unknown operation by direct construction (bypasses Literal check).
        intent = Tier1Intent.__new__(Tier1Intent)
        intent.operation = "frequency_ranking"  # type: ignore[assignment]
        intent.entity_id = "x"
        intent.entity_name = "x"
        intent.entity_type = "Place"
        intent.target_year = None
        with pytest.raises(ValueError, match="Unknown Tier 1 operation"):
            assembler.tier_1_lookup(
                intent, institution_ids=["corp_high"], permitted_levels=["public"]
            )


class TestTier1ListDocumentsForEntity:
    """Validates the LIST_DOCUMENTS_FOR_ENTITY_QUERY end-to-end."""

    def test_returns_documents_with_claim_counts(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = _build_two_corpus_writer(n_per_corpus=30, high_conf=0.85, low_conf=0.75)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="list_documents_for_entity",
            entity_id="place_anchor",
            entity_name="Anchor",
            entity_type="Place",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high", "corp_low"],
            permitted_levels=["public"],
        )
        # Two documents in the test fixture, each contributing 30 claims.
        assert result.count == 2
        assert len(result.items) == 2
        assert all("claim_count" in r for r in result.items)
        assert all("title" in r for r in result.items)

    def test_returns_empty_for_unknown_entity(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = _build_two_corpus_writer(n_per_corpus=10, high_conf=0.80, low_conf=0.70)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="list_documents_for_entity",
            entity_id="ghost_entity",
            entity_name="Ghost",
            entity_type="Place",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high", "corp_low"],
            permitted_levels=["public"],
        )
        assert result.count == 0
        assert result.items == []
        assert "No documents found" in result.answer_text


class TestTier1ListDocumentsForYear:
    """Validates the LIST_DOCUMENTS_FOR_YEAR_QUERY end-to-end.  Uses a custom
    writer fixture with explicit report_year values per document.
    """

    def _build_year_writer(self):
        from gemynd.ingest.graph import InMemoryGraphWriter
        writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
        for doc_id, year, title in [
            ("doc_1909_a", 1909, "January Report"),
            ("doc_1909_b", 1909, "December Editorial"),
            ("doc_1910_a", 1910, "Annual Summary"),
        ]:
            writer.node_store["Document"][doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "report_year": year,
                "date_start": f"{year}-01-15" if "January" in title else f"{year}-12-15",
                "institution_id": "corp_high",
                "access_level": "public",
                "deleted_at": None,
                "quarantine_status": None,
                "doc_type": "newspaper",
            }
        return writer

    def test_returns_only_target_year(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = self._build_year_writer()
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="list_documents_for_year",
            target_year=1909,
            entity_name="1909",
            entity_type="Year",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high"],
            permitted_levels=["public"],
        )
        assert result.count == 2
        assert {r["doc_id"] for r in result.items} == {"doc_1909_a", "doc_1909_b"}
        assert result.target_year == 1909

    def test_returns_empty_for_year_with_no_documents(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = self._build_year_writer()
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="list_documents_for_year",
            target_year=1850,
            entity_name="1850",
            entity_type="Year",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high"],
            permitted_levels=["public"],
        )
        assert result.count == 0
        assert result.items == []
        assert "No documents" in result.answer_text


class TestTier1YearRangeForEntity:
    """Validates the YEAR_RANGE_FOR_ENTITY_QUERY end-to-end.  Uses the standard
    two-corpus writer where documents have no report_year — so we exercise
    the no-data fallback path here, then a custom fixture for the populated
    case.
    """

    def test_returns_zero_when_no_dated_claims(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = _build_two_corpus_writer(n_per_corpus=10, high_conf=0.80, low_conf=0.70)
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="year_range_for_entity",
            entity_id="place_anchor",
            entity_name="Anchor",
            entity_type="Place",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp_high", "corp_low"],
            permitted_levels=["public"],
        )
        # Two-corpus fixture documents have no report_year, so coverage is empty.
        assert result.earliest_year is None
        assert result.latest_year is None
        assert "No dated claims" in result.answer_text

    def test_returns_year_range_from_doc_report_year(self):
        """When documents have report_year set, that's the fallback when
        observations lack IN_YEAR edges."""
        from gemynd.ingest.graph import InMemoryGraphWriter
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
        writer.node_store["Place"]["place_x"] = {"entity_id": "place_x", "name": "X"}
        for year, doc_id in [(1903, "doc_a"), (1907, "doc_b"), (1910, "doc_c")]:
            run_id = f"run_{doc_id}"
            writer.node_store["Document"][doc_id] = {
                "doc_id": doc_id,
                "title": f"Doc {year}",
                "institution_id": "corp",
                "access_level": "public",
                "deleted_at": None,
                "quarantine_status": None,
                "report_year": year,
            }
            writer.node_store["ExtractionRun"][run_id] = {
                "run_id": run_id,
                "run_timestamp": "2024-01-01T00:00:00",
            }
            writer.rel_store[("Document", doc_id, "PROCESSED_BY", "ExtractionRun", run_id, None)] = {
                "run_id": run_id,
                "run_timestamp": "2024-01-01T00:00:00",
            }
            cid = f"claim_{year}"
            writer.node_store["Claim"][cid] = {
                "claim_id": cid,
                "claim_type": "test",
                "extraction_confidence": 0.9,
                "source_sentence": "sentence",
                "run_id": run_id,
                "quarantine_status": None,
                "certainty": "observed",
            }
            page_id, sec_id, para_id = f"p_{doc_id}", f"s_{doc_id}", f"par_{doc_id}"
            writer.node_store["Page"][page_id] = {"page_id": page_id, "page_number": 1}
            writer.node_store["Section"][sec_id] = {"section_id": sec_id}
            writer.node_store["Paragraph"][para_id] = {"paragraph_id": para_id}
            writer.rel_store[("Document", doc_id, "HAS_PAGE", "Page", page_id, None)] = {}
            writer.rel_store[("Page", page_id, "HAS_SECTION", "Section", sec_id, None)] = {}
            writer.rel_store[("Section", sec_id, "HAS_PARAGRAPH", "Paragraph", para_id, None)] = {}
            writer.rel_store[("Paragraph", para_id, "HAS_CLAIM", "Claim", cid, None)] = {"run_id": run_id}
            writer.rel_store[("Claim", cid, "ABOUT", "Place", "place_x", None)] = {}

        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)
        intent = Tier1Intent(
            operation="year_range_for_entity",
            entity_id="place_x",
            entity_name="X",
            entity_type="Place",
        )
        result = assembler.tier_1_lookup(
            intent,
            institution_ids=["corp"],
            permitted_levels=["public"],
        )
        assert result.earliest_year == 1903
        assert result.latest_year == 1910
        assert result.distinct_years == 3
        assert result.count == 3
        assert "1903 to 1910" in result.answer_text
        assert "3 distinct years" in result.answer_text


class TestTier1LocationsForEntity:
    """Validates LOCATIONS_FOR_ENTITY_QUERY end-to-end."""

    def _build_writer_with_places(self):
        """Build a writer where two claims about an anchor entity also link
        OCCURRED_AT to two different Place nodes (Spokane and Cheney)."""
        from gemynd.ingest.graph import InMemoryGraphWriter
        writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)

        writer.node_store["Person"]["person_x"] = {"entity_id": "person_x", "name": "X"}
        writer.node_store["Place"]["place_spokane"] = {"entity_id": "place_spokane", "name": "Spokane"}
        writer.node_store["Place"]["place_cheney"] = {"entity_id": "place_cheney", "name": "Cheney"}

        # Doc + run
        writer.node_store["Document"]["doc_a"] = {
            "doc_id": "doc_a", "title": "Doc A", "institution_id": "corp",
            "access_level": "public", "deleted_at": None, "quarantine_status": None,
        }
        writer.node_store["ExtractionRun"]["run_a"] = {
            "run_id": "run_a", "run_timestamp": "2024-01-01T00:00:00",
        }
        writer.rel_store[("Document", "doc_a", "PROCESSED_BY", "ExtractionRun", "run_a", None)] = {
            "run_id": "run_a", "run_timestamp": "2024-01-01T00:00:00",
        }
        # Structure
        writer.node_store["Page"]["pg_a"] = {"page_id": "pg_a", "page_number": 1}
        writer.node_store["Section"]["sec_a"] = {"section_id": "sec_a"}
        writer.node_store["Paragraph"]["para_a"] = {"paragraph_id": "para_a"}
        writer.rel_store[("Document", "doc_a", "HAS_PAGE", "Page", "pg_a", None)] = {}
        writer.rel_store[("Page", "pg_a", "HAS_SECTION", "Section", "sec_a", None)] = {}
        writer.rel_store[("Section", "sec_a", "HAS_PARAGRAPH", "Paragraph", "para_a", None)] = {}

        # Three claims: two OCCURRED_AT Spokane, one at Cheney.
        for cid, place_id in [("c_1", "place_spokane"), ("c_2", "place_spokane"), ("c_3", "place_cheney")]:
            writer.node_store["Claim"][cid] = {
                "claim_id": cid, "claim_type": "test", "extraction_confidence": 0.9,
                "source_sentence": "s", "run_id": "run_a", "quarantine_status": None,
                "certainty": "observed",
            }
            writer.rel_store[("Paragraph", "para_a", "HAS_CLAIM", "Claim", cid, None)] = {"run_id": "run_a"}
            writer.rel_store[("Claim", cid, "ABOUT", "Person", "person_x", None)] = {}
            writer.rel_store[("Claim", cid, "OCCURRED_AT", "Place", place_id, None)] = {}
        return writer

    def test_returns_ranked_places(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = self._build_writer_with_places()
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="locations_for_entity",
            entity_id="person_x", entity_name="X", entity_type="Person",
        )
        result = assembler.tier_1_lookup(
            intent, institution_ids=["corp"], permitted_levels=["public"],
        )
        assert result.count == 2
        # Spokane should rank first (2 claims) above Cheney (1 claim).
        assert result.items[0]["place_name"] == "Spokane"
        assert result.items[0]["claim_count"] == 2
        assert result.items[1]["place_name"] == "Cheney"
        assert result.items[1]["claim_count"] == 1

    def test_returns_empty_for_entity_without_locations(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = self._build_writer_with_places()
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="locations_for_entity",
            entity_id="ghost_entity", entity_name="Ghost", entity_type="Person",
        )
        result = assembler.tier_1_lookup(
            intent, institution_ids=["corp"], permitted_levels=["public"],
        )
        assert result.count == 0
        assert "No locations recorded" in result.answer_text


class TestTier1CoOccurringEntities:
    """Validates CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY end-to-end."""

    def _build_writer_with_co_occurrence(self):
        """A writer where the anchor person co-occurs with two other entities
        (one species, one place) across multiple claims."""
        from gemynd.ingest.graph import InMemoryGraphWriter
        writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)

        writer.node_store["Person"]["person_anchor"] = {"entity_id": "person_anchor", "name": "Anchor"}
        writer.node_store["Species"]["species_mallard"] = {"entity_id": "species_mallard", "name": "Mallard"}
        writer.node_store["Place"]["place_a"] = {"entity_id": "place_a", "name": "Place A"}

        writer.node_store["Document"]["doc_a"] = {
            "doc_id": "doc_a", "title": "Doc A", "institution_id": "corp",
            "access_level": "public", "deleted_at": None, "quarantine_status": None,
        }
        writer.node_store["ExtractionRun"]["run_a"] = {
            "run_id": "run_a", "run_timestamp": "2024-01-01T00:00:00",
        }
        writer.rel_store[("Document", "doc_a", "PROCESSED_BY", "ExtractionRun", "run_a", None)] = {
            "run_id": "run_a", "run_timestamp": "2024-01-01T00:00:00",
        }
        writer.node_store["Page"]["pg_a"] = {"page_id": "pg_a", "page_number": 1}
        writer.node_store["Section"]["sec_a"] = {"section_id": "sec_a"}
        writer.node_store["Paragraph"]["para_a"] = {"paragraph_id": "para_a"}
        writer.rel_store[("Document", "doc_a", "HAS_PAGE", "Page", "pg_a", None)] = {}
        writer.rel_store[("Page", "pg_a", "HAS_SECTION", "Section", "sec_a", None)] = {}
        writer.rel_store[("Section", "sec_a", "HAS_PARAGRAPH", "Paragraph", "para_a", None)] = {}

        # 3 claims with anchor + mallard, 1 claim with anchor + place_a
        for i, partner in enumerate(
            [("Species", "species_mallard"), ("Species", "species_mallard"),
             ("Species", "species_mallard"), ("Place", "place_a")]
        ):
            cid = f"c_{i}"
            writer.node_store["Claim"][cid] = {
                "claim_id": cid, "claim_type": "test", "extraction_confidence": 0.9,
                "source_sentence": "s", "run_id": "run_a", "quarantine_status": None,
                "certainty": "observed",
            }
            writer.rel_store[("Paragraph", "para_a", "HAS_CLAIM", "Claim", cid, None)] = {"run_id": "run_a"}
            writer.rel_store[("Claim", cid, "ABOUT", "Person", "person_anchor", None)] = {}
            writer.rel_store[("Claim", cid, "ABOUT", partner[0], partner[1], None)] = {}
        return writer

    def test_returns_ranked_co_occurring(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = self._build_writer_with_co_occurrence()
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="co_occurring_entities_for_entity",
            entity_id="person_anchor", entity_name="Anchor", entity_type="Person",
        )
        result = assembler.tier_1_lookup(
            intent, institution_ids=["corp"], permitted_levels=["public"],
        )
        assert result.count == 2
        # Mallard should rank first (3 shared claims) above Place A (1).
        assert result.items[0]["entity_id"] == "species_mallard"
        assert result.items[0]["co_occurrence_count"] == 3
        assert result.items[0]["entity_type"] == "Species"
        assert result.items[1]["entity_id"] == "place_a"
        assert result.items[1]["co_occurrence_count"] == 1


class TestTier1FrequencyRanking:
    """Validates FREQUENCY_RANKING_FOR_TYPE_QUERY end-to-end."""

    def _build_writer_with_persons(self):
        from gemynd.ingest.graph import InMemoryGraphWriter
        writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)

        for pid, name in [("p_a", "Alice"), ("p_b", "Bob"), ("p_c", "Carol")]:
            writer.node_store["Person"][pid] = {"entity_id": pid, "name": name}

        writer.node_store["Document"]["doc_a"] = {
            "doc_id": "doc_a", "title": "Doc A", "institution_id": "corp",
            "access_level": "public", "deleted_at": None, "quarantine_status": None,
        }
        writer.node_store["ExtractionRun"]["run_a"] = {
            "run_id": "run_a", "run_timestamp": "2024-01-01T00:00:00",
        }
        writer.rel_store[("Document", "doc_a", "PROCESSED_BY", "ExtractionRun", "run_a", None)] = {
            "run_id": "run_a", "run_timestamp": "2024-01-01T00:00:00",
        }
        writer.node_store["Page"]["pg_a"] = {"page_id": "pg_a", "page_number": 1}
        writer.node_store["Section"]["sec_a"] = {"section_id": "sec_a"}
        writer.node_store["Paragraph"]["para_a"] = {"paragraph_id": "para_a"}
        writer.rel_store[("Document", "doc_a", "HAS_PAGE", "Page", "pg_a", None)] = {}
        writer.rel_store[("Page", "pg_a", "HAS_SECTION", "Section", "sec_a", None)] = {}
        writer.rel_store[("Section", "sec_a", "HAS_PARAGRAPH", "Paragraph", "para_a", None)] = {}

        # Alice: 5 claims, Bob: 2, Carol: 1
        counts = [("p_a", 5), ("p_b", 2), ("p_c", 1)]
        i = 0
        for pid, n in counts:
            for _ in range(n):
                cid = f"c_{i}"; i += 1
                writer.node_store["Claim"][cid] = {
                    "claim_id": cid, "claim_type": "test", "extraction_confidence": 0.9,
                    "source_sentence": "s", "run_id": "run_a", "quarantine_status": None,
                    "certainty": "observed",
                }
                writer.rel_store[("Paragraph", "para_a", "HAS_CLAIM", "Claim", cid, None)] = {"run_id": "run_a"}
                writer.rel_store[("Claim", cid, "ABOUT", "Person", pid, None)] = {}
        return writer

    def test_top_n_persons_ranked_by_claim_count(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = self._build_writer_with_persons()
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="frequency_ranking_for_type",
            entity_name="people", target_label="Person", limit=10,
        )
        result = assembler.tier_1_lookup(
            intent, institution_ids=["corp"], permitted_levels=["public"],
        )
        assert result.count == 3
        assert result.items[0]["name"] == "Alice"
        assert result.items[0]["claim_count"] == 5
        assert result.items[1]["name"] == "Bob"
        assert result.items[2]["name"] == "Carol"

    def test_limit_clamps_results(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = self._build_writer_with_persons()
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="frequency_ranking_for_type",
            entity_name="people", target_label="Person", limit=2,
        )
        result = assembler.tier_1_lookup(
            intent, institution_ids=["corp"], permitted_levels=["public"],
        )
        assert result.count == 2
        assert result.items[0]["name"] == "Alice"
        assert result.items[1]["name"] == "Bob"

    def test_unknown_entity_label_returns_empty(self):
        from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
        from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
        from gemynd.retrieval.models import Tier1Intent

        writer = self._build_writer_with_persons()
        executor = InMemoryQueryExecutor(writer, entity_labels=TEST_ENTITY_LABELS)
        assembler = ProvenanceContextAssembler(executor=executor)

        intent = Tier1Intent(
            operation="frequency_ranking_for_type",
            entity_name="aliens", target_label="Alien", limit=10,
        )
        result = assembler.tier_1_lookup(
            intent, institution_ids=["corp"], permitted_levels=["public"],
        )
        assert result.count == 0
        assert "No aliens" in result.answer_text

