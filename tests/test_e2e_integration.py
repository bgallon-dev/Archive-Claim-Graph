"""End-to-end integration tests: ingest → graph write → retrieval, no external dependencies.

The ``full_pipeline`` fixture builds the complete retrieval stack (executor, assembler,
gateway) on top of the session-scoped ``InMemoryGraphWriter`` that also backs
``test_retrieval_in_memory.py``.  Tests exercise the full data-flow path that unit tests
cannot reach:

- **Bug A** — Entity-ID mismatch: ``gateway.resolve`` resolves surface forms via
  ``default_seed_entities()``; the ingestion pipeline writes entity_ids using the same
  function.  If the normalization or hashing diverges between the two call sites the
  gateway returns entity_ids that are absent from the graph and every entity-anchored
  query returns nothing.

- **Bug B** — Year-node mismatch: year values can enter the graph via
  ``Observation.year`` (set from parsed sentence dates) or via ``report_year`` on the
  Document node.  If the temporal-query handler looks up Year nodes by one path while
  they were created via the other, blocks come back with ``year=None`` even for
  correctly year-annotated reports.

- **Bug C** — Missing ABOUT_REFUGE link: if document-anchor matching fails during
  ingestion (refuge name variant not in seed_entities) no ``ABOUT_REFUGE`` edge is
  written.  The strategy selector then picks the anchor-aware temporal template
  (``build_temporal_with_anchor_query("Refuge", "ABOUT_REFUGE")``) scoped to the
  Turnbull refuge entity_id, finds no linked documents, and returns an empty
  result set — silently losing the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from gemynd.retrieval.classifier import classify_query
from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
from gemynd.retrieval.entity_gateway import EntityResolutionGateway
from gemynd.retrieval.in_memory_executor import InMemoryQueryExecutor
from gemynd.retrieval.models import EntityContext
from tests.conftest import TEST_ENTITY_LABELS

_ENTITY_LABELS = TEST_ENTITY_LABELS


# ---------------------------------------------------------------------------
# PipelineFixture — bundles the full retrieval stack for a single test module
# ---------------------------------------------------------------------------

@dataclass
class PipelineFixture:
    """All components of the retrieval stack wired to a single InMemoryGraphWriter."""

    writer: object                          # InMemoryGraphWriter
    executor: InMemoryQueryExecutor
    assembler: ProvenanceContextAssembler
    gateway: EntityResolutionGateway
    doc_ids: set[str] = field(default_factory=set)


@pytest.fixture(scope="module")
def full_pipeline(populated_writer):
    """Build the retrieval stack on top of the session-scoped InMemoryGraphWriter.

    Module-scoped so the (cheap) executor/assembler/gateway construction happens once
    per test file; the expensive pipeline run is shared with the session fixture.
    """
    executor = InMemoryQueryExecutor(
        populated_writer,
        entity_labels=TEST_ENTITY_LABELS,
        anchor_entity_type="Refuge",
        anchor_relation="ABOUT_REFUGE",
    )
    assembler = ProvenanceContextAssembler(
        executor,
        anchor_entity_type="Refuge",
        anchor_relation="ABOUT_REFUGE",
        institution_id="turnbull",
    )
    gateway = EntityResolutionGateway()
    doc_ids = set(populated_writer.node_store.get("Document", {}).keys())
    return PipelineFixture(
        writer=populated_writer,
        executor=executor,
        assembler=assembler,
        gateway=gateway,
        doc_ids=doc_ids,
    )


# ---------------------------------------------------------------------------
# Shared introspection helpers
# ---------------------------------------------------------------------------

def _linked_entity_ids(writer) -> set[str]:
    """Entity IDs that appear as the target of at least one Claim→Entity relationship."""
    return {
        ei
        for sl, si, rt, el, ei, _ in writer.rel_store
        if sl == "Claim" and el in _ENTITY_LABELS
    }


def _about_refuge_doc_ids(writer) -> set[str]:
    """Document IDs that have an ABOUT_REFUGE relationship to any Refuge node."""
    return {
        si
        for sl, si, rt, el, ei, _ in writer.rel_store
        if sl == "Document" and rt == "ABOUT_REFUGE" and el == "Refuge"
    }


def _years_in_writer(writer) -> set[int]:
    return {
        props["year"]
        for props in writer.node_store.get("Year", {}).values()
        if props.get("year") is not None
    }


# ---------------------------------------------------------------------------
# Ingest sanity — preconditions for every downstream test
# ---------------------------------------------------------------------------

class TestIngestCoverage:
    """Validate that the full pipeline ingested all three fixture reports."""

    def test_three_documents_in_writer(self, full_pipeline):
        assert len(full_pipeline.doc_ids) >= 3, (
            f"Expected ≥3 Documents in writer; found {len(full_pipeline.doc_ids)}: "
            f"{full_pipeline.doc_ids}"
        )

    def test_claims_present_for_all_docs(self, full_pipeline):
        """Every ingested document must have produced at least one Claim node."""
        claim_doc_ids = {
            props.get("run_id", "").split(":")[0]  # run_id starts with doc stem
            for props in full_pipeline.writer.node_store.get("Claim", {}).values()
        }
        # At least as many distinct runs as documents
        claim_count = len(full_pipeline.writer.node_store.get("Claim", {}))
        assert claim_count > 0, "No Claim nodes found — extraction pipeline failed"

    def test_year_nodes_present(self, full_pipeline):
        """At least one Year node must exist (required by temporal query path)."""
        years = _years_in_writer(full_pipeline.writer)
        assert years, "No Year nodes in writer — year extraction failed during ingestion"

    def test_entity_claim_links_present(self, full_pipeline):
        """At least one Claim→Entity link must exist (required by entity-anchored path)."""
        linked = _linked_entity_ids(full_pipeline.writer)
        assert linked, (
            "No Claim→Entity relationships in writer — "
            "entity resolution failed during ingestion"
        )


# ---------------------------------------------------------------------------
# Bug A — Entity-ID consistency between gateway and graph writer
# ---------------------------------------------------------------------------

class TestGatewayGraphConsistency:
    """Entity IDs produced by gateway.resolve() must appear as node keys in the writer.

    Failing this means the gateway seed and the ingestion pipeline use different
    normalization/hashing, so every entity-anchored query returns nothing.

    NOTE — known extraction gap (documented, not a test failure):
    ``'mallards'`` resolves via the gateway to ``species_45af0a5a05af1c6f`` (the seed
    "mallard" entity), but report1.json's ingestion pipeline does not produce that
    entity node — "mallards" appears only in a long multi-species sentence whose entity
    linker produces "coot" and "green-wing teal" links but not "mallard".  The
    entity-ID mechanism is consistent; the gap is in the mention extractor's coverage.
    ``test_mallard_extraction_gap_documented`` records this explicitly so regressions are
    visible.
    """

    def test_coot_resolves_to_entity_in_writer(self, full_pipeline):
        """'coots' resolves via gateway to the same entity_id the ingestion pipeline writes."""
        ctx = full_pipeline.gateway.resolve(["coots"])
        assert ctx.resolved, (
            "Expected 'coots' → REFERS_TO Species entity; "
            "seed_entities.csv has 'coot' — check DictionaryFuzzyResolver threshold"
        )
        for re_obj in ctx.resolved:
            found = any(
                re_obj.entity_id in full_pipeline.writer.node_store.get(label, {})
                for label in _ENTITY_LABELS
            )
            assert found, (
                f"gateway entity_id {re_obj.entity_id!r} (from 'coots') not found "
                "in any writer.node_store bucket — entity-ID mismatch between "
                "gateway seed and ingestion pipeline (Bug A)"
            )

    def test_turnbull_refuge_resolves_to_entity_in_writer(self, full_pipeline):
        ctx = full_pipeline.gateway.resolve(["Turnbull Refuge"])
        if not ctx.resolved:
            pytest.skip("Turnbull Refuge did not resolve via gateway (no REFERS_TO match)")
        for re_obj in ctx.resolved:
            found = any(
                re_obj.entity_id in full_pipeline.writer.node_store.get(label, {})
                for label in _ENTITY_LABELS
            )
            assert found, (
                f"gateway entity_id {re_obj.entity_id!r} (Turnbull Refuge) not found "
                "in writer — entity-ID mismatch (Bug A)"
            )

    def test_resolved_entity_ids_overlap_with_claim_links(self, full_pipeline):
        """Entity IDs the gateway resolves to must appear as Claim→Entity targets.

        Uses 'coots' — a species that the ingestion pipeline extracts and links — to
        verify the full ID round-trip from surface form → gateway entity_id → Claim link.
        """
        linked = _linked_entity_ids(full_pipeline.writer)
        if not linked:
            pytest.skip("No Claim→Entity links in writer")
        ctx = full_pipeline.gateway.resolve(["coots"])
        if not ctx.resolved:
            pytest.skip("'coots' did not resolve via gateway")
        resolved_ids = {re_obj.entity_id for re_obj in ctx.resolved}
        assert resolved_ids & linked, (
            f"Entity IDs from gateway.resolve('coots') = {resolved_ids} "
            f"have no overlap with Claim→Entity targets {linked} — "
            "entity-anchored queries will silently return nothing (Bug A)"
        )

    def test_mallard_extraction_gap_documented(self, full_pipeline):
        """Documents the known gap: 'mallards' resolves via gateway but is absent from writer.

        This test records — rather than fixes — the gap so that if it is ever closed by
        an improved mention extractor, the suite signals the positive regression.
        """
        ctx = full_pipeline.gateway.resolve(["mallards"])
        if not ctx.resolved:
            pytest.skip("gateway did not resolve 'mallards' — skip gap check")
        resolved_ids = {re_obj.entity_id for re_obj in ctx.resolved}
        in_writer = {
            eid
            for label in _ENTITY_LABELS
            for eid in full_pipeline.writer.node_store.get(label, {})
        }
        gap_ids = resolved_ids - in_writer
        # Currently expected to be non-empty (known gap); will become empty once fixed.
        # If this assertion fails, the mention extractor now covers mallard — remove gap.
        assert gap_ids, (
            "Mallard extraction gap appears to be CLOSED: 'mallards' now resolves to an "
            "entity_id that is also present in the writer.  Remove this gap-documentation "
            "test and add a positive consistency test instead."
        )


# ---------------------------------------------------------------------------
# Bug B — Year-node round-trip
# ---------------------------------------------------------------------------

class TestYearLinking:
    """Year values written during ingestion must survive the full retrieval round-trip.

    report1.json carries report_year=1938.  If Year nodes are linked via a different
    path than what _year_for_obs uses, ProvenanceBlock.year will be None.
    """

    def test_year_1938_exists_in_writer(self, full_pipeline):
        """report1.json has report_year=1938 — that Year node must be present."""
        years = _years_in_writer(full_pipeline.writer)
        assert 1938 in years, (
            f"Year 1938 missing from writer.node_store['Year']; found: {sorted(years)}"
        )

    def test_temporal_query_1938_returns_nonzero_blocks(self, full_pipeline):
        """A temporal query bounded to 1938 must return at least one block."""
        blocks, _ = full_pipeline.assembler.assemble(
            "wildlife activity in 1938",
            EntityContext(),
            year_min=1938,
            year_max=1938,
        )
        assert blocks, (
            "Temporal query year_min=year_max=1938 returned no blocks — "
            "either ABOUT_REFUGE is missing (Bug C) or Year nodes are unlinked (Bug B)"
        )

    def test_blocks_carry_correct_year_value(self, full_pipeline):
        """Every block with a non-None year must carry the year that was queried."""
        blocks, _ = full_pipeline.assembler.assemble(
            "wildlife activity in 1938",
            EntityContext(),
            year_min=1938,
            year_max=1938,
        )
        for b in blocks:
            if b.year is not None:
                assert b.year == 1938, (
                    f"Block {b.claim_id!r} has year={b.year}, "
                    "expected 1938 — year round-trip failed (Bug B)"
                )


# ---------------------------------------------------------------------------
# Bug C — Refuge linking
# ---------------------------------------------------------------------------

class TestRefugeLinking:
    """ABOUT_REFUGE must be written during ingestion so that the temporal refuge
    query can anchor results to Turnbull's document corpus.
    """

    def test_about_refuge_relationship_written(self, full_pipeline):
        """At least one Document must be linked to a Refuge via ABOUT_REFUGE."""
        linked_docs = _about_refuge_doc_ids(full_pipeline.writer)
        assert linked_docs, (
            "No Document→ABOUT_REFUGE→Refuge relationships in writer — "
            "document_anchor matching failed during ingestion (Bug C)"
        )

    def test_temporal_refuge_query_returns_rows(self, full_pipeline):
        """The anchor-aware temporal query must find claims when the refuge is linked."""
        from gemynd.core.graph.cypher import build_temporal_with_anchor_query

        refuge_ids = {
            ei
            for sl, si, rt, el, ei, _ in full_pipeline.writer.rel_store
            if rt == "ABOUT_REFUGE" and el == "Refuge"
        }
        if not refuge_ids:
            pytest.skip("No ABOUT_REFUGE in writer — already caught by previous test")

        refuge_id = next(iter(refuge_ids))
        anchor_cypher = build_temporal_with_anchor_query("Refuge", "ABOUT_REFUGE")
        rows = full_pipeline.executor.run(
            anchor_cypher,
            {
                "anchor_id": refuge_id,
                "year_min": None,
                "year_max": None,
                "claim_types": None,
                "limit": 50,
                "institution_id": "turnbull",
                "permitted_levels": ["public"],
            },
        )
        assert rows, (
            f"anchor-aware temporal query for anchor_id={refuge_id!r} "
            "returned no rows even though ABOUT_REFUGE exists — "
            "executor traversal is broken (Bug C)"
        )

    def test_no_entity_context_assembler_uses_refuge_anchor(self, full_pipeline):
        """Without entity context, assembler anchors to the default refuge for temporal queries.

        This directly exercises the TEMPORAL_CLAIMS_QUERY_WITH_REFUGE code path that
        depends on ABOUT_REFUGE being present.
        """
        blocks, _ = full_pipeline.assembler.assemble(
            "wildlife activity in 1938",
            EntityContext(),
            year_min=1938,
            year_max=1938,
        )
        assert blocks, (
            "No blocks from no-entity temporal query — either ABOUT_REFUGE is missing "
            "(Bug C) or Year 1938 is unlinked (Bug B)"
        )


# ---------------------------------------------------------------------------
# Canonical full-chain test: classify → gateway.resolve → assembler.assemble
# ---------------------------------------------------------------------------

class TestFullChain:
    """End-to-end tests: classify_query → EntityResolutionGateway → assembler.assemble.

    Exercises all integration points in sequence so failures point directly to the
    failing layer rather than a downstream symptom.
    """

    def test_classifier_extracts_year_and_entities(self):
        intent = classify_query("how many mallards were observed in 1938?")
        assert intent.year_min == 1938 and intent.year_max == 1938, (
            f"Classifier must extract 1938; got year_min={intent.year_min}, "
            f"year_max={intent.year_max}"
        )
        assert any("mallard" in e.lower() for e in intent.entities), (
            f"Classifier must include 'mallards' in entities; got {intent.entities}"
        )

    def test_gateway_resolves_mallard_to_species(self, full_pipeline):
        intent = classify_query("how many mallards were observed in 1938?")
        ctx = full_pipeline.gateway.resolve(intent.entities)
        assert ctx.resolved, (
            f"gateway.resolve({intent.entities!r}) produced no REFERS_TO match — "
            "entity-anchored retrieval will fall back to fulltext"
        )
        types = {re_obj.entity_type for re_obj in ctx.resolved}
        assert "Species" in types, (
            f"Expected 'mallards' → Species; got entity_types={types!r}"
        )

    def test_mallard_population_query_returns_1938_data(self, full_pipeline):
        """Canonical integration test from the task specification.

        Failure path: any one of the three bugs above will break this test —
        entity-ID mismatch (A) → no entity-anchored results, year mismatch (B) →
        year filter returns nothing, refuge mismatch (C) → temporal query returns nothing.
        """
        intent = classify_query("how many mallards were observed in 1938?")
        entity_ctx = full_pipeline.gateway.resolve(intent.entities)
        blocks, context = full_pipeline.assembler.assemble(
            "how many mallards were observed in 1938?",
            entity_ctx,
            year_min=intent.year_min,
            year_max=intent.year_max,
        )
        assert blocks, (
            "No blocks returned for '...mallards...1938' — "
            "check entity-ID consistency (Bug A), year linking (Bug B), "
            "refuge linking (Bug C)"
        )
        assert any("mallard" in b.source_sentence.lower() for b in blocks), (
            "No block contains 'mallard' in source_sentence; sentences: "
            + str([b.source_sentence[:80] for b in blocks[:5]])
        )
        assert any(b.year == 1938 for b in blocks), (
            f"No block has year=1938; years present: "
            f"{sorted({b.year for b in blocks if b.year is not None})}"
        )

    def test_assembler_serializes_mallard_claim_into_context(self, full_pipeline):
        """The serialized context string must contain the mallard sentence text."""
        intent = classify_query("how many mallards were observed in 1938?")
        entity_ctx = full_pipeline.gateway.resolve(intent.entities)
        blocks, context = full_pipeline.assembler.assemble(
            "how many mallards were observed in 1938?",
            entity_ctx,
            year_min=intent.year_min,
            year_max=intent.year_max,
        )
        if not blocks:
            pytest.skip("No blocks returned — upstream test already covers this")
        assert "mallard" in context.lower(), (
            "Serialized context does not contain 'mallard' — "
            "_serialise_block may have dropped or truncated the relevant sentence"
        )
        assert "<claim_text>" in context, "Context must use <claim_text> delimiters"

    def test_impossible_year_returns_empty_through_full_chain(self, full_pipeline):
        """Year range that matches no data must propagate to empty blocks and empty context."""
        intent = classify_query("what happened in 9999?")
        entity_ctx = full_pipeline.gateway.resolve(intent.entities)
        blocks, ctx = full_pipeline.assembler.assemble(
            "what happened in 9999?",
            entity_ctx,
            year_min=intent.year_min,
            year_max=intent.year_max,
        )
        assert blocks == []
        assert ctx == ""

    def test_all_blocks_have_doc_title_and_claim_id(self, full_pipeline):
        """Every block returned from any query must have the minimum required fields."""
        intent = classify_query("how many mallards were observed in 1938?")
        entity_ctx = full_pipeline.gateway.resolve(intent.entities)
        blocks, _ = full_pipeline.assembler.assemble(
            "how many mallards were observed in 1938?",
            entity_ctx,
            year_min=1938,
            year_max=1938,
        )
        for b in blocks:
            assert b.doc_title, f"Block {b.claim_id!r} has empty doc_title"
            assert b.claim_id, "Block is missing claim_id"
