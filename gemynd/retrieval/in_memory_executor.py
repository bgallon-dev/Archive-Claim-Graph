"""In-memory query executor for testing the retrieval layer without Neo4j.

Implements the same ``run(cypher, params)`` interface as ``Neo4jQueryExecutor``
but executes against ``InMemoryGraphWriter``'s ``node_store`` / ``rel_store``.

The dispatcher uses Python ``is`` (object identity) to recognise the closed set
of module-level query-template constants.  Any other Cypher string (e.g. the
anchor-id startup query the assembler runs on ``__init__``) is handled by
``_generic_dispatch``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemynd.core.graph.cypher import (
    CLAIM_TYPE_SCOPED_QUERY,
    CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY,
    COUNT_CLAIMS_FOR_ENTITY_QUERY,
    DOC_TYPE_BROWSE_QUERY,
    ENTITY_ANCHORED_CLAIMS_QUERY,
    FREQUENCY_RANKING_FOR_TYPE_QUERY,
    FULLTEXT_CLAIMS_QUERY,
    LIST_DOCUMENTS_FOR_ENTITY_QUERY,
    LIST_DOCUMENTS_FOR_YEAR_QUERY,
    LOCATIONS_FOR_ENTITY_QUERY,
    MULTI_ENTITY_CLAIMS_QUERY,
    PROVENANCE_CHAIN_QUERY,
    SERIES_BROWSE_QUERY,
    TEMPORAL_CLAIMS_QUERY,
    YEAR_RANGE_FOR_ENTITY_QUERY,
    build_document_anchor_query,
    build_temporal_with_anchor_query,
)

if TYPE_CHECKING:
    from gemynd.ingest.graph import InMemoryGraphWriter


class InMemoryQueryExecutor:
    """Query executor backed by ``InMemoryGraphWriter`` stores.

    Drop-in replacement for ``Neo4jQueryExecutor`` in test contexts:
    pass this to ``ProvenanceContextAssembler`` to run the real retrieval
    logic against real pipeline-ingested data without a running database.
    """

    def __init__(
        self,
        writer: "InMemoryGraphWriter",
        *,
        entity_labels: frozenset[str],
        anchor_entity_type: str | None = None,
        anchor_relation: str | None = None,
    ) -> None:
        self._w = writer
        self._entity_labels = entity_labels
        self._anchor_entity_type = anchor_entity_type
        self._anchor_relation = anchor_relation
        if anchor_entity_type and anchor_relation:
            self._anchor_temporal_cypher: str | None = build_temporal_with_anchor_query(
                anchor_entity_type, anchor_relation
            )
            self._anchor_document_cypher: str | None = build_document_anchor_query(
                anchor_entity_type, anchor_relation
            )
        else:
            self._anchor_temporal_cypher = None
            self._anchor_document_cypher = None

    # ── public interface ────────────────────────────────────────────────────

    def run(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute *cypher* against the in-memory stores and return row dicts."""
        p = params or {}
        if cypher is ENTITY_ANCHORED_CLAIMS_QUERY:
            return self._entity_anchored(p)
        if cypher is TEMPORAL_CLAIMS_QUERY:
            return self._temporal(p)
        if (
            self._anchor_temporal_cypher is not None
            and cypher is self._anchor_temporal_cypher
        ):
            return self._temporal_with_anchor(p)
        if (
            self._anchor_document_cypher is not None
            and cypher is self._anchor_document_cypher
        ):
            return self._document_anchored(p)
        if cypher is MULTI_ENTITY_CLAIMS_QUERY:
            return self._multi_entity(p)
        if cypher is CLAIM_TYPE_SCOPED_QUERY:
            return self._claim_type_scoped(p)
        if cypher is FULLTEXT_CLAIMS_QUERY:
            return self._fulltext(p)
        if cypher is DOC_TYPE_BROWSE_QUERY:
            return self._doc_type_browse(p)
        if cypher is SERIES_BROWSE_QUERY:
            return self._series_browse(p)
        if cypher is PROVENANCE_CHAIN_QUERY:
            return self._provenance_chain(p)
        if cypher is COUNT_CLAIMS_FOR_ENTITY_QUERY:
            return self._count_claims_for_entity(p)
        if cypher is YEAR_RANGE_FOR_ENTITY_QUERY:
            return self._year_range_for_entity(p)
        if cypher is LIST_DOCUMENTS_FOR_ENTITY_QUERY:
            return self._list_documents_for_entity(p)
        if cypher is LIST_DOCUMENTS_FOR_YEAR_QUERY:
            return self._list_documents_for_year(p)
        if cypher is LOCATIONS_FOR_ENTITY_QUERY:
            return self._locations_for_entity(p)
        if cypher is CO_OCCURRING_ENTITIES_FOR_ENTITY_QUERY:
            return self._co_occurring_entities_for_entity(p)
        if cypher is FREQUENCY_RANKING_FOR_TYPE_QUERY:
            return self._frequency_ranking_for_type(p)
        return self._generic_dispatch(cypher, p)

    def _count_claims_for_entity(self, p: dict) -> list[dict]:
        """Tier 1 lookup: count active Claims linked to a given entity."""
        entity_id = p["entity_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        seen: set[str] = set()
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id):
                if self._claim_entity_rel(claim_id, entity_id) is not None:
                    seen.add(claim_id)
        return [{"n": len(seen)}]

    def _year_range_for_entity(self, p: dict) -> list[dict]:
        """Tier 1 lookup: temporal coverage for a given entity.

        Mirrors YEAR_RANGE_FOR_ENTITY_QUERY: prefers observation year, falls
        back to the document's report_year when an observation has no
        IN_YEAR link or no observation exists at all.
        """
        entity_id = p["entity_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])

        # year → set of claim_ids attributed to that year
        years_to_claims: dict[int, set[str]] = {}
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            doc_props = self._w.node_store.get("Document", {}).get(doc_id, {})
            doc_year = doc_props.get("report_year")
            for claim_id in self._claims_for_run(run_id):
                if self._claim_entity_rel(claim_id, entity_id) is None:
                    continue
                # Find observation year if present.
                obs_ids = self._observations_for_claim(claim_id)
                effective_year: int | None = None
                for obs_id in obs_ids:
                    y_node = self._year_for_obs(obs_id)
                    if y_node is not None:
                        y = y_node.get("year")
                        if y is not None:
                            effective_year = int(y)
                            break
                if effective_year is None and doc_year is not None:
                    effective_year = int(doc_year)
                if effective_year is None:
                    continue
                years_to_claims.setdefault(effective_year, set()).add(claim_id)

        if not years_to_claims:
            return [{
                "earliest_year": None,
                "latest_year": None,
                "distinct_years": 0,
                "total_claims": 0,
            }]
        return [{
            "earliest_year": min(years_to_claims),
            "latest_year": max(years_to_claims),
            "distinct_years": len(years_to_claims),
            "total_claims": sum(len(s) for s in years_to_claims.values()),
        }]

    def _list_documents_for_entity(self, p: dict) -> list[dict]:
        """Tier 1 lookup: documents containing claims linked to a given entity,
        ranked by claim count desc.
        """
        entity_id = p["entity_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])

        doc_to_claims: dict[str, set[str]] = {}
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id):
                if self._claim_entity_rel(claim_id, entity_id) is not None:
                    doc_to_claims.setdefault(doc_id, set()).add(claim_id)

        rows: list[dict] = []
        for doc_id, claim_ids in doc_to_claims.items():
            d = self._w.node_store.get("Document", {}).get(doc_id, {})
            rows.append({
                "doc_id": doc_id,
                "title": d.get("title"),
                "doc_type": d.get("doc_type"),
                "date_start": d.get("date_start"),
                "date_end": d.get("date_end"),
                "report_year": d.get("report_year"),
                "institution_id": d.get("institution_id"),
                "lccn": d.get("lccn"),
                "edition": d.get("edition"),
                "loc_sequence": d.get("loc_sequence"),
                "claim_count": len(claim_ids),
            })
        rows.sort(key=lambda r: r["claim_count"], reverse=True)
        return rows[:50]

    def _list_documents_for_year(self, p: dict) -> list[dict]:
        """Tier 1 lookup: documents whose report_year matches $target_year,
        with their claim count.  Includes documents with zero claims so the
        user sees the full year roster.
        """
        target_year = p["target_year"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            d = self._w.node_store.get("Document", {}).get(doc_id, {})
            if d.get("report_year") != target_year:
                continue
            run_id = self._latest_run_id(doc_id)
            claim_ids: set[str] = set()
            if run_id:
                for claim_id in self._claims_for_run(run_id):
                    claim_ids.add(claim_id)
            rows.append({
                "doc_id": doc_id,
                "title": d.get("title"),
                "doc_type": d.get("doc_type"),
                "date_start": d.get("date_start"),
                "date_end": d.get("date_end"),
                "report_year": d.get("report_year"),
                "institution_id": d.get("institution_id"),
                "lccn": d.get("lccn"),
                "edition": d.get("edition"),
                "loc_sequence": d.get("loc_sequence"),
                "claim_count": len(claim_ids),
            })
        rows.sort(key=lambda r: (r.get("date_start") or "", r.get("title") or ""))
        return rows[:50]

    def _locations_for_entity(self, p: dict) -> list[dict]:
        """Tier 1 lookup: places where active claims about the entity occurred.

        Mirrors LOCATIONS_FOR_ENTITY_QUERY: walks Claim -> OCCURRED_AT -> Place|Refuge,
        ranks by claim count desc, caps at 25.
        """
        entity_id = p["entity_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])

        place_to_claims: dict[str, set[str]] = {}
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id):
                if self._claim_entity_rel(claim_id, entity_id) is None:
                    continue
                # Walk OCCURRED_AT to Place/Refuge nodes.
                for sl, si, rt, el, ei, _ in self._w.rel_store:
                    if (sl == "Claim" and si == claim_id and rt == "OCCURRED_AT"
                            and el in ("Place", "Refuge")):
                        place_to_claims.setdefault(ei, set()).add(claim_id)

        rows: list[dict] = []
        for place_id, claim_ids in place_to_claims.items():
            # Find the place's actual label (Place or Refuge) and properties.
            place_props = (
                self._w.node_store.get("Place", {}).get(place_id)
                or self._w.node_store.get("Refuge", {}).get(place_id)
                or {}
            )
            place_type = "Place" if place_id in self._w.node_store.get("Place", {}) else "Refuge"
            rows.append({
                "place_id": place_id,
                "place_name": place_props.get("name"),
                "place_type": place_type,
                "claim_count": len(claim_ids),
            })
        rows.sort(key=lambda r: r["claim_count"], reverse=True)
        return rows[:25]

    def _co_occurring_entities_for_entity(self, p: dict) -> list[dict]:
        """Tier 1 lookup: entities that co-occur in the same claims as the
        anchor entity, ranked by co-occurrence count desc, capped at 25.
        """
        entity_id = p["entity_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])

        # entity_id (other) → set of claim_ids the pair appears in
        other_to_claims: dict[str, set[str]] = {}
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id):
                if self._claim_entity_rel(claim_id, entity_id) is None:
                    continue
                # Find OTHER entities this same claim points to.
                for sl, si, rt, el, ei, _ in self._w.rel_store:
                    if sl != "Claim" or si != claim_id:
                        continue
                    if ei == entity_id:
                        continue
                    if el not in self._entity_labels:
                        continue
                    other_to_claims.setdefault(ei, set()).add(claim_id)

        rows: list[dict] = []
        for other_id, claim_ids in other_to_claims.items():
            # Locate the other entity's node to read name + label.
            other_label = ""
            other_props: dict = {}
            for label in self._entity_labels:
                node = self._w.node_store.get(label, {}).get(other_id)
                if node is not None:
                    other_label = label
                    other_props = node
                    break
            rows.append({
                "entity_id": other_id,
                "name": other_props.get("name"),
                "entity_type": other_label,
                "co_occurrence_count": len(claim_ids),
            })
        rows.sort(key=lambda r: r["co_occurrence_count"], reverse=True)
        return rows[:25]

    def _frequency_ranking_for_type(self, p: dict) -> list[dict]:
        """Tier 1 lookup: top-N entities of a given label by claim count,
        scoped by access control.
        """
        entity_label = p["entity_label"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        limit = int(p.get("limit", 10))

        # Collect entity_ids of the requested label first.
        candidate_ids = list(self._w.node_store.get(entity_label, {}).keys())
        if not candidate_ids:
            return []

        entity_to_claims: dict[str, set[str]] = {}
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id):
                # For each claim, check if it links to any of our candidate entities.
                for sl, si, rt, el, ei, _ in self._w.rel_store:
                    if sl != "Claim" or si != claim_id:
                        continue
                    if el != entity_label:
                        continue
                    if ei in candidate_ids:
                        entity_to_claims.setdefault(ei, set()).add(claim_id)

        rows: list[dict] = []
        for entity_id, claim_ids in entity_to_claims.items():
            props = self._w.node_store.get(entity_label, {}).get(entity_id, {})
            rows.append({
                "entity_id": entity_id,
                "name": props.get("name"),
                "entity_type": entity_label,
                "claim_count": len(claim_ids),
            })
        rows.sort(key=lambda r: r["claim_count"], reverse=True)
        return rows[:limit]

    # ── query handlers ──────────────────────────────────────────────────────

    def _entity_anchored(self, p: dict) -> list[dict]:
        entity_id = p["entity_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        limit = p.get("limit", 20)

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id):
                rel_type = self._claim_entity_rel(claim_id, entity_id)
                if rel_type is None:
                    continue
                row = self._build_row(doc_id, claim_id, rel_type)
                year = (row.get("y") or {}).get("year")
                if year_min is not None and year is not None and year < year_min:
                    continue
                if year_max is not None and year is not None and year > year_max:
                    continue
                rows.append(row)

        rows.sort(
            key=lambda r: (r.get("c") or {}).get("extraction_confidence", 0.0),
            reverse=True,
        )
        # Per-corpus stratification mirrors the Cypher template: each institution_id
        # contributes up to `limit` candidates so the higher-confidence corpus cannot
        # monopolise the candidate pool before downstream truncation.
        by_corpus: dict[str, list[dict]] = {}
        for row in rows:
            corpus = (row.get("d") or {}).get("institution_id") or ""
            by_corpus.setdefault(corpus, []).append(row)
        return [r for corpus_rows in by_corpus.values() for r in corpus_rows[:limit]]

    def _temporal(self, p: dict) -> list[dict]:
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        claim_types = p.get("claim_types")
        limit = p.get("limit", 20)

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id, claim_types):
                for obs_id in self._observations_for_claim(claim_id):
                    year_node = self._year_for_obs(obs_id)
                    if year_node is None:
                        continue
                    year = year_node.get("year")
                    if year is None:
                        continue
                    if year_min is not None and year < year_min:
                        continue
                    if year_max is not None and year > year_max:
                        continue
                    rows.append(self._build_row(doc_id, claim_id, "IN_YEAR"))
                    break  # one row per claim

        rows.sort(
            key=lambda r: (
                (r.get("y") or {}).get("year") or 0,
                (r.get("c") or {}).get("extraction_confidence", 0.0),
            )
        )
        return rows[:limit]

    def _temporal_with_anchor(self, p: dict) -> list[dict]:
        anchor_id = p["anchor_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        claim_types = p.get("claim_types")
        limit = p.get("limit", 20)

        anchor_rel = self._anchor_relation
        anchor_label = self._anchor_entity_type

        valid = set(self._valid_docs(institution_ids, permitted_levels))
        anchor_docs: set[str] = set()
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if (
                rt == anchor_rel
                and el == anchor_label
                and ei == anchor_id
                and si in valid
            ):
                anchor_docs.add(si)

        rows: list[dict] = []
        for doc_id in anchor_docs:
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id, claim_types):
                for obs_id in self._observations_for_claim(claim_id):
                    year_node = self._year_for_obs(obs_id)
                    if year_node is None:
                        continue
                    year = year_node.get("year")
                    if year is None:
                        continue
                    if year_min is not None and year < year_min:
                        continue
                    if year_max is not None and year > year_max:
                        continue
                    rows.append(self._build_row(doc_id, claim_id, anchor_rel))
                    break

        rows.sort(
            key=lambda r: (
                (r.get("y") or {}).get("year") or 0,
                (r.get("c") or {}).get("extraction_confidence", 0.0),
            )
        )
        return rows[:limit]

    def _document_anchored(self, p: dict) -> list[dict]:
        anchor_id = p["anchor_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        claim_types = p.get("claim_types")
        limit = p.get("limit", 20)

        anchor_rel = self._anchor_relation
        anchor_label = self._anchor_entity_type

        valid = set(self._valid_docs(institution_ids, permitted_levels))
        anchor_docs: set[str] = set()
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if (
                rt == anchor_rel
                and el == anchor_label
                and ei == anchor_id
                and si in valid
            ):
                anchor_docs.add(si)

        rows: list[dict] = []
        for doc_id in anchor_docs:
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id, claim_types):
                row = self._build_row(doc_id, claim_id, anchor_rel)
                year = (row.get("y") or {}).get("year")
                if year is not None:
                    if year_min is not None and year < year_min:
                        continue
                    if year_max is not None and year > year_max:
                        continue
                rows.append(row)

        rows.sort(
            key=lambda r: (r.get("c") or {}).get("extraction_confidence", 0.0),
            reverse=True,
        )
        return rows[:limit]

    def _multi_entity(self, p: dict) -> list[dict]:
        entity_ids: list[str] = p["entity_ids"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        claim_types = p.get("claim_types")
        limit = p.get("limit", 20)
        entity_id_set = set(entity_ids)

        seen: set[str] = set()
        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id, claim_types):
                if claim_id in seen:
                    continue
                matched: list[str] = []
                first_rel: str | None = None
                for sl, si, rt, el, ei, _ in self._w.rel_store:
                    if (
                        sl == "Claim"
                        and si == claim_id
                        and ei in entity_id_set
                        and el in self._entity_labels
                    ):
                        matched.append(ei)
                        if first_rel is None:
                            first_rel = rt
                if not matched:
                    continue
                row = self._build_row(doc_id, claim_id, first_rel)
                year = (row.get("y") or {}).get("year")
                if year_min is not None and year is not None and year < year_min:
                    continue
                if year_max is not None and year is not None and year > year_max:
                    continue
                row["matched_entity_ids"] = matched
                seen.add(claim_id)
                rows.append(row)

        rows.sort(
            key=lambda r: (
                -len(r.get("matched_entity_ids") or []),
                -(r.get("c") or {}).get("extraction_confidence", 0.0),
            )
        )
        return rows[:limit]

    def _claim_type_scoped(self, p: dict) -> list[dict]:
        claim_types: list[str] = p["claim_types"]
        entity_ids = p.get("entity_ids")
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        limit = p.get("limit", 20)
        entity_id_set = set(entity_ids) if entity_ids else None

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id, claim_types):
                if entity_id_set is not None:
                    if not any(
                        sl == "Claim"
                        and si == claim_id
                        and ei in entity_id_set
                        and el in self._entity_labels
                        for sl, si, rt, el, ei, _ in self._w.rel_store
                    ):
                        continue
                claim_props = self._w.node_store.get("Claim", {}).get(claim_id, {})
                row = self._build_row(doc_id, claim_id, claim_props.get("claim_type"))
                year = (row.get("y") or {}).get("year")
                if year_min is not None and year is not None and year < year_min:
                    continue
                if year_max is not None and year is not None and year > year_max:
                    continue
                rows.append(row)

        rows.sort(
            key=lambda r: (r.get("c") or {}).get("extraction_confidence", 0.0),
            reverse=True,
        )
        return rows[:limit]

    def _fulltext(self, p: dict) -> list[dict]:
        search_text: str = p.get("search_text", "")
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        limit = p.get("limit", 20)
        needle = search_text.lower()

        rows: list[dict] = []
        for claim_id, claim_props in self._w.node_store.get("Claim", {}).items():
            sentence = (
                claim_props.get("normalized_sentence")
                or claim_props.get("source_sentence")
                or ""
            )
            if needle and needle not in sentence.lower():
                continue
            doc_id = self._doc_for_claim(claim_id)
            if doc_id is None:
                continue
            d = self._w.node_store.get("Document", {}).get(doc_id, {})
            if not self._doc_is_valid(d, institution_ids, permitted_levels):
                continue
            run_id = self._latest_run_id(doc_id)
            if claim_props.get("run_id") != run_id:
                continue
            rows.append(self._build_row(doc_id, claim_id))

        return rows[:limit]

    def _doc_type_browse(self, p: dict) -> list[dict]:
        doc_types: list[str] = p.get("doc_types") or []
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        claim_types = p.get("claim_types")
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        limit = p.get("limit", 20)

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            d = self._w.node_store.get("Document", {}).get(doc_id, {})
            if doc_types and d.get("doc_type") not in doc_types:
                continue
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id, claim_types):
                row = self._build_row(doc_id, claim_id, "DOC_TYPE_BROWSE")
                year = (row.get("y") or {}).get("year")
                if year_min is not None and year is not None and year < year_min:
                    continue
                if year_max is not None and year is not None and year > year_max:
                    continue
                rows.append(row)

        rows.sort(
            key=lambda r: (r.get("c") or {}).get("extraction_confidence", 0.0),
            reverse=True,
        )
        return rows[:limit]

    def _series_browse(self, p: dict) -> list[dict]:
        series: list[str] = p.get("series") or []
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])
        claim_types = p.get("claim_types")
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        limit = p.get("limit", 20)

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_ids, permitted_levels):
            d = self._w.node_store.get("Document", {}).get(doc_id, {})
            if series and d.get("series") not in series:
                continue
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id, claim_types):
                row = self._build_row(doc_id, claim_id, "SERIES_BROWSE")
                year = (row.get("y") or {}).get("year")
                if year_min is not None and year is not None and year < year_min:
                    continue
                if year_max is not None and year is not None and year > year_max:
                    continue
                rows.append(row)

        rows.sort(
            key=lambda r: (r.get("c") or {}).get("extraction_confidence", 0.0),
            reverse=True,
        )
        return rows[:limit]

    def _provenance_chain(self, p: dict) -> list[dict]:
        claim_id: str = p["claim_id"]
        institution_ids = p.get("institution_ids") or []
        permitted_levels = p.get("permitted_levels", ["public"])

        claim_props = self._w.node_store.get("Claim", {}).get(claim_id)
        if not claim_props:
            return []
        qs = claim_props.get("quarantine_status")
        if qs is not None and qs != "active":
            return []
        doc_id = self._doc_for_claim(claim_id)
        if doc_id is None:
            return []
        d = self._w.node_store.get("Document", {}).get(doc_id, {})
        if not self._doc_is_valid(d, institution_ids, permitted_levels):
            return []
        run_id = self._latest_run_id(doc_id)
        if claim_props.get("run_id") != run_id:
            return []
        return [self._build_row(doc_id, claim_id)]

    def _generic_dispatch(self, cypher: str, p: dict) -> list[dict]:
        # Assembler __init__ startup query to find anchor entity by label:
        # "MATCH (d:Document)-->(r:{EntityType}) WHERE d.institution_id = $inst_id
        #  RETURN r.entity_id AS eid"
        if "eid" in cypher and "Document" in cypher and "-->" in cypher:
            import re as _re
            label_match = _re.search(r'\(r:(\w+)\)', cypher)
            target_label = label_match.group(1) if label_match else self._anchor_entity_type
            if target_label is None:
                return []
            target_inst = p.get("inst_id")
            for sl, si, rt, el, ei, _ in self._w.rel_store:
                if el == target_label and sl == "Document":
                    if target_inst is not None:
                        d = self._w.node_store.get("Document", {}).get(si, {})
                        if d.get("institution_id") != target_inst:
                            continue
                    return [{"eid": ei}]
            return []
        return []

    # ── document filtering ──────────────────────────────────────────────────

    def _valid_docs(
        self, institution_ids: list[str], permitted_levels: list[str]
    ) -> list[str]:
        from gemynd.core.access_control import doc_is_valid
        return [
            doc_id
            for doc_id, props in self._w.node_store.get("Document", {}).items()
            if doc_is_valid(props, institution_ids, permitted_levels)
        ]

    def _doc_is_valid(
        self,
        props: dict,
        institution_ids: list[str],
        permitted_levels: list[str],
    ) -> bool:
        from gemynd.core.access_control import doc_is_valid
        return doc_is_valid(props, institution_ids, permitted_levels)

    # ── run / claim helpers ─────────────────────────────────────────────────

    def _latest_run_id(self, doc_id: str) -> str | None:
        best_ts: Any = None
        best_run: str | None = None
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if (
                sl == "Document"
                and si == doc_id
                and rt == "PROCESSED_BY"
                and el == "ExtractionRun"
            ):
                run_node = self._w.node_store.get("ExtractionRun", {}).get(ei, {})
                ts = run_node.get("run_timestamp")
                if best_ts is None or (ts is not None and ts > best_ts):
                    best_ts = ts
                    best_run = ei
        return best_run

    def _claims_for_run(
        self, run_id: str, claim_types: list[str] | None = None
    ) -> list[str]:
        result: list[str] = []
        for claim_id, props in self._w.node_store.get("Claim", {}).items():
            if props.get("run_id") != run_id:
                continue
            qs = props.get("quarantine_status")
            if qs is not None and qs != "active":
                continue
            if claim_types is not None and props.get("claim_type") not in claim_types:
                continue
            result.append(claim_id)
        return result

    # ── graph traversal helpers ─────────────────────────────────────────────

    def _observations_for_claim(self, claim_id: str) -> list[str]:
        obs_ids: list[str] = []
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if (
                sl == "Claim"
                and si == claim_id
                and rt == "SUPPORTS"
                and el == "Observation"
            ):
                obs_ids.append(ei)
        return obs_ids

    def _year_for_obs(self, obs_id: str) -> dict | None:
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if (
                sl == "Observation"
                and si == obs_id
                and rt == "IN_YEAR"
                and el == "Year"
            ):
                return self._w.node_store.get("Year", {}).get(ei)
        return None

    def _claim_entity_rel(self, claim_id: str, entity_id: str) -> str | None:
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if (
                sl == "Claim"
                and si == claim_id
                and ei == entity_id
                and el in self._entity_labels
            ):
                return rt
        return None

    def _paragraph_for_claim(self, claim_id: str) -> tuple[str | None, dict]:
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if rt == "HAS_CLAIM" and el == "Claim" and ei == claim_id:
                return si, self._w.node_store.get("Paragraph", {}).get(si, {})
        return None, {}

    def _section_for_paragraph(self, para_id: str) -> tuple[str | None, dict]:
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if rt == "HAS_PARAGRAPH" and el == "Paragraph" and ei == para_id:
                return si, self._w.node_store.get("Section", {}).get(si, {})
        return None, {}

    def _page_for_section(self, section_id: str) -> tuple[str | None, dict]:
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if rt == "HAS_SECTION" and el == "Section" and ei == section_id:
                return si, self._w.node_store.get("Page", {}).get(si, {})
        return None, {}

    def _doc_for_claim(self, claim_id: str) -> str | None:
        para_id, _ = self._paragraph_for_claim(claim_id)
        if para_id is None:
            return None
        section_id, _ = self._section_for_paragraph(para_id)
        if section_id is None:
            return None
        page_id, _ = self._page_for_section(section_id)
        if page_id is None:
            return None
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if rt == "HAS_PAGE" and el == "Page" and ei == page_id:
                return si
        return None

    def _measurements_for(self, obs_id: str | None, claim_id: str) -> list[dict]:
        m_ids: set[str] = set()
        measurements: list[dict] = []
        if obs_id:
            for sl, si, rt, el, ei, _ in self._w.rel_store:
                if (
                    sl == "Observation"
                    and si == obs_id
                    and rt == "HAS_MEASUREMENT"
                    and el == "Measurement"
                ):
                    m_ids.add(ei)
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if (
                sl == "Claim"
                and si == claim_id
                and rt == "HAS_MEASUREMENT"
                and el == "Measurement"
            ):
                m_ids.add(ei)
        for m_id in m_ids:
            m = self._w.node_store.get("Measurement", {}).get(m_id)
            if m:
                measurements.append(m)
        return measurements

    # ── row assembly ────────────────────────────────────────────────────────

    def _build_row(
        self,
        doc_id: str,
        claim_id: str,
        traversal_rel_type: str | None = None,
    ) -> dict[str, Any]:
        w = self._w
        d = w.node_store.get("Document", {}).get(doc_id, {})
        c = w.node_store.get("Claim", {}).get(claim_id, {})

        para_id, para = self._paragraph_for_claim(claim_id)
        section_id, sec = self._section_for_paragraph(para_id) if para_id else (None, {})
        _, pg = self._page_for_section(section_id) if section_id else (None, {})

        obs_ids = self._observations_for_claim(claim_id)
        obs_id = obs_ids[0] if obs_ids else None
        obs = w.node_store.get("Observation", {}).get(obs_id, {}) if obs_id else {}

        sp: dict = {}
        y: dict = {}
        if obs_id:
            for sl, si, rt, el, ei, _ in w.rel_store:
                if sl == "Observation" and si == obs_id:
                    if rt == "IN_YEAR" and el == "Year":
                        y = w.node_store.get("Year", {}).get(ei, {})
                    elif rt == "OF_SPECIES" and el == "Species":
                        sp = w.node_store.get("Species", {}).get(ei, {})

        measurements = self._measurements_for(obs_id, claim_id)

        row: dict[str, Any] = {
            "c": c,
            "d": d,
            "pg": pg,
            "sec": sec,
            "para": para,
            "obs": obs,
            "sp": sp,
            "y": y,
            "measurements": measurements,
        }
        if traversal_rel_type is not None:
            row["traversal_rel_type"] = traversal_rel_type
        return row
