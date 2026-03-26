"""In-memory query executor for testing the retrieval layer without Neo4j.

Implements the same ``run(cypher, params)`` interface as ``Neo4jQueryExecutor``
but executes against ``InMemoryGraphWriter``'s ``node_store`` / ``rel_store``.

The dispatcher uses Python ``is`` (object identity) to recognise the closed set
of module-level query-template constants.  Any other Cypher string (e.g. the
refuge-id startup query the assembler runs on ``__init__``) is handled by
``_generic_dispatch``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphrag_pipeline.core.graph.cypher import (
    CLAIM_TYPE_SCOPED_QUERY,
    ENTITY_ANCHORED_CLAIMS_QUERY,
    FULLTEXT_CLAIMS_QUERY,
    MULTI_ENTITY_CLAIMS_QUERY,
    PROVENANCE_CHAIN_QUERY,
    TEMPORAL_CLAIMS_QUERY,
    TEMPORAL_CLAIMS_QUERY_WITH_REFUGE,
)

if TYPE_CHECKING:
    from graphrag_pipeline.ingest.graph.writer import InMemoryGraphWriter

# Domain entity labels — mirrors DOMAIN_LABELS in writer.py without importing it
# (to avoid a retrieval→ingest dependency in production code paths).
_ENTITY_LABELS: frozenset[str] = frozenset(
    {
        "Refuge",
        "Place",
        "Person",
        "Organization",
        "Species",
        "Activity",
        "Period",
        "Habitat",
        "SurveyMethod",
    }
)


class InMemoryQueryExecutor:
    """Query executor backed by ``InMemoryGraphWriter`` stores.

    Drop-in replacement for ``Neo4jQueryExecutor`` in test contexts:
    pass this to ``ProvenanceContextAssembler`` to run the real retrieval
    logic against real pipeline-ingested data without a running database.
    """

    def __init__(self, writer: "InMemoryGraphWriter") -> None:
        self._w = writer

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
        if cypher is TEMPORAL_CLAIMS_QUERY_WITH_REFUGE:
            return self._temporal_with_refuge(p)
        if cypher is MULTI_ENTITY_CLAIMS_QUERY:
            return self._multi_entity(p)
        if cypher is CLAIM_TYPE_SCOPED_QUERY:
            return self._claim_type_scoped(p)
        if cypher is FULLTEXT_CLAIMS_QUERY:
            return self._fulltext(p)
        if cypher is PROVENANCE_CHAIN_QUERY:
            return self._provenance_chain(p)
        return self._generic_dispatch(cypher, p)

    # ── query handlers ──────────────────────────────────────────────────────

    def _entity_anchored(self, p: dict) -> list[dict]:
        entity_id = p["entity_id"]
        institution_id = p.get("institution_id", "turnbull")
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        limit = p.get("limit", 20)

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_id, permitted_levels):
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
        return rows[:limit]

    def _temporal(self, p: dict) -> list[dict]:
        institution_id = p.get("institution_id", "turnbull")
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        claim_types = p.get("claim_types")
        limit = p.get("limit", 20)

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_id, permitted_levels):
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

    def _temporal_with_refuge(self, p: dict) -> list[dict]:
        refuge_id = p["refuge_id"]
        institution_id = p.get("institution_id", "turnbull")
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        claim_types = p.get("claim_types")
        limit = p.get("limit", 20)

        valid = set(self._valid_docs(institution_id, permitted_levels))
        refuge_docs: set[str] = set()
        for sl, si, rt, el, ei, _ in self._w.rel_store:
            if rt == "ABOUT_REFUGE" and el == "Refuge" and ei == refuge_id and si in valid:
                refuge_docs.add(si)

        rows: list[dict] = []
        for doc_id in refuge_docs:
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
                    rows.append(self._build_row(doc_id, claim_id, "ABOUT_REFUGE"))
                    break

        rows.sort(
            key=lambda r: (
                (r.get("y") or {}).get("year") or 0,
                (r.get("c") or {}).get("extraction_confidence", 0.0),
            )
        )
        return rows[:limit]

    def _multi_entity(self, p: dict) -> list[dict]:
        entity_ids: list[str] = p["entity_ids"]
        institution_id = p.get("institution_id", "turnbull")
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        claim_types = p.get("claim_types")
        limit = p.get("limit", 20)
        entity_id_set = set(entity_ids)

        seen: set[str] = set()
        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_id, permitted_levels):
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
                        and el in _ENTITY_LABELS
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
        institution_id = p.get("institution_id", "turnbull")
        permitted_levels = p.get("permitted_levels", ["public"])
        year_min = p.get("year_min")
        year_max = p.get("year_max")
        limit = p.get("limit", 20)
        entity_id_set = set(entity_ids) if entity_ids else None

        rows: list[dict] = []
        for doc_id in self._valid_docs(institution_id, permitted_levels):
            run_id = self._latest_run_id(doc_id)
            if not run_id:
                continue
            for claim_id in self._claims_for_run(run_id, claim_types):
                if entity_id_set is not None:
                    if not any(
                        sl == "Claim"
                        and si == claim_id
                        and ei in entity_id_set
                        and el in _ENTITY_LABELS
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
        institution_id = p.get("institution_id", "turnbull")
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
            if not self._doc_is_valid(d, institution_id, permitted_levels):
                continue
            run_id = self._latest_run_id(doc_id)
            if claim_props.get("run_id") != run_id:
                continue
            rows.append(self._build_row(doc_id, claim_id))

        return rows[:limit]

    def _provenance_chain(self, p: dict) -> list[dict]:
        claim_id: str = p["claim_id"]
        institution_id = p.get("institution_id", "turnbull")
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
        if not self._doc_is_valid(d, institution_id, permitted_levels):
            return []
        run_id = self._latest_run_id(doc_id)
        if claim_props.get("run_id") != run_id:
            return []
        return [self._build_row(doc_id, claim_id)]

    def _generic_dispatch(self, cypher: str, p: dict) -> list[dict]:
        # Assembler __init__ startup query to find the refuge entity_id.
        if "ABOUT_REFUGE" in cypher and "eid" in cypher:
            for sl, si, rt, el, ei, _ in self._w.rel_store:
                if rt == "ABOUT_REFUGE" and el == "Refuge":
                    return [{"eid": ei}]
            return []
        # Stats queries and anything else — return empty (not needed for retrieval tests).
        return []

    # ── document filtering ──────────────────────────────────────────────────

    def _valid_docs(
        self, institution_id: str, permitted_levels: list[str]
    ) -> list[str]:
        result: list[str] = []
        for doc_id, props in self._w.node_store.get("Document", {}).items():
            if self._doc_is_valid(props, institution_id, permitted_levels):
                result.append(doc_id)
        return result

    def _doc_is_valid(
        self,
        props: dict,
        institution_id: str,
        permitted_levels: list[str],
    ) -> bool:
        if props.get("deleted_at") is not None:
            return False
        if props.get("institution_id") != institution_id:
            return False
        if props.get("access_level") not in permitted_levels:
            return False
        qs = props.get("quarantine_status")
        if qs is not None and qs != "active":
            return False
        return True

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
                and el in _ENTITY_LABELS
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
