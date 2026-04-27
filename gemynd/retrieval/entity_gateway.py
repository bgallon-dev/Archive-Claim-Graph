"""Layer 1 — Entity Resolution Gateway.

Maps extracted surface forms from a query to canonical graph entity IDs using
the existing DictionaryFuzzyResolver infrastructure.  High-confidence matches
(REFERS_TO) are passed downstream to Cypher parameterisation; low-confidence
matches (POSSIBLY_REFERS_TO) are surfaced to the caller as clarification
candidates rather than silently accepted.
"""
from __future__ import annotations

import logging

from gemynd.core.models import MentionRecord
from gemynd.core.resolver import DictionaryFuzzyResolver, ResolutionPolicy, default_seed_entities, normalize_name
from gemynd.ingest.extractors.mention_extractor import ResolutionContext
from .executor import Neo4jQueryExecutor
from .models import EntityContext, ResolvedEntity

_log = logging.getLogger(__name__)

# Sentinel values for stub MentionRecord fields that are not meaningful at
# query time (the gateway does not touch the graph for mention data).
_STUB_RUN_ID = "query"
_STUB_PARAGRAPH_ID = "query"


class EntityResolutionGateway:
    """Resolve query-time entity surface forms to canonical entity IDs.

    Parameters
    ----------
    resolver:
        If *None*, a ``DictionaryFuzzyResolver`` is constructed from the
        default seed entity vocabulary (CSV on disk).  Pass an explicit
        instance to override the vocabulary (e.g. in tests).
    policy:
        Resolution policy thresholds.  Defaults to the standard
        ``ResolutionPolicy`` (refers_to=0.85, maybe=0.65, gap=0.05).
    """

    def __init__(
        self,
        resolver: DictionaryFuzzyResolver | None = None,
        policy: ResolutionPolicy | None = None,
        *,
        executor: Neo4jQueryExecutor | None = None,
    ) -> None:
        self._resolver = resolver or DictionaryFuzzyResolver(
            seed_entities=default_seed_entities(),
            policy=policy or ResolutionPolicy(),
        )
        self._executor = executor
        # Per-gateway cache of entity_ids known to exist in the live graph.
        # Populated lazily as resolve() encounters new ids; survives the
        # process lifetime of the gateway (bounded by unique seed-entity count).
        self._live_id_cache: set[str] = set()
        self._missing_id_cache: set[str] = set()

    def _filter_live_ids(self, entity_ids: list[str]) -> set[str]:
        """Return the subset of *entity_ids* that exist in the live graph.

        Results are cached so repeated queries against the same anchor set
        do not re-issue the Cypher probe. Callers without an executor (e.g.
        tests) get all ids back unchanged.
        """
        if self._executor is None or not entity_ids:
            return set(entity_ids)
        unknown = [
            eid for eid in entity_ids
            if eid not in self._live_id_cache and eid not in self._missing_id_cache
        ]
        if unknown:
            try:
                rows = self._executor.run(
                    "MATCH (e) WHERE e.entity_id IN $ids RETURN e.entity_id AS eid",
                    {"ids": unknown},
                )
                present = {r.get("eid") for r in rows if r.get("eid")}
                for eid in unknown:
                    if eid in present:
                        self._live_id_cache.add(eid)
                    else:
                        self._missing_id_cache.add(eid)
            except Exception as exc:
                # If the probe fails we fall open rather than silently dropping
                # every entity — the executor-less code path above does the
                # same, and a transient Neo4j outage should not hide results.
                _log.warning("live-id probe failed; admitting all ids: %s", exc)
                return set(entity_ids)
        return {eid for eid in entity_ids if eid in self._live_id_cache}

    def resolve(
        self,
        surface_forms: list[str],
        entity_hints: list[str] | None = None,
        *,
        inferred_claim_types: list[str] | None = None,
    ) -> EntityContext:
        """Resolve *surface_forms* (and optional *entity_hints*) to graph IDs.

        Parameters
        ----------
        surface_forms:
            Entity surface forms extracted from the query text by the classifier.
        entity_hints:
            Additional surface forms supplied by the API caller.
        """
        all_forms = list(surface_forms)
        for hint in entity_hints or []:
            if hint not in all_forms:
                all_forms.append(hint)

        if not all_forms:
            return EntityContext()

        # Build stub MentionRecord objects (the resolver only needs surface_form
        # and normalized_form; all structural fields are stubs).
        stubs: list[MentionRecord] = []
        form_index: dict[str, str] = {}  # mention_id -> surface_form
        for idx, form in enumerate(all_forms):
            mid = f"query-mention-{idx}"
            stubs.append(
                MentionRecord(
                    mention_id=mid,
                    run_id=_STUB_RUN_ID,
                    paragraph_id=_STUB_PARAGRAPH_ID,
                    surface_form=form,
                    normalized_form=normalize_name(form),
                    start_offset=0,
                    end_offset=len(form),
                    detection_confidence=1.0,
                    ocr_suspect=False,
                )
            )
            form_index[mid] = form

        # Build a synthetic ResolutionContext so the resolver can apply the A1
        # claim-type adjustment (preferred_entity_types boost) at query time.
        # All query mentions share a single paragraph_id so one context suffices.
        contexts: dict[str, ResolutionContext] | None = None
        if inferred_claim_types:
            contexts = {
                _STUB_PARAGRAPH_ID: ResolutionContext(
                    paragraph_id=_STUB_PARAGRAPH_ID,
                    claim_types=sorted(set(inferred_claim_types)),
                ),
            }

        _, resolutions = self._resolver.resolve(stubs, contexts=contexts)

        # Build a set of mention_ids that were resolved at all.
        resolved_mids: set[str] = set()
        resolved: list[ResolvedEntity] = []
        ambiguous: list[str] = []
        maybe_resolved: list[ResolvedEntity] = []

        # Collect entity metadata for matched entity IDs.
        entity_lookup = {e.entity_id: e for e in self._resolver._seed_entities}

        for res in resolutions:
            resolved_mids.add(res.mention_id)
            entity = entity_lookup.get(res.entity_id)
            if entity is None:
                continue
            re_obj = ResolvedEntity(
                surface_form=form_index.get(res.mention_id, ""),
                entity_id=res.entity_id,
                entity_type=entity.entity_type,
                resolution_confidence=res.match_score,
                resolution_relation=res.relation_type,
            )
            if res.relation_type == "REFERS_TO":
                resolved.append(re_obj)
            else:
                # POSSIBLY_REFERS_TO — flag as ambiguous for the caller and
                # retain the resolved id so the retrieval plan can opt it into
                # multi-entity queries when a strong anchor already exists.
                ambiguous.append(form_index.get(res.mention_id, ""))
                maybe_resolved.append(re_obj)

        unresolved = [
            form_index[stub.mention_id]
            for stub in stubs
            if stub.mention_id not in resolved_mids
        ]

        # Deduplicate resolved by entity_id, keeping highest confidence
        seen: dict[str, ResolvedEntity] = {}
        for re_obj in resolved:
            existing = seen.get(re_obj.entity_id)
            if existing is None or re_obj.resolution_confidence > existing.resolution_confidence:
                seen[re_obj.entity_id] = re_obj
        resolved = list(seen.values())

        # Drop entity_ids that no longer exist in the live graph. This guards
        # against stale seed_entities.csv ids that would otherwise produce
        # silent zero-result queries downstream.
        if self._executor is not None and resolved:
            live = self._filter_live_ids([r.entity_id for r in resolved])
            dropped = [r for r in resolved if r.entity_id not in live]
            if dropped:
                _log.debug(
                    "gateway dropped %d stale entity_ids: %s",
                    len(dropped),
                    [(r.surface_form, r.entity_id) for r in dropped],
                )
                unresolved.extend(r.surface_form for r in dropped)
                resolved = [r for r in resolved if r.entity_id in live]

        # Remove from ambiguous any surface form already covered by a resolved entity
        resolved_surface_forms = {r.surface_form.lower() for r in resolved}
        resolved_normalized = {
            token
            for r in resolved
            for token in r.surface_form.lower().split()
        }
        ambiguous = [
            f for f in ambiguous
            if f.lower() not in resolved_surface_forms
            and f.lower() not in resolved_normalized
        ]

        # Dedup maybe_resolved by entity_id (highest-confidence wins) and drop
        # any that were promoted to REFERS_TO resolution.
        resolved_ids = {r.entity_id for r in resolved}
        maybe_seen: dict[str, ResolvedEntity] = {}
        for m in maybe_resolved:
            if m.entity_id in resolved_ids:
                continue
            existing = maybe_seen.get(m.entity_id)
            if existing is None or m.resolution_confidence > existing.resolution_confidence:
                maybe_seen[m.entity_id] = m
        maybe_resolved = list(maybe_seen.values())

        return EntityContext(
            resolved=resolved,
            ambiguous=ambiguous,
            unresolved=unresolved,
            maybe_resolved=maybe_resolved,
        )
