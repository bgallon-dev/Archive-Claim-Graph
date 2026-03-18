"""Layer 1 — Entity Resolution Gateway.

Maps extracted surface forms from a query to canonical graph entity IDs using
the existing DictionaryFuzzyResolver infrastructure.  High-confidence matches
(REFERS_TO) are passed downstream to Cypher parameterisation; low-confidence
matches (POSSIBLY_REFERS_TO) are surfaced to the caller as clarification
candidates rather than silently accepted.
"""
from __future__ import annotations

from ..models import MentionRecord
from ..resolver import DictionaryFuzzyResolver, ResolutionPolicy, default_seed_entities, normalize_name
from .models import EntityContext, ResolvedEntity

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
    ) -> None:
        self._resolver = resolver or DictionaryFuzzyResolver(
            seed_entities=default_seed_entities(),
            policy=policy or ResolutionPolicy(),
        )

    def resolve(
        self,
        surface_forms: list[str],
        entity_hints: list[str] | None = None,
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

        _, resolutions = self._resolver.resolve(stubs)

        # Build a set of mention_ids that were resolved at all.
        resolved_mids: set[str] = set()
        resolved: list[ResolvedEntity] = []
        ambiguous: list[str] = []

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
                # POSSIBLY_REFERS_TO — flag as ambiguous for the caller.
                ambiguous.append(form_index.get(res.mention_id, ""))

        unresolved = [
            form_index[stub.mention_id]
            for stub in stubs
            if stub.mention_id not in resolved_mids
        ]

        return EntityContext(resolved=resolved, ambiguous=ambiguous, unresolved=unresolved)
