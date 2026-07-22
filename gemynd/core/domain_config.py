"""Unified domain configuration object.

``DomainConfig`` aggregates all domain resources (seed entities, patterns,
measurement units, OCR corrections, derivation registry, etc.) into a single
dataclass that can be loaded once and threaded through the extraction pipeline.

``ClaimDerivationSpec`` is a frozen typed wrapper for derivation registry
entries, replacing raw dicts produced by ``load_derivation_registry()``.

Usage::

    from gemynd.core.domain_config import load_domain_config

    config = load_domain_config(resources_dir)
    extractor = RuleBasedClaimExtractor(config=config)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gemynd.core.models import EntityRecord

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClaimDerivationSpec:
    """Typed, frozen representation of one derivation registry entry.

    Replaces the raw ``dict`` returned by ``load_derivation_registry()`` so
    that ``build_derivation_contexts()`` can accept either form.
    """

    observation_type: str | None
    event_type: str | None
    required_entities: tuple[str, ...]
    optional_entities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoleResolver:
    """How to resolve a role entity from a claim's links.

    ``source`` is either ``"entity_links"`` (match ``claim_entity_links``
    by relation type) or ``"location_links"`` (match
    ``claim_location_links`` by ``entity_type``).
    """

    source: str
    relations: frozenset[str] = frozenset()
    entity_type: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimLinkRole:
    """One rule-based claim-link role (e.g. Species, Habitat, Location).

    Drives the config-driven loop in ``RuleBasedClaimExtractor`` that replaced
    the hardcoded per-entity-type blocks. Role order in the domain's
    ``claim_link_roles`` block is **significant** — cross-role span collisions
    are deduplicated by relation priority with an offset-stable sort, so the
    order must match the legacy block order to preserve output.

    - ``entity_type``    : the entity-type hint emitted, and the ``ROLE_POLICY``
                           lookup key ``(claim_type, entity_type)``.
    - ``seed_types``     : seed entity types whose names are scanned as terms.
    - ``default_relation``: relation when ``ROLE_POLICY`` has no entry; for a
                           location role this is the non-occurrence relation.
    - ``fallback_relation`` / ``fallback_claim_types``: when ``default_relation``
                           is ``None``, emit ``fallback_relation`` only for the
                           listed claim types (else the match is skipped).
    - ``is_location_role``: location roles use ``claim_location_relation`` when
                           the term sits in occurrence context, else
                           ``default_relation`` — they do not consult ROLE_POLICY.
    - ``hint_substring_overrides``: ordered ``(substring, label)`` pairs; the
                           first substring found in the matched surface overrides
                           the emitted entity-type hint (reproduces the legacy
                           ``"Refuge" if "refuge" in surface else "Place"``).
    """

    entity_type: str
    seed_types: tuple[str, ...]
    default_relation: str | None
    fallback_relation: str | None = None
    fallback_claim_types: frozenset[str] = frozenset()
    is_location_role: bool = False
    hint_substring_overrides: tuple[tuple[str, str], ...] = ()


def _build_claim_link_roles(
    schema: dict[str, Any],
) -> tuple[tuple[ClaimLinkRole, ...], tuple[str, ...], tuple[str, ...]]:
    """Parse the ``claim_link_roles`` block from ``domain_schema.yaml``.

    Returns ``(roles, occurrence_verbs, subject_trigger_verbs)``. A domain
    without the block gets empty tuples — no rule-based claim links and no
    occurrence/subject verb detection (never Turnbull's role taxonomy).
    """
    block = schema.get("claim_link_roles") or {}
    roles: list[ClaimLinkRole] = []
    for r in block.get("roles") or []:
        entity_type = str(r["entity_type"])
        seed_types = tuple(str(t) for t in (r.get("seed_types") or [entity_type]))
        default_relation = r.get("default_relation")
        fallback_relation = r.get("fallback_relation")
        roles.append(
            ClaimLinkRole(
                entity_type=entity_type,
                seed_types=seed_types,
                default_relation=(str(default_relation) if default_relation is not None else None),
                fallback_relation=(str(fallback_relation) if fallback_relation is not None else None),
                fallback_claim_types=frozenset(str(c) for c in (r.get("fallback_claim_types") or [])),
                is_location_role=bool(r.get("is_location_role", False)),
                hint_substring_overrides=tuple(
                    (str(k).lower(), str(v)) for k, v in (r.get("hint_substring_overrides") or {}).items()
                ),
            )
        )
    occurrence_verbs = tuple(str(v) for v in (block.get("occurrence_verbs") or []))
    subject_verbs = tuple(str(v) for v in (block.get("subject_trigger_verbs") or []))
    return tuple(roles), occurrence_verbs, subject_verbs


@dataclass(slots=True)
class DomainConfig:
    """All domain resources bundled into one place.

    Loaded by :func:`load_domain_config` and optionally passed to extractor
    and resolver constructors.  When ``config`` is provided to a constructor
    it replaces the module-level cached loaders for that instance.
    """

    seed_entities: list[EntityRecord]
    claim_type_patterns: list[tuple[str, re.Pattern, float]]
    claim_role_policy: dict[tuple[str, str], str]
    measurement_units: dict[str, tuple[str, str]]
    measurement_species: dict[str, Any]
    ocr_corrections: frozenset[str]
    ocr_correction_map: dict[str, str]
    negative_lexicon: frozenset[str]
    preferred_entity_types: dict[str, list[str]]
    compatibility_matrix: dict[str, dict[str, str]]
    derivation_registry: dict[str, ClaimDerivationSpec]
    domain_anchor: dict[str, Any] | None
    year_validation: dict[str, Any] | None
    synthesis_context: str

    # Phase 5 — externalized domain schema fields
    claim_entity_relation_precedence: tuple[str, ...]
    claim_entity_relations: frozenset[str]
    relation_to_entity_type_hints: dict[str, frozenset[str]]
    claim_location_relation: str
    entity_labels: frozenset[str]
    legacy_renames: dict[str, str]
    allowed_claim_types: frozenset[str]
    observation_eligible_types: frozenset[str]
    event_eligible_types: frozenset[str]
    concept_rules: list[tuple[str, frozenset[str], re.Pattern, float]]
    query_intent_to_claim_types: dict[str, list[str]]

    # Claim sentence validator — domain-specific finite-verb pattern
    validator_verb_re: re.Pattern[str] | None = None
    # Heading detection override — None means use default; set to a compiled
    # pattern to customise, or to a pattern that never matches to disable.
    validator_heading_re: re.Pattern[str] | None = None

    # Retrieval-layer fields
    institution_id: str = ""
    expected_claim_shares: dict[str, float] = field(default_factory=dict)
    # Maps ``entity_type`` → ``{"builder": method_name, "param": keyword_arg}``
    # so the web layer can dispatch analytical queries without hardcoded
    # species/habitat checks. Loaded from ``analytical_queries`` block in
    # ``domain_profile.yaml``. Empty dict means the domain has no analytical
    # path and the hybrid bucket collapses to the conversational path.
    analytical_queries: dict[str, dict[str, str]] = field(default_factory=dict)
    # Opt-in flag for the text-to-Cypher LLM fallback (plan B6). Default off
    # so domains have to consciously accept the additional API cost and the
    # validator-bound trust model. Read from
    # ``enable_llm_cypher`` in ``domain_profile.yaml``.
    enable_llm_cypher: bool = False

    # When true, mention→entity resolution rejects candidates whose entity_type
    # contradicts the type inferred for the mention. Off by default because it
    # is only safe for domains whose type rules decline to guess: a domain whose
    # rules end in a `fallback` (Turnbull types everything Person) would have
    # every Refuge/Species/Place resolution blocked. Read from
    # ``resolution_type_filter`` in ``domain_profile.yaml``.
    resolution_type_filter: bool = False

    # Phase 8 — derivation role externalization.  Role name → resolver
    # spec (how to find the entity from the claim's links), plus two
    # edge maps keyed by role name → (neo4j label, edge type).  Two
    # separate edge maps because the writer uses different edges per
    # record type (e.g. OF_SPECIES from Observation, INVOLVED_SPECIES
    # from Event).
    role_resolution: dict[str, RoleResolver] = field(default_factory=dict)
    observation_role_edges: dict[str, tuple[str, str]] = field(default_factory=dict)
    event_role_edges: dict[str, tuple[str, str]] = field(default_factory=dict)

    # Entity-type inference cascade (corporate/gov patterns). Consulted
    # only after seed-vocabulary lookup; see ``entity_type_inference``.
    entity_type_rules: list[dict[str, Any]] = field(default_factory=list)

    # Terms the discovery rules declare are not entities. Mentions of these
    # never resolve — see load_non_entity_terms.
    non_entity_terms: frozenset[str] = frozenset()

    # Optional human-readable label and topic hints for the chat welcome
    # message. When unset, the welcome falls back to ``institution_id`` and
    # an empty topic list. Loader is free to populate these from the
    # corpus's ``domain_profile.yaml`` later without breaking existing
    # construction sites.
    synthesis_context_label: str | None = None
    welcome_topic_hints: list[str] = field(default_factory=list)

    # Measurement scan patterns / naming rules (relocated from hardcoded
    # constants in measurement_extractor.py). All default to empty so a domain
    # without ``measurement_patterns.yaml`` gets no direct-scan patterns and
    # identity naming — never Turnbull's measurement vocabulary.
    measurement_pattern_specs: list[tuple[re.Pattern[str], str, str | None]] = field(default_factory=list)
    measurement_unit_right_re: re.Pattern[str] | None = None
    measurement_count_claim_types: frozenset[str] = frozenset()
    measurement_acre_specialization: dict[str, Any] = field(default_factory=dict)
    measurement_money: dict[str, Any] = field(default_factory=dict)

    # Rule-based claim-link roles (relocated from the hardcoded per-entity-type
    # blocks in claim_extractor.py). Empty for a domain that declares no
    # ``claim_link_roles`` block — its rule-based claim-link layer is then a
    # no-op rather than inheriting Turnbull's Species/Habitat/Location taxonomy.
    claim_link_roles: tuple[ClaimLinkRole, ...] = ()
    claim_link_occurrence_verbs: tuple[str, ...] = ()
    claim_link_subject_verbs: tuple[str, ...] = ()

    # Source-parser profile (relocated from hardcoded defaults in
    # source_parser.py). Keys: ``default_doc_type`` (used when a document's
    # payload has no doc_type), ``untitled_title`` (title fallback), and
    # ``filename_inference`` (list of named rules, e.g. ``loc_sequence``).
    # Empty dict → neutral parsing (doc_type "document", no filename inference),
    # never Turnbull's "narrative_report" / LoC heuristics.
    source_profile: dict[str, Any] = field(default_factory=dict)

    # Categorial label projection: domain entity_type -> shared categorial label
    # the graph writer also stamps on the node (e.g. {"Refuge": "Place"}).
    # Replaces the hardcoded Refuge->Place coercion in the writer. Empty for a
    # domain with no such projections.
    categorial_label_map: dict[str, str] = field(default_factory=dict)

    # Entity discovery / minting. When a claim references a proper-noun entity
    # (person, place, organization) absent from the seed gazetteer, the resolver
    # drops the link as NO_RESOLVED_MENTION because there is no entity to attach
    # to. With minting enabled, such surfaces are turned into freshly-minted
    # entities so the claim links survive. Off by default (empty dict) so a
    # domain with a comprehensive seed gazetteer — Turnbull — never mints, and
    # its output stays byte-identical. Keys (all optional):
    #   ``enabled`` (bool), ``min_surface_length`` (int, default 4),
    #   ``default_entity_type`` (str, default "Person"),
    #   ``location_entity_type`` (str, default "Place").
    entity_minting: dict[str, Any] = field(default_factory=dict)

    # Domain-aware reference vocabulary for the spelling-review queue (seed
    # names + measurement words + claim-type pattern terms). Lets the spelling
    # reviewer judge "known domain term" against the domain's own vocabulary
    # instead of Turnbull's. Empty frozenset when unavailable.
    spelling_reference_terms: frozenset[str] = frozenset()

    # Review builder-repair detector policy: which claim types expect a focus
    # relation / a location link / are method-overtrigger candidates, and the
    # entity types that satisfy them. Parsed verbatim from the ``builder_repair``
    # block in ``domain_schema.yaml``. Empty dict → the detector proposes no
    # focus/location/method repairs for this domain (it does NOT fall back to
    # Turnbull's wildlife policy; that fallback exists only when no config at
    # all is passed to the detector). Keys (all optional): ``focus_relation``,
    # ``focus_candidate_entity_types``, ``focus_expecting_types``,
    # ``location_candidate_entity_types``, ``location_expecting_types``,
    # ``method_relation``, ``method_overtrigger_types``.
    builder_repair: dict[str, Any] = field(default_factory=dict)

    # Review junk-mention detector: domain-specific header regexes appended to
    # the universal structural set (page numbers, bare digits). Parsed from the
    # ``junk_mention`` block in ``domain_schema.yaml``. Empty → only the
    # structural headers apply, never Turnbull's institutional headers ("refuge
    # manager", "fish and wildlife service", …), which move to Turnbull's config.
    junk_mention_filters: dict[str, Any] = field(default_factory=dict)

    # Query-intent classifier surface-form maps (from ``query_intent.yaml``).
    # ``doc_type_aliases``: user phrase → d.doc_type; ``series_aliases``: phrase
    # → d.series; ``tier1_type_keywords``: word → entity label for the
    # frequency-ranking fast path. All empty by default; a domain declaring none
    # leaves the classifier on its module-level defaults (retrieval unions these
    # across all registered corpora). Lets a new corpus own its query vocabulary
    # instead of inheriting the hardcoded BLM/GLO/Spokane grab-bag.
    doc_type_aliases: dict[str, str] = field(default_factory=dict)
    series_aliases: dict[str, str] = field(default_factory=dict)
    tier1_type_keywords: dict[str, str] = field(default_factory=dict)

    # Synthesis-prompt RETRIEVED_VIA semantics: relation label (or "A / B"
    # group) → one-line meaning for the LLM. Parsed from the ``relation_semantics``
    # block in ``domain_schema.yaml``. Empty → the synthesis prompt keeps its
    # default wildlife/civic block; a domain declaring its own relations here
    # replaces that block so the prompt never describes SPECIES_FOCUS etc. to a
    # non-wildlife corpus.
    relation_semantics: dict[str, str] = field(default_factory=dict)

    # EAD 2002 export: entity_type → EAD access-point element (persname,
    # corpname, geogname, subject, …). Parsed from the ``ead_tag_map`` block in
    # ``domain_schema.yaml``; overlaid on the module default in the exporter.
    # Empty → the default map applies (a foreign entity type falls back to
    # "subject").
    ead_tag_map: dict[str, str] = field(default_factory=dict)

    @property
    def extraction_stopwords(self) -> frozenset[str]:
        """Query-time stopwords for entity surface-form extraction.

        Sourced exclusively from the explicit ``extraction_stopwords`` list
        under ``document_anchor`` in ``domain_profile.yaml``.

        ``title_keywords`` and ``normalized_form`` are deliberately **not**
        consulted here, despite both naming the anchor entity: those fields
        drive ingest-time document anchoring and ID generation respectively,
        and pulling them into query stopwords causes catastrophic resolution
        failures in composite (multi-corpus) deployments. Concretely, if the
        Turnbull anchor's ``normalized_form: turnbull`` were stopworded at
        query time, a query like *"What entities are associated with turnbull
        refuge"* would have no surface forms left for the resolver, the
        gateway would return an empty entity set, and the cascade would
        fall through to ``claim_type_scoped`` against the newspaper corpus —
        returning all-Spokane content instead of Refuge content.

        Single-corpus deployments that previously relied on the implicit
        derivation can re-add the anchor name explicitly under
        ``document_anchor.extraction_stopwords`` if they need that behavior.
        """
        words: set[str] = set()
        if self.domain_anchor:
            for sw in self.domain_anchor.get("extraction_stopwords", []):
                words.add(sw.lower())
        return frozenset(words)

    @property
    def anchor_entity_id(self) -> str | None:
        """Entity ID of the primary anchor from seed_entities."""
        if not self.domain_anchor:
            return None
        anchor_type = self.domain_anchor.get("entity_type")
        anchor_norm = self.domain_anchor.get("normalized_form", "").lower()
        for e in self.seed_entities:
            if e.entity_type == anchor_type and anchor_norm in e.normalized_form:
                return e.entity_id
        return None

    @property
    def anchor_entity_type(self) -> str | None:
        """Entity type label for the domain anchor (e.g. 'Refuge')."""
        if self.domain_anchor:
            return self.domain_anchor.get("entity_type")
        return None

    @property
    def anchor_relation(self) -> str | None:
        """Document→anchor relation type (e.g. 'ABOUT_REFUGE').

        Read from ``document_anchor.relation`` in ``domain_profile.yaml``.
        ``None`` when the domain has no anchor or the relation is unset.
        """
        if self.domain_anchor:
            return self.domain_anchor.get("relation")
        return None

    @property
    def claim_entity_relation_cypher(self) -> str:
        """Cypher IN-list literal for claim-entity relation types."""
        return ", ".join(f"'{r}'" for r in self.claim_entity_relation_precedence)

    @property
    def extractor_claim_link_relations(self) -> frozenset[str]:
        """All relation types valid in extractor claim links."""
        return frozenset(self.claim_entity_relations | {self.claim_location_relation})


@dataclass(slots=True)
class CompositeDomainConfig:
    """Multi-corpus view over a list of :class:`DomainConfig` objects.

    Mergeable fields (claim types, stopwords, intent maps, concept rules,
    entity labels, relation precedence) are unioned across all corpora so
    the retrieval layer can classify, resolve, and query any corpus without
    knowing which one a claim came from.

    Non-mergeable fields (anchor, expected_claim_shares, synthesis_context)
    stay keyed by ``institution_id`` and are accessed via the per-corpus
    helper methods.
    """

    members: list[DomainConfig]

    @property
    def institution_ids(self) -> list[str]:
        return [c.institution_id for c in self.members if c.institution_id]

    @property
    def default_member(self) -> DomainConfig:
        """Return the first member — used as a fallback for callers that
        still expect a single-corpus view (e.g. single-entity search UIs).
        """
        return self.members[0]

    # ---- unioned fields --------------------------------------------------

    @property
    def allowed_claim_types(self) -> frozenset[str]:
        result: set[str] = set()
        for c in self.members:
            result |= c.allowed_claim_types
        return frozenset(result)

    @property
    def entity_labels(self) -> frozenset[str]:
        result: set[str] = set()
        for c in self.members:
            result |= c.entity_labels
        return frozenset(result)

    @property
    def extraction_stopwords(self) -> frozenset[str]:
        result: set[str] = set()
        for c in self.members:
            result |= c.extraction_stopwords
        return frozenset(result)

    @property
    def query_intent_to_claim_types(self) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for c in self.members:
            for keyword, claim_types in c.query_intent_to_claim_types.items():
                existing = merged.setdefault(keyword, [])
                for ct in claim_types:
                    if ct not in existing:
                        existing.append(ct)
        return merged

    @property
    def claim_entity_relation_precedence(self) -> tuple[str, ...]:
        # Preserve order from the first corpus, then append any relations
        # from later corpora that weren't already present.
        seen: list[str] = []
        for c in self.members:
            for rel in c.claim_entity_relation_precedence:
                if rel not in seen:
                    seen.append(rel)
        return tuple(seen)

    @property
    def claim_entity_relations(self) -> frozenset[str]:
        return frozenset(self.claim_entity_relation_precedence)

    @property
    def seed_entities(self) -> list[EntityRecord]:
        """Union of seed entities across all member corpora, deduped by entity_id.

        Retrieval-layer components (EntityResolutionGateway, DictionaryFuzzyResolver)
        need vocabulary spanning every registered corpus so that cross-corpus
        surface forms resolve at query time.
        """
        seen: dict[str, EntityRecord] = {}
        for c in self.members:
            for e in c.seed_entities:
                if e.entity_id not in seen:
                    seen[e.entity_id] = e
        return list(seen.values())

    @property
    def enable_llm_cypher(self) -> bool:
        """True iff any member corpus opts in to the LLM-Cypher fallback."""
        return any(c.enable_llm_cypher for c in self.members)

    @property
    def entity_type_rules(self) -> list[dict[str, Any]]:
        """Members' rule cascades concatenated in member order.

        First match wins within the cascade, so a member's rules keep their
        relative order and earlier corpora take precedence — the same rule the
        single-corpus loader follows.
        """
        merged: list[dict[str, Any]] = []
        for member in self.members:
            merged.extend(member.entity_type_rules)
        return merged

    @property
    def non_entity_terms(self) -> frozenset[str]:
        """Union of members' non-entity vocabularies.

        Unions rather than intersects: a word one corpus knows is not a name is
        not a name in the composite either, and the cost of over-excluding one
        common word is far lower than letting it collect resolutions.
        """
        terms: frozenset[str] = frozenset()
        for member in self.members:
            terms |= member.non_entity_terms
        return terms

    @property
    def resolution_type_filter(self) -> bool:
        """True only if EVERY member opts in.

        Unlike the other flags this merges with ``all``: the filter drops
        resolutions, and one guessing member's rules (a `fallback` type) would
        suppress correct matches across the whole composite.
        """
        return bool(self.members) and all(c.resolution_type_filter for c in self.members)

    @property
    def analytical_queries(self) -> dict[str, dict[str, str]]:
        """Union of per-corpus analytical_queries maps.

        First corpus to declare a dispatch for a given entity_type wins;
        later corpora with a different builder for the same type are
        ignored (this keeps dispatch deterministic when two corpora share
        an entity_type but have mismatched builder methods).
        """
        merged: dict[str, dict[str, str]] = {}
        for c in self.members:
            for entity_type, spec in c.analytical_queries.items():
                if entity_type not in merged:
                    merged[entity_type] = dict(spec)
        return merged

    @property
    def preferred_entity_types(self) -> dict[str, list[str]]:
        """Union of preferred_entity_types maps across all member corpora.

        For each claim_type, walk every member corpus's list. New entity_types
        are inserted into the merged list relative to the **nearest shared
        anchor** — the next entity_type in the source list that already exists
        in the merged list. The new type is placed immediately before that
        anchor, mirroring the source corpus's stated relative preference. If
        no shared anchor follows in the source list, the new type is appended
        at the end.

        Why anchor-relative insertion rather than append-at-end: the latter
        was silently lossy when a later corpus's list was a better-ordered
        superset of an earlier one (e.g. newspaper's
        ``[Refuge, Place, Organization, Person]`` vs Turnbull's
        ``[Place, Organization, Person]`` for the shared ``turnbull_prehistory``
        claim_type — Refuge would land in the least-preferred slot and the
        resolver would demote the Refuge candidate below REFERS_TO threshold).
        Anchoring on shared types lets a later corpus correctly raise a
        newly-introduced type into a prominent position when the surrounding
        ordering agrees, while preserving first-corpus-wins for fully
        disjoint additions (which simply append).
        """
        merged: dict[str, list[str]] = {}
        for c in self.members:
            for claim_type, types in c.preferred_entity_types.items():
                existing = merged.setdefault(claim_type, [])
                for src_pos, et in enumerate(types):
                    if et in existing:
                        continue
                    insert_at = len(existing)
                    for after_pos in range(src_pos + 1, len(types)):
                        anchor = types[after_pos]
                        if anchor in existing:
                            insert_at = existing.index(anchor)
                            break
                    existing.insert(insert_at, et)
        return merged

    # ---- per-corpus helpers ---------------------------------------------

    def expected_claim_shares(self) -> dict[str, dict[str, float]]:
        """Return a mapping ``institution_id -> expected share dict``."""
        return {
            c.institution_id: dict(c.expected_claim_shares)
            for c in self.members
            if c.institution_id
        }

    def anchor_for(self, institution_id: str) -> tuple[str | None, str | None, str | None]:
        """Return ``(entity_id, entity_type, relation)`` for a corpus, or ``(None, None, None)``."""
        for c in self.members:
            if c.institution_id == institution_id:
                return (c.anchor_entity_id, c.anchor_entity_type, c.anchor_relation)
        return (None, None, None)

    def anchors(self) -> dict[str, tuple[str, str, str]]:
        """Return a dict of ``institution_id -> (entity_id, entity_type, relation)``
        for corpora that have a fully-specified anchor. Corpora with a null
        or partial anchor are omitted.
        """
        out: dict[str, tuple[str, str, str]] = {}
        for c in self.members:
            if not c.institution_id:
                continue
            aid, atype, arel = c.anchor_entity_id, c.anchor_entity_type, c.anchor_relation
            if aid and atype and arel:
                out[c.institution_id] = (aid, atype, arel)
        return out

    def synthesis_context(self) -> str:
        """Return a merged synthesis context string covering all member corpora."""
        parts = [c.synthesis_context for c in self.members if c.synthesis_context]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return " / ".join(parts)


def merge_domain_configs(configs: list[DomainConfig]) -> CompositeDomainConfig:
    """Bundle one or more :class:`DomainConfig` instances into a composite.

    The result keeps references to the originals — no field copies are made,
    so mutations to the member configs are visible through the composite.
    """
    if not configs:
        raise ValueError("merge_domain_configs requires at least one DomainConfig")
    return CompositeDomainConfig(members=list(configs))


def load_all_registered_corpora(
    registry_path: Path | None = None,
) -> list[DomainConfig]:
    """Load every corpus listed in ``data/corpus_registry.yaml`` as a
    :class:`DomainConfig`. Corpora whose resources directory cannot be found
    are skipped with a warning so partial-deployment environments still boot.
    """
    from gemynd.shared.resource_loader import (
        load_corpus_registry,
        resolve_corpus_resources_dir,
    )

    entries = load_corpus_registry(registry_path)
    result: list[DomainConfig] = []
    for entry in entries:
        try:
            rdir = resolve_corpus_resources_dir(entry, registry_path)
        except ValueError as e:
            _log.warning("corpus_registry entry skipped: %s", e)
            continue
        if not rdir.exists():
            _log.warning(
                "corpus_registry entry %r resources_dir %s does not exist — skipping",
                entry.get("corpus_id"),
                rdir,
            )
            continue
        try:
            result.append(load_domain_config(rdir))
        except Exception as e:
            _log.warning(
                "failed to load domain config for corpus %r from %s: %s",
                entry.get("corpus_id"),
                rdir,
                e,
            )
    return result


def load_domain_config(resources_dir: Path | None = None) -> DomainConfig:
    """Load every domain resource file and return a validated :class:`DomainConfig`.

    A run MUST name its domain. Passing ``None`` raises rather than silently
    loading the bundled Turnbull corpus — the silent-domain fallback is the
    exact "semiotic substitution" this engine is meant to avoid. Callers that
    genuinely want Turnbull pass
    :data:`gemynd.shared.resource_loader.TURNBULL_RESOURCES_DIR` explicitly.

    All imports are deferred to avoid circular-import issues at module load
    time.  The function is safe to call multiple times; caching at call-site
    is left to the caller.
    """
    if resources_dir is None:
        raise ValueError(
            "load_domain_config requires an explicit resources_dir; the silent "
            "Turnbull default has been removed. Pass a domain resources directory "
            "(e.g. resource_loader.TURNBULL_RESOURCES_DIR for the Turnbull corpus)."
        )

    from gemynd.core.resolver import default_seed_entities
    from gemynd.shared.resource_loader import (
        load_claim_relation_compatibility,
        load_claim_role_policy,
        load_claim_type_patterns,
        load_concept_rules,
        load_derivation_registry,
        load_domain_profile,
        load_domain_schema,
        load_entity_type_rules,
        load_non_entity_terms,
        load_measurement_patterns,
        load_measurement_species,
        load_measurement_units,
        load_negative_entities,
        load_ocr_correction_map,
        load_ocr_corrections,
        load_query_intent,
        load_query_intent_maps,
        load_spelling_reference_terms,
        load_validator_verbs,
    )

    profile = load_domain_profile(resources_dir)
    compat = load_claim_relation_compatibility(resources_dir)

    raw_registry = load_derivation_registry(resources_dir)
    registry: dict[str, ClaimDerivationSpec] = {
        ct: ClaimDerivationSpec(
            observation_type=entry.get("observation_type"),
            event_type=entry.get("event_type"),
            required_entities=tuple(entry.get("required_entities") or []),
            optional_entities=tuple(entry.get("optional_entities") or []),
        )
        for ct, entry in raw_registry.items()
    }

    # Domain schema: graph vocabulary (relation types, entity labels, etc.)
    schema = load_domain_schema(resources_dir)
    claim_type_patterns = load_claim_type_patterns(resources_dir)

    relation_entries = schema.get("claim_entity_relations") or []
    relation_precedence = tuple(e["relation"] for e in relation_entries)
    relation_hints: dict[str, frozenset[str]] = {
        e["relation"]: frozenset(e.get("entity_types") or [])
        for e in relation_entries
    }
    schema_location_rel = schema.get("claim_location_relation", "OCCURRED_AT")
    location_entity_types = schema.get("claim_location_entity_types") or []
    if location_entity_types:
        relation_hints[schema_location_rel] = frozenset(location_entity_types)

    entity_labels = frozenset(schema.get("entity_labels") or [])
    legacy_renames = schema.get("legacy_renames") or {}

    # Derivation roles: resolver specs + per-record-type edge maps.
    # Missing section degrades to empty dicts so non-wildlife resource
    # bundles without this block still load.
    roles_block = schema.get("derivation_roles") or {}
    role_resolution: dict[str, RoleResolver] = {}
    for role_name, spec in (roles_block.get("resolution") or {}).items():
        role_resolution[str(role_name)] = RoleResolver(
            source=str(spec.get("source", "entity_links")),
            relations=frozenset(spec.get("relations") or []),
            entity_type=spec.get("entity_type"),
        )

    def _parse_edge_map(raw: Any) -> dict[str, tuple[str, str]]:
        result: dict[str, tuple[str, str]] = {}
        for role_name, pair in (raw or {}).items():
            if not pair or len(pair) != 2:
                _log.warning(
                    "derivation_roles edge entry %r is not a [label, edge_type] pair",
                    role_name,
                )
                continue
            result[str(role_name)] = (str(pair[0]), str(pair[1]))
        return result

    observation_role_edges = _parse_edge_map(roles_block.get("observation_edges"))
    event_role_edges = _parse_edge_map(roles_block.get("event_edges"))

    claim_link_roles, claim_link_occurrence_verbs, claim_link_subject_verbs = (
        _build_claim_link_roles(schema)
    )

    # Derive allowed_claim_types from patterns + registry + unclassified sentinel.
    from gemynd.core.claim_contract import UNCLASSIFIED_TYPE
    pattern_types = {ct for ct, _, _ in claim_type_patterns}
    registry_types = set(registry.keys())
    allowed_claim_types = frozenset(
        pattern_types | registry_types | set(legacy_renames.values()) | {UNCLASSIFIED_TYPE}
    )

    # Derive eligible types from derivation registry.
    obs_eligible = frozenset(
        ct for ct, spec in registry.items() if spec.observation_type
    )
    evt_eligible = frozenset(
        ct for ct, spec in registry.items() if spec.event_type
    )

    # Validator verb vocabulary
    verb_words = load_validator_verbs(resources_dir)
    validator_verb_re: re.Pattern[str] | None = None
    if verb_words:
        verb_alt = "|".join(re.escape(v) for v in verb_words)
        try:
            validator_verb_re = re.compile(rf"\b({verb_alt})\b", re.IGNORECASE)
        except re.error:
            _log.warning("validator_verbs produced invalid regex; using default")
            validator_verb_re = None

    # Heading detection override from domain_profile.yaml:
    #   heading_detection: false          → disable heading rejection entirely
    #   heading_detection: "^REGEX$"      → use custom regex
    #   heading_detection: (absent/true)  → use built-in default
    validator_heading_re: re.Pattern[str] | None = None
    heading_cfg = profile.get("heading_detection")
    if heading_cfg is False:
        # Explicitly disabled — use a pattern that never matches.
        validator_heading_re = re.compile(r"(?!)")  # negative lookahead: never matches
    elif isinstance(heading_cfg, str):
        try:
            validator_heading_re = re.compile(heading_cfg)
        except re.error:
            _log.warning("heading_detection regex is invalid; using default")
            validator_heading_re = None

    meas_patterns = load_measurement_patterns(resources_dir)
    qi_maps = load_query_intent_maps(resources_dir)

    config = DomainConfig(
        seed_entities=default_seed_entities(resources_dir),
        claim_type_patterns=claim_type_patterns,
        claim_role_policy=load_claim_role_policy(resources_dir),
        measurement_units=load_measurement_units(resources_dir),
        measurement_species=load_measurement_species(resources_dir),
        ocr_corrections=load_ocr_corrections(resources_dir),
        ocr_correction_map=load_ocr_correction_map(resources_dir),
        negative_lexicon=load_negative_entities(resources_dir),
        preferred_entity_types=compat.get("preferred_entity_types", {}),
        compatibility_matrix=compat.get("compatibility", {}),
        derivation_registry=registry,
        domain_anchor=profile.get("document_anchor"),
        year_validation=profile.get("year_validation"),
        synthesis_context=profile.get("synthesis_context", ""),
        claim_entity_relation_precedence=relation_precedence,
        claim_entity_relations=frozenset(relation_precedence),
        relation_to_entity_type_hints=relation_hints,
        claim_location_relation=schema_location_rel,
        entity_labels=entity_labels,
        legacy_renames=legacy_renames,
        allowed_claim_types=allowed_claim_types,
        observation_eligible_types=obs_eligible,
        event_eligible_types=evt_eligible,
        concept_rules=load_concept_rules(resources_dir),
        query_intent_to_claim_types=load_query_intent(resources_dir),
        validator_verb_re=validator_verb_re,
        validator_heading_re=validator_heading_re,
        institution_id=profile.get("institution_id", ""),
        expected_claim_shares=profile.get("expected_claim_shares") or {},
        analytical_queries=profile.get("analytical_queries") or {},
        enable_llm_cypher=bool(profile.get("enable_llm_cypher", False)),
        resolution_type_filter=bool(profile.get("resolution_type_filter", False)),
        role_resolution=role_resolution,
        observation_role_edges=observation_role_edges,
        event_role_edges=event_role_edges,
        entity_type_rules=load_entity_type_rules(resources_dir),
        non_entity_terms=load_non_entity_terms(resources_dir),
        measurement_pattern_specs=meas_patterns["pattern_specs"],
        measurement_unit_right_re=meas_patterns["unit_right_re"],
        measurement_count_claim_types=meas_patterns["count_claim_types"],
        measurement_acre_specialization=meas_patterns["acre_specialization"],
        measurement_money=meas_patterns["money"],
        claim_link_roles=claim_link_roles,
        claim_link_occurrence_verbs=claim_link_occurrence_verbs,
        claim_link_subject_verbs=claim_link_subject_verbs,
        source_profile=profile.get("source_profile") or {},
        categorial_label_map={
            str(k): str(v) for k, v in (schema.get("categorial_label_map") or {}).items()
        },
        entity_minting=dict(schema.get("entity_minting") or {}),
        spelling_reference_terms=load_spelling_reference_terms(resources_dir),
        builder_repair=dict(schema.get("builder_repair") or {}),
        junk_mention_filters=dict(schema.get("junk_mention") or {}),
        doc_type_aliases=qi_maps["doc_type_aliases"],
        series_aliases=qi_maps["series_aliases"],
        tier1_type_keywords=qi_maps["tier1_type_keywords"],
        relation_semantics={
            str(k): str(v) for k, v in (schema.get("relation_semantics") or {}).items()
        },
        ead_tag_map={
            str(k): str(v) for k, v in (schema.get("ead_tag_map") or {}).items()
        },
    )
    _validate_config(config, resources_dir)
    return config


def _validate_config(
    config: DomainConfig,
    resources_dir: Path | None = None,
) -> list[str]:
    """Check cross-resource consistency.  Returns a list of issue strings.

    Every issue is also emitted as a warning log.  The function never raises.
    """
    from gemynd.core.claim_contract import UNCLASSIFIED_TYPE

    issues: list[str] = []

    def _warn(msg: str) -> None:
        _log.warning(msg)
        issues.append(msg)

    pattern_claim_types = {ct for ct, _, _ in config.claim_type_patterns}
    seed_entity_types = {e.entity_type for e in config.seed_entities}

    # Legacy renames and the unclassified fallback intentionally lack patterns.
    _skip = set(config.legacy_renames.keys()) | {UNCLASSIFIED_TYPE}

    # --- Existing checks ---

    for ct in config.derivation_registry:
        if ct not in pattern_claim_types and ct not in _skip:
            _warn(
                f"derivation_registry entry {ct!r} has no matching claim_type_pattern"
            )

    for ct, types in config.preferred_entity_types.items():
        for et in types:
            if et not in seed_entity_types:
                _warn(
                    f"preferred_entity_types[{ct!r}] references unknown entity type {et!r}"
                )

    for (ct, _et) in config.claim_role_policy:
        if ct not in pattern_claim_types:
            _warn(
                f"claim_role_policy entry {ct!r} has no matching claim_type_pattern"
            )

    if not config.institution_id:
        _warn("domain_profile is missing 'institution_id'")
    if not config.synthesis_context:
        _warn("domain_profile is missing 'synthesis_context'")

    # --- Onboarding checks ---

    # 1. Entity labels vs. seed entity types
    if config.entity_labels:
        labels_without_seeds = config.entity_labels - seed_entity_types
        # Period is auto-created during extraction, not seeded.
        labels_without_seeds -= {"Period"}
        for label in sorted(labels_without_seeds):
            _warn(
                f"entity_labels includes {label!r} but no seed entity has that type"
            )
        seeds_without_labels = seed_entity_types - config.entity_labels
        for et in sorted(seeds_without_labels):
            _warn(
                f"seed_entities contains type {et!r} not listed in entity_labels"
            )

    # 2. Derivation registry covers all pattern-defined claim types
    for ct in sorted(pattern_claim_types):
        if ct not in config.derivation_registry and ct not in _skip:
            _warn(
                f"claim_type_pattern {ct!r} has no derivation_registry entry"
            )

    # 3. Document anchor entity type exists in seed vocabulary
    if config.domain_anchor:
        anchor_type = config.domain_anchor.get("entity_type")
        if anchor_type and anchor_type not in seed_entity_types:
            _warn(
                f"document_anchor entity_type {anchor_type!r} not found in seed_entities"
            )
        if not config.domain_anchor.get("relation"):
            _warn(
                "document_anchor is set but 'relation' is missing — "
                "retrieval cannot build an anchor-scoped temporal query"
            )

    # 4. Sensitivity vocabulary file exists
    if resources_dir is not None:
        from gemynd.shared.resource_loader import _dir
        res = _dir(resources_dir)
        sens_cfg_path = res / "sensitivity_config.yaml"
        if sens_cfg_path.exists():
            try:
                import yaml
                with sens_cfg_path.open(encoding="utf-8") as fh:
                    sens_cfg = yaml.safe_load(fh) or {}
                vocab_file = (
                    sens_cfg.get("indigenous_sensitivity", {})
                    .get("vocabulary_file")
                )
                if vocab_file:
                    vocab_path = res / vocab_file
                    if not vocab_path.exists():
                        _warn(
                            f"sensitivity_config references vocabulary_file "
                            f"{vocab_file!r} but {vocab_path} does not exist"
                        )
            except Exception:
                pass  # YAML parse failure handled elsewhere

    return issues
