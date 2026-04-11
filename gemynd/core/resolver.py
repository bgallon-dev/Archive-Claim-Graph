from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

from gemynd.core.ids import make_entity_id
from gemynd.core.models import EntityRecord, EntityResolutionRecord, MentionRecord
from gemynd.shared.resource_loader import load_claim_relation_compatibility, load_seed_entity_rows

if TYPE_CHECKING:
    from gemynd.ingest.extractors.mention_extractor import ResolutionContext
    from gemynd.core.domain_config import CompositeDomainConfig, DomainConfig


RESOLUTION_CONTRACT: dict[str, str] = {
    "REFERS_TO": (
        "match_score >= 0.85 and uniqueness_gap >= 0.05. "
        "Algorithmically unambiguous match. "
        "Treat as confirmed unless an EntityResolutionConfirmationRecord with relation_type='REFUTED_BY' overrides it."
    ),
    "POSSIBLY_REFERS_TO": (
        "match_score 0.50-0.84, or >= 0.85 without uniqueness gap. "
        "Plausible but ambiguous. Requires human review. "
        "Can be promoted to CONFIRMED_AS or explicitly rejected via EntityResolutionConfirmationRecord."
    ),
    "CONFIRMED_AS": (
        "Human-confirmed resolution. Overrides algorithmic REFERS_TO or POSSIBLY_REFERS_TO. "
        "Set via EntityResolutionConfirmationRecord with confirmed_by and confirmed_at."
    ),
    "REFUTED_BY": (
        "Human-rejected resolution. Overrides algorithmic REFERS_TO for this mention-entity pair. "
        "Set via EntityResolutionConfirmationRecord."
    ),
}


@dataclass(slots=True)
class ResolutionPolicy:
    refers_to_threshold: float = 0.85
    maybe_threshold: float = 0.65
    uniqueness_gap: float = 0.05


class EntityResolver(Protocol):
    def resolve(
        self,
        mentions: list[MentionRecord],
        contexts: dict[str, ResolutionContext] | None = None,
        document_entity_counts: Counter[str] | None = None,
    ) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        ...


def default_seed_entities(resources_dir: Path | None = None) -> list[EntityRecord]:
    """Build EntityRecord list from seed_entities.csv.

    The CSV uses a long format: one row per (entity, property) pair.
    Rows sharing the same (entity_type, name) are merged into one EntityRecord.
    """
    rows = load_seed_entity_rows(resources_dir)

    # Preserve insertion order; group properties by (entity_type, name).
    props_by_key: dict[tuple[str, str], dict[str, object]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row["entity_type"], row["name"])
        if key not in props_by_key:
            props_by_key[key] = {}
            order.append(key)
        if row["prop_key"] and row["prop_value"]:
            props_by_key[key][row["prop_key"]] = row["prop_value"]

    entities: list[EntityRecord] = []
    for entity_type, name in order:
        normalized = normalize_name(name)
        entities.append(
            EntityRecord(
                entity_id=make_entity_id(entity_type, normalized),
                entity_type=entity_type,
                name=name,
                normalized_form=normalized,
                properties=props_by_key[(entity_type, name)],
            )
        )
    return entities


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s\-]", "", name.strip().lower())
    return re.sub(r"\s+", " ", normalized)



class DictionaryFuzzyResolver:
    # Boost magnitudes indexed by position in the preferred_entity_types list.
    _POSITION_BOOSTS: tuple[float, ...] = (0.05, 0.03, 0.01)
    # Penalty when the entity type is absent from a non-empty preferred list.
    _ABSENT_PENALTY: float = -0.02

    def __init__(
        self,
        seed_entities: Iterable[EntityRecord] | None = None,
        policy: ResolutionPolicy | None = None,
        supplementary_candidates: list[EntityRecord] | None = None,
        resources_dir: Path | None = None,
        config: "DomainConfig | CompositeDomainConfig | None" = None,
    ) -> None:
        self._policy = policy or ResolutionPolicy()
        if config is not None and seed_entities is None:
            self._seed_entities = list(config.seed_entities)
        else:
            self._seed_entities = list(seed_entities) if seed_entities is not None else default_seed_entities()
        if supplementary_candidates:
            # Seed vocabulary wins: skip graph candidates whose (type, form) already
            # appear in the seed list so the seed entity_id is always preferred.
            seed_keys = {(e.entity_type, e.normalized_form) for e in self._seed_entities}
            extras = [
                e for e in supplementary_candidates
                if (e.entity_type, e.normalized_form) not in seed_keys
            ]
            self._seed_entities = self._seed_entities + extras

        # Load preferred_entity_types from claim_relation_compatibility.yaml.
        # This drives the YAML-based claim-type-conditioned score adjustments.
        if config is not None:
            self._preferred_entity_types: dict[str, list[str]] = config.preferred_entity_types
        else:
            compat_data = load_claim_relation_compatibility(resources_dir)
            self._preferred_entity_types = compat_data.get("preferred_entity_types", {})

    def resolve(
        self,
        mentions: list[MentionRecord],
        contexts: dict[str, ResolutionContext] | None = None,
        document_entity_counts: Counter[str] | None = None,
    ) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        """Resolve mentions to entities using a two-pass algorithm when contexts are provided.

        When *contexts* is None, the method runs a single pass identical to the
        original implementation (backward-compatible).

        **Two-pass algorithm** (active when contexts is not None):

        Pass 1 — high-confidence mentions only (score >= refers_to_threshold with
        uniqueness gap).  Builds a per-paragraph co-mention type counter used in
        Pass 2.  Populates context.resolved_entity_types as an output side-effect.

        Pass 2 — remaining mentions receive a +0.03 co-mention consistency boost
        for candidates whose entity_type matches the most-frequent type resolved
        in the same paragraph during Pass 1.

        The *document_entity_counts* Counter is threaded into _candidate_scores()
        (A3 frequency prior) in both passes.
        """
        if contexts is None:
            # Single-pass fallback — identical to original implementation.
            return self._resolve_single_pass(mentions, document_entity_counts)

        from collections import defaultdict

        entity_map: dict[str, EntityRecord] = {}
        resolutions: list[EntityResolutionRecord] = []
        resolved_mention_ids: set[str] = set()
        # co_mention_type_counts[paragraph_id][entity_type] = count of REFERS_TO
        co_mention_type_counts: dict[str, Counter[str]] = defaultdict(Counter)

        # ── Pass 1: high-confidence resolutions ───────────────────────────────
        for mention in mentions:
            context = contexts.get(mention.paragraph_id)
            candidates = self._candidate_scores(mention, context, document_entity_counts)
            if not candidates:
                continue
            top_entity, top_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0
            unique = top_score - second_score >= self._policy.uniqueness_gap

            if top_score >= self._policy.refers_to_threshold and unique:
                entity_map[top_entity.entity_id] = top_entity
                resolutions.append(EntityResolutionRecord(
                    mention_id=mention.mention_id,
                    entity_id=top_entity.entity_id,
                    relation_type="REFERS_TO",
                    match_score=round(top_score, 4),
                ))
                resolved_mention_ids.add(mention.mention_id)
                co_mention_type_counts[mention.paragraph_id][top_entity.entity_type] += 1

        # Populate resolved_entity_types on each context as an output.
        for pid, type_counter in co_mention_type_counts.items():
            ctx = contexts.get(pid)
            if ctx is not None:
                ctx.resolved_entity_types = sorted(type_counter.elements())

        # ── Pass 2: remaining mentions with co-mention boost ──────────────────
        for mention in mentions:
            if mention.mention_id in resolved_mention_ids:
                continue
            context = contexts.get(mention.paragraph_id)
            candidates = self._candidate_scores(mention, context, document_entity_counts)
            if not candidates:
                continue

            # Apply co-mention consistency boost (+0.03) for the most-frequent
            # entity type in this paragraph's Pass-1 resolutions.
            para_counter = co_mention_type_counts.get(mention.paragraph_id)
            if para_counter:
                most_frequent_type = para_counter.most_common(1)[0][0]
                candidates = [
                    (entity, min(score + 0.03, 1.0) if entity.entity_type == most_frequent_type else score)
                    for entity, score in candidates
                ]
                candidates.sort(key=lambda row: row[1], reverse=True)

            top_entity, top_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0
            unique = top_score - second_score >= self._policy.uniqueness_gap

            if top_score >= self._policy.refers_to_threshold and unique:
                relation = "REFERS_TO"
            elif top_score >= self._policy.maybe_threshold:
                relation = "POSSIBLY_REFERS_TO"
            else:
                continue

            entity_map[top_entity.entity_id] = top_entity
            resolutions.append(EntityResolutionRecord(
                mention_id=mention.mention_id,
                entity_id=top_entity.entity_id,
                relation_type=relation,
                match_score=round(top_score, 4),
            ))

        return list(entity_map.values()), resolutions

    def _resolve_single_pass(
        self,
        mentions: list[MentionRecord],
        document_entity_counts: Counter[str] | None = None,
    ) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        """Original single-pass resolution — used when contexts=None."""
        entity_map: dict[str, EntityRecord] = {}
        resolutions: list[EntityResolutionRecord] = []

        for mention in mentions:
            candidates = self._candidate_scores(mention, None, document_entity_counts)
            if not candidates:
                continue
            top_entity, top_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0
            unique = top_score - second_score >= self._policy.uniqueness_gap

            if top_score >= self._policy.refers_to_threshold and unique:
                relation = "REFERS_TO"
            elif top_score >= self._policy.maybe_threshold:
                relation = "POSSIBLY_REFERS_TO"
            else:
                continue

            entity_map[top_entity.entity_id] = top_entity
            resolutions.append(EntityResolutionRecord(
                mention_id=mention.mention_id,
                entity_id=top_entity.entity_id,
                relation_type=relation,
                match_score=round(top_score, 4),
            ))

        return list(entity_map.values()), resolutions

    def _claim_type_adjustment(self, entity_type: str, claim_types: list[str]) -> float:
        """Return the net score adjustment driven by preferred_entity_types.

        For each claim type in *claim_types*, compute the boost/penalty for
        *entity_type* based on its position in preferred_entity_types[claim_type].
        Return the maximum positive adjustment found, or the minimum negative
        adjustment if no positive adjustments exist.  Return 0.0 when
        preferred_entity_types has no entry for any of the claim types.
        """
        if not claim_types:
            return 0.0
        adjustments: list[float] = []
        for ct in claim_types:
            preferred = self._preferred_entity_types.get(ct)
            if preferred is None:
                # No preference data for this claim type — neutral.
                continue
            if not preferred:
                # Explicitly empty list (e.g. unclassified_assertion) — neutral.
                continue
            try:
                pos = preferred.index(entity_type)
                adj = self._POSITION_BOOSTS[pos] if pos < len(self._POSITION_BOOSTS) else 0.0
            except ValueError:
                adj = self._ABSENT_PENALTY
            adjustments.append(adj)

        if not adjustments:
            return 0.0
        if any(a > 0 for a in adjustments):
            return max(adjustments)
        return min(adjustments)

    def _candidate_scores(
        self,
        mention: MentionRecord,
        context: ResolutionContext | None = None,
        document_entity_counts: Counter[str] | None = None,
    ) -> list[tuple[EntityRecord, float]]:
        """Score all seed entities against *mention* and return sorted candidates.

        Score calculation order:
        1. Base string similarity (similarity_score).
        2. [A1] YAML-driven claim-type adjustment (boost or small penalty).
           Floor at maybe_threshold to avoid hard-filtering borderline candidates.
        3. [A3] Document-level entity frequency prior (log-scaled boost, capped at 0.06).
        Candidates below maybe_threshold after all adjustments are excluded.
        """
        scored: list[tuple[EntityRecord, float]] = []
        for entity in self._seed_entities:
            score = similarity_score(mention.normalized_form, entity.normalized_form)
            if score < self._policy.maybe_threshold:
                continue

            # A1: YAML-driven claim-type-conditioned adjustment.
            if context is not None and context.claim_types:
                net = self._claim_type_adjustment(entity.entity_type, context.claim_types)
                score = min(score + net, 1.0)
                score = max(score, self._policy.maybe_threshold)

            # A3: Document-level entity frequency prior.
            if document_entity_counts and entity.entity_id in document_entity_counts:
                n = document_entity_counts[entity.entity_id]
                freq_boost = min(0.02 * math.log1p(n), 0.06)
                score = min(score + freq_boost, 1.0)

            scored.append((entity, score))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored


def similarity_score(left: str, right: str) -> float:
    if left == right:
        return 1.0
    ratio = difflib.SequenceMatcher(a=left, b=right).ratio()
    if left in right or right in left:
        ratio = max(ratio, 0.88)
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
        ratio = max(ratio, overlap)
    return round(ratio, 4)
