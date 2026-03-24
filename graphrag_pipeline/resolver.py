from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .ids import make_entity_id
from .models import EntityRecord, EntityResolutionRecord, MentionRecord
from .resource_loader import load_seed_entity_rows


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
    def resolve(self, mentions: list[MentionRecord]) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
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
    def __init__(
        self,
        seed_entities: Iterable[EntityRecord] | None = None,
        policy: ResolutionPolicy | None = None,
        supplementary_candidates: list[EntityRecord] | None = None,
    ) -> None:
        self._policy = policy or ResolutionPolicy()
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

    def resolve(self, mentions: list[MentionRecord]) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        entity_map: dict[str, EntityRecord] = {}
        resolutions: list[EntityResolutionRecord] = []

        for mention in mentions:
            candidates = self._candidate_scores(mention.normalized_form)
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
            resolutions.append(
                EntityResolutionRecord(
                    mention_id=mention.mention_id,
                    entity_id=top_entity.entity_id,
                    relation_type=relation,
                    match_score=round(top_score, 4),
                )
            )

        return list(entity_map.values()), resolutions

    def _candidate_scores(self, mention_normalized: str) -> list[tuple[EntityRecord, float]]:
        scored: list[tuple[EntityRecord, float]] = []
        for entity in self._seed_entities:
            candidate_match_score = similarity_score(mention_normalized, entity.normalized_form)
            if candidate_match_score >= self._policy.maybe_threshold:
                scored.append((entity, candidate_match_score))
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
