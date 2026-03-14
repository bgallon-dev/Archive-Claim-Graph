from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from .ids import make_entity_id
from .models import EntityRecord, EntityResolutionRecord, MentionRecord


@dataclass(slots=True)
class ResolutionPolicy:
    refers_to_threshold: float = 0.85
    maybe_threshold: float = 0.50
    uniqueness_gap: float = 0.05


class EntityResolver(Protocol):
    def resolve(self, mentions: list[MentionRecord]) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        ...


def default_seed_entities() -> list[EntityRecord]:
    seed_data: list[tuple[str, str, dict[str, object]]] = [
        ("Refuge", "Turnbull Refuge", {"refuge_id": "turnbull_refuge"}),
        ("Place", "Pine Creek", {"place_type": "creek"}),
        ("Place", "Ice Lake", {"place_type": "lake"}),
        ("Place", "Highbridge Pothole", {"place_type": "pond"}),
        ("Place", "Cheney", {"place_type": "city"}),
        ("Place", "Spokane County", {"place_type": "county"}),
        ("Person", "John D. Connors", {"role": "refuge_manager"}),
        ("Person", "John Finley", {"role": "official"}),
        ("Person", "Theo Sheffer", {"role": "official"}),
        ("Person", "Hunter Engman", {"role": "governor"}),
        ("Organization", "Spokane County Sportsmens Assn", {"org_type": "association"}),
        ("Organization", "State Sport Council", {"org_type": "council"}),
        ("Organization", "Spokane Bird Club", {"org_type": "club"}),
        ("Organization", "WPA", {"org_type": "government"}),
        ("Species", "mallard", {"taxon_group": "bird"}),
        ("Species", "green-wing teal", {"taxon_group": "bird"}),
        ("Species", "coot", {"taxon_group": "bird"}),
        ("Species", "Canada goose", {"taxon_group": "bird"}),
        ("Species", "Chinese pheasant", {"taxon_group": "bird"}),
        ("Species", "Valley quail", {"taxon_group": "bird"}),
        ("Species", "Hungarian partridge", {"taxon_group": "bird"}),
        ("Species", "ruffed grouse", {"taxon_group": "bird"}),
        ("Species", "white-tailed deer", {"taxon_group": "mammal"}),
        ("Species", "coyote", {"taxon_group": "mammal"}),
        ("Activity", "grazing restriction", {"activity_type": "grazing_restriction"}),
        ("Activity", "haying", {"activity_type": "haying"}),
        ("Activity", "wildfire suppression", {"activity_type": "wildfire_suppression"}),
        ("Activity", "predator control", {"activity_type": "predator_control"}),
        ("Activity", "wild celery planting", {"activity_type": "planting"}),
        ("Activity", "public relations contact", {"activity_type": "public_relations_contact"}),
    ]
    entities: list[EntityRecord] = []
    for label, name, props in seed_data:
        normalized = normalize_name(name)
        entities.append(
            EntityRecord(
                entity_id=make_entity_id(label, normalized),
                label=label,
                name=name,
                normalized_name=normalized,
                properties=props,
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
    ) -> None:
        self._policy = policy or ResolutionPolicy()
        self._seed_entities = list(seed_entities) if seed_entities is not None else default_seed_entities()

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
                    score=round(top_score, 4),
                )
            )

        return list(entity_map.values()), resolutions

    def _candidate_scores(self, mention_normalized: str) -> list[tuple[EntityRecord, float]]:
        scored: list[tuple[EntityRecord, float]] = []
        for entity in self._seed_entities:
            score = similarity_score(mention_normalized, entity.normalized_name)
            if score >= self._policy.maybe_threshold:
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
