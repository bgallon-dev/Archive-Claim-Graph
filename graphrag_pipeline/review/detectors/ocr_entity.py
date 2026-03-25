"""OCR/entity cleanup detector.

Emits `merge_entities` or `create_alias` proposals for obvious duplicate
or OCR-corrupted entity variants.  Reuses the existing spelling-review
heuristics and entity-resolver fuzzy matching logic.
"""
from __future__ import annotations

import difflib
from collections import defaultdict
from typing import Any

from ...core.ids import stable_hash
from ...core.models import EntityRecord, EntityResolutionRecord, MentionRecord, SemanticBundle, StructureBundle
from ..models import DetectorProposal, ProposalTarget
from ..patch_spec import make_patch_spec

DETECTOR_NAME = "ocr_entity_detector"
DETECTOR_VERSION = "v1"


def _entity_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _group_similar_entities(
    entities: list[EntityRecord], threshold: float = 0.82,
) -> list[list[EntityRecord]]:
    """Group entities of the same type by normalized-form similarity."""
    by_type: dict[str, list[EntityRecord]] = defaultdict(list)
    for e in entities:
        by_type[e.entity_type].append(e)

    groups: list[list[EntityRecord]] = []
    for entity_type, type_entities in by_type.items():
        assigned: set[str] = set()
        for i, anchor in enumerate(type_entities):
            if anchor.entity_id in assigned:
                continue
            group = [anchor]
            assigned.add(anchor.entity_id)
            for j in range(i + 1, len(type_entities)):
                candidate = type_entities[j]
                if candidate.entity_id in assigned:
                    continue
                sim = _entity_similarity(anchor.normalized_form, candidate.normalized_form)
                if sim >= threshold:
                    group.append(candidate)
                    assigned.add(candidate.entity_id)
            if len(group) > 1:
                groups.append(group)
    return groups


def _pick_canonical(group: list[EntityRecord], mention_counts: dict[str, int]) -> EntityRecord:
    """Choose the canonical entity: prefer most mentions, then longest name, then lex-first."""
    return max(
        group,
        key=lambda e: (mention_counts.get(e.entity_id, 0), len(e.name), e.name),
    )


def _is_ocr_variant(a: str, b: str) -> bool:
    """Check if two names differ by common OCR confusion patterns."""
    if abs(len(a) - len(b)) > 2:
        return False
    a_lower, b_lower = a.lower(), b.lower()
    # Common OCR confusions: rn↔m, li↔h, 0↔o, 1↔l
    normalized_a = a_lower.replace("rn", "m").replace("li", "h").replace("0", "o").replace("1", "l")
    normalized_b = b_lower.replace("rn", "m").replace("li", "h").replace("0", "o").replace("1", "l")
    return normalized_a == normalized_b


def detect(
    structure: StructureBundle,
    semantic: SemanticBundle,
    snapshot_id: str,
) -> list[DetectorProposal]:
    """Run the OCR/entity cleanup detector.

    Returns DetectorProposal objects for merge_entities or create_alias proposals.
    """
    # Count mentions per entity for canonical selection
    mention_counts: dict[str, int] = defaultdict(int)
    for resolution in semantic.entity_resolutions:
        mention_counts[resolution.entity_id] += 1

    groups = _group_similar_entities(semantic.entities)
    proposals: list[DetectorProposal] = []

    for group in groups:
        canonical = _pick_canonical(group, mention_counts)
        others = [e for e in group if e.entity_id != canonical.entity_id]

        if not others:
            continue

        # Determine if this is an OCR variant or a true alias
        all_ocr = all(_is_ocr_variant(canonical.name, other.name) for other in others)

        if len(others) == 1 and not all_ocr:
            # Single non-OCR pair: create_alias
            other = others[0]
            issue_class = "duplicate_entity_alias"
            proposal_type = "create_alias"
            confidence = _entity_similarity(canonical.normalized_form, other.normalized_form)

            targets = [
                ProposalTarget(
                    proposal_id="",  # filled later
                    target_kind="entity",
                    target_id=canonical.entity_id,
                    target_role="canonical",
                    exists_in_snapshot=True,
                ),
                ProposalTarget(
                    proposal_id="",
                    target_kind="entity",
                    target_id=other.entity_id,
                    target_role="alias",
                    exists_in_snapshot=True,
                ),
            ]
            patch = make_patch_spec(
                "create_alias",
                canonical_entity_id=canonical.entity_id,
                alias_entity_id=other.entity_id,
                canonical_name=canonical.name,
                alias_name=other.name,
            )
            evidence = {
                "confidence": confidence,
                "issue_class": issue_class,
                "canonical_entity": canonical.to_dict(),
                "alias_entity": other.to_dict(),
                "canonical_mention_count": mention_counts.get(canonical.entity_id, 0),
                "alias_mention_count": mention_counts.get(other.entity_id, 0),
                "similarity_score": _entity_similarity(canonical.normalized_form, other.normalized_form),
                "source_file": structure.document.source_file,
            }
            proposals.append(DetectorProposal(
                anti_pattern_id="ap_duplicate_alias",
                issue_class=issue_class,
                proposal_type=proposal_type,
                confidence=confidence,
                targets=targets,
                patch_spec=patch,
                evidence_snapshot=evidence,
                reasoning_summary={"reason": f"Entity '{other.name}' appears to be an alias of '{canonical.name}'"},
                detector_name=DETECTOR_NAME,
                detector_version=DETECTOR_VERSION,
            ))
        else:
            # Multi-entity merge (OCR variants)
            issue_class = "ocr_spelling_variant"
            proposal_type = "merge_entities"
            merge_ids = sorted(e.entity_id for e in others)
            avg_sim = sum(
                _entity_similarity(canonical.normalized_form, e.normalized_form) for e in others
            ) / len(others)
            confidence = avg_sim

            targets = [
                ProposalTarget(
                    proposal_id="",
                    target_kind="entity",
                    target_id=canonical.entity_id,
                    target_role="canonical",
                    exists_in_snapshot=True,
                ),
            ]
            for other in others:
                targets.append(ProposalTarget(
                    proposal_id="",
                    target_kind="entity",
                    target_id=other.entity_id,
                    target_role="merge_source",
                    exists_in_snapshot=True,
                ))

            patch = make_patch_spec(
                "merge_entities",
                canonical_entity_id=canonical.entity_id,
                merge_entity_ids=merge_ids,
                canonical_name=canonical.name,
            )
            evidence = {
                "confidence": confidence,
                "issue_class": issue_class,
                "canonical_entity": canonical.to_dict(),
                "merge_entities": [e.to_dict() for e in others],
                "canonical_mention_count": mention_counts.get(canonical.entity_id, 0),
                "merge_mention_counts": {e.entity_id: mention_counts.get(e.entity_id, 0) for e in others},
                "average_similarity": avg_sim,
                "is_ocr_variant": all_ocr,
                "source_file": structure.document.source_file,
            }
            proposals.append(DetectorProposal(
                anti_pattern_id="ap_ocr_spelling",
                issue_class=issue_class,
                proposal_type=proposal_type,
                confidence=confidence,
                targets=targets,
                patch_spec=patch,
                evidence_snapshot=evidence,
                reasoning_summary={"reason": f"Entities {[e.name for e in others]} appear to be OCR variants of '{canonical.name}'"},
                detector_name=DETECTOR_NAME,
                detector_version=DETECTOR_VERSION,
            ))

    return proposals
