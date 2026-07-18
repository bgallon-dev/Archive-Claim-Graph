from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gemynd.core.models import ClaimRecord

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig


@dataclass(slots=True)
class ConceptAssignment:
    claim_id: str
    concept_id: str
    confidence: float
    matched_rule: str


def assign_concepts(
    claim: ClaimRecord,
    *,
    config: "DomainConfig | None" = None,
) -> list[ConceptAssignment]:
    """Return concept assignments for a claim based on rule matching.

    A claim may match multiple concepts. All matches are returned —
    the caller decides whether to write all or only the top-scoring one.
    Concepts are only assigned when the claim_type is in the allowed
    set for that concept rule, preventing cross-domain noise.

    Concept rules come exclusively from ``config.concept_rules`` (loaded from
    the domain's ``concept_rules.yaml``). A domain that declares no concept
    rules — or a call with no config — produces **no** concept assignments.
    There is deliberately no module-level fallback: falling back to one
    domain's (ecological) concept vocabulary is the silent substitution this
    engine exists to prevent.
    """
    rules = config.concept_rules if config is not None else []
    assignments: list[ConceptAssignment] = []
    sentence = claim.source_sentence or ""

    for concept_id, allowed_types, pattern, confidence in rules:
        if claim.claim_type not in allowed_types:
            continue
        if pattern.search(sentence):
            assignments.append(
                ConceptAssignment(
                    claim_id=claim.claim_id,
                    concept_id=concept_id,
                    confidence=confidence,
                    matched_rule=pattern.pattern[:60],
                )
            )

    return assignments
