"""Audit-field contract shared by every extractor draft type.

The three extractor draft dataclasses (``ClaimDraft``, ``MentionDraft``,
``MeasurementDraft``) all carry the same audit-trail fields:
``decision_trace``, ``matched_patterns``, and ``fallback_used``. Each draft
also carries a source-tag field, but that field is named differently on
``ClaimDraft``/``MeasurementDraft`` (``extraction_source``) versus
``MentionDraft`` (``detection_source``) — the naming reflects a real
semantic difference (extraction yields structured records; detection
locates spans) and is intentionally NOT unified in the Protocol below.

The Protocol is structural: any dataclass declaring these three fields
satisfies it without inheritance. Slotted-dataclass inheritance across
unrelated types is brittle, so we use a Protocol + a free helper instead
of a base class.
"""
from __future__ import annotations

from typing import Protocol


class HasAudit(Protocol):
    """Shared audit-trail fields across all extractor draft dataclasses."""

    decision_trace: list[str]
    matched_patterns: list[str]
    fallback_used: bool


def tag_hybrid_trace(draft: HasAudit, trace: str) -> None:
    """Defensively copy ``draft.decision_trace`` and append a hybrid marker.

    Replaces the two-line pattern
    ::

        draft.decision_trace = list(draft.decision_trace)
        draft.decision_trace.append(trace)

    that the hybrid claim and mention extractors both apply when one
    candidate wins over the other during the merge step. The defensive copy
    is load-bearing: it severs the list reference from the originating
    extractor's output so subsequent appends here cannot leak back into a
    cached or otherwise reused upstream list.
    """
    draft.decision_trace = list(draft.decision_trace)
    draft.decision_trace.append(trace)
