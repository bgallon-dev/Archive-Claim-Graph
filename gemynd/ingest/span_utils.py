"""Generic span-overlap resolution shared across extractors."""
from __future__ import annotations

from typing import Protocol, TypeVar


class SpanLike(Protocol):
    start_offset: int
    end_offset: int
    detection_confidence: float


T = TypeVar("T", bound=SpanLike)


def resolve_overlapping_spans(
    spans: list[T],
    *,
    longer_wins_threshold: float = 0.60,
) -> list[T]:
    """Greedy longest-match-first overlap resolution.

    Two spans overlap when they share at least one character position; adjacent
    spans (end == start) do not overlap. When a conflict is detected the longer
    span wins if its confidence is >= ``longer_wins_threshold``; otherwise the
    already-accepted (shorter) span is kept. Exact-duplicate positions (same
    start AND end) are resolved by keeping the higher-confidence span.
    """
    sorted_spans = sorted(
        spans,
        key=lambda s: (s.start_offset, -(s.end_offset - s.start_offset)),
    )
    accepted: list[T] = []

    def _overlaps(a: SpanLike, b: SpanLike) -> bool:
        return not (a.end_offset <= b.start_offset or a.start_offset >= b.end_offset)

    for candidate in sorted_spans:
        conflict_idx: int | None = None
        for i, existing in enumerate(accepted):
            if _overlaps(candidate, existing):
                conflict_idx = i
                break

        if conflict_idx is None:
            accepted.append(candidate)
            continue

        existing = accepted[conflict_idx]
        cand_len = candidate.end_offset - candidate.start_offset
        exist_len = existing.end_offset - existing.start_offset

        if cand_len > exist_len and candidate.detection_confidence >= longer_wins_threshold:
            accepted[conflict_idx] = candidate
        elif cand_len == exist_len and candidate.detection_confidence > existing.detection_confidence:
            accepted[conflict_idx] = candidate

    return sorted(accepted, key=lambda s: s.start_offset)
