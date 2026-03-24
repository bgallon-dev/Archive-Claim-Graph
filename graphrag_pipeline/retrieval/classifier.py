"""Layer 0 — Query Intent Classifier.

Rule-based classifier that maps a natural-language query to one of three
buckets (analytical / conversational / hybrid) without any LLM call.
Entity surface forms and year ranges are extracted here so that Layer 1 can
perform graph-anchored resolution before any Cypher executes.
"""
from __future__ import annotations

import re

from ..resolver import normalize_name
from .models import QueryIntent

# ---------------------------------------------------------------------------
# Signal vocabulary
# ---------------------------------------------------------------------------

_ANALYTICAL_SIGNALS: frozenset[str] = frozenset({
    "count", "total", "trend", "compare", "how many", "across years",
    "average", "sum", "tally", "change over", "over time", "per year",
    "annually", "statistics", "numbers", "population size", "how often",
    "highest", "lowest", "peak", "maximum", "minimum",
})

_CONVERSATIONAL_SIGNALS: frozenset[str] = frozenset({
    "what", "why", "describe", "explain", "did", "was", "were",
    "tell me", "what did", "what does", "what was", "how did",
    "evidence", "support", "report", "habitat", "condition",
})

# Words that must never be forwarded to the entity resolver.  Common English
# words here score 0.65–0.90 against domain entity names via fuzzy matching,
# producing spurious resolutions that corrupt the retrieval strategy selection.
_EXTRACTION_STOPWORDS: frozenset[str] = frozenset({
    "what", "when", "where", "which", "while", "were", "have", "been",
    "that", "this", "these", "those", "their", "there", "then", "than",
    "with", "from", "about", "over", "under", "into", "onto", "upon",
    "describe", "explain", "tell", "show", "give", "find", "list",
    "happening", "occurred", "recorded", "observed", "noted", "reported",
    "patterns", "changes", "trends", "during", "across", "stand", "stood",
    "decade", "period", "years", "year", "time", "times", "date", "dates",
    "significant", "important", "notable", "common", "general", "specific",
    "turnbull",  # always resolves to Refuge — handled separately by capitalized pass
})

# Patterns used to pull year values from query text.
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[012]\d)\b")
_BETWEEN_RE = re.compile(r"\bbetween\s+(1[89]\d{2}|20[012]\d)\s+and\s+(1[89]\d{2}|20[012]\d)\b")


def classify_query(
    text: str,
    year_range: tuple[int, int] | None = None,
    entity_hints: list[str] | None = None,
) -> QueryIntent:
    """Classify *text* into analytical / conversational / hybrid.

    Parameters
    ----------
    text:
        Raw natural-language query.
    year_range:
        Caller-supplied year bounds that override any years detected in *text*.
    entity_hints:
        Caller-supplied surface forms to supplement automatic entity detection.
    """
    lowered = text.lower()

    # --- bucket scoring ---
    analytical_hits = sum(1 for sig in _ANALYTICAL_SIGNALS if sig in lowered)
    conversational_hits = sum(1 for sig in _CONVERSATIONAL_SIGNALS if sig in lowered)

    if analytical_hits > 0 and conversational_hits > 0:
        bucket = "hybrid"
        # Confidence is the relative dominance of the leading signal set.
        confidence = max(analytical_hits, conversational_hits) / (analytical_hits + conversational_hits)
    elif analytical_hits > conversational_hits:
        bucket = "analytical"
        total = analytical_hits + conversational_hits or 1
        confidence = round(analytical_hits / total, 2)
    elif conversational_hits > analytical_hits:
        bucket = "conversational"
        total = analytical_hits + conversational_hits or 1
        confidence = round(conversational_hits / total, 2)
    else:
        # No clear signal: default to conversational with low confidence.
        bucket = "conversational"
        confidence = 0.5

    # --- year extraction ---
    if year_range is not None:
        year_min, year_max = year_range
    else:
        between_match = _BETWEEN_RE.search(lowered)
        if between_match:
            year_min = int(between_match.group(1))
            year_max = int(between_match.group(2))
        else:
            year_hits = [int(y) for y in _YEAR_RE.findall(text)]
            year_min = min(year_hits) if year_hits else None
            year_max = max(year_hits) if year_hits else None

    # --- entity surface form extraction ---
    # Collect tokens from the query that survive normalize_name and are longer
    # than 2 characters; caller can supply additional hints via entity_hints.
    entities: list[str] = list(entity_hints or [])
    seen_normalized: set[str] = {normalize_name(e) for e in entities}

    # First pass: capitalized phrases (high-precision for proper nouns mid-sentence).
    capitalised_re = re.compile(r"(?<![.!?]\s)(?<!\A)\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b")
    for m in capitalised_re.finditer(text):
        candidate = m.group(1)
        norm = normalize_name(candidate)
        if len(norm) > 2 and norm not in seen_normalized:
            entities.append(candidate)
            seen_normalized.add(norm)

    # Second pass: all word tokens >= 4 chars (catches lowercase names like "mallard").
    # The resolver's REFERS_TO threshold (>=0.85 with uniqueness gap) acts as filter.
    for m in re.finditer(r'\b([a-zA-Z]{4,})\b', text):
        norm = normalize_name(m.group(1))
        if (len(norm) >= 4
                and norm not in seen_normalized
                and m.group(1).lower() not in _EXTRACTION_STOPWORDS):
            entities.append(m.group(1).lower())
            seen_normalized.add(norm)

    return QueryIntent(
        bucket=bucket,
        classifier_confidence=confidence,
        entities=entities,
        year_min=year_min,
        year_max=year_max,
        claim_types=[],
    )
