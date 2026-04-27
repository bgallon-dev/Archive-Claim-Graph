"""Layer 0 — Query Intent Classifier.

Rule-based classifier that maps a natural-language query to one of three
buckets (analytical / conversational / hybrid) without any LLM call.
Entity surface forms and year ranges are extracted here so that Layer 1 can
perform graph-anchored resolution before any Cypher executes.
"""
from __future__ import annotations

import re

from gemynd.core.resolver import normalize_name
from .models import QueryIntent

# Fallback year-range bounds used when the caller does not supply domain
# bounds. Spans 1800–2029 to accept any year a historical archive query is
# plausibly asking about.
_DEFAULT_YEAR_BOUNDS: tuple[int, int] = (1800, 2029)


_SINGLE_YEAR_RE = re.compile(r"\b(\d{4})\b")
_BETWEEN_YEAR_RE = re.compile(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b")


def _vocab_matches(token: str, vocab: frozenset[str]) -> bool:
    """Return True if *token* equals, substring-hits, or is within edit
    distance 1 of any whitespace-split segment in *vocab*.

    The seed vocabulary contains multi-word surface forms ("mallard duck");
    we split on whitespace so a single-token query word can match one
    component. Edit distance 1 tolerates plural/singular drift ("mallards"
    vs. "mallard") without pulling in unrelated English words.
    """
    if not vocab:
        return False
    tok_segments: set[str] = set()
    for phrase in vocab:
        for seg in phrase.split():
            if seg:
                tok_segments.add(seg)
    if token in tok_segments:
        return True
    for seg in tok_segments:
        if abs(len(seg) - len(token)) > 1:
            continue
        if _edit_distance_le1(token, seg):
            return True
    return False


def _edit_distance_le1(a: str, b: str) -> bool:
    """Return True if Levenshtein(a, b) <= 1. Linear in min(len(a), len(b))."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # Align for single-char insertion/deletion/substitution.
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    diff = 0
    while i < la and j < lb:
        if a[i] != b[j]:
            diff += 1
            if diff > 1:
                return False
            if la == lb:
                i += 1
                j += 1
            else:
                j += 1
        else:
            i += 1
            j += 1
    if j < lb:
        diff += lb - j
    return diff <= 1

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
# Maps natural-language phrases a user might type to canonical doc_type values
# stored on Document nodes.  Keys are lowercased; values are the exact strings
# that must appear in d.doc_type in the graph.  Extend this dict as new
# doc_type values are ingested.
_DOC_TYPE_ALIASES: dict[str, str] = {
    # BLM General Land Office patents — doc_type = "land_patent" in this graph.
    # All of the following are common ways users refer to the same document set.
    "land patent": "land_patent",
    "land patents": "land_patent",
    "land grant": "land_patent",
    "land grants": "land_patent",
    "glo record": "land_patent",
    "glo records": "land_patent",
    "glo patent": "land_patent",
    "glo patents": "land_patent",
    "blm record": "land_patent",
    "blm records": "land_patent",
    "blm patent": "land_patent",
    "blm patents": "land_patent",
    "general land office record": "land_patent",
    "general land office records": "land_patent",
    "general land office patent": "land_patent",
    "general land office patents": "land_patent",
    "homestead entry": "land_patent",
    "homestead entries": "land_patent",
    "homestead patent": "land_patent",
    "homestead patents": "land_patent",
    "cash entry": "land_patent",
    "cash entries": "land_patent",
    "donation land claim": "land_patent",
    "donation land claims": "land_patent",
    "patent": "land_patent",
    "patents": "land_patent",
    # Newspaper articles
    "newspaper article": "newspaper_article",
    "newspaper articles": "newspaper_article",
    "newspaper": "newspaper_article",
    "newspapers": "newspaper_article",
    "news article": "newspaper_article",
    "news articles": "newspaper_article",
    "press article": "newspaper_article",
    "press articles": "newspaper_article",
    # Other doc types — extend as new doc_type values are ingested.
    "survey plat": "survey_plat",
    "survey plats": "survey_plat",
    "plat map": "survey_plat",
    "plat maps": "survey_plat",
    "township plat": "survey_plat",
    "township plats": "survey_plat",
    "field note": "field_notes",
    "field notes": "field_notes",
    "narrative report": "narrative_report",
    "narrative reports": "narrative_report",
    "annual report": "annual_report",
    "annual reports": "annual_report",
    "census record": "census_record",
    "census records": "census_record",
    "deed": "deed",
    "deeds": "deed",
}


# Maps natural-language phrases to the exact d.series string stored in the graph.
# Series represent archival collections that may span multiple doc_types
# (e.g. a GLO collection could later include plats and field notes alongside patents).
# Keys are lowercased; values must match d.series exactly (case-sensitive in Neo4j).
_SERIES_ALIASES: dict[str, str] = {
    # BLM General Land Office Records series
    "blm glo records": "BLM GLO Records",
    "blm glo record": "BLM GLO Records",
    "glo series": "BLM GLO Records",
    "blm series": "BLM GLO Records",
    "blm records": "BLM GLO Records",
    "blm record": "BLM GLO Records",
    "general land office series": "BLM GLO Records",
    "bureau of land management records": "BLM GLO Records",
    # The Spokane Press newspaper series
    "the spokane press": "The Spokane Press",
    "spokane press": "The Spokane Press",
    "spokane newspaper": "The Spokane Press",
    "spokane newspapers": "The Spokane Press",
    # Add additional series here as new archival collections are ingested.
}


def _extract_series_hints(text: str) -> list[str]:
    """Return canonical series strings detected in *text* via alias matching.

    Longest-match first, same strategy as _extract_doc_type_hints.
    """
    lowered = text.lower()
    matched: dict[str, str] = {}
    for phrase in sorted(_SERIES_ALIASES, key=len, reverse=True):
        if phrase in lowered:
            canonical = _SERIES_ALIASES[phrase]
            if canonical not in matched.values():
                matched[phrase] = canonical
    return list(dict.fromkeys(matched.values()))


def _extract_doc_type_hints(text: str) -> list[str]:
    """Return canonical doc_type values detected in *text* via alias matching.

    Checks multi-word phrases first (longest match wins) to avoid partial hits.
    """
    lowered = text.lower()
    matched: dict[str, str] = {}
    for phrase in sorted(_DOC_TYPE_ALIASES, key=len, reverse=True):
        if phrase in lowered:
            canonical = _DOC_TYPE_ALIASES[phrase]
            if canonical not in matched.values():
                matched[phrase] = canonical
    return list(dict.fromkeys(matched.values()))


_EXTRACTION_STOPWORDS: frozenset[str] = frozenset({
    "what", "when", "where", "which", "while", "were", "have", "been",
    "that", "this", "these", "those", "their", "there", "then", "than",
    "with", "from", "about", "over", "under", "into", "onto", "upon",
    "describe", "explain", "tell", "show", "give", "find", "list",
    "during", "across",
    "decade", "period", "years", "year", "time", "times", "date", "dates",
    "common", "general", "specific",
    # Doc-type and series words — caught by _extract_doc_type_hints /
    # _extract_series_hints and must not reach the entity resolver.
    "patent", "patents", "grant", "grants", "land", "deed", "deeds",
    "plat", "plats", "record", "records", "series", "collection",
    "newspaper", "newspapers", "article", "articles", "press",
})

def classify_query(
    text: str,
    year_range: tuple[int, int] | None = None,
    entity_hints: list[str] | None = None,
    extra_stopwords: frozenset[str] | None = None,
    *,
    year_bounds: tuple[int, int] | None = None,
    seed_vocab: frozenset[str] | None = None,
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
    _lo, _hi = year_bounds or _DEFAULT_YEAR_BOUNDS

    def _in_bounds(y: int) -> bool:
        return _lo <= y <= _hi

    if year_range is not None:
        year_min, year_max = year_range
    else:
        between_match = _BETWEEN_YEAR_RE.search(lowered)
        if between_match and _in_bounds(int(between_match.group(1))) and _in_bounds(int(between_match.group(2))):
            year_min = int(between_match.group(1))
            year_max = int(between_match.group(2))
        else:
            year_hits = [int(y) for y in _SINGLE_YEAR_RE.findall(text) if _in_bounds(int(y))]
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
    # When *seed_vocab* is provided, we gate candidates against the normalized
    # seed vocabulary (exact hit or edit-distance-1 hit). This prevents every
    # 4+ char token from reaching the fuzzy resolver, where the 0.85 threshold
    # produces spurious matches on common English words.
    stopwords = _EXTRACTION_STOPWORDS | (extra_stopwords or frozenset())
    for m in re.finditer(r'\b([a-zA-Z]{4,})\b', text):
        candidate = m.group(1)
        norm = normalize_name(candidate)
        if len(norm) < 4 or norm in seen_normalized or candidate.lower() in stopwords:
            continue
        if seed_vocab is not None and not _vocab_matches(norm, seed_vocab):
            continue
        entities.append(candidate.lower())
        seen_normalized.add(norm)

    return QueryIntent(
        bucket=bucket,
        classifier_confidence=confidence,
        entities=entities,
        year_min=year_min,
        year_max=year_max,
        claim_types=[],
        doc_type_hints=_extract_doc_type_hints(text),
        series_hints=_extract_series_hints(text),
    )
