"""Layer 0 — Query Intent Classifier.

Rule-based classifier that maps a natural-language query to one of three
buckets (analytical / conversational / hybrid) without any LLM call.
Entity surface forms and year ranges are extracted here so that Layer 1 can
perform graph-anchored resolution before any Cypher executes.
"""
from __future__ import annotations

import re

from gemynd.core.resolver import normalize_name
from .models import EntityContext, QueryIntent, Tier1Intent

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


def _extract_series_hints(text: str, aliases: dict[str, str] | None = None) -> list[str]:
    """Return canonical series strings detected in *text* via alias matching.

    Longest-match first, same strategy as _extract_doc_type_hints. *aliases*
    (from the domain's ``query_intent.yaml``) replaces the module default when
    supplied.
    """
    alias_map = aliases if aliases is not None else _SERIES_ALIASES
    lowered = text.lower()
    matched: dict[str, str] = {}
    for phrase in sorted(alias_map, key=len, reverse=True):
        if phrase in lowered:
            canonical = alias_map[phrase]
            if canonical not in matched.values():
                matched[phrase] = canonical
    return list(dict.fromkeys(matched.values()))


def _extract_doc_type_hints(text: str, aliases: dict[str, str] | None = None) -> list[str]:
    """Return canonical doc_type values detected in *text* via alias matching.

    Checks multi-word phrases first (longest match wins) to avoid partial hits.
    *aliases* (from the domain's ``query_intent.yaml``) replaces the module
    default when supplied.
    """
    alias_map = aliases if aliases is not None else _DOC_TYPE_ALIASES
    lowered = text.lower()
    matched: dict[str, str] = {}
    for phrase in sorted(alias_map, key=len, reverse=True):
        if phrase in lowered:
            canonical = alias_map[phrase]
            if canonical not in matched.values():
                matched[phrase] = canonical
    return list(dict.fromkeys(matched.values()))


# Cross-corpus overlap detection — fires when the query asks for entities
# appearing in both of two named corpora.  Triggers the CROSS_CORPUS_OVERLAP
# cascade entry, which uses cross_corpus_ner REFERS_TO/POSSIBLY_REFERS_TO edges
# to find Persons mentioned in one corpus AND subjects of land_grant claims in
# another.  Patterns must combine an "in both" / "also appear" intent with at
# least one corpus reference (doc_type or series alias).  Pure "X and Y" without
# a join verb stays in the normal cascade.
_CROSS_CORPUS_OVERLAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "individuals/people/persons/names ... also appear / appear in both"
    re.compile(
        r"\b(individuals?|people|persons?|names?|figures?)\b.{0,80}?"
        r"\b(also\s+appear|appear\s+in\s+both|named\s+in\s+both|"
        r"appear\s+as|listed\s+in\s+both|mentioned\s+in\s+both|"
        r"shared\s+between|common\s+to\s+both)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # "who appears in both X and Y" / "which names show up in both"
    re.compile(
        r"\b(who|which)\b.{0,50}?\b(appears?|shows?\s+up|named|listed|mentioned)\b"
        r".{0,50}?\bin\s+both\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # "overlap between X and Y" / "intersection of X and Y"
    re.compile(
        r"\b(overlap|intersection|crossover)\b.{0,30}?\bbetween\b",
        re.IGNORECASE | re.DOTALL,
    ),
)


def _detect_cross_corpus_overlap(text: str, corpora_hits: int) -> bool:
    """Return True when the query phrasing asks for entities appearing across
    two corpora AND at least two distinct corpus references are present.

    The corpora-count gate prevents false positives on phrasings like "people
    who appear in both photographs and writings" where "both" is generic.
    """
    if corpora_hits < 2:
        return False
    return any(p.search(text) for p in _CROSS_CORPUS_OVERLAP_PATTERNS)


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
    doc_type_aliases: dict[str, str] | None = None,
    series_aliases: dict[str, str] | None = None,
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
    doc_type_aliases / series_aliases:
        Domain surface-form maps (from ``query_intent.yaml``). When supplied
        they replace the module-level defaults so a corpus recognises its own
        document-type / series vocabulary instead of the hardcoded grab-bag.
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

    doc_type_hints = _extract_doc_type_hints(text, doc_type_aliases)
    series_hints = _extract_series_hints(text, series_aliases)
    # Cross-corpus overlap requires at least two distinct corpus references in
    # the query (combined doc_type + series hints, deduped).  A single corpus
    # plus generic "both" wording (e.g. "in both 1908 and 1909") must not fire.
    corpora_hits = len({*doc_type_hints, *series_hints})
    cross_corpus_overlap = _detect_cross_corpus_overlap(text, corpora_hits)

    return QueryIntent(
        bucket=bucket,
        classifier_confidence=confidence,
        entities=entities,
        year_min=year_min,
        year_max=year_max,
        claim_types=[],
        doc_type_hints=doc_type_hints,
        series_hints=series_hints,
        cross_corpus_overlap=cross_corpus_overlap,
    )


# ---------------------------------------------------------------------------
# Tier 1 — structural-lookup intent detection (skips synthesis entirely)
# ---------------------------------------------------------------------------

# Minimum entity-resolution confidence required to short-circuit.  Entity must
# also be a definitive REFERS_TO match (not POSSIBLY_REFERS_TO).
#
# Threshold matches the resolver's own REFERS_TO floor (0.85, see
# core/resolver.py ResolutionPolicy).  REFERS_TO already requires a
# uniqueness gap of ≥0.05 over the next-best candidate, so by the time we
# get a REFERS_TO match the resolver has already proven the result is
# unambiguous.  Setting the Tier 1 threshold higher than the REFERS_TO
# floor gates out legitimate surface-form variants (e.g. "Turnbull Refuge"
# resolves to the canonical Turnbull entity at 0.88 via substring match)
# without buying any additional safety.
_TIER1_MIN_CONFIDENCE = 0.85


# Mapping from natural-language type words → graph entity labels for the
# frequency-ranking operation.  Only entries here will route to Tier 1; words
# not in this map fall through to synthesis (a query asking for "top 10
# something" where "something" isn't a known label is too ambiguous to short-
# circuit).  Singular and plural forms both map.
_TIER1_TYPE_KEYWORDS: dict[str, str] = {
    "person": "Person",
    "people": "Person",
    "place": "Place",
    "places": "Place",
    "location": "Place",
    "locations": "Place",
    "species": "Species",
    "organization": "Organization",
    "organizations": "Organization",
    "refuge": "Refuge",
    "refuges": "Refuge",
}

# Frequency-ranking pattern: matches "top N people", "most common places",
# "which species appear most", etc.  Captures (optional) limit and the type
# keyword.  Type keyword must be in _TIER1_TYPE_KEYWORDS or the match is
# discarded.
_TIER1_FREQ_RE = re.compile(
    r"^\s*(?:"
    r"top\s+(?P<n>\d+)\s+(?P<type1>[a-z]+)"                       # "top 10 people"
    r"|most\s+(?:documented|mentioned|frequent|common)\s+(?P<type2>[a-z]+)"  # "most common species"
    r"|which\s+(?P<type3>[a-z]+)\s+appear\s+most"                 # "which people appear most"
    r")\b",
    re.IGNORECASE,
)


# Pattern registry. Each entry is (operation, regex, requirement). Matching
# anchors at the start of the query so a buried "how many" inside a longer
# narrative question still falls through to normal synthesis.
#
# requirement is one of:
#   "one_entity"  — exactly one resolved entity at REFERS_TO ≥ 0.90
#   "exact_year"  — year_min == year_max and both non-None (single-year filter)
#   "type_keyword" — query matches _TIER1_FREQ_RE AND the captured type word
#                    is in _TIER1_TYPE_KEYWORDS (handled by detect logic, not
#                    in the simple registry loop)
_TIER1_PATTERNS: list[tuple[str, "re.Pattern[str]", str]] = [
    # "How many claims/documents/mentions about X?"
    (
        "count_claims_for_entity",
        re.compile(r"^\s*how\s+many\b", re.IGNORECASE),
        "one_entity",
    ),
    # "What years was X documented?" / "When was X documented?" / "Date range for X"
    (
        "year_range_for_entity",
        re.compile(
            r"^\s*(what\s+years?|when\s+was|date\s+range\s+for|time\s+period\s+for)\b",
            re.IGNORECASE,
        ),
        "one_entity",
    ),
    # "List documents about X" / "Show me documents mentioning X" / "Which documents mention X"
    # The required object word ("documents") + connective ("about|mentioning|that mention")
    # gates this away from generic listing queries that aren't entity-anchored.
    (
        "list_documents_for_entity",
        re.compile(
            r"^\s*(list|show\s+me|which|what)\s+documents?\s+"
            r"(about|mentioning|that\s+mention|on|covering|discuss(?:ing)?|featuring)\b",
            re.IGNORECASE,
        ),
        "one_entity",
    ),
    # "List documents from 1909" / "Show documents in 1909" / "What documents were published in 1909"
    # Year is captured by the existing year-range extractor; we only fire when
    # exactly one year was identified (year_min == year_max).
    (
        "list_documents_for_year",
        re.compile(
            r"^\s*(list|show\s+me|which|what)\s+documents?\s+"
            r"(from|in|published\s+in|dated|written\s+in)\s+\d{4}\b",
            re.IGNORECASE,
        ),
        "exact_year",
    ),
    # "Where does X appear?" / "What places are associated with X?" / "Where is X mentioned?"
    # The verb forms ("does .* appear", "is .* mentioned") and noun forms
    # ("places", "locations") all gate to the same operation.
    (
        "locations_for_entity",
        re.compile(
            r"^\s*(where\s+(does|is)|what\s+(places|locations)|which\s+(places|locations))\b",
            re.IGNORECASE,
        ),
        "one_entity",
    ),
    # "Who appears with X?" / "What entities are associated with X?" / "Co-occur with X"
    # Strict pattern — needs an explicit relational verb so a vague "who is X"
    # falls through to synthesis (which is the right move for biographical Qs).
    (
        "co_occurring_entities_for_entity",
        re.compile(
            r"^\s*(who\s+(appears|is\s+mentioned|co-?occurs?)\s+with"
            r"|what\s+entities\s+(appear|are\s+associated|co-?occur)\s+with"
            r"|which\s+entities\s+(appear|are\s+associated|co-?occur)\s+with)\b",
            re.IGNORECASE,
        ),
        "one_entity",
    ),
]


def detect_tier1_intent(
    query_text: str,
    entity_ctx: "EntityContext",
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    type_keywords: dict[str, str] | None = None,
) -> "Tier1Intent | None":
    """Return a Tier1Intent when the query matches a structural-lookup pattern
    AND its required filters resolve cleanly.  Returns ``None`` otherwise —
    the caller falls through to the normal synthesis path.

    Conservative by design: the cost of a false positive (terse graph output
    when the user wanted narrative analysis) is worse than the cost of a
    false negative (an extra few seconds of synthesis on a structural query).
    Patterns must anchor at the start of the query and required filters must
    bind cleanly; any ambiguity falls through.
    """
    if not query_text:
        return None
    # Frequency-ranking is checked separately because it requires keyword
    # extraction (entity-type word from a closed vocabulary, optional limit N).
    freq_intent = _detect_frequency_ranking(query_text, type_keywords)
    if freq_intent is not None:
        return freq_intent
    for operation, regex, requirement in _TIER1_PATTERNS:
        if not regex.match(query_text):
            continue
        if requirement == "one_entity":
            if len(entity_ctx.resolved) != 1:
                continue
            entity = entity_ctx.resolved[0]
            if entity.resolution_relation != "REFERS_TO":
                continue
            if entity.resolution_confidence < _TIER1_MIN_CONFIDENCE:
                continue
            return Tier1Intent(
                operation=operation,  # type: ignore[arg-type]
                entity_id=entity.entity_id,
                entity_name=entity.surface_form,
                entity_type=entity.entity_type,
            )
        if requirement == "exact_year":
            if year_min is None or year_min != year_max:
                continue
            return Tier1Intent(
                operation=operation,  # type: ignore[arg-type]
                entity_id="",
                entity_name=str(year_min),
                entity_type="Year",
                target_year=year_min,
            )
    return None


def _detect_frequency_ranking(
    query_text: str,
    type_keywords: dict[str, str] | None = None,
) -> "Tier1Intent | None":
    """Match top-N / most-X / which-X-appear-most patterns and resolve the
    type keyword to a graph entity label.  Returns None when the query
    doesn't match or the type word isn't in the closed vocabulary.

    *type_keywords* (from the domain's ``query_intent.yaml``) replaces the
    module default when supplied, so a corpus routes only its own entity-type
    words (e.g. a newspaper corpus never short-circuits "top 10 refuges").
    """
    keyword_map = type_keywords if type_keywords is not None else _TIER1_TYPE_KEYWORDS
    m = _TIER1_FREQ_RE.match(query_text)
    if m is None:
        return None
    type_word = (m.group("type1") or m.group("type2") or m.group("type3") or "").lower()
    label = keyword_map.get(type_word)
    if label is None:
        return None  # type word not in closed vocab → fall through to synthesis
    n_capture = m.group("n")
    limit = int(n_capture) if n_capture else 10
    # Sanity-cap the limit so a hostile query like "top 99999 people" can't
    # ask Neo4j for an unbounded result; the frontend table also won't render
    # more than a few hundred rows usefully.
    limit = max(1, min(limit, 100))
    return Tier1Intent(
        operation="frequency_ranking_for_type",
        entity_id="",
        entity_name=type_word,        # surface form for the answer text
        entity_type=label,             # canonical entity label
        target_label=label,
        limit=limit,
    )
