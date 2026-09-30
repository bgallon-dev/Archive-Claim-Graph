"""Query-term handling shared by the evidence backends and coverage scoring.

Agents tend to send natural-language questions ("Who was Jimmy Durkin in
Spokane?").  Matching every token lets function words such as "who", "was",
and "in" dominate lexical retrieval, so only content terms are searched.
"""

from __future__ import annotations

import math
import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORD_TEXT = """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    else ever few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no nor not
    now of off on once only or other our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom
    whose why will with would you your yours yourself yourselves
    tell show find give know explain describe list please information info anything
    something happened happen history historical record records evidence source sources
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def query_terms(text: str) -> list[str]:
    """Return ordered, de-duplicated content terms for a search question."""
    seen: dict[str, None] = {}
    for token in tokens(text):
        if token in STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        seen.setdefault(token, None)
    return list(seen)


def row_text(row: dict[str, Any]) -> str:
    """Searchable text of a raw or normalized evidence row."""
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    parts = [
        str(row.get(key) or "")
        for key in ("claim", "excerpt", "source_sentence", "document", "title", "text")
    ]
    parts.append(str(source.get("document") or ""))
    for subject in row.get("subjects") or []:
        if isinstance(subject, dict):
            parts.append(str(subject.get("name") or ""))
    parts.extend(str(item) for item in row.get("place") or [] if item is not None)
    return " ".join(parts)


def matched_terms(terms: list[str], row: dict[str, Any]) -> list[str]:
    present = set(tokens(row_text(row)))
    return [term for term in terms if term in present]


def strong_match_threshold(terms: list[str]) -> int:
    """Terms a row must contain to count as a strong match for the question."""
    return max(1, math.ceil(0.6 * len(terms)))
