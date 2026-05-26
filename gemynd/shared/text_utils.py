from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def normalize_for_match(text: str) -> str:
    return collapse_whitespace(text.lower())
