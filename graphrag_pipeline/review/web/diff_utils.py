"""Diff and highlight utilities for the source comparison view."""
from __future__ import annotations

import difflib
import html
import re


def char_diff_html(a: str, b: str) -> tuple[str, str]:
    """Return (html_a, html_b) with character-level diff spans.

    Characters only in *a* get ``<span class="diff-del">``,
    characters only in *b* get ``<span class="diff-ins">``.
    Unchanged characters are HTML-escaped.
    """
    sm = difflib.SequenceMatcher(None, a, b)
    parts_a: list[str] = []
    parts_b: list[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts_a.append(html.escape(a[i1:i2]))
            parts_b.append(html.escape(b[j1:j2]))
        elif tag == "delete":
            parts_a.append(f'<span class="diff-del">{html.escape(a[i1:i2])}</span>')
        elif tag == "insert":
            parts_b.append(f'<span class="diff-ins">{html.escape(b[j1:j2])}</span>')
        elif tag == "replace":
            parts_a.append(f'<span class="diff-del">{html.escape(a[i1:i2])}</span>')
            parts_b.append(f'<span class="diff-ins">{html.escape(b[j1:j2])}</span>')

    return "".join(parts_a), "".join(parts_b)


def highlight_in_text(text: str, term: str, max_len: int = 2000) -> str:
    """Wrap case-insensitive occurrences of *term* in ``<mark>`` tags.

    The rest of the text is HTML-escaped.  Truncates *text* to *max_len*
    characters before processing.
    """
    text = text[:max_len]
    if not term:
        return html.escape(text)

    pattern = re.compile(re.escape(term), re.IGNORECASE)
    parts: list[str] = []
    last = 0
    for m in pattern.finditer(text):
        parts.append(html.escape(text[last:m.start()]))
        parts.append(f'<mark class="source-highlight">{html.escape(m.group())}</mark>')
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)
