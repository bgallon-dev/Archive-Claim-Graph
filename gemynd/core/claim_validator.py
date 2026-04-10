from __future__ import annotations

import re

_DEFAULT_VERB_RE = re.compile(
    r"\b(was|were|is|are|had|have|has|been|being|be|did|do|does)\b",
    re.IGNORECASE,
)

# Structural section heading: short line that is EITHER all-uppercase words
# OR title-case with no lowercase-only words (excluding tiny function words).
# Must end with optional colon/period.  Lines with 2+ proper nouns or mixed
# case (e.g. newspaper headlines like "Big Fire in Hillyard") are NOT matched.
#
# The previous pattern (^[A-Z][A-Za-z\s]{0,40}[:\.]?\s*$) was too broad and
# would misclassify newspaper headlines, entity names, and title-case prose
# fragments as section headings.
_HEADING_RE = re.compile(
    r"^(?:"
    # All-caps line with ≤5 words (structural labels like "WILDLIFE MANAGEMENT")
    r"(?:[A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})"
    r"|"
    # Title-case line with ≤4 words, all starting uppercase, no digits
    # (e.g. "Introduction", "Fire Control", "Wildlife Management Plan")
    r"(?:[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|and|of|the|for|in|on|at|to|by)){0,3})"
    r")[:\.]?\s*$"
)
_LIST_ARTIFACT_RE = re.compile(r"^[A-Z\-\•\*\d][\-\.\)]\s{0,3}\w{0,15}\.?\s*$")
_TRUNCATED_HONORIFIC_RE = re.compile(
    r"\b(Mr|Mrs|Dr|Gov|Col|Lt|Sgt|St|Jr|Sr|Mgr|Dept|Approx|"
    r"Oct|Nov|Dec|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?)\.\s*$",
    re.IGNORECASE,
)
_ALPHA_THRESHOLD = 0.50
MIN_CHARS = 20
MIN_WORDS = 3


def is_valid_claim_sentence(
    sentence: str,
    verb_re: re.Pattern[str] | None = None,
    heading_re: re.Pattern[str] | None = _HEADING_RE,
) -> tuple[bool, str]:
    effective_verb_re = verb_re if verb_re is not None else _DEFAULT_VERB_RE
    text = sentence.strip()
    if len(text) < MIN_CHARS:
        return False, f"too_short:{len(text)}_chars"
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < MIN_WORDS:
        return False, f"too_few_words:{word_count}"
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < _ALPHA_THRESHOLD:
        return False, f"low_alpha_density:{alpha_ratio:.2f}"
    if heading_re is not None and heading_re.match(text) and not effective_verb_re.search(text):
        return False, "section_heading"
    if _LIST_ARTIFACT_RE.match(text):
        return False, "list_artifact"
    if not effective_verb_re.search(text):
        return False, "no_finite_verb"
    if _TRUNCATED_HONORIFIC_RE.search(text):
        return False, "truncated_at_honorific"
    return True, ""
