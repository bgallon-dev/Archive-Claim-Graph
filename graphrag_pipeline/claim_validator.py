from __future__ import annotations

import re

_VERB_RE = re.compile(
    r"\b(was|were|is|are|had|have|has|been|being|be|"
    r"burned|started|fell|dried|came|reared|taken|seen|"
    r"made|consisted|allowed|felt|needed|noted|handled|"
    r"occurred|controlled|inhabited|planted|attended|"
    r"continued|provided|lasted|reached|produced|caused|"
    r"resulted|reported|recorded|conducted|completed|"
    r"observed|detected|identified|received|increased|decreased)\b",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^[A-Z][A-Za-z\s]{0,40}[:\.]?\s*$")
_LIST_ARTIFACT_RE = re.compile(r"^[A-Z\-\•\*\d][\-\.\)]\s{0,3}\w{0,15}\.?\s*$")
_TRUNCATED_HONORIFIC_RE = re.compile(
    r"\b(Mr|Mrs|Dr|Gov|Col|Lt|Sgt|St|Jr|Sr|Mgr|Dept|Approx|"
    r"Oct|Nov|Dec|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?)\.\s*$",
    re.IGNORECASE,
)
_ALPHA_THRESHOLD = 0.50
MIN_CHARS = 20
MIN_WORDS = 3


def is_valid_claim_sentence(sentence: str) -> tuple[bool, str]:
    text = sentence.strip()
    if len(text) < MIN_CHARS:
        return False, f"too_short:{len(text)}_chars"
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < MIN_WORDS:
        return False, f"too_few_words:{word_count}"
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < _ALPHA_THRESHOLD:
        return False, f"low_alpha_density:{alpha_ratio:.2f}"
    if _HEADING_RE.match(text) and not _VERB_RE.search(text):
        return False, "section_heading"
    if _LIST_ARTIFACT_RE.match(text):
        return False, "list_artifact"
    if not _VERB_RE.search(text):
        return False, "no_finite_verb"
    if _TRUNCATED_HONORIFIC_RE.search(text):
        return False, "truncated_at_honorific"
    return True, ""
