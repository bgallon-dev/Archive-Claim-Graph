"""Junk mention suppression detector.

Emits `suppress_mention` proposals for headers, boilerplate, short generic
tokens, and clear OCR garbage mentions.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from ...ingest.extractors.mention_extractor import get_ocr_flags
from ...core.models import MentionRecord, ParagraphRecord, SemanticBundle, StructureBundle
from ..models import DetectorProposal, ProposalTarget
from ..patch_spec import make_patch_spec

if TYPE_CHECKING:
    from ...core.domain_config import DomainConfig

_log = logging.getLogger(__name__)

DETECTOR_NAME = "junk_mention_detector"
DETECTOR_VERSION = "v1"

# Universal structural header/footer patterns — page numbers and bare digits.
# These apply to every domain regardless of config.
_STRUCTURAL_HEADER_PATTERNS = [
    re.compile(r"^page\s+\d+", re.IGNORECASE),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^-\s*\d+\s*-$"),
]

# Turnbull federal-agency report headers. Applied only on the ``config is None``
# fallback path; for a real run these come from the domain's ``junk_mention``
# block (Turnbull declares exactly these). A domain that declares none gets only
# the structural patterns above — never these institutional strings.
_TURNBULL_HEADER_PATTERNS = [
    re.compile(r"united\s+states\s+department", re.IGNORECASE),
    re.compile(r"fish\s+and\s+wildlife\s+service", re.IGNORECASE),
    re.compile(r"bureau\s+of\s+(?:sport\s+)?fisheries", re.IGNORECASE),
    re.compile(r"department\s+of\s+(?:the\s+)?interior", re.IGNORECASE),
    re.compile(r"annual\s+narrative\s+report", re.IGNORECASE),
    re.compile(r"refuge\s+manager", re.IGNORECASE),
]

# Full list used when no config is supplied (byte-identical to legacy behaviour).
_HEADER_PATTERNS = _STRUCTURAL_HEADER_PATTERNS + _TURNBULL_HEADER_PATTERNS


@lru_cache(maxsize=16)
def _compile_domain_header_patterns(raw_patterns: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    """Compile domain-supplied header regexes (case-insensitive), skipping any
    that are invalid rather than crashing the whole detector run."""
    compiled: list[re.Pattern] = []
    for raw in raw_patterns:
        try:
            compiled.append(re.compile(raw, re.IGNORECASE))
        except re.error:
            _log.warning("junk_mention header pattern is invalid regex; skipping: %r", raw)
    return tuple(compiled)

# Boilerplate phrases
_BOILERPLATE_PATTERNS = [
    re.compile(r"submitted\s+by", re.IGNORECASE),
    re.compile(r"prepared\s+by", re.IGNORECASE),
    re.compile(r"approved\s+by", re.IGNORECASE),
    re.compile(r"date\s*:", re.IGNORECASE),
    re.compile(r"signature\s*:", re.IGNORECASE),
    re.compile(r"copies\s+to\s*:", re.IGNORECASE),
    re.compile(r"distribution\s*:", re.IGNORECASE),
    re.compile(r"table\s+of\s+contents", re.IGNORECASE),
]

# Short generic tokens
_SHORT_GENERIC_TOKENS = frozenset({
    "the", "this", "that", "they", "them", "been", "were", "have", "area",
    "also", "made", "used", "year", "june", "july", "work", "much", "many",
    "some", "most", "each", "more", "last", "only", "same", "well", "time",
    "very", "good", "high", "both", "will", "from", "with", "into", "over",
    "under", "near", "here", "there",
})

# OCR garbage pattern: high proportion of non-alpha chars
_GARBAGE_RE = re.compile(r"^[^a-zA-Z]*$")
_MOSTLY_DIGITS_RE = re.compile(r"^\d{2,}[a-zA-Z]{0,2}$")


def _is_header_paragraph(
    paragraph: ParagraphRecord,
    page_paragraphs: list[ParagraphRecord],
    header_patterns: "tuple[re.Pattern, ...] | list[re.Pattern]" = _HEADER_PATTERNS,
) -> bool:
    """Check if paragraph is in the first or last position on a page (likely header/footer)."""
    if not page_paragraphs:
        return False
    text = (paragraph.clean_text or paragraph.raw_ocr_text).strip()
    if len(text) < 10:
        return True
    sorted_paras = sorted(page_paragraphs, key=lambda p: p.paragraph_index)
    if paragraph.paragraph_id in (sorted_paras[0].paragraph_id, sorted_paras[-1].paragraph_id):
        return any(pat.search(text) for pat in header_patterns)
    return False


def _is_boilerplate_paragraph(paragraph: ParagraphRecord) -> bool:
    text = (paragraph.clean_text or paragraph.raw_ocr_text).strip()
    return any(pat.search(text) for pat in _BOILERPLATE_PATTERNS)


def _is_short_generic_mention(mention: MentionRecord) -> bool:
    normalized = mention.normalized_form.strip().lower()
    if len(normalized) < 4:
        return True
    if normalized in _SHORT_GENERIC_TOKENS:
        return True
    return False


def _is_ocr_garbage_mention(
    mention: MentionRecord,
    known_ocr_errors: frozenset[str] | None = None,
    lexicon_hints: dict[str, list[str]] | None = None,
) -> bool:
    text = mention.surface_form.strip()
    if not text:
        return True
    if _GARBAGE_RE.match(text):
        return True
    if _MOSTLY_DIGITS_RE.match(text):
        return True
    # High ratio of non-alpha characters
    alpha = sum(1 for c in text if c.isalpha())
    if len(text) > 2 and alpha / len(text) < 0.4:
        return True
    flags = get_ocr_flags(text, known_ocr_errors=known_ocr_errors, lexicon_hints=lexicon_hints)
    if "ocr_suspect_list" in flags and len(text) <= 4:
        return True
    return False


def detect(
    structure: StructureBundle,
    semantic: SemanticBundle,
    snapshot_id: str,
    config: "DomainConfig | None" = None,
) -> list[DetectorProposal]:
    """Run the junk mention suppression detector.

    When *config* is provided, the OCR-suspect check uses the domain's OCR
    correction set and seed lexicon rather than Turnbull's. Without a config,
    ``get_ocr_flags`` falls back to the explicit Turnbull reference vocabulary.
    """
    known_ocr_errors: frozenset[str] | None = None
    lexicon_hints: dict[str, list[str]] | None = None
    if config is not None:
        known_ocr_errors = config.ocr_corrections
        lexicon_hints = {}
        for entity in config.seed_entities:
            lexicon_hints.setdefault(entity.name.lower(), []).append(entity.entity_type)

    # Header patterns: structural (universal) + domain-specific from config.
    # No config → full legacy list (structural + Turnbull institutional).
    if config is not None:
        header_patterns: tuple[re.Pattern, ...] = tuple(_STRUCTURAL_HEADER_PATTERNS) + _compile_domain_header_patterns(
            tuple(config.junk_mention_filters.get("header_patterns") or ())
        )
    else:
        header_patterns = tuple(_HEADER_PATTERNS)

    paragraphs_by_id = {p.paragraph_id: p for p in structure.paragraphs}
    pages_by_id = {p.page_id: p for p in structure.pages}
    paragraphs_by_page: dict[str, list[ParagraphRecord]] = defaultdict(list)
    for p in structure.paragraphs:
        paragraphs_by_page[p.page_id].append(p)

    # Group mentions by suppression reason
    suppress_groups: dict[str, list[tuple[MentionRecord, str, dict[str, Any]]]] = {
        "header_contamination": [],
        "boilerplate_contamination": [],
        "short_generic_token": [],
        "ocr_garbage": [],
    }

    for mention in semantic.mentions:
        paragraph = paragraphs_by_id.get(mention.paragraph_id)
        if not paragraph:
            continue
        page = pages_by_id.get(paragraph.page_id)

        evidence_context = {
            "mention_id": mention.mention_id,
            "surface_form": mention.surface_form,
            "normalized_form": mention.normalized_form,
            "paragraph_id": mention.paragraph_id,
            "page_id": paragraph.page_id if paragraph else None,
            "page_number": paragraph.page_number if paragraph else None,
            "image_ref": page.image_ref if page else None,
            "source_file": structure.document.source_file,
            "paragraph_raw_ocr_text": (paragraph.raw_ocr_text or "")[:2000] if paragraph else "",
            "paragraph_clean_text": (paragraph.clean_text or "")[:2000] if paragraph else "",
        }

        page_paras = paragraphs_by_page.get(paragraph.page_id, [])
        if _is_header_paragraph(paragraph, page_paras, header_patterns):
            suppress_groups["header_contamination"].append(
                (mention, "header_contamination", evidence_context)
            )
        elif _is_boilerplate_paragraph(paragraph):
            suppress_groups["boilerplate_contamination"].append(
                (mention, "boilerplate_contamination", evidence_context)
            )
        elif _is_ocr_garbage_mention(mention, known_ocr_errors, lexicon_hints):
            suppress_groups["ocr_garbage"].append(
                (mention, "ocr_garbage", evidence_context)
            )
        elif _is_short_generic_mention(mention):
            suppress_groups["short_generic_token"].append(
                (mention, "short_generic_token", evidence_context)
            )

    proposals: list[DetectorProposal] = []

    issue_class_map = {
        "header_contamination": "header_contamination",
        "boilerplate_contamination": "boilerplate_contamination",
        "short_generic_token": "short_generic_token",
        "ocr_garbage": "ocr_garbage_mention",
    }
    anti_pattern_map = {
        "header_contamination": "ap_header_contamination",
        "boilerplate_contamination": "ap_boilerplate_contamination",
        "short_generic_token": "ap_short_generic",
        "ocr_garbage": "ap_ocr_garbage",
    }

    for reason, items in suppress_groups.items():
        if not items:
            continue

        mention_ids = sorted(m.mention_id for m, _, _ in items)
        confidence = 0.85 if reason in ("header_contamination", "boilerplate_contamination") else 0.70

        targets = [
            ProposalTarget(
                proposal_id="",
                target_kind="mention",
                target_id=m.mention_id,
                target_role="suppressed",
                exists_in_snapshot=True,
            )
            for m, _, _ in items
        ]

        patch = make_patch_spec(
            "suppress_mention",
            mention_ids=mention_ids,
            suppression_reason=reason,
            scope="semantic_only",
        )

        evidence = {
            "confidence": confidence,
            "issue_class": issue_class_map[reason],
            "impact_size": len(items),
            "suppression_reason": reason,
            "affected_mentions": [ctx for _, _, ctx in items],
            "source_file": structure.document.source_file,
        }

        proposals.append(DetectorProposal(
            anti_pattern_id=anti_pattern_map[reason],
            issue_class=issue_class_map[reason],
            proposal_type="suppress_mention",
            confidence=confidence,
            targets=targets,
            patch_spec=patch,
            evidence_snapshot=evidence,
            reasoning_summary={"reason": f"{len(items)} mention(s) flagged as {reason}"},
            detector_name=DETECTOR_NAME,
            detector_version=DETECTOR_VERSION,
        ))

    return proposals
