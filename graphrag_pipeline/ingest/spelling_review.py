from __future__ import annotations

import difflib
import re
from collections import defaultdict
from typing import Any

from .extractors.mention_extractor import get_ocr_flags
from graphrag_pipeline.core.ids import stable_hash
from graphrag_pipeline.core.models import MentionRecord, SemanticBundle, StructureBundle
from graphrag_pipeline.shared.resource_loader import (
    load_ocr_correction_map,
    load_seed_entity_rows,
    load_spelling_reference_terms,
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_MONTHS = frozenset({
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
    "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december",
})
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "from", "with", "into", "onto",
    "near", "only", "about", "around", "after", "before", "during", "under",
    "over", "were", "was", "are", "is", "this", "that", "these", "those", "for",
    "not", "have", "had", "has", "been", "being", "their", "there", "which",
    "while", "where", "when", "also", "than", "then",
})
_OCR_CORRECTION_MAP = load_ocr_correction_map()
_OCR_CORRECTIONS = frozenset(_OCR_CORRECTION_MAP)
_DOMAIN_REFERENCE_TERMS = load_spelling_reference_terms()
_SEED_REFERENCE_TERMS = frozenset(
    token.lower()
    for row in load_seed_entity_rows()
    for token in _TOKEN_RE.findall(str(row.get("name", "")).lower())
)
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


def _normalize_token(token: str) -> str:
    return re.sub(r"\s+", " ", token.strip().lower())


def _claim_span_start_end(evidence_start: int | None, evidence_end: int | None, paragraph_text: str) -> tuple[int, int]:
    if evidence_start is not None and evidence_end is not None:
        return evidence_start, evidence_end
    return 0, len(paragraph_text)


def _iter_tokens(text: str) -> list[str]:
    return [match.group(0) for match in _TOKEN_RE.finditer(text)]


def _best_match(
    token: str,
    candidates: frozenset[str],
    *,
    threshold: float,
    require_same_initial: bool = True,
) -> tuple[str, float] | tuple[None, float]:
    best_term: str | None = None
    best_ratio = 0.0
    for candidate in candidates:
        if candidate == token:
            continue
        if abs(len(candidate) - len(token)) > 2:
            continue
        if require_same_initial and candidate[:1] != token[:1]:
            continue
        ratio = difflib.SequenceMatcher(None, token, candidate).ratio()
        if ratio > best_ratio:
            best_term = candidate
            best_ratio = ratio
    if best_term is None or best_ratio < threshold:
        return None, 0.0
    return best_term, best_ratio


def _suggest_correction(token: str) -> tuple[str, float]:
    exact = _OCR_CORRECTION_MAP.get(token)
    if exact:
        return exact, 1.0

    seed_match, seed_ratio = _best_match(token, _SEED_REFERENCE_TERMS, threshold=0.84)
    if seed_match:
        return seed_match, seed_ratio

    domain_match, domain_ratio = _best_match(token, _DOMAIN_REFERENCE_TERMS, threshold=0.90)
    if domain_match:
        return domain_match, domain_ratio

    return "", 0.0


def _token_is_ignorable(token: str) -> bool:
    normalized = _normalize_token(token)
    if len(normalized) < 4:
        return True
    if normalized in _STOPWORDS or normalized in _MONTHS:
        return True
    return False


def _is_known_domain_term(token: str) -> bool:
    if token in _DOMAIN_REFERENCE_TERMS:
        return True
    candidates = {token}
    if token.endswith("s") and len(token) > 4:
        candidates.add(token[:-1])
    if token.endswith("es") and len(token) > 5:
        candidates.add(token[:-2])
    if token.endswith("ies") and len(token) > 5:
        candidates.add(f"{token[:-3]}y")
    return any(candidate in _DOMAIN_REFERENCE_TERMS for candidate in candidates)


def _issue_severity(flags: set[str], suggestion: str) -> str:
    if "ocr_suspect_list" in flags:
        return "high"
    if suggestion and any(flag.startswith("near_seed_term:") for flag in flags):
        return "high"
    if "mention_ocr_suspect" in flags or "digit_in_token" in flags or "rn_m_confusion" in flags or len(flags) >= 2:
        return "medium"
    return "low"


def _merge_issue_row(existing: dict[str, Any], flags: set[str], severity: str, suggested_correction: str) -> None:
    merged_flags = set(existing["flags"])
    merged_flags.update(flags)
    existing["flags"] = sorted(merged_flags)
    if _SEVERITY_RANK[severity] > _SEVERITY_RANK[existing["severity"]]:
        existing["severity"] = severity
    if not existing["suggested_correction"] and suggested_correction:
        existing["suggested_correction"] = suggested_correction


def _issue_row(
    *,
    structure: StructureBundle,
    claim: Any,
    paragraph: Any,
    page: Any,
    suspect_text: str,
    normalized_suspect_text: str,
    flags: set[str],
    suggested_correction: str,
) -> dict[str, Any]:
    review_target_type = "image_ref" if page and page.image_ref else "pdf_page"
    return {
        "issue_id": f"spell_{stable_hash(claim.claim_id, normalized_suspect_text)}",
        "issue_type": "spelling_candidate",
        "severity": _issue_severity(flags, suggested_correction),
        "doc_id": structure.document.doc_id,
        "run_id": claim.run_id,
        "claim_id": claim.claim_id,
        "paragraph_id": claim.paragraph_id,
        "page_id": paragraph.page_id if paragraph else None,
        "page_number": paragraph.page_number if paragraph else None,
        "claim_type": claim.claim_type,
        "source_sentence": claim.source_sentence,
        "normalized_sentence": claim.normalized_sentence,
        "suspect_text": suspect_text,
        "normalized_suspect_text": normalized_suspect_text,
        "flags": sorted(flags),
        "suggested_correction": suggested_correction,
        "review_target_type": review_target_type,
        "image_ref": page.image_ref if page else None,
        "source_file": structure.document.source_file,
    }


def _mention_flags_by_token(mentions: list[MentionRecord]) -> dict[str, set[str]]:
    flags_by_token: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        if not mention.ocr_suspect:
            continue
        for token in _iter_tokens(mention.surface_form):
            token_flags = set(get_ocr_flags(token))
            if not token_flags:
                continue
            token_flags.add("mention_ocr_suspect")
            flags_by_token[_normalize_token(token)].update(token_flags)
    return flags_by_token


def build_spelling_review_queue(structure: StructureBundle, semantic: SemanticBundle) -> list[dict[str, Any]]:
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in structure.paragraphs}
    page_by_id = {page.page_id: page for page in structure.pages}
    mentions_by_paragraph: dict[str, list[MentionRecord]] = defaultdict(list)
    for mention in semantic.mentions:
        mentions_by_paragraph[mention.paragraph_id].append(mention)

    issues: dict[tuple[str, str], dict[str, Any]] = {}

    for claim in semantic.claims:
        paragraph = paragraph_by_id.get(claim.paragraph_id)
        if paragraph is None:
            continue
        page = page_by_id.get(paragraph.page_id)
        paragraph_text = paragraph.clean_text or paragraph.raw_ocr_text
        span_start, span_end = _claim_span_start_end(claim.evidence_start, claim.evidence_end, paragraph_text)
        mention_flags = _mention_flags_by_token(
            [
                mention
                for mention in mentions_by_paragraph.get(claim.paragraph_id, [])
                if mention.start_offset >= span_start and mention.end_offset <= span_end
            ]
        )

        for token in _iter_tokens(claim.source_sentence):
            normalized = _normalize_token(token)
            if not normalized:
                continue

            flags = set(get_ocr_flags(token))
            flags.update(mention_flags.get(normalized, set()))
            suggested_correction, _ = _suggest_correction(normalized)
            if flags and flags.issubset({"rn_m_confusion", "mention_ocr_suspect"}) and not suggested_correction:
                continue
            has_hard_ocr_signal = "ocr_suspect_list" in flags or "digit_in_token" in flags
            if _is_known_domain_term(normalized) and normalized not in _OCR_CORRECTIONS and not has_hard_ocr_signal:
                continue

            if not flags:
                if _token_is_ignorable(token) or _is_known_domain_term(normalized):
                    continue
                if not suggested_correction:
                    continue
                flags.add(f"spellcheck_candidate:{suggested_correction}")

            elif _is_known_domain_term(normalized) and normalized not in _OCR_CORRECTIONS and not has_hard_ocr_signal:
                continue

            key = (claim.claim_id, normalized)
            if key in issues:
                _merge_issue_row(issues[key], flags, _issue_severity(flags, suggested_correction), suggested_correction)
                continue
            issues[key] = _issue_row(
                structure=structure,
                claim=claim,
                paragraph=paragraph,
                page=page,
                suspect_text=token,
                normalized_suspect_text=normalized,
                flags=flags,
                suggested_correction=suggested_correction,
            )

    rows = list(issues.values())
    rows.sort(key=lambda row: (row["doc_id"], row["page_number"] or 0, row["claim_id"], row["normalized_suspect_text"]))
    return rows
