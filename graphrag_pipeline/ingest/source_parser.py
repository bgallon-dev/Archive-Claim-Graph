from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from graphrag_pipeline.core.ids import (
    make_annotation_id,
    make_doc_id,
    make_page_id,
    make_paragraph_id,
    make_section_id,
)
from graphrag_pipeline.core.models import (
    AnnotationRecord,
    DocumentRecord,
    PageRecord,
    ParagraphRecord,
    SectionRecord,
    StructureBundle,
)

_SECTION_NUMBER_RE = re.compile(r"^\s*(\d{1,2})[\.\)]\s+(.+?)\s*$")
_SECTION_LETTER_RE = re.compile(r"^\s*([A-Z])[\.\)]\s+(.+?)\s*$")

_MIN_PARA_CHARS: int = 150  # merge chunks smaller than this with their neighbor
_MAX_PARA_CHARS: int = 600  # split chunks larger than this at sentence boundaries

# Filename year inference: YYYYMM (e.g. 200601 → 2006) takes priority over bare years.
_YYYYMM_RE = re.compile(r"(?<!\d)(1[89]\d{2}|20[0-2]\d)\d{2}(?!\d)")
_YEAR_ONLY_RE = re.compile(r"(?<!\d)(1[89]\d{2}|20[0-2]\d)(?!\d)")


def _canonicalize_source_path(raw: str) -> str:
    """Resolve *raw* to an absolute path and verify it is within the CWD.

    Raises ValueError if the resolved path escapes the current working directory,
    preventing path traversal via a crafted metadata.source_file value.
    """
    resolved = Path(raw).resolve()
    try:
        resolved.relative_to(Path.cwd())
    except ValueError:
        raise ValueError(
            f"source_file path escapes working directory: {raw!r}"
        )
    return str(resolved)


def _infer_year_from_filename(source_file: str) -> int | None:
    """Extract the earliest (or only) year from a source filename stem.

    Handles common corpus patterns:
      - ``TBL_1938a_...``    → 1938 (letter suffix, no word-boundary issue)
      - ``TBL_1940-1941_...``  → 1940 (earliest of range)
      - ``TBL_200601_...``   → 2006 (YYYYMM format)
    """
    stem = Path(source_file).stem
    # YYYYMM check first: 6-digit sequence whose first 4 look like a year.
    yyyymm = _YYYYMM_RE.search(stem)
    if yyyymm:
        return int(yyyymm.group(1))
    years = [int(m.group(1)) for m in _YEAR_ONLY_RE.finditer(stem)]
    return min(years) if years else None


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    compact = "\n".join(line for line in lines if line)
    return re.sub(r"[ \t]+", " ", compact).strip()


def normalize_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading.strip().lower())


def detect_heading(line: str) -> tuple[int | None, str | None, str] | None:
    line = line.strip()
    if not line:
        return None
    number_match = _SECTION_NUMBER_RE.match(line)
    if number_match:
        return int(number_match.group(1)), None, number_match.group(2).strip()
    letter_match = _SECTION_LETTER_RE.match(line)
    if letter_match:
        return None, letter_match.group(1), letter_match.group(2).strip()
    return None


def split_paragraphs(raw_text: str) -> list[str]:
    normalized = raw_text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    chunks = re.split(r"\n\s*\n+", normalized)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_at_sentences(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result, current = [], ""
    for sent in sentences:
        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_chars:
            current += " " + sent
        else:
            result.append(current)
            current = sent
    if current:
        result.append(current)
    return result or [text]


def _merge_small_chunks(chunks: list[str], min_chars: int) -> list[str]:
    result = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        while len(current) < min_chars and i + 1 < len(chunks):
            i += 1
            current = current + " " + chunks[i]
        result.append(current)
        i += 1
    return result


def normalize_paragraph_sizes(
    chunks: list[str],
    min_chars: int = _MIN_PARA_CHARS,
    max_chars: int = _MAX_PARA_CHARS,
) -> list[str]:
    split: list[str] = []
    for chunk in chunks:
        split.extend(_split_at_sentences(chunk, max_chars) if len(chunk) > max_chars else [chunk])
    return _merge_small_chunks(split, min_chars)


def parse_source_payload(payload: dict[str, Any], source_file: str | None = None) -> StructureBundle:
    metadata = payload.get("metadata", {})
    page_payloads = sorted(payload.get("pages", []), key=lambda row: int(row.get("page_number", 0)))
    title = metadata.get("title") or metadata.get("doc_title") or "Untitled Narrative Report"
    date_start = metadata.get("date_start")
    date_end = metadata.get("date_end")
    report_year = metadata.get("report_year")
    if report_year is None and isinstance(date_start, str) and len(date_start) >= 4 and date_start[:4].isdigit():
        report_year = int(date_start[:4])

    _meta_source = metadata.get("source_file")
    resolved_source = source_file or _meta_source
    if report_year is None and resolved_source:
        report_year = _infer_year_from_filename(resolved_source)
    doc_id = metadata.get("doc_id") or make_doc_id(title, date_start, date_end, resolved_source)
    raw_ocr_text = "\n\n".join(str(row.get("raw_ocr_text") or row.get("raw_text", "")) for row in page_payloads)
    clean_text = normalize_text(raw_ocr_text)
    file_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    language = metadata.get("language", "en")

    document = DocumentRecord(
        doc_id=doc_id,
        title=title,
        doc_type=metadata.get("doc_type", "narrative_report"),
        series=metadata.get("series"),
        date_start=date_start,
        date_end=date_end,
        report_year=report_year,
        source_file=resolved_source,
        archive_ref=metadata.get("archive_ref"),
        raw_ocr_text=raw_ocr_text,
        clean_text=clean_text,
        language=language,
        file_hash=file_hash,
        page_count=len(page_payloads),
    )

    pages: list[PageRecord] = []
    sections: list[SectionRecord] = []
    paragraphs: list[ParagraphRecord] = []
    annotations: list[AnnotationRecord] = []

    section_index = 0
    paragraph_index = 0
    current_section_id: str | None = None
    current_section_heading: str | None = None

    for page_item in page_payloads:
        page_number = int(page_item.get("page_number"))
        raw_page_ocr_text = str(page_item.get("raw_ocr_text") or page_item.get("raw_text", ""))
        clean_page_text = normalize_text(raw_page_ocr_text)
        page_id = make_page_id(doc_id, page_number)
        page = PageRecord(
            page_id=page_id,
            doc_id=doc_id,
            page_number=page_number,
            raw_ocr_text=raw_page_ocr_text,
            clean_text=clean_page_text,
            ocr_confidence=page_item.get("ocr_confidence"),
            image_ref=page_item.get("image_ref"),
        )
        pages.append(page)

        for line in raw_page_ocr_text.splitlines():
            heading_match = detect_heading(line)
            if not heading_match:
                continue
            section_number, section_letter, heading = heading_match
            normalized_heading = normalize_heading(heading)
            if normalized_heading != current_section_heading:
                section_index += 1
                current_section_id = make_section_id(doc_id, section_index, heading)
                sections.append(
                    SectionRecord(
                        section_id=current_section_id,
                        doc_id=doc_id,
                        heading=heading,
                        section_number=section_number,
                        section_letter=section_letter,
                        normalized_heading=normalized_heading,
                        page_start=page_number,
                        page_end=page_number,
                    )
                )
                current_section_heading = normalized_heading
            elif sections:
                sections[-1].page_end = page_number

        if current_section_id is None:
            section_index += 1
            fallback_heading = f"Page {page_number}"
            current_section_id = make_section_id(doc_id, section_index, fallback_heading)
            sections.append(
                SectionRecord(
                    section_id=current_section_id,
                    doc_id=doc_id,
                    heading=fallback_heading,
                    section_number=section_index,
                    section_letter=None,
                    normalized_heading=normalize_heading(fallback_heading),
                    page_start=page_number,
                    page_end=page_number,
                )
            )

        raw_chunks = [
            chunk for chunk in split_paragraphs(raw_page_ocr_text)
            if not (detect_heading(chunk) and len(chunk.splitlines()) == 1)
        ]
        for paragraph_text in normalize_paragraph_sizes(raw_chunks):
            paragraph_index += 1
            clean_paragraph = normalize_text(paragraph_text)
            paragraph_id = make_paragraph_id(doc_id, paragraph_index, page_number)
            paragraphs.append(
                ParagraphRecord(
                    paragraph_id=paragraph_id,
                    doc_id=doc_id,
                    page_id=page_id,
                    section_id=current_section_id,
                    paragraph_index=paragraph_index,
                    page_number=page_number,
                    raw_ocr_text=paragraph_text,
                    clean_text=clean_paragraph,
                    char_count=len(clean_paragraph),
                )
            )

        for idx, annotation in enumerate(page_item.get("annotations", []), start=1):
            kind = str(annotation.get("kind", "note"))
            text = str(annotation.get("text", ""))
            annotations.append(
                AnnotationRecord(
                    annotation_id=make_annotation_id(doc_id, page_number, idx, kind, text),
                    doc_id=doc_id,
                    page_id=page_id,
                    page_number=page_number,
                    kind=kind,
                    text=text,
                    bbox=annotation.get("bbox"),
                )
            )

    if not sections:
        section_index += 1
        fallback_heading = "General"
        fallback_id = make_section_id(doc_id, section_index, fallback_heading)
        start_page = pages[0].page_number if pages else 1
        end_page = pages[-1].page_number if pages else 1
        sections.append(
            SectionRecord(
                section_id=fallback_id,
                doc_id=doc_id,
                heading=fallback_heading,
                section_number=1,
                section_letter=None,
                normalized_heading=normalize_heading(fallback_heading),
                page_start=start_page,
                page_end=end_page,
            )
        )
        for paragraph in paragraphs:
            paragraph.section_id = fallback_id

    return StructureBundle(
        document=document,
        pages=pages,
        sections=sections,
        paragraphs=paragraphs,
        annotations=annotations,
    )


def parse_source_file(path: str | Path) -> StructureBundle:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return parse_source_payload(payload, source_file=str(source_path))
