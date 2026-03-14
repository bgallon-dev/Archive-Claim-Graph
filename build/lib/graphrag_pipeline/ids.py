from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def _normalize_part(value: object) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def stable_hash(*parts: object, size: int = 16) -> str:
    normalized = "||".join(_normalize_part(part) for part in parts)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest[:size]


def make_doc_id(title: str, date_start: str | None, date_end: str | None, source_file: str | None) -> str:
    return f"doc_{stable_hash(title, date_start, date_end, source_file)}"


def make_page_id(doc_id: str, page_number: int) -> str:
    return f"{doc_id}:page:{page_number:04d}"


def make_section_id(doc_id: str, section_index: int, heading: str) -> str:
    return f"{doc_id}:section:{section_index:03d}:{stable_hash(heading, size=8)}"


def make_paragraph_id(doc_id: str, paragraph_index: int, page_number: int) -> str:
    return f"{doc_id}:paragraph:{paragraph_index:05d}:p{page_number:04d}"


def make_annotation_id(doc_id: str, page_number: int, index: int, kind: str, text: str) -> str:
    return f"{doc_id}:annotation:{page_number:04d}:{index:03d}:{stable_hash(kind, text, size=8)}"


def make_run_id(ocr_engine: str = "unknown", now: datetime | None = None) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}_{stable_hash(ocr_engine, ts, size=8)}"


def make_claim_id(run_id: str, paragraph_id: str, claim_index: int, normalized_sentence: str) -> str:
    return f"{run_id}:claim:{claim_index:03d}:{stable_hash(paragraph_id, normalized_sentence)}"


def make_measurement_id(
    run_id: str,
    claim_id: str,
    measurement_index: int,
    name: str,
    raw_value: str,
) -> str:
    return f"{run_id}:measurement:{measurement_index:03d}:{stable_hash(claim_id, name, raw_value)}"


def make_mention_id(
    run_id: str,
    paragraph_id: str,
    start_offset: int,
    end_offset: int,
    surface_form: str,
) -> str:
    return f"{run_id}:mention:{stable_hash(paragraph_id, start_offset, end_offset, surface_form)}"


def make_entity_id(label: str, normalized_name: str) -> str:
    return f"{label.lower()}_{stable_hash(label, normalized_name)}"


def make_period_id(start_date: str | None, end_date: str | None, label: str | None = None) -> str:
    return f"period_{stable_hash(start_date, end_date, label)}"
