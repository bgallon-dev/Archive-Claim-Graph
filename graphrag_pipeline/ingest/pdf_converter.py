"""PDF-to-JSON converter for the ingest pipeline.

Refactored from tools/pdf_to_json.py so it can be imported by the web layer.
The core extraction logic is unchanged; the public interface is the single
``convert_pdf_to_json`` function.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ConversionError(RuntimeError):
    """Raised when PDF-to-JSON conversion fails."""


# ---------------------------------------------------------------------------
# Page text extraction
# ---------------------------------------------------------------------------

def _extract_pages_pymupdf_text(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text from PDF pages using PyMuPDF's built-in text layer."""
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    pages = []
    for i, pdf_page in enumerate(doc, start=1):
        text = pdf_page.get_text()
        pages.append({
            "page_number": i,
            "raw_text": text,
            "ocr_confidence": None,
            "_has_text": bool(text.strip()),
        })
    doc.close()
    return pages


def _ocr_page_image(pil_image: Any) -> tuple[str, float]:
    """OCR a PIL image with pytesseract; return (text, mean_confidence)."""
    try:
        import pytesseract  # type: ignore
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    except Exception:
        return ("", 0.0)
    words = [
        (data["text"][i], int(data["conf"][i]))
        for i in range(len(data["text"]))
        if data["text"][i].strip() and int(data["conf"][i]) >= 0
    ]
    text = " ".join(w for w, _ in words)
    confidence = sum(c for _, c in words) / len(words) / 100.0 if words else 0.0
    return text, round(confidence, 3)


def _extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text using PyMuPDF; OCR any image-only pages with pytesseract."""
    pages = _extract_pages_pymupdf_text(pdf_path)

    image_only = [p for p in pages if not p["_has_text"]]
    if not image_only:
        return pages

    try:
        import fitz  # noqa: F401
        from PIL import Image
    except ImportError:
        # No OCR fallback available — image pages will have empty text.
        return pages

    doc = fitz.open(str(pdf_path))
    image_page_nums = {p["page_number"] for p in image_only}
    for pdf_page, page in zip(doc, pages):
        if page["page_number"] in image_page_nums:
            pix = pdf_page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text, confidence = _ocr_page_image(img)
            page["raw_text"] = text
            page["ocr_confidence"] = confidence
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Metadata inference
# ---------------------------------------------------------------------------

def _meta_from_filename(pdf_path: Path) -> dict[str, Any]:
    """Infer document metadata from the PDF filename (regex-based)."""
    meta: dict[str, Any] = {}
    stem = pdf_path.stem

    year_match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", stem)
    if year_match:
        meta["report_year"] = int(year_match.group(1))

    meta["archive_ref"] = re.sub(r"[^A-Za-z0-9]", "_", stem).upper()
    meta["title"] = stem.replace("_", " ").replace("-", " ").title()
    return meta


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def _build_payload(
    pdf_path: Path,
    pages: list[dict[str, Any]],
    meta_extra: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "doc_type": "narrative_report",
        "language": "en",
        "source_file": str(pdf_path),
    }
    metadata.update(meta_extra)
    if "title" not in metadata:
        metadata["title"] = pdf_path.stem.replace("_", " ").replace("-", " ").title()

    clean_pages = []
    for p in pages:
        page: dict[str, Any] = {
            "page_number": p["page_number"],
            "raw_text": p["raw_text"],
        }
        if p.get("ocr_confidence") is not None:
            page["ocr_confidence"] = p["ocr_confidence"]
        clean_pages.append(page)

    return {"metadata": metadata, "pages": clean_pages}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_pdf_to_json(
    pdf_path: Path,
    out_dir: Path,
    metadata_overrides: dict[str, Any] | None = None,
) -> Path:
    """Convert a PDF to the OCR JSON format expected by the pipeline.

    Uses PyMuPDF for text extraction; falls back to pytesseract OCR for
    image-only pages.  Metadata is inferred from the filename; pass
    ``metadata_overrides`` to supply or override specific fields.

    Returns the path of the written .json file.
    Raises ``ConversionError`` on failure.
    """
    try:
        import fitz  # noqa: F401 — verify pymupdf is available before doing any work
    except ImportError as exc:
        raise ConversionError(
            "pymupdf is required for PDF conversion. "
            "Install with: pip install -e .[ingest]"
        ) from exc

    try:
        pages = _extract_pages(pdf_path)
    except Exception as exc:
        raise ConversionError(
            f"Failed to extract pages from {pdf_path.name}: {exc}"
        ) from exc

    meta = _meta_from_filename(pdf_path)
    if metadata_overrides:
        meta.update(metadata_overrides)

    payload = _build_payload(pdf_path, pages, meta)

    dest_dir = out_dir / pdf_path.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / (pdf_path.stem + ".json")
    try:
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        raise ConversionError(f"Failed to write output JSON: {exc}") from exc

    return out_path
