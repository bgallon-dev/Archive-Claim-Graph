"""
pdf_to_json.py — Convert OCR'd PDFs to the structured JSON format
expected by the Graphrag pipeline.

Usage:
    python tools/pdf_to_json.py path/to/report.pdf [more.pdf ...] [options]

Options:
    --out-dir DIR          Directory to write output JSON files (default: same dir as PDF)
    --meta-claude          Use Claude API to extract metadata from page 1 text
    --meta-prompt          Interactively prompt for metadata fields
    --meta-filename        Parse metadata from the PDF filename (regex-based)
    --doc-type TYPE        Override doc_type field (default: narrative_report)
    --series SERIES        Set the series field
    --language LANG        Set the language field (default: en)
    --validate             Validate output JSON through source_parser before writing

Requirements (install as needed):
    pip install pdfplumber pdf2image pytesseract anthropic
    # Also requires Tesseract and Poppler binaries on your PATH
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_pages_pymupdf_text(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text from PDF pages using PyMuPDF's built-in text extraction."""
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


def _extract_pages_ocr(pdf_path: Path) -> list[dict[str, Any]]:
    """Convert PDF pages to images via PyMuPDF and OCR each one."""
    import fitz  # pymupdf
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    pages = []
    for i, pdf_page in enumerate(doc, start=1):
        pix = pdf_page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text, confidence = _ocr_page_image(img)
        pages.append({
            "page_number": i,
            "raw_text": text,
            "ocr_confidence": confidence,
            "_has_text": bool(text.strip()),
        })
    doc.close()
    return pages


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Extract text using PyMuPDF. For image-only pages, attempt OCR
    via pytesseract (if Tesseract is installed), otherwise warn and skip.
    """
    pages = _extract_pages_pymupdf_text(pdf_path)

    image_only = [p for p in pages if not p["_has_text"]]
    if not image_only:
        return pages

    # Attempt OCR for image-only pages
    print(f"  {len(image_only)} image-only page(s) detected — attempting OCR ...")
    try:
        import fitz  # pymupdf
        from PIL import Image
    except ImportError:
        print("  [warn] pymupdf or Pillow not installed — image-only pages will have empty text")
        return pages

    doc = fitz.open(str(pdf_path))
    image_page_nums = {p["page_number"] for p in image_only}
    ocr_failed = False
    for pdf_page, page in zip(doc, pages):
        if page["page_number"] in image_page_nums:
            pix = pdf_page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text, confidence = _ocr_page_image(img)
            if not text and not ocr_failed:
                ocr_failed = True
                print("  [warn] OCR returned no text — Tesseract may not be installed")
            page["raw_text"] = text
            page["ocr_confidence"] = confidence
    doc.close()

    return pages


# ---------------------------------------------------------------------------
# Metadata extraction modes
# ---------------------------------------------------------------------------

def _meta_from_claude(first_page_text: str) -> dict[str, Any]:
    """Ask Claude to extract document metadata from the first page."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        print("  [warn] anthropic package not installed — skipping --meta-claude")
        return {}

    client = anthropic.Anthropic()
    prompt = f"""Extract document metadata from the following text (the first page of a historical narrative report).
Return ONLY a JSON object with these fields (omit any you cannot determine):
  title        — full document title (string)
  date_start   — start date in YYYY-MM-DD format (string)
  date_end     — end date in YYYY-MM-DD format (string)
  report_year  — four-digit year (integer)
  series       — report series name (string)
  archive_ref  — archive reference code (string)

Text:
\"\"\"
{first_page_text[:2000]}
\"\"\"

JSON:"""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [warn] Claude returned unparseable metadata: {raw!r}")
        return {}


def _meta_from_prompt(pdf_path: Path) -> dict[str, Any]:
    """Interactively prompt the user for each metadata field."""
    print(f"\n  Metadata for: {pdf_path.name}")
    print("  (Press Enter to leave a field blank)\n")

    def ask(field: str, hint: str = "") -> str:
        suffix = f"  ({hint})" if hint else ""
        return input(f"    {field}{suffix}: ").strip()

    meta: dict[str, Any] = {}

    title = ask("title")
    if title:
        meta["title"] = title

    date_start = ask("date_start", "YYYY-MM-DD")
    if date_start:
        meta["date_start"] = date_start

    date_end = ask("date_end", "YYYY-MM-DD")
    if date_end:
        meta["date_end"] = date_end

    report_year = ask("report_year", "integer")
    if report_year and report_year.isdigit():
        meta["report_year"] = int(report_year)

    series = ask("series")
    if series:
        meta["series"] = series

    archive_ref = ask("archive_ref")
    if archive_ref:
        meta["archive_ref"] = archive_ref

    return meta


def _meta_from_filename(pdf_path: Path) -> dict[str, Any]:
    """
    Infer metadata from the PDF filename using common patterns.

    Examples:
      narrative_report_1939.pdf      → report_year=1939
      turnbull_1938a_q3.pdf          → archive_ref based on stem, report_year=1938
      1940_july_oct_report.pdf       → report_year=1940
    """
    meta: dict[str, Any] = {}
    stem = pdf_path.stem

    # Four-digit year anywhere in the stem
    year_match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", stem)
    if year_match:
        meta["report_year"] = int(year_match.group(1))

    # Build a rough archive_ref from the stem (uppercase, underscores)
    meta["archive_ref"] = re.sub(r"[^A-Za-z0-9]", "_", stem).upper()

    # Use stem as title placeholder
    meta["title"] = stem.replace("_", " ").replace("-", " ").title()

    return meta


# ---------------------------------------------------------------------------
# Assemble output payload
# ---------------------------------------------------------------------------

def build_payload(
    pdf_path: Path,
    pages: list[dict[str, Any]],
    meta_extra: dict[str, Any],
    doc_type: str,
    series: str | None,
    language: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "doc_type": doc_type,
        "language": language,
        "source_file": str(pdf_path),
    }
    metadata.update(meta_extra)
    if series and "series" not in metadata:
        metadata["series"] = series
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
# Validation (optional)
# ---------------------------------------------------------------------------

def validate_payload(payload: dict[str, Any], pdf_path: Path) -> bool:
    try:
        # Add project root to path so graphrag_pipeline is importable
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from graphrag_pipeline.ingest.source_parser import parse_source_payload  # type: ignore
        parse_source_payload(payload, source_file=str(pdf_path))
        return True
    except Exception as exc:
        print(f"  [error] Validation failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    out_dir: Path | None,
    meta_mode: str,
    doc_type: str,
    series: str | None,
    language: str,
    validate: bool,
) -> Path | None:
    print(f"\nProcessing: {pdf_path.name}")

    # 1. Extract page text
    pages = extract_pages(pdf_path)
    print(f"  Extracted {len(pages)} page(s)")

    # 2. Gather metadata
    first_text = pages[0]["raw_text"] if pages else ""
    if meta_mode == "claude":
        print("  Extracting metadata via Claude API ...")
        meta_extra = _meta_from_claude(first_text)
    elif meta_mode == "prompt":
        meta_extra = _meta_from_prompt(pdf_path)
    else:  # filename
        meta_extra = _meta_from_filename(pdf_path)

    # 3. Build output payload
    payload = build_payload(pdf_path, pages, meta_extra, doc_type, series, language)

    # 4. Optionally validate
    if validate:
        ok = validate_payload(payload, pdf_path)
        if not ok:
            print("  Skipping write due to validation error.")
            return None

    # 5. Write JSON
    base_dir = out_dir if out_dir else pdf_path.parent
    dest_dir = base_dir / pdf_path.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / (pdf_path.stem + ".json")
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Written: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OCR'd PDFs to structured JSON for the Graphrag pipeline."
    )
    parser.add_argument("pdfs", nargs="+", type=Path, help="Input PDF file(s)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--meta-claude",
        dest="meta_mode",
        action="store_const",
        const="claude",
        help="Use Claude API to extract metadata",
    )
    parser.add_argument(
        "--meta-prompt",
        dest="meta_mode",
        action="store_const",
        const="prompt",
        help="Interactively enter metadata",
    )
    parser.add_argument(
        "--meta-filename",
        dest="meta_mode",
        action="store_const",
        const="filename",
        help="Infer metadata from filename (default)",
    )
    parser.set_defaults(meta_mode="filename")
    parser.add_argument("--doc-type", default="narrative_report")
    parser.add_argument("--series", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output through source_parser before writing",
    )

    args = parser.parse_args()

    # Expand any directory arguments into their contained PDFs
    pdf_paths: list[Path] = []
    for p in args.pdfs:
        if p.is_dir():
            found = sorted(p.rglob("*.pdf"))
            if not found:
                print(f"[warn] No PDFs found in directory: {p}", file=sys.stderr)
            pdf_paths.extend(found)
        else:
            pdf_paths.append(p)

    written: list[Path] = []
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"[warn] File not found: {pdf_path}", file=sys.stderr)
            continue
        result = process_pdf(
            pdf_path=pdf_path,
            out_dir=args.out_dir,
            meta_mode=args.meta_mode,
            doc_type=args.doc_type,
            series=args.series,
            language=args.language,
            validate=args.validate,
        )
        if result:
            written.append(result)

    print(f"\nDone. {len(written)} file(s) written.")
    if written:
        print("\nNext step — run through the pipeline:")
        inputs = " ".join(str(p) for p in written)
        print(f"  graphrag run-e2e --inputs {inputs} --out-dir out --backend memory")


if __name__ == "__main__":
    main()
