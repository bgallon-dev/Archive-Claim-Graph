"""Ingest the Spokane / Chronicling America newspaper corpus through gemynd.

The corpus at ``C:\\Users\\Benjamin\\Desktop\\Tutorstructure\\corpus`` holds raw
*article* JSONs (``{article_id, article, date, lccn, scoring, ...}``), which are
NOT gemynd's source format. They are converted to gemynd's
``{metadata, pages:[{raw_text}]}`` payload via the project's own converter
(``corpus-curator/converter.py: article_to_gemynd_payload``) before ingestion.

Per-document ``*.structure.json`` / ``*.semantic.json`` bundles are written to
the out-dir (memory backend by default).

Examples:
    py scripts/ingest_tutorstructure.py --limit 300                 # LLM-on smoke (Haiku)
    py scripts/ingest_tutorstructure.py --limit 300 --no-llm        # free rule-based smoke
    py scripts/ingest_tutorstructure.py --full --workers 4          # whole corpus
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Load .env so keys/endpoints live in one place: ANTHROPIC_API_KEY for the
# Anthropic backend, or CLAIM_LLM_BASE_URL / CLAIM_LLM_MODEL to route claim
# extraction at a self-hosted OpenAI-compatible server (vLLM/Ollama/Colab).
# Does not override variables already set in the shell.
from gemynd.shared.env import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

CORPUS = Path(r"C:\Users\Benjamin\Desktop\Tutorstructure\corpus")
CURATOR = Path(r"C:\Users\Benjamin\Desktop\Tutorstructure\corpus-curator")

# Real article filenames embed a LoC date, e.g. ``ca_1883-05-19_p4_sn...json``.
# Excludes stray checkpoint/index dotfiles (``.ca_checkpoint.json`` etc.).
_ARTICLE_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_")


def _all_article_files() -> list[Path]:
    return sorted(
        f for f in CORPUS.rglob("*.json")
        if not f.name.startswith(".") and _ARTICLE_RE.search(f.name)
    )


def _select(files: list[Path], mode: str, n: int, full: bool) -> list[Path]:
    """Pick documents by mode.

    - articles : digit-prefixed American Stories segments (short, clean, cheap)
    - pages    : ca_* Chronicling America full pages (long, noisy, expensive)
    - stratified: every k-th across the whole sorted list (spans years + both types)
    - head     : first N sorted (earliest years; mostly 1883 full pages)
    """
    if mode == "articles":
        files = [f for f in files if f.name[:1].isdigit()]
    elif mode == "pages":
        files = [f for f in files if f.name.startswith("ca_")]
    if full:
        return files
    if mode == "stratified" and len(files) > n:
        step = len(files) // n
        return files[::step][:n]
    return files[:n]


def _inline_to_gemynd_payload(article: dict) -> dict | None:
    """Faithful mirror of corpus-curator/converter.py:article_to_gemynd_payload.

    Used only if the canonical converter can't be imported (its module-level
    imports pull in the curator's config/store).
    """
    date = article.get("date", "")
    if not date or len(date) < 4:
        return None
    try:
        year = int(date[:4])
    except ValueError:
        return None
    headline = (article.get("headline") or "").strip()
    newspaper = article.get("newspaper", "")
    page_num = article.get("page", "1")
    title = headline or f"{newspaper} — {date} — p{page_num}"
    return {
        "metadata": {
            "title": title,
            "doc_type": "newspaper_article",
            "date_start": date,
            "date_end": date,
            "report_year": year,
            "source_file": article.get("article_id", ""),
            "archive_ref": article.get("article_id", ""),
            "series": newspaper,
            "language": "en",
            "lccn": article.get("lccn", ""),
            "newspaper_name": article.get("newspaper_name", ""),
            "edition": article.get("edition", ""),
            "keyword_scoring": article.get("scoring", {}),
            "llm_review": article.get("llm_review", {}),
        },
        "pages": [{"page_number": 1, "raw_text": article.get("article", "")}],
    }


def _get_converter():
    """Prefer the project's canonical converter; fall back to the inline mirror."""
    try:
        sys.path.insert(0, str(CURATOR))
        from converter import article_to_gemynd_payload  # type: ignore
        print("[ingest] using corpus-curator/converter.py: article_to_gemynd_payload")
        return article_to_gemynd_payload
    except Exception as exc:  # noqa: BLE001
        print(f"[ingest] canonical converter import failed ({exc}); using inline mirror")
        return _inline_to_gemynd_payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--limit", type=int, default=300, help="Select N documents (default 300).")
    mode.add_argument("--full", action="store_true", help="Ingest the entire corpus.")
    p.add_argument(
        "--mode", choices=("stratified", "articles", "pages", "head"), default="stratified",
        help="Document selection: stratified (default, spans years + both types), "
             "articles (short clean segments), pages (full noisy pages), head (first-N).",
    )
    p.add_argument("--out-dir", type=Path, default=REPO / "out" / "newspapers_smoke")
    p.add_argument("--backend", choices=("memory", "neo4j"), default="memory")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-llm", action="store_true", help="Rule-based only (free, no API calls).")
    p.add_argument("--model", default="claude-haiku-4-5", help="Claim-extraction model (cheap by default).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not CORPUS.is_dir():
        raise SystemExit(f"Corpus not found: {CORPUS}")

    if not args.no_llm:
        os.environ.setdefault("CLAIM_EXTRACTION_MODEL", args.model)

    files = _all_article_files()
    if not files:
        raise SystemExit(f"No article .json files under {CORPUS}")
    selected = _select(files, args.mode, args.limit, args.full)

    # Convert raw articles → gemynd payloads in a staging dir, then ingest those.
    to_payload = _get_converter()
    stage_dir = args.out_dir / "_gemynd_input"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    skipped = 0
    for f in selected:
        try:
            article = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        payload = to_payload(article)
        if payload is None:
            skipped += 1
            continue
        aid = str(article.get("article_id") or f.stem).replace("/", "_").replace("\\", "_")
        out = stage_dir / f"{aid}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.append(str(out))

    print(f"[ingest] corpus       = {CORPUS}")
    print(f"[ingest] out_dir      = {args.out_dir}")
    print(f"[ingest] selected     = {len(selected)} (mode={args.mode}{'/FULL' if args.full else f', limit {args.limit}'})")
    print(f"[ingest] staged       = {len(staged)} (skipped {skipped} unconvertible)")
    print(f"[ingest] backend      = {args.backend}  workers = {args.workers}")
    print(f"[ingest] llm          = {'off' if args.no_llm else 'on (' + args.model + ')'}")
    sys.stdout.flush()
    if not staged:
        raise SystemExit("Nothing staged for ingestion.")

    from gemynd.ingest.pipeline import run_e2e

    domain_dir = REPO / "gemynd" / "resources_newspaper"
    started = time.time()
    summary = run_e2e(
        staged,
        args.out_dir,
        backend=args.backend,
        workers=args.workers,
        resources_dir=domain_dir,
        no_llm=args.no_llm,
    )
    elapsed = time.time() - started

    printable = {k: v for k, v in summary.items() if k != "writer"}
    outputs = printable.pop("outputs", [])
    print("[ingest] ---- summary ----")
    print(json.dumps(printable, indent=2, default=str))
    print(f"[ingest] output bundles: {len(outputs)}")
    print(f"[ingest] elapsed: {elapsed:.1f}s ({elapsed / max(1, len(staged)):.2f}s/doc)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
