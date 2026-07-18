"""
run_under_extracted.py — Find documents with significantly fewer claims than
the corpus average for their length band. These are sources worth a deep
dive (re-OCR, re-extraction, or manual review).

By default, newspaper docs are categorized via
`tools/categorize_newspaper_extraction.py` and content-type artifacts
(classifieds, fragments, page-level docs, schedules, petitions, etc.) are
filtered out so the result surfaces only genuine recall opportunities.
Turnbull and other non-newspaper docs are returned unfiltered.

Usage:
    python tools/run_under_extracted.py [options]

Options:
    --output FILE          Output CSV (default: under_extracted_docs.csv)
    --min-chars N          Min char_count to consider a doc (default 1000)
    --min-bucket-docs N    Min docs per length bucket for a meaningful avg
                           (default 3)
    --deficit-ratio FLOAT  Max actual/expected ratio (default 0.6 — 40% below
                           expected). Set to 1.0 for all under-extracted.
    --limit N              Override LIMIT in the cypher (default 100). Note
                           that artifact filtering happens AFTER the cypher
                           returns, so the final row count may be much
                           smaller than --limit.
    --include-artifacts    Disable the artifact filter — return raw deficit
                           ranking including classifieds, page-level docs,
                           etc. Useful when investigating false positives.

Requires NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (loaded from .env).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from gemynd.shared.env import load_dotenv
from gemynd.shared.settings import Settings
from gemynd.ingest.graph import build_driver_kwargs as _build_driver_kwargs

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("neo4j package is not installed. Install with: pip install neo4j")

# Reuse the categorizer's scoring + per-doc paragraph fetch.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from categorize_newspaper_extraction import (  # noqa: E402
    _score_doc,
    categorize,
    fetch_paragraphs,
)

CYPHER_PATH = Path(__file__).resolve().parent / "under_extracted_docs.cypher"

# Categories that represent content-type artifacts rather than genuine
# extraction-recall opportunities. Kept categories: "ocr_damage" (fixable
# upstream), "other" (real prose), and "n/a" (couldn't categorize, so don't
# silently drop). The two parser-level damage categories
# (column_flattened, non_prose_dilution) are filtered because re-extraction
# won't help — those need upstream parser/source fixes.
_ARTIFACT_CATEGORIES = frozenset({
    "list",
    "classified",
    "market_data",
    "petition",
    "legal",
    "schedule",
    "continuation",
    "page",
    "fragment",
    "column_flattened",
    "non_prose_dilution",
})


def run_query(
    settings: Settings,
    min_chars: int,
    min_bucket_docs: int,
    deficit_ratio_max: float,
    limit: int | None,
) -> list[dict]:
    cypher = CYPHER_PATH.read_text(encoding="utf-8")
    if limit is not None:
        cypher = cypher.replace("LIMIT 100", f"LIMIT {int(limit)}")

    driver_kwargs = _build_driver_kwargs(
        uri=settings.neo4j_uri,
        trust_mode=settings.neo4j_trust,
        ca_cert_path=settings.neo4j_ca_cert,
    )
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        **driver_kwargs,
    )
    rows: list[dict] = []
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(
            cypher,
            min_chars=min_chars,
            min_bucket_docs=min_bucket_docs,
            deficit_ratio_max=deficit_ratio_max,
        )
        for record in result:
            rows.append(dict(record))
    driver.close()
    return rows


def categorize_and_filter(
    settings: Settings, rows: list[dict], filter_artifacts: bool
) -> tuple[list[dict], dict[str, int]]:
    """Annotate each row with a `category` and (optionally) drop artifact
    categories. Categorization runs on all docs — newspaper-style detectors
    (classified, schedule, etc.) won't fire on narrative reports because
    their score thresholds are content-specific, while the cross-cutting
    detectors (column_flattened, non_prose_dilution) catch parser-level
    damage in any institution."""
    out: list[dict] = []
    counts: dict[str, int] = {}
    for row in rows:
        paras = fetch_paragraphs(settings, row["doc_id"])
        scores = _score_doc(
            paras,
            title=row.get("title") or "",
            doc_id=row.get("doc_id") or "",
        )
        category = categorize(scores)
        row["category"] = category
        counts[category] = counts.get(category, 0) + 1
        if filter_artifacts and category in _ARTIFACT_CATEGORIES:
            continue
        out.append(row)
    return out, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank documents by how far below the corpus-average claim count "
            "they fall, comparing within length-band peer groups."
        ),
    )
    parser.add_argument("--output", default="under_extracted_docs.csv")
    parser.add_argument("--min-chars", type=int, default=1000)
    parser.add_argument("--min-bucket-docs", type=int, default=3)
    parser.add_argument("--deficit-ratio", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help=(
            "Disable artifact filtering — include classifieds, page-level "
            "docs, fragments, etc. in the output."
        ),
    )
    args = parser.parse_args()

    load_dotenv(_REPO_ROOT / ".env")
    settings = Settings.from_env()
    if not settings.neo4j_uri:
        sys.exit("NEO4J_URI not set. Export Neo4j connection env vars first.")

    raw_rows = run_query(
        settings,
        min_chars=args.min_chars,
        min_bucket_docs=args.min_bucket_docs,
        deficit_ratio_max=args.deficit_ratio,
        limit=args.limit,
    )
    if not raw_rows:
        print(
            "No under-extracted docs found at this threshold. "
            "Try raising --deficit-ratio (e.g. 1.0 to see all docs below their bucket average)."
        )
        return

    print(f"Cypher returned {len(raw_rows)} deficit-flagged docs.")
    rows, cat_counts = categorize_and_filter(
        settings, raw_rows, filter_artifacts=not args.include_artifacts
    )

    if cat_counts:
        print("Category counts (over the raw cypher result):")
        for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
            mark = "  [kept]   " if cat not in _ARTIFACT_CATEGORIES else "  [filtered]"
            print(f"  {mark} {cat:13s} {n}")

    if not rows:
        print(
            "All flagged docs filtered out as content-type artifacts. "
            "Pass --include-artifacts to see the raw deficit ranking."
        )
        return

    out_path = Path(args.output)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    suffix = " (artifacts NOT filtered)" if args.include_artifacts else ""
    print(f"Wrote {len(rows)} under-extracted docs to {out_path}{suffix}")


if __name__ == "__main__":
    main()
