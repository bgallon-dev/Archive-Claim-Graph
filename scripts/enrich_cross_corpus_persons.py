"""CLI driver for the cross-corpus person NER enrichment pass.

Scans Spokane Press newspaper paragraphs for mentions of GLO patent grantees
and writes REFERS_TO / POSSIBLY_REFERS_TO relationships to Neo4j.

Usage::

    python scripts/enrich_cross_corpus_persons.py \\
        [--neo4j-uri bolt://localhost:7687] \\
        [--neo4j-user neo4j] \\
        [--neo4j-password <pw>] \\
        [--resources-dir gemynd/resources] \\
        [--review-store path/to/review.db] \\
        [--min-ocr-prob 0.08] \\
        [--dry-run] \\
        [--output-report matches.csv]

Environment variable fallbacks:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_log = logging.getLogger(__name__)

# Make the project root importable when run as a script.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env before reading os.environ so that argument defaults pick up the values.
from gemynd.shared.env import load_dotenv  # noqa: E402
load_dotenv()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-corpus person NER enrichment pass.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"),
        help="Neo4j URI (default: neo4j://localhost:7687 or $NEO4J_URI)",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: neo4j or $NEO4J_USER)",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password (default: $NEO4J_PASSWORD)",
    )
    parser.add_argument(
        "--resources-dir",
        default=None,
        help="Path to gemynd/resources/ directory for OCR correction maps.",
    )
    parser.add_argument(
        "--review-store",
        default=None,
        help="Path to review store SQLite database for learned OCR confusions.",
    )
    parser.add_argument(
        "--min-ocr-prob",
        type=float,
        default=0.08,
        help="Minimum OCR confusion probability for variant inclusion (default 0.08).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and report matches but write nothing to Neo4j.",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Write match report CSV to this path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        _log.error("neo4j driver not installed. Run: pip install neo4j")
        return 1

    from gemynd.ingest.enrichment.cross_corpus_ner import CrossCorpusNERPass

    resources_dir: Path | None = None
    if args.resources_dir:
        resources_dir = Path(args.resources_dir)
    else:
        candidate = _project_root / "gemynd" / "resources"
        if candidate.exists():
            resources_dir = candidate
            _log.info("Using resources directory: %s", resources_dir)

    _log.info(
        "Connecting to Neo4j at %s as %s%s",
        args.neo4j_uri,
        args.neo4j_user,
        " [DRY RUN]" if args.dry_run else "",
    )

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )

    try:
        ner = CrossCorpusNERPass(
            driver=driver,
            resources_dir=resources_dir,
            review_store=args.review_store,
            min_ocr_prob=args.min_ocr_prob,
        )
        output_report = Path(args.output_report) if args.output_report else None
        rows = ner.run(dry_run=args.dry_run, output_report=output_report)

        refers = sum(1 for r in rows if r["relation_type"] == "REFERS_TO")
        possibly = sum(1 for r in rows if r["relation_type"] == "POSSIBLY_REFERS_TO")
        ocr = sum(1 for r in rows if r["ocr_suspect"])

        _log.info(
            "Results: %d REFERS_TO, %d POSSIBLY_REFERS_TO, %d with ocr_suspect flag",
            refers, possibly, ocr,
        )
        if args.dry_run and rows:
            _log.info("Top matches (dry-run preview):")
            for row in rows[:20]:
                _log.info(
                    "  [%s %.3f%s] %r → %s",
                    row["relation_type"],
                    row["score"],
                    " OCR" if row["ocr_suspect"] else "",
                    row["surface_form"],
                    row["entity_name"],
                )
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
