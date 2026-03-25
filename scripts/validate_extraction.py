"""Validate the full extraction pipeline against a single source report JSON.

Usage:
    python -m scripts.validate_extraction tests/fixtures/report1.json
    python -m scripts.validate_extraction tests/fixtures/report1.json --verbose

Exits 0 on success, 1 on any extraction failure.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def _count_label(items: list, key: str) -> Counter:
    return Counter(getattr(item, key) for item in items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run parse + extract on a source report JSON and print a validation summary."
    )
    parser.add_argument("source", help="Path to source report JSON (e.g. tests/fixtures/report1.json)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each claim sentence")
    args = parser.parse_args(argv)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[error] File not found: {source_path}", file=sys.stderr)
        return 1

    try:
        from graphrag_pipeline.ingest.pipeline import parse_source, extract_semantic
    except ImportError as exc:
        print(f"[error] Could not import graphrag_pipeline: {exc}", file=sys.stderr)
        print("  Run: pip install -e . from the project root", file=sys.stderr)
        return 1

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        structure = parse_source(str(source_path))
    except Exception as exc:
        print(f"[error] parse_source failed: {exc}", file=sys.stderr)
        return 1

    doc = structure.document
    print(f"\nDocument : {doc.title}")
    print(f"Dates    : {doc.date_start} to {doc.date_end}  (year={doc.report_year})")
    print(f"Archive  : {doc.archive_ref}")
    print(f"Pages    : {len(structure.pages)}")
    print(f"Sections : {len(structure.sections)}")
    print(f"Paragraphs: {len(structure.paragraphs)}")

    # ── Extract ──────────────────────────────────────────────────────────────
    try:
        semantic = extract_semantic(structure)
    except Exception as exc:
        print(f"\n[error] extract_semantic failed: {exc}", file=sys.stderr)
        return 1

    print(f"\n-- Extraction results ------------------------------------------")
    print(f"Claims           : {len(semantic.claims)}")
    print(f"Measurements     : {len(semantic.measurements)}")
    print(f"Mentions         : {len(semantic.mentions)}")
    print(f"Entities         : {len(semantic.entities)}")
    print(f"Observations     : {len(semantic.observations)}")
    print(f"Events           : {len(semantic.events)}")
    print(f"Concept links    : {len(semantic.claim_concept_links)}")

    if semantic.claims:
        print(f"\n-- Claims by type ----------------------------------------------")
        for claim_type, count in sorted(_count_label(semantic.claims, "claim_type").items()):
            print(f"  {claim_type:<35} {count:>3}")

    if semantic.entities:
        print(f"\n-- Entities by type --------------------------------------------")
        for entity_type, count in sorted(_count_label(semantic.entities, "entity_type").items()):
            print(f"  {entity_type:<35} {count:>3}")

    if semantic.claim_concept_links:
        print(f"\n-- Concept links by concept ------------------------------------")
        for concept_id, count in sorted(
            Counter(lnk.concept_id for lnk in semantic.claim_concept_links).items()
        ):
            print(f"  {concept_id:<40} {count:>3}")

    if args.verbose and semantic.claims:
        print(f"\n-- Claim sentences ---------------------------------------------")
        for claim in semantic.claims:
            print(f"  [{claim.claim_type}] {claim.source_sentence}")

    if semantic.claim_link_diagnostics:
        unresolved = [d for d in semantic.claim_link_diagnostics if d.diagnostic_code == "NO_CANDIDATE"]
        if unresolved:
            print(f"\n[warn] {len(unresolved)} unresolved mention(s) (NO_CANDIDATE)")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
