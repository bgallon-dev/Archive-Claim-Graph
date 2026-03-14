"""
Audit false-positive clusters in claim entity links.

Reads CSV exports from csv_out/ (produced by json_to_csv.py) and emits a
ranked report of suspicious (claim_type, relation_type, entity_name) triplets.

Usage:
    python scripts/audit_false_positives.py [--csv-dir csv_out] [--out-csv audit_fp.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Suspicious pattern definitions
# ---------------------------------------------------------------------------

# (claim_type_set, relation_type, entity_name_fragment) → reason string
# claim_type_set=None means "any claim type"
SUSPICIOUS_PATTERNS: list[tuple[frozenset[str] | None, str, str | None, str]] = [
    # banding appearing as METHOD_FOCUS on claim types that don't survey populations
    (
        frozenset({
            "predator_control", "economic_use", "fire_incident", "public_contact",
            "development_activity", "habitat_condition", "management_action",
            "weather_observation", "unclassified_assertion",
        }),
        "METHOD_FOCUS",
        "banding",
        "banding as METHOD_FOCUS on non-survey claim type",
    ),
    # coot flagged as MANAGEMENT_TARGET under predator_control
    (
        frozenset({"predator_control"}),
        "MANAGEMENT_TARGET",
        "coot",
        "coot as MANAGEMENT_TARGET under predator_control (usually species_presence / population_estimate)",
    ),
    # LOCATION_FOCUS on public_contact or economic_use
    (
        frozenset({"public_contact", "economic_use"}),
        "LOCATION_FOCUS",
        None,
        "LOCATION_FOCUS on public_contact/economic_use — check if SUBJECT_OF_CLAIM/MANAGEMENT_TARGET is more appropriate",
    ),
    # METHOD_FOCUS on fire_incident or public_contact at all
    (
        frozenset({"fire_incident", "public_contact"}),
        "METHOD_FOCUS",
        None,
        "METHOD_FOCUS on fire_incident/public_contact — likely a false-positive",
    ),
    # SPECIES_FOCUS on public_contact
    (
        frozenset({"public_contact"}),
        "SPECIES_FOCUS",
        None,
        "SPECIES_FOCUS on public_contact — likely misclassified",
    ),
]


def _is_suspicious(claim_type: str, relation_type: str, entity_name: str) -> list[str]:
    """Return list of reasons this triplet is suspicious (empty = clean)."""
    reasons: list[str] = []
    entity_lc = entity_name.lower()
    for allowed_claim_types, pat_relation, pat_entity, reason in SUSPICIOUS_PATTERNS:
        if pat_relation != relation_type:
            continue
        if allowed_claim_types is not None and claim_type not in allowed_claim_types:
            continue
        if pat_entity is not None and pat_entity not in entity_lc:
            continue
        reasons.append(reason)
    return reasons


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _find_pairs(csv_dir: Path) -> list[tuple[Path, Path]]:
    """Return (claims.csv, claim_entity_links.csv) path pairs per subfolder."""
    pairs: list[tuple[Path, Path]] = []
    for sub in sorted(csv_dir.iterdir()):
        if not sub.is_dir():
            continue
        claims_path = sub / "claims.csv"
        links_path = sub / "claim_entity_links.csv"
        if claims_path.exists() and links_path.exists():
            pairs.append((claims_path, links_path))
    return pairs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(csv_dir: Path) -> list[dict]:
    """
    Returns rows sorted by (suspicious_flag desc, count desc) with keys:
      claim_type, relation_type, entity_name, count, suspicious, reasons, doc_ids, sample_sentence
    """
    pairs = _find_pairs(csv_dir)
    if not pairs:
        return []

    # Accumulate (claim_type, relation_type, entity_name) → {count, doc_ids, sample_sentence}
    Counter = dict[tuple[str, str, str], dict]
    counter: Counter = {}

    for claims_path, links_path in pairs:
        claims_rows = _load_csv(claims_path)
        links_rows = _load_csv(links_path)

        claim_info: dict[str, dict[str, str]] = {
            row["claim_id"]: row for row in claims_rows if "claim_id" in row
        }

        for link in links_rows:
            claim_id = link.get("claim_id", "")
            relation_type = link.get("relation_type", "")
            entity_name = link.get("entity_id", link.get("entity_name", link.get("normalized_form", "")))

            claim = claim_info.get(claim_id, {})
            claim_type = claim.get("claim_type", "")
            source_sentence = claim.get("source_sentence", "")
            doc_id = claim.get("run_id", claims_path.parent.name)

            key = (claim_type, relation_type, entity_name)
            if key not in counter:
                counter[key] = {"count": 0, "doc_ids": set(), "sample_sentence": source_sentence}
            counter[key]["count"] += 1
            counter[key]["doc_ids"].add(doc_id)
            if not counter[key]["sample_sentence"] and source_sentence:
                counter[key]["sample_sentence"] = source_sentence

    rows: list[dict] = []
    for (claim_type, relation_type, entity_name), info in counter.items():
        reasons = _is_suspicious(claim_type, relation_type, entity_name)
        rows.append({
            "claim_type": claim_type,
            "relation_type": relation_type,
            "entity_name": entity_name,
            "count": info["count"],
            "suspicious": bool(reasons),
            "reasons": "; ".join(reasons),
            "doc_ids": ", ".join(sorted(info["doc_ids"])),
            "sample_sentence": info["sample_sentence"],
        })

    # Sort: suspicious first, then by count descending
    rows.sort(key=lambda r: (not r["suspicious"], -r["count"]))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "claim_type": 25,
    "relation_type": 20,
    "entity_name": 30,
    "count": 6,
    "suspicious": 10,
}


def _print_table(rows: list[dict]) -> None:
    sep = "-" * 120
    header = (
        f"{'CLAIM_TYPE':<25}  {'RELATION_TYPE':<20}  {'ENTITY':<30}  {'CNT':>5}  "
        f"{'SUSP?':<7}  REASON / SENTENCE"
    )
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        flag = "*** YES" if row["suspicious"] else "no"
        detail = row["reasons"] if row["suspicious"] else row["sample_sentence"][:60]
        print(
            f"{row['claim_type']:<25}  {row['relation_type']:<20}  {row['entity_name']:<30}  "
            f"{row['count']:>5}  {flag:<7}  {detail}"
        )
    print(sep)
    n_susp = sum(1 for r in rows if r["suspicious"])
    print(f"\n{len(rows)} triplets total — {n_susp} suspicious")


def _write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("")
        return
    fieldnames = ["suspicious", "claim_type", "relation_type", "entity_name", "count", "reasons", "doc_ids", "sample_sentence"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit false-positive clusters in claim entity links.")
    parser.add_argument("--csv-dir", default="csv_out", help="Root CSV output directory (default: csv_out)")
    parser.add_argument("--out-csv", default=None, help="Optional path to write results CSV")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        print(f"CSV directory not found: {csv_dir}", file=sys.stderr)
        sys.exit(1)

    rows = analyse(csv_dir)
    if not rows:
        print("No (claims.csv + claim_entity_links.csv) pairs found under", csv_dir, file=sys.stderr)
        sys.exit(1)

    _print_table(rows)

    if args.out_csv:
        _write_csv(rows, Path(args.out_csv))


if __name__ == "__main__":
    main()
