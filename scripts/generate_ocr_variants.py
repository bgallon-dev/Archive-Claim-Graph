"""Generate OCR spelling variants for seed entities.

Learns an OCR confusion matrix from built-in character-pair statistics,
``known_errors`` in ``ocr_corrections.yaml``, and (optionally) accepted
corrections in a review store SQLite database.  Outputs variant candidates
for each seed entity in csv, yaml, or seed_patch format.

Usage::

    python scripts/generate_ocr_variants.py \\
        --resources-dir gemynd/resources \\
        [--review-store path/to/review.db] \\
        [--min-prob 0.05] \\
        [--output-format csv] \\
        [--output variants.csv]
"""
from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ── Built-in confusion pairs ──────────────────────────────────────────────────
# Format: {(correct_text, ocr_misread): probability}
# (correct_text, ocr_misread) means: when *correct_text* appears in a seed
# entity name, an OCR scanner might read it as *ocr_misread*.
BUILTIN_CONFUSIONS: dict[tuple[str, str], float] = {
    ("rn", "m"):  0.15,   # "rn" commonly collapsed to "m"
    ("m",  "rn"): 0.10,   # "m" expanded to "rn"
    ("li", "h"):  0.10,   # "li" misread as "h"
    ("h",  "li"): 0.05,
    ("o",  "0"):  0.15,   # letter "o" misread as digit "0"
    ("0",  "o"):  0.08,
    ("l",  "1"):  0.10,   # letter "l" misread as digit "1"
    ("1",  "l"):  0.08,
    ("d",  "cl"): 0.08,   # "d" misread as "cl"
    ("cl", "d"):  0.05,
}


def _char_pairs(wrong: str, correct: str) -> list[tuple[str, str]]:
    """Return ``(correct_chunk, wrong_chunk)`` substitution pairs.

    Each pair means: when *correct_chunk* appears in a name, it might be
    scanned as *wrong_chunk*.  Uses :class:`difflib.SequenceMatcher` to align
    the strings and extract contiguous changed regions.
    """
    pairs: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(None, wrong, correct)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            wrong_chunk = wrong[i1:i2]
            correct_chunk = correct[j1:j2]
            if correct_chunk and wrong_chunk:
                pairs.append((correct_chunk, wrong_chunk))
    return pairs


def _load_from_review_store(db_path: str) -> dict[tuple[str, str], float]:
    """Return confusion pairs learned from accepted OCR corrections in the review store.

    Review store schema: ``proposal.current_revision_id`` → ``proposal_revision.revision_id``
    (there is no ``is_current`` column).  Pairs from the review store are weighted
    at 0.25 (higher than the built-in heuristics).
    """
    import sqlite3

    sql = """
        SELECT pr.patch_spec_json
        FROM proposal p
        JOIN correction_event ce ON p.proposal_id = ce.proposal_id
        JOIN proposal_revision pr ON pr.revision_id = p.current_revision_id
        WHERE p.issue_class = 'ocr_spelling_variant' AND ce.action = 'accepted'
    """
    pairs: dict[tuple[str, str], float] = {}
    try:
        con = sqlite3.connect(db_path)
        for (patch_json,) in con.execute(sql):
            try:
                patch = json.loads(patch_json)
                wrong = patch.get("wrong", "")
                correct = patch.get("correct", "")
                if wrong and correct:
                    for orig, repl in _char_pairs(wrong, correct):
                        pairs[(orig, repl)] = max(pairs.get((orig, repl), 0.0), 0.25)
            except (json.JSONDecodeError, TypeError):
                continue
        con.close()
    except Exception:  # noqa: BLE001 — DB unavailable or wrong schema
        pass
    return pairs


@dataclass
class OCRConfusionMatrix:
    """Character-pair substitution probability table for OCR variant generation.

    ``pairs`` maps ``(correct_text, ocr_misread)`` to a probability in (0, 1].
    """

    pairs: dict[tuple[str, str], float]

    @classmethod
    def from_sources(
        cls,
        known_errors: dict[str, str] | None = None,
        review_store_path: str | None = None,
    ) -> "OCRConfusionMatrix":
        """Build matrix from built-ins, a known_errors dict, and an optional review store.

        Parameters
        ----------
        known_errors:
            Dict mapping ``wrong_form → correct_form`` (e.g. from
            ``load_ocr_correction_map()``).  Pairs inferred from this source
            are weighted at 0.20.
        review_store_path:
            Path to the review store SQLite database.  Accepted OCR corrections
            are weighted at 0.25.
        """
        pairs: dict[tuple[str, str], float] = dict(BUILTIN_CONFUSIONS)

        if known_errors:
            for wrong, correct in known_errors.items():
                for orig, repl in _char_pairs(wrong, correct):
                    pairs[(orig, repl)] = max(pairs.get((orig, repl), 0.0), 0.20)

        if review_store_path:
            for key, prob in _load_from_review_store(review_store_path).items():
                pairs[key] = max(pairs.get(key, 0.0), prob)

        return cls(pairs=pairs)


def generate_variants(
    name: str,
    matrix: OCRConfusionMatrix,
    min_prob: float = 0.05,
) -> list[tuple[str, float]]:
    """Return ``[(variant_string, probability), ...]`` sorted by descending probability.

    Applies all single-substitution pairs from *matrix* to *name*.  Variants
    that are identical to the original name (case-insensitive) and those with
    probability below *min_prob* are excluded.  When the same variant is
    produced by multiple pairs the highest probability is kept.
    """
    name_lower = name.lower()
    best: dict[str, float] = {}

    for (orig, repl), prob in matrix.pairs.items():
        if prob < min_prob:
            continue
        start = 0
        while True:
            idx = name_lower.find(orig, start)
            if idx == -1:
                break
            variant = name[:idx] + repl + name[idx + len(orig):]
            if variant.lower() != name_lower:
                if variant not in best or prob > best[variant]:
                    best[variant] = prob
            start = idx + 1

    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


# ── Output writers ─────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict[str, Any]], out: Any) -> None:
    writer = csv.DictWriter(out, fieldnames=["entity_name", "variant", "probability"])
    writer.writeheader()
    writer.writerows(rows)


def _write_yaml(rows: list[dict[str, Any]], out: Any) -> None:
    entries = [
        {"name": r["entity_name"], "variant": r["variant"], "probability": r["probability"]}
        for r in rows
    ]
    yaml.dump({"entries": entries}, out, default_flow_style=False, allow_unicode=True)


def _write_seed_patch(
    seed_rows: list[dict[str, str]],
    variant_rows: list[dict[str, Any]],
    out: Any,
) -> None:
    """Write variant candidates in seed_entities.csv long-format.

    Each variant row uses the same ``entity_type`` and properties as the
    original entity, with the variant spelling as the ``name``.
    """
    # Build a lookup: name → (entity_type, [(prop_key, prop_value)])
    props_by_name: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for row in seed_rows:
        name = row["name"]
        if name not in props_by_name:
            props_by_name[name] = (row["entity_type"], [])
        if row.get("prop_key") and row.get("prop_value"):
            props_by_name[name][1].append((row["prop_key"], row["prop_value"]))

    writer = csv.DictWriter(out, fieldnames=["entity_type", "name", "prop_key", "prop_value"])
    writer.writeheader()
    for vrow in variant_rows:
        info = props_by_name.get(vrow["entity_name"])
        if info is None:
            continue
        entity_type, props = info
        variant = vrow["variant"]
        if props:
            for prop_key, prop_value in props:
                writer.writerow({
                    "entity_type": entity_type,
                    "name": variant,
                    "prop_key": prop_key,
                    "prop_value": prop_value,
                })
        else:
            writer.writerow({
                "entity_type": entity_type,
                "name": variant,
                "prop_key": "",
                "prop_value": "",
            })


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate OCR spelling variants for seed entities."
    )
    parser.add_argument(
        "--resources-dir",
        default=None,
        help="Path to resources/ directory. Defaults to gemynd/resources/.",
    )
    parser.add_argument(
        "--review-store",
        default=None,
        help="Path to review store SQLite database (optional).",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.05,
        help="Minimum confusion probability for a variant to be included (default 0.05).",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "yaml", "seed_patch"],
        default="csv",
        help="Output format: csv, yaml, or seed_patch (default csv).",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path. Use '-' for stdout (default).",
    )
    args = parser.parse_args(argv)

    # Resolve resources directory
    resources_dir: Path | None = None
    if args.resources_dir:
        resources_dir = Path(args.resources_dir)
    else:
        candidate = Path(__file__).parent.parent / "gemynd" / "resources"
        if candidate.exists():
            resources_dir = candidate

    # Make sure gemynd is importable
    _project_root = Path(__file__).parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from gemynd.shared.resource_loader import (
        load_ocr_correction_map,
        load_seed_entity_rows,
    )

    seed_rows = load_seed_entity_rows(resources_dir)
    known_errors = load_ocr_correction_map(resources_dir)

    matrix = OCRConfusionMatrix.from_sources(
        known_errors=known_errors,
        review_store_path=args.review_store,
    )

    seen_names: set[str] = set()
    all_variant_rows: list[dict[str, Any]] = []
    for row in seed_rows:
        name = row["name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        for variant, prob in generate_variants(name, matrix, args.min_prob):
            all_variant_rows.append({
                "entity_name": name,
                "variant": variant,
                "probability": round(prob, 4),
            })

    if args.output == "-":
        out = sys.stdout
        close_out = False
    else:
        out = open(args.output, "w", encoding="utf-8", newline="")
        close_out = True

    try:
        if args.output_format == "csv":
            _write_csv(all_variant_rows, out)
        elif args.output_format == "yaml":
            _write_yaml(all_variant_rows, out)
        else:  # seed_patch
            _write_seed_patch(seed_rows, all_variant_rows, out)
    finally:
        if close_out:
            out.close()


if __name__ == "__main__":
    main()
