"""Generate and iteratively refine domain resource files from a sample corpus.

Uses the Anthropic API to extract candidate claim-type patterns and seed
entities from raw text, then runs an automated refinement loop that validates
the draft configuration, identifies gaps, and asks the LLM to fix them —
repeating until the unclassified-claim rate drops below a threshold or
convergence is detected.

Usage:
    python -m scripts.bootstrap_domain \\
        --samples input/new_domain/*.json \\
        --out-dir domains/new_domain/resources \\
        [--n-samples 10] \\
        [--domain-name my_domain] \\
        [--model claude-sonnet-4-6] \\
        [--threshold 0.20] \\
        [--max-iterations 5] \\
        [--no-refine] \\
        [--resume path/to/bootstrap_manifest.yaml]

Outputs (written to --out-dir):
    claim_type_patterns.yaml   — draft patterns with bootstrap_confidence
    seed_entities.csv          — draft entities with bootstrap_confidence column
    domain_profile.yaml        — skeleton profile referencing the above files
    bootstrap_manifest.yaml    — audit trail of every iteration + review recs
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLAIM_PATTERNS_PROMPT = """\
You are analyzing a corpus of documents. Below are excerpts from {n} sample documents.

Your task: identify recurring types of factual claims made in this corpus.
For each claim type you identify, provide:
- claim_type: a snake_case label (e.g. "species_count", "weather_observation")
- regex: a simple case-insensitive regex fragment that reliably matches sentences of this type
- weight: a float 1.0–2.0 indicating how distinctive the pattern is (2.0 = very distinctive)
- bootstrap_confidence: float 0.0–1.0, your confidence that this pattern is genuinely domain-relevant
- example_sentences: list of 1–3 example sentences from the text

IMPORTANT — do NOT create patterns for any of the following:
- Table-of-contents lines (e.g. "Page 3 ... Introduction")
- Section headers or titles (e.g. "SECTION II: METHODOLOGY")
- Page numbers, running headers, footers, or boilerplate
- Bibliographic references or citation lines
- Metadata fields (date stamps, document IDs, file names)

Focus on substantive factual assertions — sentences that state something happened,
was observed, was measured, or was decided.

Output a JSON array of objects with these fields. Aim for 8–15 claim types.
Output ONLY the JSON array, no prose, no markdown fences.

DOCUMENT EXCERPTS:
{text}
"""

_ENTITY_TYPES_PROMPT = """\
You are analyzing a corpus of documents. Below are excerpts from {n} sample documents.

Your task: identify the categories of named entities that recur across these documents.
Do NOT list individual entities — only the *types* (categories) that would be useful
for a knowledge graph of this domain.

For each entity type, provide:
- entity_type: a CamelCase label (e.g. "Species", "AircraftModel", "GeographicFeature")
- description: one sentence explaining what this type represents in this corpus

Return 6–12 entity types. Output ONLY a JSON array of objects, no prose, no markdown fences.

DOCUMENT EXCERPTS:
{text}
"""

_ENTITY_INSTANCES_PROMPT = """\
You are analyzing a corpus of documents. Below are excerpts from {n} sample documents.

The following entity types have been identified for this domain:
{entity_types_block}

Your task: for each entity type above, list the specific named entities that appear
in the text. For each entity provide:
- entity_type: one of the types listed above (use the exact CamelCase label)
- name: canonical display name
- prop_key: a meaningful property key (e.g. "species_name", "place_name", "model_designation")
- prop_value: the value for that property (usually same as or close to name, lowercased/normalized)
- bootstrap_confidence: float 0.0–1.0, your confidence this entity genuinely recurs

Aim for 20–60 entities total. Output ONLY a JSON array of objects, no prose, no markdown fences.

DOCUMENT EXCERPTS:
{text}
"""

_REFINEMENT_PROMPT = """\
You are refining domain configuration for a document analysis pipeline.

## Current claim_type_patterns.yaml
```yaml
{patterns_yaml}
```

## Quality metrics from the last validation pass
- Total claims extracted: {claim_count}
- Unclassified claims: {unclassified_count} ({unclassified_rate_pct:.1f}%)
- Claim-entity link diagnostics: {diagnostics_block}

## Unclassified sentences (not matched by any pattern)
{unclassified_block}

## Low-confidence sentences (matched but with extraction_confidence < 0.5)
{low_confidence_block}

## Your task
Analyze the unclassified and low-confidence sentences and propose changes to the
claim_type_patterns.yaml so that more sentences are correctly classified.

Return a JSON object with three arrays:

1. "pattern_additions" — new patterns to add:
   [{{"claim_type": "...", "regex": "...", "weight": 1.0, "bootstrap_confidence": 0.7,
     "rationale": "why this pattern is needed", "example_sentences": ["..."]}}]

2. "pattern_modifications" — changes to existing patterns:
   [{{"claim_type": "...", "old_regex": "...", "new_regex": "...", "new_weight": 1.2,
     "rationale": "why this change improves classification"}}]

3. "entity_additions" — new seed entities to add for better entity linking:
   [{{"entity_type": "...", "name": "...", "prop_key": "...", "prop_value": "...",
     "bootstrap_confidence": 0.7, "rationale": "why this entity is needed"}}]

Constraints:
- Do not add more than 5 new claim types in one iteration (prefer refining regex first)
- Regex patterns must be valid Python regex (re module)
- Prefer broad patterns that cover sentence clusters, not one-off sentences
- Each change MUST include a rationale

Output ONLY the JSON object, no prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_CHARS = 24_000  # ~6 000 tokens of text to stay within context limits


def _collect_text_stratified(sample_paths: list[Path], n: int) -> tuple[str, int]:
    """Load up to *n* files, selecting first/middle/last paragraphs from each.

    Returns ``(concatenated_text, doc_count)``.
    """
    from gemynd.ingest.source_parser import parse_source_file

    paragraphs: list[str] = []
    docs_read = 0
    for path in sample_paths[:n]:
        try:
            bundle = parse_source_file(path)
        except Exception as exc:
            print(f"[warn] Could not parse {path}: {exc}", file=sys.stderr)
            continue
        docs_read += 1
        texts = [
            (p.clean_text or p.raw_ocr_text or "").strip()
            for p in bundle.paragraphs
        ]
        texts = [t for t in texts if t]
        if not texts:
            continue
        # Select first, middle, last — plus up to 2 random interior paragraphs.
        indices = {0, len(texts) // 2, len(texts) - 1}
        interior = [i for i in range(1, len(texts) - 1) if i not in indices]
        if interior:
            indices.update(random.sample(interior, min(2, len(interior))))
        for idx in sorted(indices):
            paragraphs.append(texts[idx])

    joined = "\n\n".join(paragraphs)
    if len(joined) > _MAX_CHARS:
        joined = joined[:_MAX_CHARS]
        last_period = joined.rfind(". ")
        if last_period > _MAX_CHARS // 2:
            joined = joined[: last_period + 1]
    return joined, docs_read


def _call_claude(client: object, model: str, prompt: str) -> str:
    """Send a single-turn message to Claude and return the response text."""
    message = client.messages.create(  # type: ignore[attr-defined]
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _parse_json_response(raw: str, label: str) -> list | dict:
    """Parse a JSON array or object from the model response."""
    text = raw
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[warn] Could not parse {label} response as JSON: {exc}", file=sys.stderr)
        print(f"       Raw response (first 300 chars): {raw[:300]!r}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _write_claim_patterns(out_dir: Path, patterns: list[dict]) -> None:
    """Write draft claim_type_patterns.yaml."""
    entries = []
    for p in patterns:
        entry: dict = {
            "claim_type": str(p.get("claim_type", "unknown")),
            "regex": str(p.get("regex", "")),
            "weight": float(p.get("weight", 1.0)),
            "bootstrap_confidence": float(p.get("bootstrap_confidence", 0.5)),
        }
        examples = p.get("example_sentences")
        if isinstance(examples, list) and examples:
            entry["example_sentences"] = [str(s) for s in examples[:3]]
        entries.append(entry)

    payload = {"version": "1", "patterns": entries}
    out_path = out_dir / "claim_type_patterns.yaml"
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(payload, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"  Wrote {len(entries)} claim patterns → {out_path}")


def _write_seed_entities(out_dir: Path, entities: list[dict]) -> None:
    """Write draft seed_entities.csv."""
    out_path = out_dir / "seed_entities.csv"
    fieldnames = ["entity_type", "name", "prop_key", "prop_value", "bootstrap_confidence"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in entities:
            writer.writerow({
                "entity_type": str(e.get("entity_type", "Unknown")),
                "name": str(e.get("name", "")),
                "prop_key": str(e.get("prop_key", "name")),
                "prop_value": str(e.get("prop_value", e.get("name", ""))),
                "bootstrap_confidence": float(e.get("bootstrap_confidence", 0.5)),
            })
    print(f"  Wrote {len(entities)} seed entities → {out_path}")


def _write_domain_profile(out_dir: Path, domain_name: str) -> None:
    """Write a skeleton domain_profile.yaml."""
    profile = {
        "version": "1",
        "domain": domain_name,
        "description": f"TODO: describe the {domain_name} corpus.",
        "resources": {
            "seed_entities": "seed_entities.csv",
            "claim_type_patterns": "claim_type_patterns.yaml",
            "claim_role_policy": "claim_role_policy.yaml",
            "measurement_units": "measurement_units.yaml",
            "measurement_species": "measurement_species.yaml",
            "ocr_corrections": "ocr_corrections.yaml",
            "claim_relation_compatibility": "claim_relation_compatibility.yaml",
        },
        "document_anchor": {
            "__comment": "Remove or fill in if documents share a primary subject entity.",
            "title_keyword": "TODO",
            "entity_type": "TODO",
            "name": "TODO",
            "normalized_form": "TODO",
        },
        "synthesis_context": f"TODO: describe the {domain_name} corpus for the synthesis prompt.",
    }
    out_path = out_dir / "domain_profile.yaml"
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(profile, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"  Wrote domain profile skeleton → {out_path}")


def _read_file_text(path: Path) -> str:
    """Read a file as UTF-8 text, returning '' if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# Phase 1 — Draft generation
# ---------------------------------------------------------------------------


def _generate_draft(
    client: object,
    model: str,
    corpus_text: str,
    n_docs: int,
    out_dir: Path,
    domain_name: str,
) -> None:
    """Generate initial claim patterns + seed entities + domain profile."""
    # Call 1 — claim type patterns (with negative-example guidance).
    print("Calling Claude for claim type patterns…")
    patterns_prompt = _CLAIM_PATTERNS_PROMPT.format(n=n_docs, text=corpus_text)
    patterns_raw = _call_claude(client, model, patterns_prompt)
    patterns = _parse_json_response(patterns_raw, "claim patterns")
    if not isinstance(patterns, list):
        patterns = []

    # Call 2 — entity types (stage 1: types only).
    print("Calling Claude for entity types…")
    types_prompt = _ENTITY_TYPES_PROMPT.format(n=n_docs, text=corpus_text)
    types_raw = _call_claude(client, model, types_prompt)
    entity_types = _parse_json_response(types_raw, "entity types")
    if not isinstance(entity_types, list):
        entity_types = []

    # Call 3 — entity instances (stage 2: given types).
    types_block = "\n".join(
        f"- {t.get('entity_type', '?')}: {t.get('description', '')}"
        for t in entity_types
    )
    if not types_block:
        types_block = "- Unknown: no types identified"

    print("Calling Claude for entity instances…")
    instances_prompt = _ENTITY_INSTANCES_PROMPT.format(
        n=n_docs, text=corpus_text, entity_types_block=types_block,
    )
    entities_raw = _call_claude(client, model, instances_prompt)
    entities = _parse_json_response(entities_raw, "seed entities")
    if not isinstance(entities, list):
        entities = []

    # Write outputs.
    print(f"Writing output files to {out_dir}/")
    _write_claim_patterns(out_dir, patterns)
    _write_seed_entities(out_dir, entities)
    _write_domain_profile(out_dir, domain_name)


# ---------------------------------------------------------------------------
# Phase 2 — Validation
# ---------------------------------------------------------------------------


def _run_validation(
    sample_paths: list[Path],
    out_dir: Path,
) -> tuple[dict, list[tuple]]:
    """Run extract_semantic + quality_report on all samples.

    Returns ``(aggregate_metrics, doc_bundles)`` where *doc_bundles* is a list
    of ``(StructureBundle, SemanticBundle)`` tuples.
    """
    from gemynd.ingest.pipeline import (
        extract_semantic,
        quality_report,
    )
    from gemynd.ingest.source_parser import parse_source

    aggregate_claims = 0
    aggregate_unclassified = 0
    diagnostic_totals: dict[str, int] = {}
    per_document: list[dict] = []
    doc_bundles: list[tuple] = []

    for sample_path in sample_paths:
        structure = parse_source(sample_path)
        semantic = extract_semantic(structure, resources_dir=out_dir, no_llm=True)
        metrics = quality_report(structure, semantic)

        aggregate_claims += metrics.get("claim_count", 0)
        aggregate_unclassified += metrics.get("unclassified_claim_count", 0)
        for code, count in metrics.get("claim_link_diagnostic_counts", {}).items():
            diagnostic_totals[code] = diagnostic_totals.get(code, 0) + count

        per_document.append({
            "file": str(sample_path),
            "claim_count": metrics.get("claim_count", 0),
            "unclassified_claim_count": metrics.get("unclassified_claim_count", 0),
        })
        doc_bundles.append((structure, semantic))

    safe_claims = max(1, aggregate_claims)
    unclassified_rate = aggregate_unclassified / safe_claims

    aggregate_metrics = {
        "claim_count": aggregate_claims,
        "unclassified_claim_count": aggregate_unclassified,
        "unclassified_rate": round(unclassified_rate, 4),
        "claim_link_diagnostic_counts": dict(sorted(diagnostic_totals.items())),
        "per_document": per_document,
    }
    return aggregate_metrics, doc_bundles


# ---------------------------------------------------------------------------
# Phase 3 — Refinement
# ---------------------------------------------------------------------------


def _collect_unclassified_sentences(
    doc_bundles: list[tuple],
    max_unclassified: int = 20,
    max_low_confidence: int = 20,
) -> tuple[list[str], list[str], set[str]]:
    """Sample unclassified and low-confidence sentences.

    Returns ``(unclassified_sentences, low_confidence_sentences,
    unclassified_normalized_set)``.  The third element is the full
    (un-sampled) set of normalized unclassified sentences for convergence
    checking.
    """
    unclassified: dict[str, str] = {}  # normalized → source_sentence
    low_conf: dict[str, str] = {}

    for _structure, semantic in doc_bundles:
        for claim in semantic.claims:
            norm = claim.normalized_sentence
            if claim.claim_type == "unclassified_assertion":
                unclassified.setdefault(norm, claim.source_sentence)
            elif claim.extraction_confidence < 0.5:
                low_conf.setdefault(norm, claim.source_sentence)

    full_unclassified_set = set(unclassified.keys())

    unc_list = list(unclassified.values())
    lc_list = list(low_conf.values())
    if len(unc_list) > max_unclassified:
        unc_list = random.sample(unc_list, max_unclassified)
    if len(lc_list) > max_low_confidence:
        lc_list = random.sample(lc_list, max_low_confidence)

    return unc_list, lc_list, full_unclassified_set


def _build_refinement_prompt(
    current_patterns_yaml: str,
    aggregate_metrics: dict,
    unclassified_sentences: list[str],
    low_confidence_sentences: list[str],
) -> str:
    """Assemble the refinement prompt from validation results."""
    claim_count = aggregate_metrics.get("claim_count", 0)
    unc_count = aggregate_metrics.get("unclassified_claim_count", 0)
    unc_rate = aggregate_metrics.get("unclassified_rate", 0.0)

    diag = aggregate_metrics.get("claim_link_diagnostic_counts", {})
    if diag:
        diagnostics_block = "\n".join(f"  {k}: {v}" for k, v in diag.items())
    else:
        diagnostics_block = "  (none)"

    if unclassified_sentences:
        unc_block = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(unclassified_sentences, 1)
        )
    else:
        unc_block = "  (none)"

    if low_confidence_sentences:
        lc_block = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(low_confidence_sentences, 1)
        )
    else:
        lc_block = "  (none)"

    return _REFINEMENT_PROMPT.format(
        patterns_yaml=current_patterns_yaml,
        claim_count=claim_count,
        unclassified_count=unc_count,
        unclassified_rate_pct=unc_rate * 100,
        diagnostics_block=diagnostics_block,
        unclassified_block=unc_block,
        low_confidence_block=lc_block,
    )


def _apply_refinements(out_dir: Path, refinements: dict) -> dict:
    """Merge LLM-returned refinements into config files on disk.

    Returns a summary ``{patterns_added, patterns_modified, entities_added,
    rationales}``.
    """
    from gemynd.core.domain_config import load_domain_config

    patterns_path = out_dir / "claim_type_patterns.yaml"
    entities_path = out_dir / "seed_entities.csv"

    # Backup current files for rollback.
    patterns_backup = _read_file_text(patterns_path)
    entities_backup = _read_file_text(entities_path)

    # --- Patterns -----------------------------------------------------------
    with patterns_path.open("r", encoding="utf-8") as fh:
        patterns_data = yaml.safe_load(fh) or {}
    existing_patterns = patterns_data.get("patterns", [])
    existing_types = {p["claim_type"] for p in existing_patterns}

    patterns_added = 0
    patterns_modified = 0
    rationales: list[str] = []

    # Additions.
    for addition in refinements.get("pattern_additions", []):
        ct = str(addition.get("claim_type", ""))
        regex_str = str(addition.get("regex", ""))
        if not ct or not regex_str:
            continue
        # Validate regex.
        try:
            re.compile(regex_str)
        except re.error as exc:
            print(f"  [warn] Skipping invalid regex for {ct}: {exc}", file=sys.stderr)
            continue
        if ct in existing_types:
            print(f"  [warn] Skipping duplicate claim_type {ct}", file=sys.stderr)
            continue
        entry = {
            "claim_type": ct,
            "regex": regex_str,
            "weight": float(addition.get("weight", 1.0)),
            "bootstrap_confidence": float(addition.get("bootstrap_confidence", 0.5)),
        }
        examples = addition.get("example_sentences")
        if isinstance(examples, list) and examples:
            entry["example_sentences"] = [str(s) for s in examples[:3]]
        existing_patterns.append(entry)
        existing_types.add(ct)
        patterns_added += 1
        if addition.get("rationale"):
            rationales.append(f"Added pattern '{ct}': {addition['rationale']}")

    if patterns_added > 5:
        print(
            f"  [warn] {patterns_added} new claim types added in one iteration — "
            "review for overfitting.",
            file=sys.stderr,
        )

    # Modifications.
    type_to_idx = {p["claim_type"]: i for i, p in enumerate(existing_patterns)}
    for mod in refinements.get("pattern_modifications", []):
        ct = str(mod.get("claim_type", ""))
        if ct not in type_to_idx:
            print(f"  [warn] Cannot modify unknown pattern {ct}", file=sys.stderr)
            continue
        idx = type_to_idx[ct]
        new_regex = mod.get("new_regex")
        if new_regex:
            try:
                re.compile(str(new_regex))
            except re.error as exc:
                print(f"  [warn] Skipping invalid regex for {ct}: {exc}", file=sys.stderr)
                continue
            existing_patterns[idx]["regex"] = str(new_regex)
        new_weight = mod.get("new_weight")
        if new_weight is not None:
            existing_patterns[idx]["weight"] = float(new_weight)
        patterns_modified += 1
        if mod.get("rationale"):
            rationales.append(f"Modified pattern '{ct}': {mod['rationale']}")

    patterns_data["patterns"] = existing_patterns
    with patterns_path.open("w", encoding="utf-8") as fh:
        yaml.dump(patterns_data, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # --- Entities -----------------------------------------------------------
    entities_added = 0
    new_rows: list[dict] = []
    for ent in refinements.get("entity_additions", []):
        name = str(ent.get("name", ""))
        if not name:
            continue
        new_rows.append({
            "entity_type": str(ent.get("entity_type", "Unknown")),
            "name": name,
            "prop_key": str(ent.get("prop_key", "name")),
            "prop_value": str(ent.get("prop_value", name)),
            "bootstrap_confidence": float(ent.get("bootstrap_confidence", 0.5)),
        })
        entities_added += 1
        if ent.get("rationale"):
            rationales.append(f"Added entity '{name}': {ent['rationale']}")

    if new_rows:
        fieldnames = ["entity_type", "name", "prop_key", "prop_value", "bootstrap_confidence"]
        with entities_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            for row in new_rows:
                writer.writerow(row)

    # Smoke-test: ensure load_domain_config can still parse the files.
    try:
        load_domain_config(out_dir)
    except Exception as exc:
        print(f"  [error] Config broken after refinement — reverting: {exc}", file=sys.stderr)
        patterns_path.write_text(patterns_backup, encoding="utf-8")
        entities_path.write_text(entities_backup, encoding="utf-8")
        return {"patterns_added": 0, "patterns_modified": 0, "entities_added": 0, "rationales": []}

    return {
        "patterns_added": patterns_added,
        "patterns_modified": patterns_modified,
        "entities_added": entities_added,
        "rationales": rationales,
    }


# ---------------------------------------------------------------------------
# Phase 4 — Convergence
# ---------------------------------------------------------------------------


def _check_convergence(
    iteration: int,
    metrics_history: list[dict],
    unclassified_sets: list[set[str]],
    threshold: float,
    max_iterations: int,
    min_improvement_pp: float = 2.0,
) -> tuple[bool, str]:
    """Return ``(should_stop, reason)``."""
    current = metrics_history[-1]
    rate = current.get("unclassified_rate", 1.0)

    if rate < threshold:
        return True, "threshold_met"
    if iteration >= max_iterations:
        return True, "max_iterations"
    if len(metrics_history) >= 2:
        prev_rate = metrics_history[-2].get("unclassified_rate", 1.0)
        improvement = (prev_rate - rate) * 100  # percentage points
        if improvement < min_improvement_pp:
            return True, "plateau"
    if len(unclassified_sets) >= 2 and unclassified_sets[-1] == unclassified_sets[-2]:
        return True, "set_stable"

    return False, ""


# ---------------------------------------------------------------------------
# Phase 5 — Manifest
# ---------------------------------------------------------------------------


def _config_snapshot(out_dir: Path) -> dict[str, str]:
    """Capture current config file contents for the manifest."""
    return {
        "claim_type_patterns_yaml": _read_file_text(out_dir / "claim_type_patterns.yaml"),
        "seed_entities_csv": _read_file_text(out_dir / "seed_entities.csv"),
    }


def _build_review_recommendations(
    iterations: list[dict],
    final_unclassified: list[str],
) -> list[dict]:
    """Build human-readable review recommendations for the manifest."""
    recs: list[dict] = []

    # Collect all rationales from refinement iterations.
    for it in iterations:
        if it.get("phase") != "refinement":
            continue
        for rat in it.get("changes", {}).get("rationales", []):
            if "Added pattern" in rat:
                ct = rat.split("'")[1] if "'" in rat else "unknown"
                recs.append({
                    "type": "verify_pattern",
                    "claim_type": ct,
                    "rationale": rat,
                })

    # Flag remaining unclassified.
    if final_unclassified:
        recs.append({
            "type": "manual_review_needed",
            "issue": f"{len(final_unclassified)} sentences remain unclassified",
            "sample_sentences": final_unclassified[:10],
            "suggestion": "These may represent claim types not in the current taxonomy.",
        })

    return recs


def _save_manifest(
    out_dir: Path,
    domain_name: str,
    iterations: list[dict],
    convergence_reason: str,
    final_metrics: dict,
    sample_paths: list[str],
    model: str,
    final_unclassified_sentences: list[str],
) -> Path:
    """Write bootstrap_manifest.yaml."""
    recs = _build_review_recommendations(iterations, final_unclassified_sentences)

    manifest = {
        "version": "1",
        "domain": domain_name,
        "model": model,
        "sample_paths": sample_paths,
        "started_at": iterations[0].get("timestamp", "") if iterations else "",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "convergence_reason": convergence_reason,
        "iterations": iterations,
        "final_metrics": {
            "claim_count": final_metrics.get("claim_count", 0),
            "unclassified_claim_count": final_metrics.get("unclassified_claim_count", 0),
            "unclassified_rate": final_metrics.get("unclassified_rate", 0.0),
        },
        "review_recommendations": recs,
    }
    out_path = out_dir / "bootstrap_manifest.yaml"
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(manifest, fh, allow_unicode=True, sort_keys=False, default_flow_style=False,
                  width=120)
    print(f"  Wrote manifest → {out_path}")
    return out_path


def _load_manifest(manifest_path: Path) -> dict:
    """Load a previous manifest for --resume."""
    with manifest_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _restore_from_manifest(manifest: dict, out_dir: Path) -> tuple[list[dict], list[dict], list[set[str]]]:
    """Restore config snapshot and history from a manifest.

    Returns ``(iterations, metrics_history, unclassified_sets)``.
    """
    iterations = manifest.get("iterations", [])
    if not iterations:
        raise ValueError("Manifest has no iterations to resume from.")

    # Restore config files from last iteration's snapshot.
    last = iterations[-1]
    snapshot = last.get("config_snapshot", {})
    patterns_yaml = snapshot.get("claim_type_patterns_yaml", "")
    entities_csv = snapshot.get("seed_entities_csv", "")
    if patterns_yaml:
        (out_dir / "claim_type_patterns.yaml").write_text(patterns_yaml, encoding="utf-8")
    if entities_csv:
        (out_dir / "seed_entities.csv").write_text(entities_csv, encoding="utf-8")

    metrics_history = [it.get("metrics", {}) for it in iterations]
    # We cannot reconstruct unclassified_sets from the manifest, so start fresh.
    unclassified_sets: list[set[str]] = []

    print(f"  Restored config from iteration {last.get('iteration', '?')}")
    return iterations, metrics_history, unclassified_sets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap domain resource files from sample corpus documents."
    )
    parser.add_argument(
        "--samples", nargs="+", required=True,
        help="Paths to source report JSON files (same format as pipeline input).",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory to write the generated resource files.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Maximum number of sample files to read (default: 10).",
    )
    parser.add_argument(
        "--domain-name", default=None,
        help="Domain identifier for domain_profile.yaml (default: stem of --out-dir).",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Anthropic model to use (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key (falls back to Anthropic_API_Key / ANTHROPIC_API_KEY env vars).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.20,
        help="Stop refining when unclassified rate drops below this (default: 0.20).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Maximum refinement iterations (default: 5).",
    )
    parser.add_argument(
        "--no-refine", action="store_true",
        help="Skip refinement loop — only generate the initial draft (Phase 1).",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a bootstrap_manifest.yaml from a previous run to resume.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_name = args.domain_name or out_dir.stem
    sample_paths = [Path(s) for s in args.samples]

    # Resolve API key.
    api_key = (
        args.api_key
        or os.environ.get("Anthropic_API_Key")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        print("[error] No Anthropic API key found. Set Anthropic_API_Key or ANTHROPIC_API_KEY.",
              file=sys.stderr)
        return 1

    try:
        import anthropic as _anthropic
    except ImportError:
        print("[error] anthropic package not installed. Run: pip install -e .[tools]",
              file=sys.stderr)
        return 1

    client = _anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Resume or fresh start
    # ------------------------------------------------------------------
    iterations: list[dict] = []
    metrics_history: list[dict] = []
    unclassified_sets: list[set[str]] = []
    convergence_reason = "not_started"
    final_metrics: dict = {}
    final_unclassified_sentences: list[str] = []

    if args.resume:
        print(f"Resuming from {args.resume}…")
        manifest = _load_manifest(Path(args.resume))
        iterations, metrics_history, unclassified_sets = _restore_from_manifest(
            manifest, out_dir,
        )
        start_iteration = iterations[-1].get("iteration", 0) + 1
        # Override model/domain from manifest if not explicitly given.
        if not args.domain_name and manifest.get("domain"):
            domain_name = manifest["domain"]
    else:
        # Phase 1 — Draft generation.
        print(f"Reading up to {args.n_samples} sample documents…")
        corpus_text, n_docs = _collect_text_stratified(sample_paths, args.n_samples)
        if not corpus_text.strip():
            print("[error] No text extracted from samples.", file=sys.stderr)
            return 1
        print(f"  Collected {len(corpus_text):,} characters from {n_docs} documents.")

        _generate_draft(client, args.model, corpus_text, n_docs, out_dir, domain_name)

        # Phase 2 — Validate the draft.
        print("Validating initial draft…")
        metrics, doc_bundles = _run_validation(sample_paths, out_dir)
        unc_sents, lc_sents, unc_set = _collect_unclassified_sentences(doc_bundles)

        metrics_history.append(metrics)
        unclassified_sets.append(unc_set)
        final_unclassified_sentences = unc_sents

        iteration_record = {
            "iteration": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "draft",
            "metrics": {
                "claim_count": metrics["claim_count"],
                "unclassified_claim_count": metrics["unclassified_claim_count"],
                "unclassified_rate": metrics["unclassified_rate"],
            },
            "changes": {
                "patterns_added": len(
                    (yaml.safe_load(_read_file_text(out_dir / "claim_type_patterns.yaml")) or {})
                    .get("patterns", [])
                ),
                "entities_added": sum(
                    1 for _ in csv.DictReader(
                        io.StringIO(_read_file_text(out_dir / "seed_entities.csv"))
                    )
                ),
            },
            "config_snapshot": _config_snapshot(out_dir),
        }
        iterations.append(iteration_record)
        final_metrics = metrics
        start_iteration = 2

        rate = metrics["unclassified_rate"]
        print(f"  Initial unclassified rate: {rate:.1%} "
              f"({metrics['unclassified_claim_count']}/{metrics['claim_count']} claims)")

        if args.no_refine:
            convergence_reason = "no_refine"
            _save_manifest(
                out_dir, domain_name, iterations, convergence_reason,
                final_metrics, [str(p) for p in sample_paths], args.model,
                final_unclassified_sentences,
            )
            print("\nBootstrap complete (--no-refine). Next steps:")
            print(f"  1. Review {out_dir}/claim_type_patterns.yaml")
            print(f"  2. Review {out_dir}/seed_entities.csv")
            print(f"  3. Fill in {out_dir}/domain_profile.yaml")
            print(f"  4. Run: gemynd validate-domain --samples <files> --domain-dir {out_dir}")
            return 0

    # ------------------------------------------------------------------
    # Refinement loop (Phases 2–4)
    # ------------------------------------------------------------------
    for i in range(start_iteration, args.max_iterations + 2):  # +2 because range is exclusive
        print(f"\n{'='*60}")
        print(f"Refinement iteration {i}")
        print(f"{'='*60}")

        # Phase 2 — Validate.
        print("  Validating current config…")
        metrics, doc_bundles = _run_validation(sample_paths, out_dir)
        unc_sents, lc_sents, unc_set = _collect_unclassified_sentences(doc_bundles)

        metrics_history.append(metrics)
        unclassified_sets.append(unc_set)
        final_metrics = metrics
        final_unclassified_sentences = unc_sents

        rate = metrics["unclassified_rate"]
        print(f"  Unclassified rate: {rate:.1%} "
              f"({metrics['unclassified_claim_count']}/{metrics['claim_count']} claims)")

        # Phase 4 — Convergence check.
        converged, reason = _check_convergence(
            i, metrics_history, unclassified_sets,
            args.threshold, args.max_iterations,
        )
        if converged:
            convergence_reason = reason
            print(f"  Converged: {reason}")
            break

        # Phase 3 — Refine.
        print("  Building refinement prompt…")
        current_yaml = _read_file_text(out_dir / "claim_type_patterns.yaml")
        prompt = _build_refinement_prompt(current_yaml, metrics, unc_sents, lc_sents)

        print("  Calling Claude for refinements…")
        raw = _call_claude(client, args.model, prompt)
        refinements = _parse_json_response(raw, "refinements")
        if isinstance(refinements, list):
            print("  [warn] Expected JSON object, got array — wrapping.", file=sys.stderr)
            refinements = {}

        print("  Applying refinements…")
        changes = _apply_refinements(out_dir, refinements)
        print(f"    Patterns added: {changes['patterns_added']}, "
              f"modified: {changes['patterns_modified']}, "
              f"entities added: {changes['entities_added']}")

        iteration_record = {
            "iteration": i,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "refinement",
            "metrics": {
                "claim_count": metrics["claim_count"],
                "unclassified_claim_count": metrics["unclassified_claim_count"],
                "unclassified_rate": metrics["unclassified_rate"],
            },
            "changes": changes,
            "config_snapshot": _config_snapshot(out_dir),
        }
        iterations.append(iteration_record)
    else:
        # Loop completed without break — max iterations reached.
        convergence_reason = "max_iterations"

    # ------------------------------------------------------------------
    # Phase 5 — Manifest
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Writing manifest…")
    manifest_path = _save_manifest(
        out_dir, domain_name, iterations, convergence_reason,
        final_metrics, [str(p) for p in sample_paths], args.model,
        final_unclassified_sentences,
    )

    total_iters = len(iterations)
    final_rate = final_metrics.get("unclassified_rate", 0.0)
    print(f"\nBootstrap complete after {total_iters} iteration(s).")
    print(f"  Convergence reason: {convergence_reason}")
    print(f"  Final unclassified rate: {final_rate:.1%}")
    print(f"\nNext steps:")
    print(f"  1. Review {manifest_path} for change rationales and recommendations")
    print(f"  2. Tune {out_dir}/claim_type_patterns.yaml as needed")
    print(f"  3. Review {out_dir}/seed_entities.csv")
    print(f"  4. Fill in {out_dir}/domain_profile.yaml (document_anchor, synthesis_context)")
    print(f"  5. Run: gemynd validate-domain --samples <files> --domain-dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
