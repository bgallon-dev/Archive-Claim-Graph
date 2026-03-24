"""Generate starter domain resource files from a sample of corpus documents.

Uses the Anthropic API to extract candidate claim-type patterns and seed
entities from raw text, then writes draft YAML/CSV files that can be reviewed
and tuned before running the full pipeline.

Usage:
    python -m scripts.bootstrap_domain \\
        --samples input/new_domain/*.json \\
        --out-dir domains/new_domain/resources \\
        [--n-samples 10] \\
        [--domain-name my_domain] \\
        [--model claude-sonnet-4-6]

Outputs (written to --out-dir):
    claim_type_patterns.yaml   — draft patterns with bootstrap_confidence
    seed_entities.csv          — draft entities with bootstrap_confidence column
    domain_profile.yaml        — skeleton profile referencing the above files
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLAIM_PATTERNS_PROMPT = """\
You are analyzing a corpus of historical documents. Below are excerpts from {n} sample documents.

Your task: identify recurring types of factual claims made in this corpus.
For each claim type you identify, provide:
- claim_type: a snake_case label (e.g. "species_count", "weather_observation")
- regex: a simple case-insensitive regex fragment that reliably matches sentences of this type
- weight: a float 1.0–2.0 indicating how distinctive the pattern is (2.0 = very distinctive)
- bootstrap_confidence: float 0.0–1.0, your confidence that this pattern is genuinely domain-relevant
- example_sentences: list of 1–3 example sentences from the text

Output a JSON array of objects with these fields. Aim for 8–15 claim types.
Output ONLY the JSON array, no prose, no markdown fences.

DOCUMENT EXCERPTS:
{text}
"""

_SEED_ENTITIES_PROMPT = """\
You are analyzing a corpus of historical documents. Below are excerpts from {n} sample documents.

Your task: identify recurring named entities that appear across multiple documents.
For each entity, provide:
- entity_type: one of Species, Habitat, Place, Refuge, Organization, Person, SurveyMethod, Activity
- name: canonical display name
- prop_key: a meaningful property key (e.g. "species_name", "place_name")
- prop_value: the value for that property (usually same as or close to name, lowercased/normalized)
- bootstrap_confidence: float 0.0–1.0, your confidence this entity is genuinely recurrent

Output a JSON array of objects with these fields. Aim for 20–60 entities.
Output ONLY the JSON array, no prose, no markdown fences.

DOCUMENT EXCERPTS:
{text}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_CHARS = 24_000  # ~6 000 tokens of text to stay within context limits


def _collect_text(sample_paths: list[Path], n: int) -> str:
    """Load up to n files via parse_source_file and return concatenated paragraph text."""
    # Import here so the module can be imported without the full pipeline installed.
    from graphrag_pipeline.source_parser import parse_source_file

    paragraphs: list[str] = []
    for path in sample_paths[:n]:
        try:
            bundle = parse_source_file(path)
        except Exception as exc:
            print(f"[warn] Could not parse {path}: {exc}", file=sys.stderr)
            continue
        for para in bundle.paragraphs:
            text = para.clean_text or para.raw_ocr_text
            if text and text.strip():
                paragraphs.append(text.strip())

    joined = "\n\n".join(paragraphs)
    if len(joined) > _MAX_CHARS:
        joined = joined[:_MAX_CHARS]
        # Trim to last complete sentence to avoid cutting mid-word.
        last_period = joined.rfind(". ")
        if last_period > _MAX_CHARS // 2:
            joined = joined[: last_period + 1]
    return joined


def _call_claude(client: object, model: str, prompt: str) -> str:  # type: ignore[misc]
    """Send a single-turn message to Claude and return the response text."""
    message = client.messages.create(  # type: ignore[attr-defined]
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _parse_json_response(raw: str, label: str) -> list[dict]:
    """Parse a JSON array from the model response; warn and return [] on failure."""
    # Strip any accidental markdown fences.
    text = raw
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        result = json.loads(text)
        if not isinstance(result, list):
            raise ValueError("Expected a JSON array")
        return result
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[warn] Could not parse {label} response as JSON: {exc}", file=sys.stderr)
        print(f"       Raw response (first 300 chars): {raw[:300]!r}", file=sys.stderr)
        return []


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
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain_name = args.domain_name or out_dir.stem

    # Resolve API key.
    api_key = (
        args.api_key
        or os.environ.get("Anthropic_API_Key")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        print("[error] No Anthropic API key found. Set Anthropic_API_Key or ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1

    try:
        import anthropic as _anthropic
    except ImportError:
        print("[error] anthropic package not installed. Run: pip install -e .[tools]", file=sys.stderr)
        return 1

    client = _anthropic.Anthropic(api_key=api_key)

    # Collect text from sample documents.
    sample_paths = [Path(s) for s in args.samples]
    print(f"Reading up to {args.n_samples} sample documents…")
    corpus_text = _collect_text(sample_paths, args.n_samples)
    if not corpus_text.strip():
        print("[error] No text extracted from samples.", file=sys.stderr)
        return 1
    print(f"  Collected {len(corpus_text):,} characters of corpus text.")

    n_docs = min(len(sample_paths), args.n_samples)

    # Call 1 — claim type patterns.
    print("Calling Claude for claim type patterns…")
    patterns_prompt = _CLAIM_PATTERNS_PROMPT.format(n=n_docs, text=corpus_text)
    patterns_raw = _call_claude(client, args.model, patterns_prompt)
    patterns = _parse_json_response(patterns_raw, "claim patterns")

    # Call 2 — seed entities.
    print("Calling Claude for seed entities…")
    entities_prompt = _SEED_ENTITIES_PROMPT.format(n=n_docs, text=corpus_text)
    entities_raw = _call_claude(client, args.model, entities_prompt)
    entities = _parse_json_response(entities_raw, "seed entities")

    # Write outputs.
    print(f"Writing output files to {out_dir}/")
    _write_claim_patterns(out_dir, patterns)
    _write_seed_entities(out_dir, entities)
    _write_domain_profile(out_dir, domain_name)

    print()
    print("Bootstrap complete. Next steps:")
    print(f"  1. Review and tune {out_dir}/claim_type_patterns.yaml")
    print(f"  2. Review and trim {out_dir}/seed_entities.csv")
    print(f"  3. Fill in {out_dir}/domain_profile.yaml (document_anchor, synthesis_context)")
    print(f"  4. Copy over claim_role_policy.yaml, measurement_units.yaml, etc. from a base domain")
    print(f"  5. Run: graphrag validate-domain --samples <files> --domain-dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
