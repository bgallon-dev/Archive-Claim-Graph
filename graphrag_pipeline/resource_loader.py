"""Typed loaders for externalized domain resources (YAML / CSV).

All loaders accept an optional *resources_dir* argument so tests can point at
fixture directories without touching the real files.  The default resolves to
the ``resources/`` sub-directory sitting next to this module.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_RESOURCES_DIR: Path = Path(__file__).parent / "resources"


def _safe_compile(pattern: str, flags: int = 0) -> re.Pattern:
    """Compile a regex, raising ValueError on malformed patterns.

    Prevents a tampered/malformed resource YAML from injecting a pattern that
    would raise an uncaught exception or cause a ReDoS at extraction time.
    """
    try:
        return re.compile(pattern, flags)
    except re.error as e:
        raise ValueError(f"Invalid regex in resource file: {pattern!r}") from e


def _dir(resources_dir: Path | None) -> Path:
    return resources_dir if resources_dir is not None else _DEFAULT_RESOURCES_DIR


def _active_profile(resources_dir: Path | None) -> dict[str, Any]:
    """Return domain_profile.yaml payload, or {} if absent (backward compat)."""
    p = _dir(resources_dir) / "domain_profile.yaml"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resource_path(key: str, resources_dir: Path | None) -> Path:
    """Resolve a resource filename via domain_profile.yaml, falling back to the key."""
    filename = _active_profile(resources_dir).get("resources", {}).get(key, key)
    return _dir(resources_dir) / filename


# ── Seed entities ─────────────────────────────────────────────────────────────

def load_seed_entity_rows(resources_dir: Path | None = None) -> list[dict[str, str]]:
    """Return raw CSV rows from seed_entities.csv.

    Each row has keys: ``entity_type``, ``name``, ``prop_key``, ``prop_value``.
    The caller is responsible for grouping multi-property entities.
    """
    path = _resource_path("seed_entities", resources_dir)
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


# ── Claim type patterns ───────────────────────────────────────────────────────

def load_claim_type_patterns(
    resources_dir: Path | None = None,
) -> list[tuple[str, re.Pattern[str], float]]:
    """Return ``[(claim_type, compiled_pattern, weight), ...]`` in file order."""
    path = _resource_path("claim_type_patterns", resources_dir)
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    result: list[tuple[str, re.Pattern[str], float]] = []
    for entry in data["patterns"]:
        compiled = _safe_compile(entry["regex"], re.IGNORECASE)
        result.append((str(entry["claim_type"]), compiled, float(entry["weight"])))
    return result


# ── Claim role policy ─────────────────────────────────────────────────────────

def load_claim_role_policy(
    resources_dir: Path | None = None,
) -> dict[tuple[str, str], str]:
    """Return ``{(claim_type, entity_type): relation_type}`` mapping."""
    path = _resource_path("claim_role_policy", resources_dir)
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return {
        (str(e["claim_type"]), str(e["entity_type"])): str(e["relation_type"])
        for e in data["policy"]
    }


# ── Measurement units ─────────────────────────────────────────────────────────

def load_measurement_units(
    resources_dir: Path | None = None,
) -> dict[str, tuple[str, str]]:
    """Return ``{unit_word: (measurement_name, unit_string)}`` mapping."""
    path = _resource_path("measurement_units", resources_dir)
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return {k: (str(v["name"]), str(v["unit"])) for k, v in data["units"].items()}


# ── Measurement species ───────────────────────────────────────────────────────

def load_measurement_species(
    resources_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the full measurement_species.yaml payload.

    Relevant keys:
    - ``type_hints``: ``{singular_key: category}`` for ``target_entity_type_hint``
    - ``immediate_patterns``: list of regex fragments for ``_immediate_species_re``
    """
    path = _resource_path("measurement_species", resources_dir)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── OCR corrections ───────────────────────────────────────────────────────────

def load_ocr_corrections(
    resources_dir: Path | None = None,
) -> frozenset[str]:
    """Return the set of known OCR error tokens (lowercased)."""
    path = _resource_path("ocr_corrections", resources_dir)
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    known_errors = data["known_errors"]
    if isinstance(known_errors, dict):
        return frozenset(str(token).lower() for token in known_errors)
    return frozenset(str(token).lower() for token in known_errors)


def load_ocr_correction_map(
    resources_dir: Path | None = None,
) -> dict[str, str]:
    """Return ``{known_error: suggested_correction}`` mapping."""
    path = _resource_path("ocr_corrections", resources_dir)
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    known_errors = data["known_errors"]
    if isinstance(known_errors, dict):
        return {
            str(token).lower(): str(correction).lower()
            for token, correction in known_errors.items()
        }
    return {str(token).lower(): "" for token in known_errors}


_TERM_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _split_reference_terms(text: str) -> set[str]:
    normalized = text.replace("_", " ").strip().lower()
    if not normalized:
        return set()
    terms = {normalized}
    terms.update(token.lower() for token in _TERM_RE.findall(normalized))
    return {term for term in terms if len(term) >= 2}


def _extract_pattern_terms(pattern_source: str) -> set[str]:
    simplified = (
        pattern_source
        .replace(r"\b", " ")
        .replace(r"\s+", " ")
        .replace(r"(?:", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("|", " ")
        .replace("?", "")
        .replace("+", " ")
        .replace("*", " ")
    )
    return {token.lower() for token in re.findall(r"[A-Za-z]{3,}", simplified)}


def load_spelling_reference_terms(
    resources_dir: Path | None = None,
) -> frozenset[str]:
    """Return domain-aware reference terms for conservative spelling checks."""
    terms: set[str] = set()

    for row in load_seed_entity_rows(resources_dir):
        terms.update(_split_reference_terms(str(row.get("name", ""))))

    for unit_word, (measurement_name, unit) in load_measurement_units(resources_dir).items():
        terms.update(_split_reference_terms(unit_word))
        terms.update(_split_reference_terms(measurement_name))
        terms.update(_split_reference_terms(unit))

    for claim_type, compiled, _ in load_claim_type_patterns(resources_dir):
        terms.update(_split_reference_terms(claim_type))
        terms.update(_extract_pattern_terms(compiled.pattern))

    return frozenset(sorted(terms))


# ── Domain profile ────────────────────────────────────────────────────────────

def load_domain_profile(
    resources_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the full domain_profile.yaml payload."""
    path = _dir(resources_dir) / "domain_profile.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Claim relation compatibility ──────────────────────────────────────────────

def load_claim_relation_compatibility(
    resources_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the full claim_relation_compatibility.yaml payload.

    Relevant keys:
    - ``compatibility``: ``{claim_type: {relation_type: "strong"|"weak"|"forbidden"}}``
    - ``preferred_entity_types``: ``{claim_type: [entity_type, ...]}``
    """
    path = _resource_path("claim_relation_compatibility", resources_dir)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
