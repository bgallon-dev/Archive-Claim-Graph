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

_DEFAULT_RESOURCES_DIR: Path = Path(__file__).parent.parent / "resources"


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


def load_negative_entities(resources_dir: Path | None = None) -> frozenset[str]:
    """Return the set of normalized surface forms that must never be emitted as candidates."""
    path = _resource_path("negative_entities", resources_dir)
    if not path.exists():
        return frozenset()
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return frozenset(str(e).strip().lower() for e in (data.get("entries") or []) if e)


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


# ── Derivation registry ───────────────────────────────────────────────────────

def load_derivation_registry(resources_dir: Path | None = None) -> dict[str, dict]:
    """Return the derivation registry keyed by claim_type, or {} if file absent.

    Each value is a dict with keys: observation_type, event_type,
    required_entities, optional_entities.
    """
    path = _resource_path("derivation_registry", resources_dir)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return data.get("entries") or {}


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


# ── Domain schema ────────────────────────────────────────────────────────────

def load_domain_schema(
    resources_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the full domain_schema.yaml payload.

    Keys: ``claim_entity_relations``, ``claim_location_relation``,
    ``claim_location_entity_types``, ``entity_labels``, ``legacy_renames``.
    """
    path = _resource_path("domain_schema", resources_dir)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── Concept rules ────────────────────────────────────────────────────────────

def load_concept_rules(
    resources_dir: Path | None = None,
) -> list[tuple[str, frozenset[str], re.Pattern[str], float]]:
    """Return ``[(concept_id, allowed_claim_types, compiled_regex, confidence), ...]``."""
    path = _resource_path("concept_rules", resources_dir)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    result: list[tuple[str, frozenset[str], re.Pattern[str], float]] = []
    for entry in data.get("rules") or []:
        compiled = _safe_compile(entry["regex"], re.IGNORECASE)
        result.append((
            str(entry["concept_id"]),
            frozenset(entry["claim_types"]),
            compiled,
            float(entry["confidence"]),
        ))
    return result


# ── Query intent ─────────────────────────────────────────────────────────────

def load_query_intent(
    resources_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Return ``{keyword: [claim_type, ...]}`` for retrieval strategy selection."""
    path = _resource_path("query_intent", resources_dir)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    raw = data.get("intent_to_claim_types") or {}
    return {str(k): [str(v) for v in vs] for k, vs in raw.items()}


# ── Corpus registry ──────────────────────────────────────────────────────────

# Default location of the multi-corpus registry, relative to the repo root.
# Resolved as ``<repo_root>/data/corpus_registry.yaml``.
_DEFAULT_REGISTRY_PATH: Path = (
    Path(__file__).parent.parent.parent / "data" / "corpus_registry.yaml"
)


def load_corpus_registry(
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the list of ``corpora`` entries from ``corpus_registry.yaml``.

    Each entry has keys ``corpus_id``, ``display_name``, ``resources_dir`` and
    optional ``description``, ``created_at``, ``is_default``. ``resources_dir``
    may be relative (resolved against the registry file's parent-parent, i.e.
    the repo root) or absolute.
    """
    path = registry_path if registry_path is not None else _DEFAULT_REGISTRY_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    entries = data.get("corpora") or []
    return [dict(e) for e in entries if e.get("corpus_id")]


def resolve_corpus_resources_dir(
    entry: dict[str, Any],
    registry_path: Path | None = None,
) -> Path:
    """Resolve a corpus entry's ``resources_dir`` to an absolute path.

    Relative paths are resolved against the repo root (the parent of the
    ``data/`` directory that holds the registry file).
    """
    raw = str(entry.get("resources_dir") or "")
    if not raw:
        raise ValueError(f"corpus entry {entry.get('corpus_id')!r} has no resources_dir")
    p = Path(raw)
    if p.is_absolute():
        return p
    base = (
        registry_path.parent.parent
        if registry_path is not None
        else _DEFAULT_REGISTRY_PATH.parent.parent
    )
    return (base / p).resolve()


# ── Validator verbs ──────────────────────────────────────────────────────────

def load_validator_verbs(
    resources_dir: Path | None = None,
) -> tuple[str, ...]:
    """Return domain-specific finite-verb vocabulary for claim sentence validation.

    If the resource file does not exist, returns an empty tuple — the caller
    is expected to fall back to a hardcoded default list.
    """
    path = _resource_path("validator_verbs", resources_dir)
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return tuple(str(v).strip().lower() for v in (data.get("verbs") or []) if v)
