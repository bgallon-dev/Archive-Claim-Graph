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

# The bundled ecological "Turnbull" corpus. Exposed as a named constant so
# callers that genuinely want this corpus opt in *visibly* — there is no
# silent default. ``load_domain_config(None)`` / ``default_seed_entities(None)``
# raise; anything wanting Turnbull passes this explicitly.
TURNBULL_RESOURCES_DIR: Path = Path(__file__).parent.parent / "resources"


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
    return resources_dir if resources_dir is not None else TURNBULL_RESOURCES_DIR


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


# ── Measurement scan patterns / naming rules ──────────────────────────────────

def load_measurement_patterns(
    resources_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the measurement scan patterns and naming rules for a domain.

    Keys in the returned dict:
    - ``pattern_specs``: ``[(compiled (number)(unit) pattern, name, unit), ...]``
      in file order (drives the direct-scan ``_pattern_specs``).
    - ``unit_right_re``: compiled ``^\\s*(<alt>)\\b`` over the ``range_right``
      fragments, or ``None`` (drives ``_unit_right_re``).
    - ``count_claim_types``: ``frozenset[str]`` of count-eligible claim types.
    - ``acre_specialization``: dict with ``base_name``, ``by_claim_type``,
      ``by_sentence_substring``.
    - ``money``: dict with ``by_claim_type`` and ``default``.

    A domain that ships no ``measurement_patterns.yaml`` gets empty patterns and
    identity naming — never Turnbull's measurement vocabulary.
    """
    empty: dict[str, Any] = {
        "pattern_specs": [],
        "unit_right_re": None,
        "count_claim_types": frozenset(),
        "acre_specialization": {},
        "money": {},
    }
    path = _resource_path("measurement_patterns", resources_dir)
    if not path.exists():
        return empty
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    pattern_specs: list[tuple[re.Pattern[str], str, str | None]] = []
    right_alts: list[str] = []
    for entry in data.get("patterns") or []:
        match_frag = str(entry["match"])
        compiled = _safe_compile(rf"(\d+(?:\.\d+)?)\s*({match_frag})\b", re.IGNORECASE)
        unit = entry.get("unit")
        pattern_specs.append((compiled, str(entry["name"]), str(unit) if unit is not None else None))
        rr = entry.get("range_right")
        if rr:
            right_alts.append(str(rr))

    unit_right_re = (
        _safe_compile(r"^\s*(" + "|".join(right_alts) + r")\b", re.IGNORECASE)
        if right_alts
        else None
    )
    acre = data.get("acre_specialization") or {}
    money = data.get("money") or {}
    return {
        "pattern_specs": pattern_specs,
        "unit_right_re": unit_right_re,
        "count_claim_types": frozenset(str(c) for c in (data.get("count_claim_types") or [])),
        "acre_specialization": {
            "base_name": str(acre.get("base_name", "acres")),
            "by_claim_type": {str(k): str(v) for k, v in (acre.get("by_claim_type") or {}).items()},
            "by_sentence_substring": {
                str(k).lower(): str(v) for k, v in (acre.get("by_sentence_substring") or {}).items()
            },
        },
        "money": {
            "by_claim_type": {str(k): str(v) for k, v in (money.get("by_claim_type") or {}).items()},
            "default": (str(money["default"]) if money.get("default") is not None else None),
        },
    }


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


def load_query_intent_maps(
    resources_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Return the classifier's domain-specific surface-form maps from
    ``query_intent.yaml``: ``doc_type_aliases`` (phrase → doc_type),
    ``series_aliases`` (phrase → series), and ``tier1_type_keywords``
    (word → entity label). All keys are lowercased to match the classifier's
    lowercased query text. Missing blocks yield empty dicts, so a domain that
    declares none contributes nothing (the classifier then falls back to its
    module-level defaults).
    """
    empty = {"doc_type_aliases": {}, "series_aliases": {}, "tier1_type_keywords": {}}
    path = _resource_path("query_intent", resources_dir)
    if not path.exists():
        return empty
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    def _str_map(key: str) -> dict[str, str]:
        raw = data.get(key) or {}
        return {str(k).lower(): str(v) for k, v in raw.items()}

    return {
        "doc_type_aliases": _str_map("doc_type_aliases"),
        "series_aliases": _str_map("series_aliases"),
        "tier1_type_keywords": _str_map("tier1_type_keywords"),
    }


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


# ── Entity type inference rules ──────────────────────────────────────────────

def load_entity_type_rules(
    resources_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return ordered rules cascade for entity-type inference.

    Each rule dict has ``name``, ``target_type``, ``match_mode``, and
    (depending on mode) ``keywords`` or ``patterns``.  Regex patterns are
    pre-compiled so the extractor does not recompile on every sentence.
    Returns ``[]`` if the file is absent — callers may then fall back to
    hardcoded defaults.
    """
    path = _resource_path("entity_type_rules", resources_dir)
    if not path.exists():
        path = _dir(resources_dir) / "entity_type_rules.yaml"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    result: list[dict[str, Any]] = []
    for entry in data.get("rules") or []:
        mode = str(entry.get("match_mode", "")).strip()
        rule: dict[str, Any] = {
            "name": str(entry.get("name", "")),
            "target_type": str(entry.get("target_type", "Person")),
            "match_mode": mode,
        }
        if mode == "any_token_uppercase":
            rule["keywords"] = frozenset(
                str(k).strip().upper() for k in (entry.get("keywords") or []) if k
            )
        elif mode == "regex_any":
            rule["patterns"] = tuple(
                _safe_compile(str(p)) for p in (entry.get("patterns") or []) if p
            )
        result.append(rule)
    return result
