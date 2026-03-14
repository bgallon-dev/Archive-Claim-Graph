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


def _dir(resources_dir: Path | None) -> Path:
    return resources_dir if resources_dir is not None else _DEFAULT_RESOURCES_DIR


# ── Seed entities ─────────────────────────────────────────────────────────────

def load_seed_entity_rows(resources_dir: Path | None = None) -> list[dict[str, str]]:
    """Return raw CSV rows from seed_entities.csv.

    Each row has keys: ``entity_type``, ``name``, ``prop_key``, ``prop_value``.
    The caller is responsible for grouping multi-property entities.
    """
    path = _dir(resources_dir) / "seed_entities.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


# ── Claim type patterns ───────────────────────────────────────────────────────

def load_claim_type_patterns(
    resources_dir: Path | None = None,
) -> list[tuple[str, re.Pattern[str], float]]:
    """Return ``[(claim_type, compiled_pattern, weight), ...]`` in file order."""
    path = _dir(resources_dir) / "claim_type_patterns.yaml"
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    result: list[tuple[str, re.Pattern[str], float]] = []
    for entry in data["patterns"]:
        compiled = re.compile(entry["regex"], re.IGNORECASE)
        result.append((str(entry["claim_type"]), compiled, float(entry["weight"])))
    return result


# ── Claim role policy ─────────────────────────────────────────────────────────

def load_claim_role_policy(
    resources_dir: Path | None = None,
) -> dict[tuple[str, str], str]:
    """Return ``{(claim_type, entity_type): relation_type}`` mapping."""
    path = _dir(resources_dir) / "claim_role_policy.yaml"
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
    path = _dir(resources_dir) / "measurement_units.yaml"
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
    path = _dir(resources_dir) / "measurement_species.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── OCR corrections ───────────────────────────────────────────────────────────

def load_ocr_corrections(
    resources_dir: Path | None = None,
) -> frozenset[str]:
    """Return the set of known OCR error tokens (lowercased)."""
    path = _dir(resources_dir) / "ocr_corrections.yaml"
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return frozenset(str(t) for t in data["known_errors"])


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
    path = _dir(resources_dir) / "claim_relation_compatibility.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
