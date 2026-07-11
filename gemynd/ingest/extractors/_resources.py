"""Shared default-resource loaders for the extractor family.

Each entry below wraps a ``load_*`` helper from
:mod:`gemynd.shared.resource_loader` in :func:`functools.cache` so the
first call materializes the resource and every subsequent call returns the
cached instance. Replaces the per-module ``_default_*`` helpers that
previously triplicated this pattern across the three extractor modules.

**Caching is process-global.** Tests that need custom resources must pass
``resources_dir=`` or ``config=`` to the extractor constructor — the
default-resource path is shared across the whole process and is not
parameterized by ``resources_dir``. (This is the same behavior that the
prior per-module ``@functools.cache`` decorators had; consolidation makes
the shared cache more visible without changing its semantics.)
"""
from __future__ import annotations

import functools

from gemynd.core.resolver import default_seed_entities
from gemynd.shared.resource_loader import (
    load_claim_role_policy,
    load_claim_type_patterns,
    load_measurement_species,
    load_measurement_units,
    load_negative_entities,
    load_ocr_corrections,
)

default_claim_type_patterns = functools.cache(load_claim_type_patterns)
default_claim_role_policy = functools.cache(load_claim_role_policy)
default_ocr_corrections = functools.cache(load_ocr_corrections)
default_negative_entities = functools.cache(load_negative_entities)
default_measurement_units = functools.cache(load_measurement_units)
default_measurement_species = functools.cache(load_measurement_species)


@functools.cache
def seed_terms_for(*entity_types: str) -> tuple[str, ...]:
    """Cached, length-descending tuple of default seed-entity names for the given types.

    Replaces the five per-type ``_default_*_terms`` helpers
    (``_default_species_terms``, ``_default_habitat_terms``,
    ``_default_method_terms``, ``_default_location_terms``,
    ``_default_activity_terms``) that previously lived in
    ``claim_extractor.py``. Tuple return is hash-stable so the cache key
    (``entity_types``) plays well with ``functools.cache``.
    """
    labels = set(entity_types)
    return tuple(sorted(
        {entity.name for entity in default_seed_entities() if entity.entity_type in labels},
        key=len,
        reverse=True,
    ))


@functools.cache
def default_lexicon_hints() -> tuple[dict[str, list[str]], list[str]]:
    """Return ``(hints_dict, sorted_terms)`` from the default seed entities.

    ``hints_dict`` maps each entity name (lowercased) to its list of
    ``entity_type`` strings; ``sorted_terms`` is the same set of names
    sorted by length descending (so longest-match-first regexes work).
    """
    hints: dict[str, list[str]] = {}
    for entity in default_seed_entities():
        hints.setdefault(entity.name.lower(), []).append(entity.entity_type)
    return hints, sorted(hints, key=len, reverse=True)
