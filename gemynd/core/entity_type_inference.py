"""Entity-type inference used during claim-link extraction.

The extractor assigns ``entity_type_hint`` before resolution, and that hint
then filters resolution candidates.  Guessing the type from a short keyword
list produces wrong hints for corporate/governmental surface forms that
never enter the seed vocabulary, and those wrong hints then block the
correct resolution match.

This module implements a cascade that consults the seed vocabulary first
(exact, then fuzzy) and only falls back to pattern rules for novel
surface forms.  Seed vocabulary always wins so "John Church" stays a
Person even though "CHURCH" is an Organization keyword.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gemynd.core.models import EntityRecord


_FUZZY_THRESHOLD: float = 0.88


def infer_entity_type(
    surface_form: str,
    *,
    seed_entities: list["EntityRecord"] | None = None,
    rules: list[dict[str, Any]] | None = None,
    default: str = "Person",
) -> str:
    """Infer an entity type label for ``surface_form``.

    Priority:
      1. Exact match against seed vocabulary (by ``normalized_form`` or
         lowercased ``name``) → seed entity's ``entity_type``.
      2. Fuzzy match against seed vocabulary (similarity ≥ 0.88) →
         best-matching seed entity's ``entity_type``.
      3. Configured rules cascade (first match wins).
      4. ``default`` (falls back to ``"Person"``).
    """
    surface = (surface_form or "").strip()
    if not surface:
        return default
    normalized = surface.lower()
    upper = surface.upper()

    if seed_entities:
        for entity in seed_entities:
            if entity.normalized_form == normalized or entity.name.lower() == normalized:
                return entity.entity_type

        from gemynd.core.resolver import similarity_score
        best_score = 0.0
        best_type: str | None = None
        for entity in seed_entities:
            score = similarity_score(normalized, entity.normalized_form)
            if score > best_score:
                best_score = score
                best_type = entity.entity_type
        if best_score >= _FUZZY_THRESHOLD and best_type:
            return best_type

    if rules:
        upper_tokens = set(upper.split())
        for rule in rules:
            mode = rule.get("match_mode", "")
            target = rule.get("target_type", default)
            if mode == "any_token_uppercase":
                keywords = rule.get("keywords") or frozenset()
                if upper_tokens & keywords:
                    return target
            elif mode == "regex_any":
                for pattern in rule.get("patterns") or ():
                    if pattern.search(upper):
                        return target
            elif mode == "fallback":
                return target

    return default
