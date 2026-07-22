"""Opt-in type filter on mention→entity resolution.

Nothing previously stopped a mention typed Person resolving to a Place: at
checkpoint 1 "F. J. Cheney" resolved to the town of Cheney, and publication
names ("Condon Globe", "London Punch", "Baker City Democrat") resolved to the
towns they are named after.

The filter is opt-in because it is only safe for domains whose type rules
decline to guess. Turnbull's rules end in a `fallback` that types everything
Person, so enabling it there would block every Refuge/Species resolution.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gemynd.core.domain_config import load_domain_config
from gemynd.core.models import MentionRecord
from gemynd.core.resolver import DictionaryFuzzyResolver
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR

_NEWSPAPER_DIR = (
    Path(__file__).resolve().parents[1] / "gemynd" / "resources_newspaper"
)


def _mention(surface: str) -> MentionRecord:
    return MentionRecord(
        mention_id="m1",
        run_id="run",
        paragraph_id="p1",
        surface_form=surface,
        normalized_form=surface.lower(),
        start_offset=0,
        end_offset=len(surface),
        detection_confidence=0.9,
        ocr_suspect=False,
    )


@pytest.fixture(scope="module")
def newspaper_config():
    return load_domain_config(_NEWSPAPER_DIR)


def test_newspaper_domain_enables_the_filter(newspaper_config) -> None:
    assert newspaper_config.resolution_type_filter is True


def test_turnbull_leaves_the_filter_off() -> None:
    """Turnbull's rules guess, so the filter must not apply to it."""
    assert load_domain_config(TURNBULL_RESOURCES_DIR).resolution_type_filter is False


def _resolved_types(config, surface: str) -> set[str]:
    resolver = DictionaryFuzzyResolver(config=config)
    entities, resolutions = resolver.resolve([_mention(surface)])
    by_id = {e.entity_id: e for e in entities}
    return {by_id[r.entity_id].entity_type for r in resolutions if r.entity_id in by_id}


def test_person_mention_does_not_resolve_to_a_place(newspaper_config) -> None:
    """"F. J. Cheney" is a person; Cheney is a town in the seed vocabulary."""
    assert "Place" not in _resolved_types(newspaper_config, "F. J. Cheney")


def test_publication_does_not_resolve_to_the_town_it_is_named_for(newspaper_config) -> None:
    for title in ("Spokane Chronicle", "Condon Globe"):
        assert "Place" not in _resolved_types(newspaper_config, title), title


def test_non_entity_words_do_not_resolve(newspaper_config) -> None:
    """The discovery rules say these are not names; the resolver must agree.

    Each of these attached to a look-alike entity at checkpoint 1:
    american->Americans (124 edges), monday->Mondays (80), october->Ootober (59),
    price->Prince (48), good->Guy Good (58), english->Fred English (51).
    """
    assert newspaper_config.non_entity_terms, "discovery stopwords should load"
    for word in ("american", "monday", "october", "price", "good", "english", "first"):
        assert not _resolved_types(newspaper_config, word), word


def test_untyped_mentions_are_not_filtered(newspaper_config) -> None:
    """The newspaper rules have no fallback, so most forms stay untyped.

    An untyped mention must keep resolving — otherwise the filter would silently
    delete the bulk of the graph's edges.
    """
    assert _resolved_types(newspaper_config, "Portland")
