"""Every resolver the pipeline builds must carry the domain config.

The resolver's precision gates — the opt-in type filter and the non-entity
vocabulary — are both read from ``DomainConfig``. A resolver constructed with
only ``seed_entities`` and ``resources_dir`` silently has neither.

That is not hypothetical. ``pipeline.py`` re-resolves with a frequency prior and
assigns the result over ``semantic.entity_resolutions``, so a config-less
resolver there discarded every gated resolution and recomputed it ungated — for
every document after the first. The gates tested green in isolation while having
no effect on the graph, which cost several checkpoint rounds to spot.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gemynd.core.domain_config import load_domain_config
from gemynd.core.resolver import DictionaryFuzzyResolver

_PIPELINE = Path(__file__).resolve().parents[1] / "gemynd" / "ingest" / "pipeline.py"
_NEWSPAPER_DIR = Path(__file__).resolve().parents[1] / "gemynd" / "resources_newspaper"


def _resolver_constructions(source: str) -> list[tuple[int, set[str]]]:
    tree = ast.parse(source)
    found: list[tuple[int, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "DictionaryFuzzyResolver":
            continue
        found.append((node.lineno, {kw.arg for kw in node.keywords if kw.arg}))
    return found


def test_pipeline_resolvers_all_receive_config() -> None:
    constructions = _resolver_constructions(_PIPELINE.read_text(encoding="utf-8"))
    assert constructions, "expected DictionaryFuzzyResolver to be built in pipeline.py"
    missing = [line for line, kwargs in constructions if "config" not in kwargs]
    assert not missing, (
        f"DictionaryFuzzyResolver built without config= at pipeline.py lines {missing}. "
        "Such a resolver has no type filter and no non-entity vocabulary."
    )


@pytest.fixture(scope="module")
def newspaper_config():
    return load_domain_config(_NEWSPAPER_DIR)


def test_a_configless_resolver_really_does_lose_the_gates(newspaper_config) -> None:
    """Pins down *why* the assertion above matters."""
    with_config = DictionaryFuzzyResolver(config=newspaper_config)
    without = DictionaryFuzzyResolver(seed_entities=list(newspaper_config.seed_entities))

    assert with_config._non_entity_terms, "config carries the non-entity vocabulary"
    assert not without._non_entity_terms
    assert with_config._type_filter is True
    assert without._type_filter is False
