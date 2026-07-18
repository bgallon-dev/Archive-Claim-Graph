"""Adversarial tests for domain-agnosticism of the query-intent classifier.

The classifier previously hardcoded three surface-form maps (BLM/GLO/Spokane
doc-type + series aliases, and Tier-1 entity-type keywords incl. "refuge").
These are now overridable from a domain's ``query_intent.yaml`` while the module
maps remain the default. Retrieval has no golden-parity guard, so these tests
are the no-Turnbull-leak assertion for this layer.
"""
from __future__ import annotations

import textwrap

from gemynd.retrieval import classifier as C
from gemynd.shared.resource_loader import load_query_intent_maps


# --------------------------------------------------------------------------- #
# PARITY — module defaults unchanged when no domain map is supplied
# --------------------------------------------------------------------------- #
class TestClassifierParity:
    def test_default_doc_type_alias(self):
        intent = C.classify_query("show me land patents from 1909")
        assert intent.doc_type_hints == ["land_patent"]

    def test_default_series_alias(self):
        intent = C.classify_query("coverage in the spokane press")
        assert intent.series_hints == ["The Spokane Press"]

    def test_default_tier1_keyword(self):
        got = C._detect_frequency_ranking("top 10 refuges")
        assert got is not None and got.target_label == "Refuge"

    def test_none_is_the_module_map(self):
        # Passing None must be identical to omitting the arg.
        assert C._extract_doc_type_hints("land grant", None) == C._extract_doc_type_hints("land grant")


# --------------------------------------------------------------------------- #
# AGNOSTIC — a domain's own map replaces the module grab-bag (no leak)
# --------------------------------------------------------------------------- #
class TestClassifierAgnostic:
    def test_domain_map_does_not_inherit_grabbag(self):
        # A domain whose doc_type map lacks "land patent" must NOT recognise it.
        intent = C.classify_query("show me land patents", doc_type_aliases={"gazette page": "gazette"})
        assert intent.doc_type_hints == []

    def test_domain_map_recognises_own_vocab(self):
        intent = C.classify_query("read the gazette page", doc_type_aliases={"gazette page": "gazette"})
        assert intent.doc_type_hints == ["gazette"]

    def test_domain_series_map_replaces_default(self):
        intent = C.classify_query("the spokane press", series_aliases={"weekly gazette": "Weekly Gazette"})
        assert intent.series_hints == []  # Spokane Press not in this domain's map

    def test_tier1_newspaper_map_excludes_refuge(self):
        news_kw = {"person": "Person", "people": "Person", "place": "Place", "places": "Place"}
        # newspaper corpus has no Refuge → "top 10 refuges" must fall through
        assert C._detect_frequency_ranking("top 10 refuges", news_kw) is None
        # ...but its own entity types still route
        got = C._detect_frequency_ranking("top 5 people", news_kw)
        assert got is not None and got.target_label == "Person"

    def test_empty_map_yields_no_hints(self):
        # An explicit empty map is honoured (no fallback to module default).
        assert C._extract_doc_type_hints("land patents and deeds", {}) == []
        assert C._detect_frequency_ranking("top 10 species", {}) is None


# --------------------------------------------------------------------------- #
# HOSTILE / EDGE
# --------------------------------------------------------------------------- #
class TestClassifierHostile:
    def test_unknown_tier1_word_falls_through(self):
        assert C._detect_frequency_ranking("top 10 sandwiches", {"person": "Person"}) is None

    def test_case_insensitive_domain_alias(self):
        # classifier lowercases query text; domain keys are lowercased at load.
        intent = C.classify_query("The Gazette Page", doc_type_aliases={"gazette page": "gazette"})
        assert intent.doc_type_hints == ["gazette"]


# --------------------------------------------------------------------------- #
# CONFIG LOADING — query_intent.yaml alias blocks parse (lowercased keys)
# --------------------------------------------------------------------------- #
    # Resource files resolve through domain_profile.yaml's `resources:` remap
    # (that's how key "query_intent" becomes "query_intent.yaml"), so a fixture
    # dir needs a minimal profile — same requirement as every other loader.
    _PROFILE = 'version: "1"\nresources:\n  query_intent: query_intent.yaml\n'

    def test_loads_and_lowercases(self, tmp_path):
        (tmp_path / "domain_profile.yaml").write_text(self._PROFILE, encoding="utf-8")
        (tmp_path / "query_intent.yaml").write_text(textwrap.dedent("""
            version: "1"
            doc_type_aliases:
              "Gazette Page": gazette_page
              "news clipping": gazette_page
            series_aliases:
              "The Weekly Gazette": Weekly Gazette
            tier1_type_keywords:
              person: Person
              PLACE: Place
        """), encoding="utf-8")
        maps = load_query_intent_maps(tmp_path)
        assert maps["doc_type_aliases"] == {"gazette page": "gazette_page", "news clipping": "gazette_page"}
        assert maps["series_aliases"] == {"the weekly gazette": "Weekly Gazette"}
        assert maps["tier1_type_keywords"] == {"person": "Person", "place": "Place"}

    def test_missing_blocks_yield_empty(self, tmp_path):
        (tmp_path / "domain_profile.yaml").write_text(self._PROFILE, encoding="utf-8")
        (tmp_path / "query_intent.yaml").write_text('version: "1"\n', encoding="utf-8")
        maps = load_query_intent_maps(tmp_path)
        assert maps == {"doc_type_aliases": {}, "series_aliases": {}, "tier1_type_keywords": {}}

    def test_real_domains_declare_no_aliases_yet(self):
        # Both shipped domains currently rely on the module defaults (empty
        # blocks) — this pins the zero-regression state until the grab-bag is
        # migrated per-domain (follow-up).
        from pathlib import Path
        from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR
        for rd in (TURNBULL_RESOURCES_DIR, Path("gemynd/resources_newspaper")):
            maps = load_query_intent_maps(rd)
            assert maps["doc_type_aliases"] == {}
            assert maps["series_aliases"] == {}
            assert maps["tier1_type_keywords"] == {}
