"""Adversarial tests for domain-agnosticism of the synthesis system prompt.

The RETRIEVED_VIA semantics paragraph previously hardcoded wildlife/civic
relations (SPECIES_FOCUS / MANAGEMENT_TARGET, a Spokane/Colorado/Idaho example)
into the LLM system prompt. It is now generated from the domain's
``relation_semantics`` block, with the original text as the default.
"""
from __future__ import annotations

from pathlib import Path

from gemynd.retrieval.synthesis import (
    _DEFAULT_RETRIEVED_VIA,
    _DEFAULT_SYNTHESIS_CONTEXT,
    _build_system_prompt,
    build_retrieved_via_block,
)
from gemynd.core.domain_config import load_domain_config
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR


class TestRetrievedViaParity:
    def test_none_and_empty_return_default(self):
        assert build_retrieved_via_block(None) == _DEFAULT_RETRIEVED_VIA
        assert build_retrieved_via_block({}) == _DEFAULT_RETRIEVED_VIA

    def test_default_prompt_preserves_wildlife_block(self):
        prompt = _build_system_prompt(_DEFAULT_SYNTHESIS_CONTEXT)
        # the original hardcoded semantics survive verbatim as the default
        assert "SPECIES_FOCUS / MANAGEMENT_TARGET" in prompt
        assert "SUBJECT_OF_CLAIM / TOPIC_OF_CLAIM" in prompt
        assert _DEFAULT_RETRIEVED_VIA in prompt

    def test_no_placeholder_leaks_into_prompt(self):
        prompt = _build_system_prompt(_DEFAULT_SYNTHESIS_CONTEXT)
        assert "{retrieved_via}" not in prompt
        assert "{synthesis_context}" not in prompt

    def test_shipped_domains_declare_none(self):
        for rd in (TURNBULL_RESOURCES_DIR, Path("gemynd/resources_newspaper")):
            assert load_domain_config(rd).relation_semantics == {}


class TestRetrievedViaAgnostic:
    _CUSTOM = {
        "MENTIONS_PERSON": "the entity is a person named in the article",
        "DATELINE_AT": "the entity is the dateline location, not the subject",
    }

    def test_custom_block_lists_domain_relations(self):
        block = build_retrieved_via_block(self._CUSTOM)
        assert block.startswith("RETRIEVED_VIA semantics:")
        assert "MENTIONS_PERSON: the entity is a person named in the article" in block
        assert "DATELINE_AT:" in block

    def test_custom_block_has_no_wildlife_leak(self):
        block = build_retrieved_via_block(self._CUSTOM)
        for leaked in ("SPECIES_FOCUS", "MANAGEMENT_TARGET", "Spokane", "Colorado", "Idaho"):
            assert leaked not in block

    def test_custom_prompt_replaces_default_block(self):
        prompt = _build_system_prompt("newspaper archives", self._CUSTOM)
        assert "MENTIONS_PERSON" in prompt
        assert "SPECIES_FOCUS / MANAGEMENT_TARGET" not in prompt
        assert _DEFAULT_RETRIEVED_VIA not in prompt

    def test_ordering_follows_declaration(self):
        block = build_retrieved_via_block(self._CUSTOM)
        assert block.index("MENTIONS_PERSON") < block.index("DATELINE_AT")


class TestSynthesisEngineSeam:
    def test_engine_prompt_uses_relation_semantics(self):
        import pytest
        try:
            from gemynd.retrieval.synthesis import SynthesisEngine
            engine = SynthesisEngine(api_key="test-key", relation_semantics={"MENTIONS_PERSON": "named person"})
        except Exception as exc:  # optional LLM deps missing in some envs
            pytest.skip(f"SynthesisEngine unconstructable here: {exc}")
        assert "MENTIONS_PERSON: named person" in engine._system_prompt
        assert "SPECIES_FOCUS / MANAGEMENT_TARGET" not in engine._system_prompt
