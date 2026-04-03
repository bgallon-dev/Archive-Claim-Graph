"""Unit tests for Layer 1: entity resolution gateway."""
from __future__ import annotations

import pytest

from gemynd.core.models import EntityRecord, EntityResolutionRecord, MentionRecord
from gemynd.core.resolver import DictionaryFuzzyResolver, ResolutionPolicy
from gemynd.retrieval.entity_gateway import EntityResolutionGateway
from gemynd.retrieval.models import EntityContext


def _make_entity(entity_id: str, entity_type: str, name: str, normalized_form: str) -> EntityRecord:
    return EntityRecord(
        entity_id=entity_id,
        entity_type=entity_type,
        name=name,
        normalized_form=normalized_form,
        properties={},
    )


@pytest.fixture()
def gateway():
    """Gateway with a minimal controlled seed vocabulary."""
    seed = [
        _make_entity("sp-mallard", "Species", "Mallard", "mallard"),
        _make_entity("sp-canvasback", "Species", "Canvasback", "canvasback"),
        _make_entity("h-pothole", "Habitat", "Pothole Wetland", "pothole wetland"),
    ]
    resolver = DictionaryFuzzyResolver(seed_entities=seed, policy=ResolutionPolicy())
    return EntityResolutionGateway(resolver=resolver)


class TestResolve:
    def test_refers_to_match(self, gateway):
        ctx = gateway.resolve(["mallard"])
        assert len(ctx.resolved) == 1
        assert ctx.resolved[0].entity_id == "sp-mallard"
        assert ctx.resolved[0].resolution_relation == "REFERS_TO"
        assert ctx.ambiguous == []
        assert ctx.unresolved == []

    def test_unresolved_form(self, gateway):
        ctx = gateway.resolve(["xyzunknownbird"])
        assert ctx.resolved == []
        assert ctx.unresolved == ["xyzunknownbird"]

    def test_empty_input(self, gateway):
        ctx = gateway.resolve([])
        assert ctx.resolved == []
        assert ctx.ambiguous == []
        assert ctx.unresolved == []

    def test_entity_hints_merged(self, gateway):
        ctx = gateway.resolve([], entity_hints=["canvasback"])
        assert any(r.entity_id == "sp-canvasback" for r in ctx.resolved)

    def test_no_duplicate_hints(self, gateway):
        ctx = gateway.resolve(["mallard"], entity_hints=["mallard"])
        # mallard should appear exactly once in resolved
        resolved_ids = [r.entity_id for r in ctx.resolved]
        assert resolved_ids.count("sp-mallard") == 1

    def test_multi_word_entity(self, gateway):
        ctx = gateway.resolve(["pothole wetland"])
        assert any(r.entity_id == "h-pothole" for r in ctx.resolved)

    def test_resolution_confidence_populated(self, gateway):
        ctx = gateway.resolve(["mallard"])
        assert ctx.resolved[0].resolution_confidence > 0.0

    def test_entity_type_populated(self, gateway):
        ctx = gateway.resolve(["mallard"])
        assert ctx.resolved[0].entity_type == "Species"
