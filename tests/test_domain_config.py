"""Tests for DomainConfig (Phase 4A) and _get_spec dual-mode registry."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from graphrag_pipeline.core.domain_config import (
    ClaimDerivationSpec,
    DomainConfig,
    _validate_config,
    load_domain_config,
)
from graphrag_pipeline.core.models import ClaimRecord
from graphrag_pipeline.ingest.derivation_context import _get_spec, build_derivation_contexts
from graphrag_pipeline.ingest.extractors.claim_extractor import (
    RuleBasedClaimExtractor,
    _LOADED_TYPE_PATTERNS,
)

_RESOURCES_DIR = Path(__file__).parent.parent / "graphrag_pipeline" / "resources"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_domain_config(
    *,
    derivation_registry: dict | None = None,
    preferred_entity_types: dict | None = None,
) -> DomainConfig:
    """Load real config and optionally override fields for isolation tests."""
    base = load_domain_config(_RESOURCES_DIR)
    return DomainConfig(
        seed_entities=base.seed_entities,
        claim_type_patterns=base.claim_type_patterns,
        claim_role_policy=base.claim_role_policy,
        measurement_units=base.measurement_units,
        measurement_species=base.measurement_species,
        ocr_corrections=base.ocr_corrections,
        ocr_correction_map=base.ocr_correction_map,
        negative_lexicon=base.negative_lexicon,
        preferred_entity_types=preferred_entity_types if preferred_entity_types is not None else base.preferred_entity_types,
        compatibility_matrix=base.compatibility_matrix,
        derivation_registry=derivation_registry if derivation_registry is not None else base.derivation_registry,
        domain_anchor=base.domain_anchor,
        year_validation=base.year_validation,
        synthesis_context=base.synthesis_context,
    )


# ---------------------------------------------------------------------------
# 4A-1: load_domain_config returns a populated DomainConfig
# ---------------------------------------------------------------------------

def test_load_domain_config_returns_populated_config() -> None:
    config = load_domain_config(_RESOURCES_DIR)
    assert len(config.seed_entities) > 0, "seed_entities must not be empty"
    assert len(config.claim_type_patterns) > 0, "claim_type_patterns must not be empty"
    assert len(config.derivation_registry) > 0, "derivation_registry must not be empty"


# ---------------------------------------------------------------------------
# 4A-2: derivation_registry values are ClaimDerivationSpec with tuple fields
# ---------------------------------------------------------------------------

def test_derivation_registry_values_are_claim_derivation_spec() -> None:
    config = load_domain_config(_RESOURCES_DIR)
    assert config.derivation_registry, "derivation_registry must not be empty"
    for ct, spec in config.derivation_registry.items():
        assert isinstance(spec, ClaimDerivationSpec), (
            f"Registry entry {ct!r} is {type(spec)}, expected ClaimDerivationSpec"
        )
        assert isinstance(spec.required_entities, tuple), (
            f"{ct}.required_entities must be tuple, got {type(spec.required_entities)}"
        )
        assert isinstance(spec.optional_entities, tuple), (
            f"{ct}.optional_entities must be tuple, got {type(spec.optional_entities)}"
        )


# ---------------------------------------------------------------------------
# 4A-3: _validate_config warns when a registry entry has no matching pattern
# ---------------------------------------------------------------------------

def test_validate_config_warns_on_registry_entry_without_pattern(caplog: pytest.LogCaptureFixture) -> None:
    bad_registry = _minimal_domain_config().derivation_registry.copy()
    bad_registry["nonexistent_claim_type_xyz"] = ClaimDerivationSpec("obs", "evt", (), ())
    bad_config = _minimal_domain_config(derivation_registry=bad_registry)
    with caplog.at_level(logging.WARNING, logger="graphrag_pipeline.core.domain_config"):
        _validate_config(bad_config)
    assert "nonexistent_claim_type_xyz" in caplog.text


# ---------------------------------------------------------------------------
# 4A-4: _validate_config warns when preferred_entity_types references unknown type
# ---------------------------------------------------------------------------

def test_validate_config_warns_on_unknown_preferred_entity_type(caplog: pytest.LogCaptureFixture) -> None:
    bad_preferred = {"population_estimate": ["UnknownEntityTypeXYZ"]}
    bad_config = _minimal_domain_config(preferred_entity_types=bad_preferred)
    with caplog.at_level(logging.WARNING, logger="graphrag_pipeline.core.domain_config"):
        _validate_config(bad_config)
    assert "UnknownEntityTypeXYZ" in caplog.text


# ---------------------------------------------------------------------------
# 4A-5: Extractor with config uses config's patterns
# ---------------------------------------------------------------------------

def test_extractor_with_config_uses_config_patterns() -> None:
    """When created with a stub config, the extractor's instance patterns come from config."""
    real_config = load_domain_config(_RESOURCES_DIR)
    sentinel_pat = re.compile(r"\bsentinel_token_xyz\b", re.IGNORECASE)
    stub_patterns = [(real_config.claim_type_patterns[0][0], sentinel_pat, 2.0)]
    stub_config = DomainConfig(
        seed_entities=real_config.seed_entities,
        claim_type_patterns=stub_patterns,
        claim_role_policy=real_config.claim_role_policy,
        measurement_units=real_config.measurement_units,
        measurement_species=real_config.measurement_species,
        ocr_corrections=real_config.ocr_corrections,
        ocr_correction_map=real_config.ocr_correction_map,
        negative_lexicon=real_config.negative_lexicon,
        preferred_entity_types=real_config.preferred_entity_types,
        compatibility_matrix=real_config.compatibility_matrix,
        derivation_registry=real_config.derivation_registry,
        domain_anchor=real_config.domain_anchor,
        year_validation=real_config.year_validation,
        synthesis_context=real_config.synthesis_context,
    )
    extractor = RuleBasedClaimExtractor(config=stub_config)
    assert extractor._type_scored_patterns is stub_patterns


# ---------------------------------------------------------------------------
# 4A-6: Extractor without config uses module-level defaults
# ---------------------------------------------------------------------------

def test_extractor_without_config_uses_module_defaults() -> None:
    extractor = RuleBasedClaimExtractor()
    assert extractor._type_scored_patterns is _LOADED_TYPE_PATTERNS


# ---------------------------------------------------------------------------
# 4A-7: build_derivation_contexts accepts ClaimDerivationSpec registry values
# ---------------------------------------------------------------------------

def test_build_derivation_contexts_accepts_claim_derivation_spec_registry() -> None:
    registry = {
        "population_estimate": ClaimDerivationSpec(
            observation_type="population_count",
            event_type="SurveyEvent",
            required_entities=("Species",),
            optional_entities=("Place", "Refuge"),
        ),
    }
    claim = ClaimRecord(
        claim_id="c1",
        run_id="r1",
        paragraph_id="p1",
        claim_type="population_estimate",
        source_sentence="50 mallards observed",
        normalized_sentence="50 mallards observed",
        certainty="certain",
        extraction_confidence=0.9,
    )
    contexts = build_derivation_contexts(
        claims=[claim],
        measurements=[],
        claim_entity_links=[],
        claim_location_links=[],
        claim_period_links=[],
        entity_lookup={},
        run_id="r1",
        report_year=1956,
        registry=registry,
    )
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx.observation_type == "population_count"
    assert ctx.event_type == "SurveyEvent"


# ---------------------------------------------------------------------------
# 4A-8: load_domain_config round-trips without error on real resources
# ---------------------------------------------------------------------------

def test_load_domain_config_round_trips_on_real_resources() -> None:
    config = load_domain_config(_RESOURCES_DIR)
    assert isinstance(config.ocr_corrections, frozenset)
    assert isinstance(config.negative_lexicon, frozenset)
    assert config.domain_anchor is not None


# ---------------------------------------------------------------------------
# _get_spec dual-mode tests
# ---------------------------------------------------------------------------

def test_get_spec_with_claim_derivation_spec() -> None:
    spec = ClaimDerivationSpec("obs", "evt", ("Species",), ("Place",))
    obs, evt, req, opt = _get_spec({"ct": spec}, "ct")
    assert obs == "obs"
    assert evt == "evt"
    assert req == ("Species",)
    assert opt == ("Place",)


def test_get_spec_with_raw_dict() -> None:
    registry = {
        "ct": {
            "observation_type": "obs",
            "event_type": "evt",
            "required_entities": ["Species"],
            "optional_entities": ["Place"],
        }
    }
    obs, evt, req, opt = _get_spec(registry, "ct")
    assert obs == "obs"
    assert evt == "evt"
    assert req == ("Species",)
    assert opt == ("Place",)


def test_get_spec_returns_nones_for_missing_entry() -> None:
    obs, evt, req, opt = _get_spec({"other": {}}, "ct")
    assert obs is None and evt is None
    assert req == () and opt == ()


def test_get_spec_returns_nones_for_none_registry() -> None:
    obs, evt, req, opt = _get_spec(None, "ct")
    assert obs is None and evt is None
    assert req == () and opt == ()
