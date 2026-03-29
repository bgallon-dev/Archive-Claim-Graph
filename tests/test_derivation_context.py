"""Tests for derivation_context.py — 3A, 3B, and 3C."""
from __future__ import annotations

import importlib

import pytest

from graphrag_pipeline.core.models import (
    ClaimEntityLinkRecord,
    ClaimLocationLinkRecord,
    ClaimPeriodLinkRecord,
    ClaimRecord,
    EntityRecord,
    MeasurementRecord,
)
from graphrag_pipeline.ingest.derivation_context import (
    DerivationContext,
    _check_year_plausibility,
    _extract_year,
    build_derivation_contexts,
)
from graphrag_pipeline.ingest.observation_builder import build_observations
from graphrag_pipeline.ingest.event_builder import build_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim(
    claim_id: str,
    claim_type: str,
    paragraph_id: str = "para_1",
    claim_date: str | None = "1956-04-15",
    certainty: str = "certain",
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        run_id="run_1",
        paragraph_id=paragraph_id,
        claim_type=claim_type,
        source_sentence="test sentence",
        normalized_sentence="test sentence",
        certainty=certainty,
        extraction_confidence=0.8,
        claim_date=claim_date,
    )


def _entity(entity_id: str, entity_type: str, name: str = "test") -> EntityRecord:
    return EntityRecord(
        entity_id=entity_id,
        entity_type=entity_type,
        name=name,
        normalized_form=name.lower(),
    )


def _measurement(measurement_id: str, claim_id: str) -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=measurement_id,
        claim_id=claim_id,
        run_id="run_1",
        name="individual_count",
        raw_value="100",
        numeric_value=100.0,
        unit="individuals",
    )


def _build(
    claims,
    measurements=None,
    entity_links=None,
    location_links=None,
    period_links=None,
    entity_lookup=None,
    report_year: int | None = 1956,
    **kwargs,
) -> list[DerivationContext]:
    return build_derivation_contexts(
        claims=claims,
        measurements=measurements or [],
        claim_entity_links=entity_links or [],
        claim_location_links=location_links or [],
        claim_period_links=period_links or [],
        entity_lookup=entity_lookup or {},
        run_id="run_1",
        report_year=report_year,
        **kwargs,
    )


# ===========================================================================
# 3A — DerivationContext construction
# ===========================================================================

class TestDerivationContextBasics:
    def test_population_estimate_routing(self):
        """3A-1: population_estimate claim → correct types and measurement_owner."""
        species = _entity("sp1", "Species", "mallard")
        ctx_list = _build(
            claims=[_claim("c1", "population_estimate")],
            entity_links=[ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="SPECIES_FOCUS")],
            entity_lookup={"sp1": species},
        )
        assert len(ctx_list) == 1
        ctx = ctx_list[0]
        assert ctx.observation_type == "population_count"
        assert ctx.event_type == "SurveyEvent"
        assert ctx.species_id == "sp1"
        assert ctx.measurement_owner == "observation"

    def test_fire_incident_routing(self):
        """3A-2: fire_incident → observation_type is None, event_type=FireEvent, measurement_owner=event."""
        ctx_list = _build(claims=[_claim("c1", "fire_incident")])
        ctx = ctx_list[0]
        assert ctx.observation_type is None
        assert ctx.event_type == "FireEvent"
        assert ctx.measurement_owner == "event"

    def test_unclassified_assertion_routing(self):
        """3A-3: unclassified_assertion → both None, measurement_owner=none."""
        ctx_list = _build(claims=[_claim("c1", "unclassified_assertion")])
        ctx = ctx_list[0]
        assert ctx.observation_type is None
        assert ctx.event_type is None
        assert ctx.measurement_owner == "none"

    def test_entity_ids_populated(self):
        """Species/Habitat/SurveyMethod/Refuge/Place resolved into context."""
        species = _entity("sp1", "Species", "mallard")
        habitat = _entity("h1", "Habitat", "marsh")
        method = _entity("sm1", "SurveyMethod", "point count")
        refuge = _entity("r1", "Refuge", "Turnbull")
        place = _entity("p1", "Place", "Pine Creek")
        entity_lookup = {e.entity_id: e for e in [species, habitat, method, refuge, place]}

        ctx_list = _build(
            claims=[_claim("c1", "population_estimate")],
            entity_links=[
                ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="SPECIES_FOCUS"),
                ClaimEntityLinkRecord(claim_id="c1", entity_id="h1", relation_type="HABITAT_FOCUS"),
                ClaimEntityLinkRecord(claim_id="c1", entity_id="sm1", relation_type="METHOD_FOCUS"),
            ],
            location_links=[
                ClaimLocationLinkRecord(claim_id="c1", entity_id="r1"),
                ClaimLocationLinkRecord(claim_id="c1", entity_id="p1"),
            ],
            entity_lookup=entity_lookup,
        )
        ctx = ctx_list[0]
        assert ctx.species_id == "sp1"
        assert ctx.habitat_id == "h1"
        assert ctx.survey_method_id == "sm1"
        assert ctx.refuge_id == "r1"
        assert ctx.place_id == "p1"

    def test_management_target_populates_species(self):
        """MANAGEMENT_TARGET relation feeds species_id."""
        species = _entity("sp1", "Species", "coyote")
        ctx_list = _build(
            claims=[_claim("c1", "predator_control")],
            entity_links=[ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="MANAGEMENT_TARGET")],
            entity_lookup={"sp1": species},
        )
        assert ctx_list[0].species_id == "sp1"

    def test_measurement_ids_collected(self):
        """measurement_ids on context contains all measurements for the claim."""
        m1 = _measurement("m1", "c1")
        m2 = _measurement("m2", "c1")
        ctx_list = _build(
            claims=[_claim("c1", "population_estimate")],
            measurements=[m1, m2],
        )
        assert sorted(ctx_list[0].measurement_ids) == ["m1", "m2"]

    def test_observation_id_starts_as_none(self):
        """3A-5: observation_id is None before build_observations() runs."""
        ctx_list = _build(claims=[_claim("c1", "population_estimate")])
        assert ctx_list[0].observation_id is None

    def test_observation_id_set_after_build_observations(self):
        """3A-5: observation_id is populated after build_observations() with _contexts."""
        ctx_list = _build(claims=[_claim("c1", "population_estimate")])
        build_observations(
            claims=[ctx_list[0].claim],
            measurements=[],
            claim_entity_links=[],
            claim_location_links=[],
            claim_period_links=[],
            entity_lookup={},
            run_id="run_1",
            report_year=1956,
            _contexts=ctx_list,
        )
        assert ctx_list[0].observation_id is not None

    def test_event_builder_does_not_import_from_observation_builder(self):
        """3A-6: event_builder imports _extract_year from derivation_context, not observation_builder."""
        import graphrag_pipeline.ingest.event_builder as eb
        src = eb.__file__
        with open(src, encoding="utf-8") as fh:
            source = fh.read()
        assert "from .observation_builder import" not in source
        assert "from graphrag_pipeline.ingest.derivation_context import _extract_year" in source

    def test_contexts_path_produces_identical_observations(self):
        """3A-4: build_observations with _contexts matches the no-contexts baseline."""
        claim = _claim("c1", "population_estimate")
        species = _entity("sp1", "Species", "mallard")
        entity_lookup = {"sp1": species}
        entity_links = [ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="SPECIES_FOCUS")]

        shared_kwargs = dict(
            claims=[claim],
            measurements=[],
            claim_entity_links=entity_links,
            claim_location_links=[],
            claim_period_links=[],
            entity_lookup=entity_lookup,
            run_id="run_1",
            report_year=1956,
        )

        baseline_obs, baseline_years, _, _ = build_observations(**shared_kwargs)

        ctx_list = _build(
            claims=[claim],
            entity_links=entity_links,
            entity_lookup=entity_lookup,
        )
        ctx_obs, ctx_years, _, _ = build_observations(**shared_kwargs, _contexts=ctx_list)

        assert len(ctx_obs) == len(baseline_obs)
        assert ctx_obs[0].observation_type == baseline_obs[0].observation_type
        assert ctx_obs[0].species_id == baseline_obs[0].species_id
        assert ctx_obs[0].year == baseline_obs[0].year
        assert ctx_obs[0].year_source == baseline_obs[0].year_source

    def test_contexts_path_produces_identical_events(self):
        """3A-4: build_events with _contexts matches the no-contexts baseline."""
        claim = _claim("c1", "population_estimate")
        species = _entity("sp1", "Species", "mallard")
        entity_lookup = {"sp1": species}
        entity_links = [ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="SPECIES_FOCUS")]

        shared_kwargs = dict(
            claims=[claim],
            measurements=[],
            claim_entity_links=entity_links,
            claim_location_links=[],
            claim_period_links=[],
            entity_lookup=entity_lookup,
            run_id="run_1",
            report_year=1956,
        )
        baseline_obs, _, _, _ = build_observations(**shared_kwargs)
        baseline_events, _, _, _ = build_events(**shared_kwargs, observations=baseline_obs)

        ctx_list = _build(
            claims=[claim],
            entity_links=entity_links,
            entity_lookup=entity_lookup,
        )
        build_observations(**shared_kwargs, _contexts=ctx_list)
        ctx_events, _, _, _ = build_events(**shared_kwargs, observations=baseline_obs, _contexts=ctx_list)

        assert len(ctx_events) == len(baseline_events)
        assert ctx_events[0].event_type == baseline_events[0].event_type
        assert ctx_events[0].species_id == baseline_events[0].species_id


# ===========================================================================
# 3B — Registry override
# ===========================================================================

class TestDerivationRegistry:
    _REGISTRY = {
        "custom_type": {
            "observation_type": "custom_obs",
            "event_type": "CustomEvent",
            "required_entities": ["Species"],
            "optional_entities": [],
        }
    }

    def test_registry_overrides_constants(self):
        """3B-1: observation_type and event_type come from registry when provided."""
        ctx_list = _build(
            claims=[_claim("c1", "custom_type")],
            registry=self._REGISTRY,
        )
        assert ctx_list[0].observation_type == "custom_obs"
        assert ctx_list[0].event_type == "CustomEvent"

    def test_constants_fallback_when_no_registry(self):
        """3B-2: Without registry, constants from claim_contract are used."""
        ctx_list = _build(claims=[_claim("c1", "population_estimate")])
        assert ctx_list[0].observation_type == "population_count"
        assert ctx_list[0].event_type == "SurveyEvent"

    def test_empty_registry_uses_constants(self):
        """3B-2: Empty registry dict (load absent file) falls back to constants."""
        ctx_list = _build(claims=[_claim("c1", "population_estimate")], registry=None)
        assert ctx_list[0].observation_type == "population_count"

    def test_missing_required_entities_populated(self):
        """3B-3: missing_required_entities contains Species when no Species link present."""
        registry = {
            "population_estimate": {
                "observation_type": "population_count",
                "event_type": "SurveyEvent",
                "required_entities": ["Species"],
                "optional_entities": [],
            }
        }
        ctx_list = _build(
            claims=[_claim("c1", "population_estimate")],
            registry=registry,
        )
        assert "Species" in ctx_list[0].missing_required_entities

    def test_no_missing_when_required_entity_present(self):
        """3B-3: missing_required_entities is empty when Species link present."""
        registry = {
            "population_estimate": {
                "observation_type": "population_count",
                "event_type": "SurveyEvent",
                "required_entities": ["Species"],
                "optional_entities": [],
            }
        }
        species = _entity("sp1", "Species", "mallard")
        ctx_list = _build(
            claims=[_claim("c1", "population_estimate")],
            entity_links=[ClaimEntityLinkRecord(claim_id="c1", entity_id="sp1", relation_type="SPECIES_FOCUS")],
            entity_lookup={"sp1": species},
            registry=registry,
        )
        assert ctx_list[0].missing_required_entities == []

    def test_load_derivation_registry_returns_empty_when_absent(self, tmp_path):
        """3B-2: load_derivation_registry() returns {} when file is absent."""
        from graphrag_pipeline.shared.resource_loader import load_derivation_registry
        result = load_derivation_registry(resources_dir=tmp_path)
        assert result == {}

    def test_load_derivation_registry_loads_entries(self, tmp_path):
        """load_derivation_registry() returns entries dict from YAML."""
        yaml_content = (
            "version: '1'\n"
            "entries:\n"
            "  population_estimate:\n"
            "    observation_type: population_count\n"
            "    event_type: SurveyEvent\n"
            "    required_entities: [Species]\n"
            "    optional_entities: []\n"
        )
        # Register via domain_profile so _resource_path resolves correctly
        (tmp_path / "domain_profile.yaml").write_text(
            "resources:\n  derivation_registry: derivation_registry.yaml\n",
            encoding="utf-8",
        )
        (tmp_path / "derivation_registry.yaml").write_text(yaml_content, encoding="utf-8")
        from graphrag_pipeline.shared.resource_loader import load_derivation_registry
        result = load_derivation_registry(resources_dir=tmp_path)
        assert "population_estimate" in result
        assert result["population_estimate"]["observation_type"] == "population_count"


# ===========================================================================
# 3C — Year plausibility validation
# ===========================================================================

class TestYearPlausibility:
    def test_year_within_window_unchanged(self):
        """3C-1: Year within tolerance window → unchanged."""
        year, source = _check_year_plausibility(
            year=1955, year_source="claim_date",
            doc_date_start="1950-01-01", doc_date_end="1960-12-31",
            report_year=1955,
            cfg={"tolerance": 5, "action": "flag"},
        )
        assert year == 1955
        assert source == "claim_date"

    def test_flag_action_outside_window(self):
        """3C-2: action=flag + outside window → year_source='suspect', year preserved."""
        year, source = _check_year_plausibility(
            year=1900, year_source="claim_date",
            doc_date_start="1950-01-01", doc_date_end="1960-12-31",
            report_year=1955,
            cfg={"tolerance": 5, "action": "flag"},
        )
        assert year == 1900
        assert source == "suspect"

    def test_exclude_action_outside_window(self):
        """3C-3: action=exclude + outside window → year=None, year_source='unknown'."""
        year, source = _check_year_plausibility(
            year=1900, year_source="claim_date",
            doc_date_start="1950-01-01", doc_date_end="1960-12-31",
            report_year=1955,
            cfg={"tolerance": 5, "action": "exclude"},
        )
        assert year is None
        assert source == "unknown"

    def test_no_validation_cfg_skips_check(self):
        """3C-4: year_validation_cfg=None → year unchanged from _extract_year."""
        ctx_list = _build(
            claims=[_claim("c1", "population_estimate", claim_date="1900-01-01")],
            report_year=1956,
            year_validation_cfg=None,
        )
        assert ctx_list[0].year == 1900
        assert ctx_list[0].year_source == "claim_date"

    def test_window_from_report_year_when_dates_absent(self):
        """3C-5: doc_date_start/end=None → window derived from report_year ± tolerance."""
        year, source = _check_year_plausibility(
            year=1900, year_source="claim_date",
            doc_date_start=None, doc_date_end=None,
            report_year=1955,
            cfg={"tolerance": 5, "action": "flag"},
        )
        assert source == "suspect"   # 1900 outside [1950, 1960]

    def test_window_from_report_year_plausible(self):
        """3C-5: Year within report_year ± tolerance → unchanged."""
        year, source = _check_year_plausibility(
            year=1953, year_source="claim_date",
            doc_date_start=None, doc_date_end=None,
            report_year=1955,
            cfg={"tolerance": 5, "action": "flag"},
        )
        assert year == 1953
        assert source == "claim_date"

    def test_exclude_nullifies_year_id_in_context(self):
        """3C-3: year_id is also None when year is excluded."""
        ctx_list = _build(
            claims=[_claim("c1", "population_estimate", claim_date="1900-01-01")],
            report_year=1956,
            doc_date_start="1950-01-01",
            doc_date_end="1960-12-31",
            year_validation_cfg={"tolerance": 5, "action": "exclude"},
        )
        ctx = ctx_list[0]
        assert ctx.year is None
        assert ctx.year_id is None
        assert ctx.year_source == "unknown"

    def test_flag_preserves_year_id_in_context(self):
        """3C-2: year_id still set when year is flagged (year value kept)."""
        ctx_list = _build(
            claims=[_claim("c1", "population_estimate", claim_date="1900-01-01")],
            report_year=1956,
            doc_date_start="1950-01-01",
            doc_date_end="1960-12-31",
            year_validation_cfg={"tolerance": 5, "action": "flag"},
        )
        ctx = ctx_list[0]
        assert ctx.year == 1900
        assert ctx.year_id is not None
        assert ctx.year_source == "suspect"

    def test_no_window_when_both_dates_and_report_year_absent(self):
        """No plausibility window → year returned unchanged."""
        year, source = _check_year_plausibility(
            year=1900, year_source="claim_date",
            doc_date_start=None, doc_date_end=None,
            report_year=None,
            cfg={"tolerance": 5, "action": "flag"},
        )
        assert year == 1900
        assert source == "claim_date"
