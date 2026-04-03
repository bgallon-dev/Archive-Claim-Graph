from pathlib import Path

from gemynd.ingest.checkpoint import CHECKPOINT_FILENAME
from gemynd.shared.io_utils import load_semantic_bundle
from gemynd.ingest.pipeline import run_e2e


def test_e2e_parallel(fixtures_dir: Path, tmp_path: Path) -> None:
    inputs = [
        fixtures_dir / "report1.json",
        fixtures_dir / "report2.json",
        fixtures_dir / "report3.json",
    ]
    summary = run_e2e([str(path) for path in inputs], tmp_path, backend="memory", workers=2)
    assert summary["documents_processed"] == 3
    for output in summary["outputs"]:
        gates = output["quality"]["quality_gates"]
        assert gates["claims_have_evidence"]
        assert gates["measurements_linked"]
        assert gates["no_duplicate_ids"]
        assert gates["mention_offsets_valid"]


def test_e2e_on_three_reports(fixtures_dir: Path, tmp_path: Path) -> None:
    inputs = [
        fixtures_dir / "report1.json",
        fixtures_dir / "report2.json",
        fixtures_dir / "report3.json",
    ]
    summary = run_e2e([str(path) for path in inputs], tmp_path, backend="memory")
    assert summary["documents_processed"] == 3

    for output in summary["outputs"]:
        quality = output["quality"]
        gates = quality["quality_gates"]
        assert gates["claims_have_evidence"]
        assert gates["measurements_linked"]
        assert gates["no_duplicate_ids"]
        assert gates["mention_offsets_valid"]

    report1_semantic = load_semantic_bundle(tmp_path / "report1.semantic.json")
    assert report1_semantic.claims, "Expected claims in report1."
    first_claim_payload = report1_semantic.claims[0].to_dict()
    assert "epistemic_status" in first_claim_payload
    assert "certainty" not in first_claim_payload
    period_entities = [entity for entity in report1_semantic.entities if entity.entity_type == "Period"]
    assert period_entities, "Expected Period entities in report1."
    for entity in period_entities:
        assert "source_title" in entity.properties
        assert "label" not in entity.properties
    fire_claim_ids = [claim.claim_id for claim in report1_semantic.claims if claim.claim_type == "fire_incident"]
    assert fire_claim_ids, "Expected fire claims in report1."
    fire_measurements = [
        measurement.name
        for measurement in report1_semantic.measurements
        if measurement.claim_id in set(fire_claim_ids)
    ]
    assert "acres_burned" in fire_measurements
    assert "suppression_cost" in fire_measurements

    assert report1_semantic.claim_entity_links, "Expected typed claim-entity links in report1."
    relation_types = {link.relation_type for link in report1_semantic.claim_entity_links}
    assert "ABOUT" not in relation_types
    assert "SPECIES_FOCUS" in relation_types
    assert "LOCATION_FOCUS" in relation_types
    assert "MANAGEMENT_TARGET" in relation_types

    report3_semantic = load_semantic_bundle(tmp_path / "report3.semantic.json")
    assert any(mention.ocr_suspect for mention in report3_semantic.mentions)
    assert any(resolution.relation_type == "POSSIBLY_REFERS_TO" for resolution in report3_semantic.entity_resolutions)

    economic_claim_ids = [claim.claim_id for claim in report1_semantic.claims if claim.claim_type == "economic_use"]
    economic_measurements = [
        measurement.name
        for measurement in report1_semantic.measurements
        if measurement.claim_id in set(economic_claim_ids)
    ]
    for name in ("farmers_count", "hay_cut", "land_area", "revenue"):
        assert name in economic_measurements

    # --- Observation layer assertions ---
    assert report1_semantic.observations, "Expected observations in report1."
    obs_types = {obs.observation_type for obs in report1_semantic.observations}
    assert obs_types, "Expected at least one observation type."

    # Year nodes created
    assert report1_semantic.years, "Expected year nodes in report1."
    year_values = {y.year for y in report1_semantic.years}
    assert year_values, "Expected at least one year value."

    # Observation-measurement links point to valid observations
    if report1_semantic.observation_measurement_links:
        obs_ids = {obs.observation_id for obs in report1_semantic.observations}
        for link in report1_semantic.observation_measurement_links:
            assert link.observation_id in obs_ids

    # Quality gates include new observation checks
    for output in summary["outputs"]:
        quality = output["quality"]
        assert "observation_count" in quality
        assert "observations_have_evidence" in quality["quality_gates"]
        assert "claim_link_diagnostic_counts" in quality
        assert quality["typed_claim_entity_link_share"] >= 0.0


def test_e2e_single_worker_checkpoint_resume(fixtures_dir: Path, tmp_path: Path) -> None:
    """Single-worker interleaved path skips extraction for checkpointed docs."""
    inputs = [str(fixtures_dir / "report1.json")]
    summary1 = run_e2e(inputs, tmp_path, backend="memory")
    assert summary1["documents_processed"] == 1
    assert (tmp_path / CHECKPOINT_FILENAME).exists()

    # Re-run: checkpoint exists → doc fully skipped (no re-extraction).
    summary2 = run_e2e(inputs, tmp_path, backend="memory")
    assert summary2["documents_processed"] == 0
