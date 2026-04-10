from pathlib import Path

from gemynd.ingest.graph.writer import InMemoryGraphWriter
from gemynd.ingest.pipeline import extract_semantic
from gemynd.ingest.source_parser import parse_source_file
from tests.conftest import TEST_ENTITY_LABELS

# DOMAIN_LABELS was removed; this test fixture uses the Turnbull label set.
DOMAIN_LABELS = TEST_ENTITY_LABELS


def _has_rel(
    writer: InMemoryGraphWriter,
    start_label: str,
    start_id: str,
    rel_type: str,
    end_label: str,
    end_id: str,
) -> bool:
    return any(
        key[:5] == (start_label, start_id, rel_type, end_label, end_id)
        for key in writer.rel_store
    )


def _has_rel_with_any_end_label(
    writer: InMemoryGraphWriter,
    start_label: str,
    start_id: str,
    rel_type: str,
    end_id: str,
) -> bool:
    return any(
        key[0] == start_label
        and key[1] == start_id
        and key[2] == rel_type
        and key[4] == end_id
        for key in writer.rel_store
    )


def test_inmemory_upsert_idempotent_and_historical(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic_run_1 = extract_semantic(
        structure,
        run_overrides={"run_id": "run_test_1", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )
    semantic_run_2 = extract_semantic(
        structure,
        run_overrides={"run_id": "run_test_2", "run_timestamp": "2026-03-10T01:00:00+00:00"},
    )

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
    writer.create_schema()

    writer.load_structure(structure)
    writer.load_semantic(structure, semantic_run_1)
    first_claim_count = len(writer.node_store["Claim"])
    first_run_count = len(writer.node_store["ExtractionRun"])

    writer.load_structure(structure)
    writer.load_semantic(structure, semantic_run_1)
    assert len(writer.node_store["Claim"]) == first_claim_count
    assert len(writer.node_store["ExtractionRun"]) == first_run_count

    writer.load_semantic(structure, semantic_run_2)
    assert len(writer.node_store["Claim"]) > first_claim_count
    assert len(writer.node_store["ExtractionRun"]) == 2


def test_observation_and_year_nodes_created(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_obs_test", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)

    assert "Observation" in writer.node_store, "Expected Observation nodes."
    assert "Year" in writer.node_store, "Expected Year nodes."

    # Observations should have SUPPORTS relationships from Claims
    supports_rels = [k for k in writer.rel_store if k[2] == "SUPPORTS"]
    assert supports_rels, "Expected SUPPORTS relationships from Claims to Observations."

    # Observation->HAS_MEASUREMENT should exist for wildlife observations
    obs_measurement_rels = [k for k in writer.rel_store if k[2] == "HAS_MEASUREMENT" and k[0] == "Observation"]
    # Non-observation measurements keep Claim->HAS_MEASUREMENT
    claim_measurement_rels = [k for k in writer.rel_store if k[2] == "HAS_MEASUREMENT" and k[0] == "Claim"]
    # At least one path should have measurements
    assert obs_measurement_rels or claim_measurement_rels, "Expected HAS_MEASUREMENT relationships."

    # Year should have COVERS_YEAR from Document
    covers_year_rels = [k for k in writer.rel_store if k[2] == "COVERS_YEAR"]
    assert covers_year_rels, "Expected COVERS_YEAR relationships."


def test_doc_id_not_stored_on_structural_nodes(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
    writer.create_schema()
    writer.load_structure(structure)

    for label in ("Page", "Section", "Paragraph", "Annotation"):
        for node_props in writer.node_store.get(label, {}).values():
            assert "doc_id" not in node_props, (
                f"{label} node must not store doc_id as a property "
                "(authoritative as a graph edge)"
            )


def test_group_a_redundant_fk_properties_not_stored_on_nodes(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_group_a", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)

    forbidden_by_label = {
        "Page": {"doc_id"},
        "Section": {"doc_id"},
        "Paragraph": {"doc_id", "page_id", "section_id"},
        "Annotation": {"doc_id", "page_id"},
        "Claim": {"paragraph_id"},
        "Mention": {"paragraph_id"},
        "Observation": {
            "paragraph_id",
            "species_id",
            "refuge_id",
            "place_id",
            "year_id",
            "habitat_id",
            "survey_method_id",
        },
        "Event": {
            "paragraph_id",
            "species_id",
            "refuge_id",
            "place_id",
            "year_id",
            "habitat_id",
            "survey_method_id",
        },
    }

    for label, forbidden_keys in forbidden_by_label.items():
        for node_props in writer.node_store.get(label, {}).values():
            unexpected = sorted(forbidden_keys.intersection(node_props))
            assert not unexpected, f"{label} nodes must not persist redundant FK props: {unexpected}"


def test_group_a_edges_replace_removed_fk_properties(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_group_a_edges", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)

    entity_labels = {entity.entity_id: entity.entity_type for entity in semantic.entities}

    for page in structure.pages:
        assert _has_rel(writer, "Document", structure.document.doc_id, "HAS_PAGE", "Page", page.page_id)

    for paragraph in structure.paragraphs:
        if paragraph.section_id:
            assert _has_rel(writer, "Section", paragraph.section_id, "HAS_PARAGRAPH", "Paragraph", paragraph.paragraph_id)

    for annotation in structure.annotations:
        assert _has_rel(writer, "Page", annotation.page_id, "HAS_ANNOTATION", "Annotation", annotation.annotation_id)

    for claim in semantic.claims:
        assert _has_rel(writer, "Paragraph", claim.paragraph_id, "HAS_CLAIM", "Claim", claim.claim_id)
        assert _has_rel(writer, "Claim", claim.claim_id, "EVIDENCED_BY", "Paragraph", claim.paragraph_id)

    for mention in semantic.mentions:
        assert _has_rel(writer, "Paragraph", mention.paragraph_id, "CONTAINS_MENTION", "Mention", mention.mention_id)

    for resolution in semantic.entity_resolutions:
        assert entity_labels.get(resolution.entity_id) is not None
        assert _has_rel_with_any_end_label(
            writer,
            "Mention",
            resolution.mention_id,
            resolution.relation_type,
            resolution.entity_id,
        )

    claim_entity_rel_types = {
        key[2]
        for key in writer.rel_store
        if key[0] == "Claim" and key[3] in {"Species", "Habitat", "SurveyMethod", "Activity", "Place", "Refuge"}
    }
    assert "ABOUT" not in claim_entity_rel_types
    assert "SPECIES_FOCUS" in claim_entity_rel_types
    assert "LOCATION_FOCUS" in claim_entity_rel_types
    assert "MANAGEMENT_TARGET" in claim_entity_rel_types

    for link in semantic.observation_measurement_links:
        assert _has_rel(writer, "Observation", link.observation_id, "HAS_MEASUREMENT", "Measurement", link.measurement_id)

    for obs in semantic.observations:
        assert _has_rel(writer, "Observation", obs.observation_id, "EVIDENCED_BY", "Paragraph", obs.paragraph_id)
        if obs.species_id:
            assert _has_rel(writer, "Observation", obs.observation_id, "OF_SPECIES", "Species", obs.species_id)
        if obs.refuge_id:
            assert _has_rel(writer, "Observation", obs.observation_id, "AT_REFUGE", "Refuge", obs.refuge_id)
        if obs.place_id:
            assert _has_rel(writer, "Observation", obs.observation_id, "AT_PLACE", "Place", obs.place_id)
        if obs.year_id:
            assert _has_rel(writer, "Observation", obs.observation_id, "IN_YEAR", "Year", obs.year_id)
        if obs.habitat_id:
            assert _has_rel(writer, "Observation", obs.observation_id, "IN_HABITAT", "Habitat", obs.habitat_id)
        if obs.survey_method_id:
            assert _has_rel(writer, "Observation", obs.observation_id, "USED_METHOD", "SurveyMethod", obs.survey_method_id)

    for evt in semantic.events:
        assert _has_rel(writer, "Claim", evt.claim_id, "TRIGGERED", "Event", evt.event_id)
        assert _has_rel(writer, "Event", evt.event_id, "SOURCED_FROM", "Paragraph", evt.paragraph_id)
        if evt.species_id:
            assert _has_rel(writer, "Event", evt.event_id, "INVOLVED_SPECIES", "Species", evt.species_id)
        if evt.refuge_id:
            assert _has_rel(writer, "Event", evt.event_id, "OCCURRED_AT", "Refuge", evt.refuge_id)
        if evt.place_id:
            assert _has_rel(writer, "Event", evt.event_id, "OCCURRED_AT", "Place", evt.place_id)
        if evt.year_id:
            assert _has_rel(writer, "Event", evt.event_id, "IN_YEAR", "Year", evt.year_id)
        if evt.habitat_id:
            assert _has_rel(writer, "Event", evt.event_id, "IN_HABITAT", "Habitat", evt.habitat_id)
        if evt.survey_method_id:
            assert _has_rel(writer, "Event", evt.event_id, "USED_METHOD", "SurveyMethod", evt.survey_method_id)

    for link in semantic.event_measurement_links:
        assert _has_rel(writer, "Event", link.event_id, "PRODUCED_MEASUREMENT", "Measurement", link.measurement_id)

    for link in semantic.document_refuge_links:
        assert _has_rel(writer, "Document", link.doc_id, "ABOUT_REFUGE", "Refuge", link.refuge_id)

    for link in semantic.document_year_links:
        assert _has_rel(writer, "Document", link.doc_id, "COVERS_YEAR", "Year", link.year_id)


def test_entity_resolution_edges_store_match_score_not_score(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_match_score", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)

    resolution_props = [
        props
        for key, props in writer.rel_store.items()
        if key[2] in {"REFERS_TO", "POSSIBLY_REFERS_TO"}
    ]
    assert resolution_props, "Expected entity resolution relationships."
    for props in resolution_props:
        assert "match_score" in props
        assert "score" not in props


def test_domain_entities_also_stored_under_entity_label(fixtures_dir: Path) -> None:
    """Every entity with a domain label must also appear under the 'Entity'
    key in node_store, so retrieval queries matching (e:Entity) will find them."""
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_label_test", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)

    for entity in semantic.entities:
        if entity.entity_type in DOMAIN_LABELS:
            assert entity.entity_id in writer.node_store.get("Entity", {}), (
                f"Entity {entity.entity_id} ({entity.entity_type}) not found under 'Entity' label"
            )


def test_located_in_refuge_edge_emitted(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_refuge_test", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    writer = InMemoryGraphWriter(entity_labels=TEST_ENTITY_LABELS)
    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)

    rel_types = {k[2] for k in writer.rel_store}
    assert "LOCATED_IN_REFUGE" in rel_types, "Expected LOCATED_IN_REFUGE relationship."
    assert "PART_OF" not in rel_types, "PART_OF must not be emitted; use LOCATED_IN_REFUGE."
