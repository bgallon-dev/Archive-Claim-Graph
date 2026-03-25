from graphrag_pipeline.core.ids import (
    make_claim_id,
    make_doc_id,
    make_entity_id,
    make_measurement_id,
    make_mention_id,
    make_observation_id,
    make_year_id,
    stable_hash,
)


def test_stable_hash_is_deterministic() -> None:
    first = stable_hash("Turnbull Refuge", 1938, "report.json")
    second = stable_hash("turnbull refuge", "1938", "report.json")
    assert first == second


def test_document_id_is_deterministic() -> None:
    doc_id_a = make_doc_id("Narrative Report", "1938-07-16", "1938-10-31", "report1.json")
    doc_id_b = make_doc_id("Narrative Report", "1938-07-16", "1938-10-31", "report1.json")
    assert doc_id_a == doc_id_b


def test_run_scoped_ids_include_run() -> None:
    claim_a = make_claim_id("run_1", "para_1", 1, "a fire occurred")
    claim_b = make_claim_id("run_2", "para_1", 1, "a fire occurred")
    assert claim_a != claim_b

    measurement_a = make_measurement_id("run_1", claim_a, 1, "acres_burned", "100 acres")
    measurement_b = make_measurement_id("run_2", claim_b, 1, "acres_burned", "100 acres")
    assert measurement_a != measurement_b

    mention_a = make_mention_id("run_1", "para_1", 0, 7, "Turnbull")
    mention_b = make_mention_id("run_2", "para_1", 0, 7, "Turnbull")
    assert mention_a != mention_b


def test_entity_id_stable() -> None:
    assert make_entity_id("Place", "pine creek") == make_entity_id("Place", "pine creek")


def test_observation_id_deterministic() -> None:
    a = make_observation_id("run_1", "claim_1", 1, "population_count")
    b = make_observation_id("run_1", "claim_1", 1, "population_count")
    assert a == b


def test_observation_id_varies_by_run() -> None:
    a = make_observation_id("run_1", "claim_1", 1, "population_count")
    b = make_observation_id("run_2", "claim_1", 1, "population_count")
    assert a != b


def test_year_id_simple() -> None:
    assert make_year_id(1938) == "year_1938"


def test_year_id_global_merge() -> None:
    assert make_year_id(1956) == make_year_id(1956)
