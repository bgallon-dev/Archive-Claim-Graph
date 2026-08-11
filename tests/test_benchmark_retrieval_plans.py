from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark_retrieval_plans import (
    DEFAULT_CASES,
    EXPECTED_FAMILY_COUNTS,
    ManifestValidationError,
    evidence_matches_facet,
    evaluate_evidence,
    evaluate_routes,
    load_manifest,
    redact_sensitive,
    validate_manifest,
)


def test_coverage_manifest_has_exact_checkpoint_zero_shape() -> None:
    cases = load_manifest(DEFAULT_CASES)

    assert len(cases) == 40
    assert {
        family: sum(case["family"] == family for case in cases)
        for family in EXPECTED_FAMILY_COUNTS
    } == EXPECTED_FAMILY_COUNTS
    assert len({case["id"] for case in cases}) == 40
    assert all(
        case["required_evidence_facets"]
        or any(value for value in case["scope"].values())
        for case in cases
    )


def test_manifest_validation_rejects_family_count_drift() -> None:
    cases = load_manifest(DEFAULT_CASES)
    broken = deepcopy(cases)
    broken[0]["family"] = "mention_only_entity"

    with pytest.raises(ManifestValidationError, match="family counts differ"):
        validate_manifest(broken)


def test_evidence_matcher_combines_all_active_axes() -> None:
    block = SimpleNamespace(
        claim_type="free_speech_event",
        source_sentence="The IWW continued its free speech campaign.",
        doc_title="Spokane Press — 1909-11-03",
        doc_title_raw="The Spokane Press",
        doc_date_start="1909-11-03",
        doc_date_end=None,
        year=None,
        institution_id="spokane_newspaper",
        series=None,
    )
    facet = {
        "claim_types": ["free_speech_event"],
        "source_terms_all": ["IWW", "free speech"],
        "exact_date": "1909-11-03",
        "institution_ids": ["spokane_newspaper"],
    }

    assert evidence_matches_facet(block, facet) is True
    assert evidence_matches_facet(
        block,
        {**facet, "exact_date": "1909-11-04"},
    ) is False


def test_evidence_evaluation_records_rank_and_missing_facets() -> None:
    blocks = [
        {
            "claim_type": "labor_action",
            "source_sentence": "A strike was called.",
            "doc_date_start": "1909-01-01",
            "institution_id": "spokane_newspaper",
        },
        {
            "claim_type": "free_speech_event",
            "source_sentence": "The IWW held a street meeting.",
            "doc_date_start": "1909-11-03",
            "institution_id": "spokane_newspaper",
        },
    ]
    facets = [
        {"source_terms_any": ["IWW"]},
        {"source_terms_any": ["nonexistent phrase"]},
    ]

    matched, missing = evaluate_evidence(facets, blocks)

    assert matched == [
        {
            "facet_index": 0,
            "facet": facets[0],
            "matched_block_rank": 2,
        }
    ]
    assert missing == [{"facet_index": 1, "facet": facets[1]}]


def test_multi_turn_case_survives_json_serialization(tmp_path: Path) -> None:
    cases = load_manifest(DEFAULT_CASES)
    original = next(
        case for case in cases if case["id"] == "followup_free_speech_people"
    )
    path = tmp_path / "case.json"
    path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))

    assert restored["turns"] == original["turns"]
    assert len(restored["turns"]) == 2
    assert restored["turns"][-1]["text"] == "Who were the main people involved?"


def test_report_redaction_removes_keys_env_values_and_uri_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHMARK_TEST_PASSWORD", "unique-secret-value")
    payload = {
        "neo4j_password": "unique-secret-value",
        "nested": {
            "message": (
                "could not connect to bolt://neo4j:unique-secret-value@database:7687"
            ),
            "safe": "keep-me",
        },
    }

    rendered = json.dumps(redact_sensitive(payload))

    assert "unique-secret-value" not in rendered
    assert "neo4j:" not in rendered
    assert "[REDACTED]" in rendered
    assert "keep-me" in rendered


def test_forbidden_fallback_causes_route_gate_failure() -> None:
    evaluation = evaluate_routes(
        expected=["entity_anchored"],
        forbidden=["entity_anchored:drop_entity_anchor"],
        attempted_labels=[
            "entity_anchored",
            "entity_anchored:drop_entity_anchor",
        ],
        selected_label="entity_anchored:drop_entity_anchor",
    )

    assert evaluation["expected_route_seen"] is True
    assert evaluation["forbidden_route_absent"] is False
    assert evaluation["matched_forbidden_routes"] == [
        {
            "pattern": "entity_anchored:drop_entity_anchor",
            "labels": ["entity_anchored:drop_entity_anchor"],
        }
    ]
