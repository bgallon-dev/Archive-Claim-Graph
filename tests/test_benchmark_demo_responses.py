from scripts.benchmark_demo_responses import evaluate_response


def _case() -> dict:
    return {
        "expected_claim_types": ["labor_action", "free_speech_event"],
        "answer_term_groups": [["labor", "IWW"], ["organizer", "speech"]],
        "year_min": 1908,
        "year_max": 1909,
        "min_context_claims": 2,
        "min_citations": 2,
        "min_in_range_years": 2,
    }


def _response() -> dict:
    return {
        "answer": (
            "Labor organizers and IWW speakers challenged local restrictions "
            "during the two-year period covered by the cited newspapers."
        ),
        "confidence_assessment": "MEDIUM",
        "supporting_claim_ids": ["c1", "c2"],
        "supporting_claims_detail": [
            {
                "claim_id": "c1",
                "claim_type": "labor_action",
                "doc_title": "Spokane Press — 1908-05-01",
            },
            {
                "claim_id": "c2",
                "claim_type": "free_speech_event",
                "doc_title": "Spokane Press — 1909-11-02",
            },
        ],
        "retrieval_stats": {
            "candidates_retrieved": 8,
            "claims_in_context": 2,
            "documents_in_context": 2,
            "plan_trace": {
                "selected_label": "claim_type_scoped:fulltext",
                "attempts": [],
            },
        },
        "synthesis_available": True,
    }


def test_evaluate_response_passes_representative_answer() -> None:
    result = evaluate_response(_case(), _response())
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["metrics"]["relevant_claim_ratio"] == 1.0
    assert result["metrics"]["evidence_years_in_requested_range"] == [1908, 1909]


def test_evaluate_response_exposes_specific_quality_failures() -> None:
    response = _response()
    response["answer"] = "No relevant material was found."
    response["supporting_claim_ids"] = []
    response["supporting_claims_detail"][1]["claim_type"] = "crime_report"
    response["retrieval_stats"]["claims_in_context"] = 1

    result = evaluate_response(_case(), response)
    assert result["passed"] is False
    assert result["checks"]["answer_present"] is False
    assert result["checks"]["context_claims"] is False
    assert result["checks"]["citations"] is False
    assert result["checks"]["claim_type_relevance"] is False
