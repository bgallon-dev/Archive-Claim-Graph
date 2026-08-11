"""Offline contract tests for the deep-research HTTP surface."""
from __future__ import annotations

from gemynd.retrieval.web.app import QueryResponse, create_app
from gemynd.shared.settings import Settings


def test_deep_routes_are_registered_without_starting_external_services():
    app = create_app()
    by_path = {getattr(route, "path", ""): route for route in app.routes}

    assert "/query/deep" in by_path
    assert "/v1/query/deep" in by_path
    assert "POST" in by_path["/query/deep"].methods
    assert "POST" in by_path["/v1/query/deep"].methods
    assert by_path["/query/deep"].include_in_schema is False
    assert by_path["/query/deep"].deprecated is True
    assert by_path["/v1/query/deep"].include_in_schema is True


def test_query_response_research_trace_is_backward_compatible():
    ordinary = QueryResponse(
        answer="answer",
        confidence_assessment="confidence",
        supporting_claim_ids=[],
        caveats=[],
    )
    assert ordinary.research_trace is None

    deep = ordinary.model_copy(update={
        "research_trace": {
            "stop_reason": "no_new_evidence",
            "attempts": [],
            "unique_claims": 0,
        }
    })
    assert deep.research_trace["stop_reason"] == "no_new_evidence"


def test_deep_rate_limit_defaults_and_environment(monkeypatch):
    defaults = Settings.from_env()
    assert defaults.deep_research_rate_limit_max == 4
    assert defaults.deep_research_rate_limit_period == 60.0

    monkeypatch.setenv("DEEP_RESEARCH_RATE_LIMIT_MAX", "7")
    monkeypatch.setenv("DEEP_RESEARCH_RATE_LIMIT_PERIOD", "90")
    configured = Settings.from_env()
    assert configured.deep_research_rate_limit_max == 7
    assert configured.deep_research_rate_limit_period == 90.0

