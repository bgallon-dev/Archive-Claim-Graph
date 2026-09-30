from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemynd.mcp.payments import (
    HTTPPaymentVerifier,
    PaymentError,
    PaymentLedger,
    PaymentVerification,
)
from gemynd.mcp.server import (
    HTTPQueryBackend,
    JsonEvidenceBackend,
    KettleMCPService,
    MCPToolError,
    RequestRateLimiter,
    SnapshotEvidenceBackend,
    UsageMeter,
    _authenticated_tenant,
    create_mcp_app,
    dispatch_jsonrpc,
)
from gemynd.mcp.snapshot import build_snapshot
from gemynd.mcp.sqlite_snapshot import SQLiteEvidenceSnapshotBackend, build_sqlite_snapshot


class FixtureBackend:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def search(self, question: str, *, time_range: dict | None, limit: int) -> list[dict]:
        return self.rows[:limit]


@pytest.fixture
def service(tmp_path: Path) -> KettleMCPService:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "corpora:\n"
        "  - corpus_id: spokane_newspaper\n"
        "    display_name: Spokane Historical Newspapers\n"
        "    description: Regional Spokane newspaper records\n",
        encoding="utf-8",
    )
    rows = [
        {
            "evidence_id": "ev-1",
            "claim": "The Spokane council approved the river plan in 1912.",
            "year": 1912,
            "subjects": [{"entity_id": "person:1", "name": "Ada Example"}],
            "relationships": [{"from": "person:1", "to": "org:1", "type": "served_on"}],
            "source": {
                "institution": "Spokane Public Library",
                "document": "Spokane Daily Chronicle",
                "page_or_record": "p. 4",
                "source_url": "https://example.test/ev-1",
            },
            "excerpt": "The Spokane council approved the river plan in 1912.",
            "confidence": 0.91,
        },
        {
            "evidence_id": "ev-2",
            "claim": "A later report disputed the river plan's cost estimate.",
            "year": 1913,
            "attestation": "uncertain",
            "source": {"document": "Municipal report"},
        },
    ]
    return KettleMCPService(
        backend=FixtureBackend(rows),
        registry_path=registry,
        meter=UsageMeter(daily_cap_usd=0.10),
    )


def test_coverage_probe_is_measured_against_the_archive(service: KettleMCPService) -> None:
    result = service.coverage_probe("What happened in Spokane in 1912?")
    assert result["coverage_basis"] == "archive_search"
    assert result["query_terms"] == ["spokane", "1912"]
    # One record holds both key terms: real but thin evidence.
    assert result["strong_matches"] == 1
    assert result["coverage_status"] == "partial"
    assert "spokane_newspaper" in result["relevant_corpora"]
    assert result["recommended_tool"] == "search_evidence"
    assert result["sample_evidence"][0]["evidence_id"] == "ev-1"
    assert result["sample_evidence"][0]["matched_terms"] == ["spokane", "1912"]


def test_coverage_probe_reports_strong_only_with_matching_records(tmp_path: Path) -> None:
    rows = [
        {
            "evidence_id": f"ev-{index}",
            "claim": f"Jimmy Durkin opened saloon number {index} in Spokane.",
            "source": {"document": f"Spokane Press {index}", "date_start": f"1905-03-0{index}"},
        }
        for index in range(1, 4)
    ]
    svc = KettleMCPService(backend=FixtureBackend(rows), registry_path=tmp_path / "none.yaml")
    strong = svc.coverage_probe("Who was Jimmy Durkin?")
    assert strong["coverage_status"] == "strong"
    assert strong["recommended_tool"] == "research_subject"
    assert strong["sample_evidence"][0]["date"] == "1905-03-01"

    unrelated = svc.coverage_probe("Boston tea party")
    assert unrelated["coverage_status"] == "weak"
    assert unrelated["external_search_recommended"] is True

    empty = KettleMCPService(backend=FixtureBackend([]), registry_path=tmp_path / "none.yaml")
    assert empty.coverage_probe("Jimmy Durkin")["coverage_status"] == "none"
    assert empty.coverage_probe("Jimmy Durkin")["recommended_tool"] is None


def test_coverage_probe_explains_questions_without_search_terms(
    service: KettleMCPService,
) -> None:
    result = service.coverage_probe("Who was he?")
    assert result["query_terms"] == []
    assert result["coverage_status"] == "none"
    assert "no searchable terms" in result["known_gaps"][0]


def test_coverage_probe_flags_years_outside_the_archive(tmp_path: Path) -> None:
    backend = JsonEvidenceBackend.__new__(JsonEvidenceBackend)
    backend.records = [
        {
            "evidence_id": "a",
            "claim": "Mill fire in Spokane",
            "year": 1905,
            "source": {"document": "D"},
        },
        {
            "evidence_id": "b",
            "claim": "Mill rebuilt in Spokane",
            "year": 1910,
            "source": {"document": "D"},
        },
    ]
    svc = KettleMCPService(backend=backend, registry_path=tmp_path / "none.yaml")
    result = svc.coverage_probe("Spokane mill", time_range={"start": 1950, "end": 1960})
    assert result["corpus"]["year_min"] == 1905 and result["corpus"]["year_max"] == 1910
    assert result["temporal_fit"] == 0.0
    assert any("outside the archive span" in gap for gap in result["known_gaps"])


def test_search_normalizes_evidence_and_is_free(service: KettleMCPService) -> None:
    result = service.search_evidence("Spokane river plan")
    evidence = result["evidence_objects"]
    assert [item["evidence_id"] for item in evidence] == ["ev-1", "ev-2"]
    assert evidence[0]["source"]["source_url"] == "https://example.test/ev-1"
    assert result["research_receipt"]["billing_status"] == "free"


def test_paid_call_is_metered_and_cap_is_enforced(service: KettleMCPService) -> None:
    first = service.call("research_subject", {"question": "Spokane river plan"}, tenant_id="t1")
    assert first["status"] == "ok"
    assert first["research_receipt"]["billing_status"] == "metered"
    with pytest.raises(MCPToolError, match="daily research cap"):
        service.call("research_subject", {"question": "Spokane river plan"}, tenant_id="t1")


def test_no_result_is_free_and_payment_mode_fails_closed(service: KettleMCPService) -> None:
    empty = KettleMCPService(backend=FixtureBackend([]), registry_path=service.registry_path)
    result = empty.call("research_subject", {"question": "Spokane river plan"})
    assert result["status"] == "no_result"
    assert result["research_receipt"]["price_usd"] == 0.0
    assert empty.meter.spent == {}

    service.require_payment = True
    with pytest.raises(MCPToolError, match="payment authorization"):
        service.call("research_subject", {"question": "Spokane river plan"})
    assert service.meter.spent == {}


def test_http_backend_does_not_forward_synthesized_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b'{"answer":"generated prose", "supporting_claim_ids":["claim-1"]}'

    monkeypatch.setattr("gemynd.mcp.server.urlopen", lambda *args, **kwargs: Response())
    backend = HTTPQueryBackend("https://example.test")
    with pytest.raises(MCPToolError, match="source-linked evidence details"):
        backend.search("question", time_range=None, limit=10)


def test_jsonrpc_lists_and_calls_tools(service: KettleMCPService) -> None:
    listed = dispatch_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, service)
    assert listed and len(listed["result"]["tools"]) == 4
    called = dispatch_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "coverage_probe", "arguments": {"question": "Spokane history"}},
        },
        service,
    )
    coverage = called and called["result"]["structuredContent"]
    assert coverage["coverage_basis"] == "archive_search"
    assert coverage["coverage_status"] == "partial"
    assert all(tool["annotations"]["readOnlyHint"] for tool in listed["result"]["tools"])


def test_initialize_negotiates_version_and_describes_the_corpus(
    service: KettleMCPService,
) -> None:
    response = dispatch_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        service,
    )
    result = response and response["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert "coverage_probe first" in result["instructions"]
    assert "Spokane Historical Newspapers" in result["instructions"]


def test_bad_arguments_are_tool_errors_not_crashes(service: KettleMCPService) -> None:
    def call(arguments: dict) -> dict:
        response = dispatch_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search_evidence", "arguments": arguments},
            },
            service,
        )
        assert response
        return response["result"]

    typo = call({"question": "Spokane river", "limt": 5})
    assert typo["isError"] and "limt" in typo["structuredContent"]["message"]
    bad_type = call({"question": "Spokane river", "limit": "ten"})
    assert bad_type["isError"]
    assert bad_type["structuredContent"]["error_code"] == "invalid_arguments"
    with pytest.raises(MCPToolError, match="depth must be one of"):
        service.research_subject("Spokane river", depth="extreme")
    # Filters advertised on coverage_probe must be accepted.
    probe = dispatch_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "coverage_probe",
                "arguments": {"question": "Spokane river", "document": "Chronicle"},
            },
        },
        service,
    )
    assert probe and not probe["result"].get("isError")


def test_research_response_carries_each_evidence_object_once(
    service: KettleMCPService,
) -> None:
    packet = service.build_evidence_packet("Spokane river plan")
    ids = [item["evidence_id"] for item in packet["evidence_objects"]]
    assert sorted(entry["evidence_id"] for entry in packet["timeline"]) == sorted(ids)
    assert packet["supporting_evidence"] == ids
    assert packet["contradictory_evidence"] == ["ev-2"]
    first = packet["evidence_objects"][0]
    assert first["excerpt"] == ""  # identical to the claim, so not repeated
    assert first["matched_terms"] == ["spokane", "river", "plan"]


def test_direct_trace_relationship_is_explicitly_unsupported(service: KettleMCPService) -> None:
    response = dispatch_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "trace_relationship", "arguments": {"question": "Spokane"}},
        },
        service,
    )
    assert (
        response
        and response["result"]["structuredContent"]["error_code"] == "unsupported_operation"
    )


def test_http_endpoint_requires_optional_api_key(
    service: KettleMCPService, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KETTLE_MCP_API_KEY", "test-key")
    client = TestClient(create_mcp_app(service))
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert client.post("/mcp", json=payload).status_code == 401
    response = client.post("/mcp", json=payload, headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert json.loads(response.text)["result"]["tools"]


def test_snapshot_backend_verifies_manifest_and_serves_without_neo4j(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        "evidence_id": "ev-standalone",
                        "claim": "A source-backed standalone claim.",
                        "source": {"document": "Example Gazette", "page_or_record": "p. 1"},
                        "excerpt": "A source-backed standalone claim.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    snapshot_dir = tmp_path / "snapshot"
    manifest = build_snapshot(source, snapshot_dir, snapshot_id="test-001")
    backend = SnapshotEvidenceBackend(snapshot_dir)
    assert backend.manifest["snapshot_id"] == "test-001"
    assert (
        backend.search("standalone claim", time_range=None, limit=1)[0]["evidence_id"]
        == "ev-standalone"
    )
    evidence = snapshot_dir / "evidence.jsonl"
    evidence.write_text(evidence.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        SnapshotEvidenceBackend(snapshot_dir)


def test_sqlite_snapshot_backend_searches_large_snapshot_format(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "sqlite-snapshot"
    manifest = build_sqlite_snapshot(
        [
            {
                "evidence_id": "ev-sqlite",
                "claim": "A searchable Spokane evidence record.",
                "excerpt": "A searchable Spokane evidence record.",
                "year": 1909,
                "subjects": [{"entity_id": "person:1", "name": "Ada Example"}],
                "relationships": [{"type": "served_on"}],
                "place": ["Spokane"],
                "provenance_id": "fingerprint-1",
                "source": {"document": "Example Gazette", "page_or_record": "p. 2"},
            }
        ],
        snapshot_dir,
        snapshot_id="sqlite-test-001",
    )
    backend = SQLiteEvidenceSnapshotBackend(snapshot_dir)
    assert manifest["record_count"] == 1
    row = backend.search("Spokane evidence", time_range=None, limit=5)[0]
    assert row["evidence_id"] == "ev-sqlite"
    assert row["provenance_id"] == "fingerprint-1"
    assert row["excerpt"] == row["claim"]
    assert row["subjects"][0]["name"] == "Ada Example"
    assert backend.stats()["year_min"] == 1909
    assert backend.search("Ada", time_range=None, limit=5)[0]["evidence_id"] == "ev-sqlite"
    assert manifest["format"] == "kettle-evidence-sqlite-snapshot-v2"
    assert manifest["size_bytes"] > 0
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(create_mcp_app(KettleMCPService(backend=backend)))
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_evidence",
                "arguments": {"question": "Spokane evidence"},
            },
        },
    )
    assert response.status_code == 200
    assert (
        response.json()["result"]["structuredContent"]["evidence_objects"][0]["evidence_id"]
        == "ev-sqlite"
    )
    backend.close()
    database = snapshot_dir / "evidence.sqlite"
    with database.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        SQLiteEvidenceSnapshotBackend(snapshot_dir)


def test_authenticated_tenant_ignores_spoofable_header() -> None:
    keys = ("secret-key",)
    first = _authenticated_tenant("Bearer secret-key", keys, "tenant-a")
    second = _authenticated_tenant("Bearer secret-key", keys, "tenant-b")
    assert first == second
    assert first and first.startswith("api_key_")
    assert "secret-key" not in first
    assert _authenticated_tenant("Bearer wrong", keys, "tenant-a") is None


def test_request_rate_limiter_rejects_after_window_budget() -> None:
    limiter = RequestRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("tenant-a")
    assert limiter.allow("tenant-a")
    assert not limiter.allow("tenant-a")
    assert limiter.allow("tenant-b")


def test_unsupported_claims_are_never_returned(service: KettleMCPService) -> None:
    service.backend = FixtureBackend(
        [
            {"claim_id": "unsupported", "source_sentence": "No source identity"},
            {
                "evidence_id": "supported",
                "claim": "A source-backed claim",
                "source": {"document": "Gazette"},
            },
        ]
    )
    result = service.search_evidence("source-backed")
    assert [item["evidence_id"] for item in result["evidence_objects"]] == ["supported"]


def test_filters_and_offset_pagination_are_applied(service: KettleMCPService) -> None:
    service.backend = FixtureBackend(
        [
            {
                "evidence_id": "a",
                "claim": "River event",
                "year": 1909,
                "place": ["Spokane"],
                "source": {"document": "Press A", "page_or_record": "p. 1"},
            },
            {
                "evidence_id": "b",
                "claim": "River event",
                "year": 1910,
                "place": ["Seattle"],
                "source": {"document": "Press B", "page_or_record": "p. 2"},
            },
            {
                "evidence_id": "c",
                "claim": "River event",
                "year": 1911,
                "place": ["Spokane"],
                "source": {"document": "Press A", "page_or_record": "p. 3"},
            },
        ]
    )
    result = service.search_evidence(
        "River event",
        time_range={"start": 1909, "end": 1911},
        location="Spokane",
        document="Press A",
        limit=1,
        offset=1,
    )
    assert [item["evidence_id"] for item in result["evidence_objects"]] == ["c"]
    assert result["pagination"] == {"offset": 1, "limit": 1, "has_more": False, "next_offset": None}


def test_golden_queries_have_expected_top_evidence_and_complete_provenance() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "kettle_evidence_golden.json").read_text(
            encoding="utf-8"
        )
    )
    for case in fixture["cases"]:
        svc = KettleMCPService(backend=FixtureBackend(case["records"]))
        result = svc.search_evidence(case["query"], limit=10)
        ids = [item["evidence_id"] for item in result["evidence_objects"]]
        assert ids and ids[0] in case["expected_top_ids"]
        assert all(
            item["claim"] and (item["source"]["document"] or item["source"]["page_or_record"])
            for item in result["evidence_objects"]
        )


def test_ambiguous_identity_can_be_narrowed_by_document(service: KettleMCPService) -> None:
    service.backend = FixtureBackend(
        [
            {
                "evidence_id": "spokane-john",
                "claim": "John Smith served in Spokane.",
                "source": {"document": "Spokane Press"},
            },
            {
                "evidence_id": "seattle-john",
                "claim": "John Smith served in Seattle.",
                "source": {"document": "Seattle Times"},
            },
        ]
    )
    broad = service.search_evidence("John Smith")
    narrowed = service.search_evidence("John Smith", document="Seattle Times")
    assert len(broad["evidence_objects"]) == 2
    assert [item["evidence_id"] for item in narrowed["evidence_objects"]] == ["seattle-john"]


def test_malformed_queries_and_ranges_fail_closed(service: KettleMCPService) -> None:
    with pytest.raises(MCPToolError, match="at least 3"):
        service.search_evidence("  ")
    with pytest.raises(MCPToolError, match="must be <="):
        service.search_evidence("valid query", time_range={"start": 1912, "end": 1900})


def test_paid_request_is_idempotent_and_records_snapshot(
    tmp_path: Path, service: KettleMCPService
) -> None:
    class Verifier:
        calls = 0

        def verify(self, **_kwargs):
            self.calls += 1
            return PaymentVerification(provider_reference="payment-ref-1")

    verifier = Verifier()
    service.backend.manifest = {"snapshot_id": "snapshot-123"}
    service.require_payment = True
    service.payment_verifier = verifier
    service.payment_ledger = PaymentLedger(tmp_path / "payments.db")
    first = service.call(
        "research_subject",
        {"question": "Spokane river plan"},
        tenant_id="tenant-1",
        idempotency_key="request-1",
        payment_receipt="receipt-1",
    )
    replay = service.call(
        "research_subject",
        {"question": "Spokane river plan"},
        tenant_id="tenant-1",
        idempotency_key="request-1",
        payment_receipt="receipt-1",
    )
    receipt = first["research_receipt"]
    assert receipt["billing_status"] == "authorized"
    assert receipt["snapshot_id"] == "snapshot-123"
    assert receipt["price_usd"] == 0.06
    assert receipt["payment_reference"] == "payment-ref-1"
    assert "receipt-1" not in json.dumps(first)
    assert replay["research_receipt"]["idempotent_replay"] is True
    assert verifier.calls == 1


def test_payment_receipt_cannot_be_reused(tmp_path: Path, service: KettleMCPService) -> None:
    class Verifier:
        def verify(self, **_kwargs):
            return PaymentVerification(provider_reference="payment-ref")

    service.require_payment = True
    service.payment_verifier = Verifier()
    service.payment_ledger = PaymentLedger(tmp_path / "payments.db")
    service.call(
        "research_subject",
        {"question": "Spokane history"},
        tenant_id="t1",
        idempotency_key="key-1",
        payment_receipt="same-receipt",
    )
    with pytest.raises(MCPToolError, match="already used"):
        service.call(
            "research_subject",
            {"question": "Spokane history"},
            tenant_id="t1",
            idempotency_key="key-2",
            payment_receipt="same-receipt",
        )


def test_api_key_rotation_accepts_old_and_new_keys(
    service: KettleMCPService, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("KETTLE_MCP_API_KEY", raising=False)
    monkeypatch.setenv("KETTLE_MCP_API_KEYS", "old-key,new-key")
    client = TestClient(create_mcp_app(service))
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert (
        client.post("/mcp", json=payload, headers={"Authorization": "Bearer old-key"}).status_code
        == 200
    )
    assert (
        client.post("/mcp", json=payload, headers={"Authorization": "Bearer new-key"}).status_code
        == 200
    )
    assert (
        client.post("/mcp", json=payload, headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )


def test_payment_verifier_errors_do_not_expose_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from urllib.error import URLError

    monkeypatch.setattr(
        "gemynd.mcp.payments.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("secret-receipt-value")),
    )
    verifier = HTTPPaymentVerifier("https://payments.example/verify", token="secret-token")
    with pytest.raises(PaymentError) as caught:
        verifier.verify(
            payment_receipt="secret-receipt-value",
            idempotency_key="request-1",
            tenant_id="tenant-1",
            operation="research_subject",
            price_usd=0.06,
        )
    message = str(caught.value)
    assert "secret-token" not in message
    assert "secret-receipt-value" not in message


def test_sqlite_search_ranks_full_matches_before_single_term_matches(tmp_path: Path) -> None:
    filler = [
        {
            "evidence_id": f"noise-{index}",
            "claim": "Durkin Durkin Durkin was mentioned again and again.",
            "source": {"document": f"Noise {index}"},
        }
        for index in range(5)
    ]
    target = {
        "evidence_id": "target",
        "claim": "The council granted Jimmy Durkin a saloon license on Riverside avenue.",
        "source": {"document": "Spokane Press", "date_start": "1907-06-02"},
        "year": 1907,
    }
    snapshot_dir = tmp_path / "ranked"
    build_sqlite_snapshot([*filler, target], snapshot_dir, snapshot_id="ranked-001")
    backend = SQLiteEvidenceSnapshotBackend(snapshot_dir)
    try:
        rows = backend.search("Who was Jimmy Durkin, saloon license?", time_range=None, limit=3)
        assert rows[0]["evidence_id"] == "target"
        assert len(rows) == 3  # single-term matches fill the remainder
        assert backend.search("who was he", time_range=None, limit=3) == []
        svc = KettleMCPService(backend=backend, registry_path=tmp_path / "none.yaml")
        evidence = svc.search_evidence("Jimmy Durkin saloon license")["evidence_objects"][0]
        assert evidence["time"] == {"start": "1907-06-02", "end": None, "precision": "day"}
    finally:
        backend.close()


def test_json_backend_ignores_stopwords_and_partial_words() -> None:
    backend = JsonEvidenceBackend.__new__(JsonEvidenceBackend)
    backend.records = [
        {
            "evidence_id": "a",
            "claim": "They tell me the locks are half built.",
            "source": {"document": "D"},
        },
        {"evidence_id": "b", "claim": "Moonlight on the river.", "source": {"document": "D"}},
    ]
    assert backend.search("Tell me about the moon landing", time_range=None, limit=5) == []


def test_lccn_sources_are_labeled_and_filterable_by_publication_title(
    service: KettleMCPService,
) -> None:
    service.backend = FixtureBackend(
        [
            {
                "evidence_id": "press",
                "claim": "Free speech fight continues in Spokane.",
                "source": {
                    "collection": "sn88085947",
                    "document": "sn88085947 — 1909-11-20 — pp1",
                    "date_start": "1909-11-20",
                },
            },
            {
                "evidence_id": "other",
                "claim": "Free speech fight reported in Portland.",
                "source": {"collection": "Local papers", "document": "Portland notes"},
            },
        ]
    )
    result = service.search_evidence("free speech fight", document="Spokane Press")
    [item] = result["evidence_objects"]
    assert item["source"]["publication"] == "The Spokane Press"
    assert item["source"]["document"] == "The Spokane Press — 1909-11-20 — pp1"
    assert item["source"]["collection"] == "sn88085947"  # provenance key is preserved
