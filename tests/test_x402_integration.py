from __future__ import annotations

from typing import Any

import pytest

from gemynd.mcp.server import KettleMCPService, create_mcp_app
from gemynd.mcp.x402 import (
    BASE_SEPOLIA,
    PAID_ROUTES,
    X402Settings,
    _normalize_paid_arguments,
    estimate_purchase,
)


class CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def search(
        self, question: str, *, time_range: dict[str, int] | None, limit: int
    ) -> list[dict[str, Any]]:
        self.calls += 1
        return [
            {
                "evidence_id": "ev-x402",
                "claim": "Spokane civic leaders supported riverfront redevelopment.",
                "year": 1909,
                "source": {"document": "Spokane Press", "page_or_record": "p. 1"},
            }
        ]


def _settings() -> X402Settings:
    return X402Settings(
        enabled=True,
        network=BASE_SEPOLIA,
        pay_to="0x1111111111111111111111111111111111111111",
        public_base_url="https://research.example.com",
        cdp_api_key_id="test-key-id",
        cdp_api_key_secret="test-key-secret",
    )


def test_x402_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "KETTLE_X402_ENABLED",
        "KETTLE_X402_NETWORK",
        "KETTLE_X402_PAY_TO",
        "KETTLE_X402_PUBLIC_BASE_URL",
        "CDP_API_KEY_ID",
        "CDP_API_KEY_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = X402Settings.from_env()
    assert settings.enabled is False
    assert settings.network == BASE_SEPOLIA


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"pay_to": "not-an-address"}, "20-byte EVM address"),
        ({"pay_to": "0x" + "0" * 40}, "nonzero"),
        ({"network": "eip155:1"}, "KETTLE_X402_NETWORK"),
        ({"cdp_api_key_secret": ""}, "configured together"),
        ({"public_base_url": ""}, "absolute HTTP"),
    ],
)
def test_x402_enabled_configuration_fails_closed(replacement: dict[str, str], message: str) -> None:
    values = _settings().__dict__ | replacement
    with pytest.raises(RuntimeError, match=message):
        X402Settings(**values).validate()


def test_fixed_price_routes_are_stable() -> None:
    assert [(route.path, route.price, route.max_evidence_objects) for route in PAID_ROUTES] == [
        ("/paid/find-evidence", "$0.03", 10),
        ("/paid/research-subject", "$0.06", 40),
        ("/paid/investigate-subject", "$0.15", 100),
        ("/paid/compile-evidence-packet", "$0.25", 100),
    ]


def test_paid_route_evidence_budget_cannot_be_bypassed() -> None:
    with pytest.raises(ValueError, match="between 1 and 10"):
        _normalize_paid_arguments(
            {"question": "Spokane riverfront leaders", "max_evidence_objects": 11},
            max_evidence_objects=10,
        )
    normalized = _normalize_paid_arguments(
        {"question": "Spokane riverfront leaders"},
        max_evidence_objects=10,
    )
    assert normalized["max_evidence_objects"] == 10


def test_free_estimate_performs_no_retrieval_and_recommends_by_intent() -> None:
    packet = estimate_purchase({"goal": "compile_evidence_packet", "max_evidence_objects": 30})
    assert packet["recommended"]["path"] == "/paid/compile-evidence-packet"
    assert packet["recommended"]["price_usd"] == 0.25
    assert packet["retrieval_performed"] is False
    deep = estimate_purchase({"goal": "find_evidence", "max_evidence_objects": 80})
    assert deep["recommended"]["path"] == "/paid/investigate-subject"


def test_public_funnel_needs_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KETTLE_MCP_API_KEY", "private-mcp-key")
    backend = CountingBackend()
    client = TestClient(create_mcp_app(KettleMCPService(backend=backend)))

    capabilities = client.get("/public/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["positioning"].startswith("Use Kettle")
    assert capabilities.json()["paid_operations"][0]["path"] == "/paid/find-evidence"

    coverage = client.post(
        "/public/coverage",
        json={"question": "Who shaped Spokane's riverfront development?"},
    )
    assert coverage.status_code == 200
    assert "coverage_status" in coverage.json()

    estimate = client.post(
        "/public/estimate",
        json={"goal": "research_subject", "max_evidence_objects": 25},
    )
    assert estimate.status_code == 200
    assert estimate.json()["recommended"]["path"] == "/paid/research-subject"
    assert estimate.json()["payment_available"] is False
    assert backend.calls == 0

    mcp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert mcp.status_code == 401


def test_x402_verifies_before_work_and_settles_before_releasing_result() -> None:
    pytest.importorskip("x402")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from x402 import x402ResourceServer
    from x402.http import (
        decode_payment_required_header,
        decode_payment_response_header,
        encode_payment_signature_header,
    )
    from x402.mechanisms.evm.exact import ExactEvmServerScheme
    from x402.schemas import (
        PaymentPayload,
        SettleResponse,
        SupportedKind,
        SupportedResponse,
        VerifyResponse,
    )

    events: list[str] = []

    class FakeCoinbaseFacilitator:
        def get_supported(self) -> SupportedResponse:
            events.append("supported")
            return SupportedResponse(
                kinds=[
                    SupportedKind(
                        x402_version=2,
                        scheme="exact",
                        network=BASE_SEPOLIA,
                    )
                ],
                extensions=["bazaar"],
            )

        async def verify(self, payload, requirements) -> VerifyResponse:
            events.append("verify")
            return VerifyResponse(
                is_valid=True,
                payer="0x2222222222222222222222222222222222222222",
            )

        async def settle(self, payload, requirements) -> SettleResponse:
            events.append("settle")
            return SettleResponse(
                success=True,
                payer="0x2222222222222222222222222222222222222222",
                transaction="0x" + "ab" * 32,
                network=BASE_SEPOLIA,
                amount=requirements.amount,
            )

    facilitator = FakeCoinbaseFacilitator()
    resource_server = x402ResourceServer(facilitator)
    resource_server.register(BASE_SEPOLIA, ExactEvmServerScheme())
    backend = CountingBackend()
    app = create_mcp_app(
        KettleMCPService(backend=backend),
        x402_settings=_settings(),
        x402_resource_server=resource_server,
    )
    client = TestClient(app)
    request_body = {"question": "Spokane riverfront leaders"}

    unpaid = client.post("/paid/research-subject", json=request_body)
    assert unpaid.status_code == 402
    assert backend.calls == 0
    required_header = unpaid.headers["payment-required"]
    required = decode_payment_required_header(required_header)
    assert required.accepts[0].network == BASE_SEPOLIA
    assert required.accepts[0].amount == "60000"
    assert required.accepts[0].pay_to == _settings().pay_to
    assert required.resource.url == "https://research.example.com/paid/research-subject"
    assert required.extensions and "bazaar" in required.extensions
    bazaar = required.extensions["bazaar"]
    body_schema = bazaar["schema"]["properties"]["input"]["properties"]["body"]
    assert body_schema["properties"]["max_evidence_objects"]["maximum"] == 40
    assert "Use for a bounded historical investigation" in required.resource.description

    payload = PaymentPayload(
        payload={"signature": "integration-test-signature"},
        accepted=required.accepts[0],
        resource=required.resource,
        extensions=required.extensions,
    )
    paid = client.post(
        "/paid/research-subject",
        json=request_body,
        headers={"Payment-Signature": encode_payment_signature_header(payload)},
    )
    assert paid.status_code == 200, paid.text
    assert events.index("verify") < events.index("settle")
    assert backend.calls == 1, paid.text
    receipt = paid.json()["research_receipt"]
    assert receipt["billing_status"] == "settled_by_x402"
    assert receipt["price_usd"] == 0.06
    assert receipt["payment_network"] == BASE_SEPOLIA
    settlement = decode_payment_response_header(paid.headers["payment-response"])
    assert settlement.success is True
    assert settlement.transaction == "0x" + "ab" * 32


def test_x402_settlement_failure_does_not_release_research_body() -> None:
    pytest.importorskip("x402")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from x402 import x402ResourceServer
    from x402.http import decode_payment_required_header, encode_payment_signature_header
    from x402.mechanisms.evm.exact import ExactEvmServerScheme
    from x402.schemas import (
        PaymentPayload,
        SettleResponse,
        SupportedKind,
        SupportedResponse,
        VerifyResponse,
    )

    class FailedSettlementFacilitator:
        def get_supported(self) -> SupportedResponse:
            return SupportedResponse(
                kinds=[
                    SupportedKind(
                        x402_version=2,
                        scheme="exact",
                        network=BASE_SEPOLIA,
                    )
                ],
                extensions=["bazaar"],
            )

        async def verify(self, payload, requirements) -> VerifyResponse:
            return VerifyResponse(is_valid=True)

        async def settle(self, payload, requirements) -> SettleResponse:
            return SettleResponse(
                success=False,
                error_reason="test_failure",
                error_message="Settlement intentionally failed",
                transaction="",
                network=BASE_SEPOLIA,
            )

    server = x402ResourceServer(FailedSettlementFacilitator())
    server.register(BASE_SEPOLIA, ExactEvmServerScheme())
    backend = CountingBackend()
    client = TestClient(
        create_mcp_app(
            KettleMCPService(backend=backend),
            x402_settings=_settings(),
            x402_resource_server=server,
        )
    )
    body = {"question": "Spokane riverfront leaders"}
    unpaid = client.post("/paid/find-evidence", json=body)
    required = decode_payment_required_header(unpaid.headers["payment-required"])
    payload = PaymentPayload(
        payload={"signature": "integration-test-signature"},
        accepted=required.accepts[0],
        resource=required.resource,
        extensions=required.extensions,
    )
    failed = client.post(
        "/paid/find-evidence",
        json=body,
        headers={"Payment-Signature": encode_payment_signature_header(payload)},
    )
    assert backend.calls == 1, failed.text
    assert failed.status_code == 402
    assert "evidence_objects" not in failed.text
    assert "Spokane Press" not in failed.text
