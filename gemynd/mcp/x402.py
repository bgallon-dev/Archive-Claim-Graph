"""Coinbase CDP/x402 payment routes for the public Kettle marketplace API.

The x402 dependency is optional.  Nothing in this module imports it until the
integration is explicitly enabled, so a private MCP-only deployment does not
need to carry the blockchain/payment SDKs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import KettleMCPService


BASE_SEPOLIA = "eip155:84532"
BASE_MAINNET = "eip155:8453"
SUPPORTED_NETWORKS = {BASE_SEPOLIA, BASE_MAINNET}
_EVM_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class X402Settings:
    """Validated settings for Kettle's x402 resource server."""

    enabled: bool = False
    network: str = BASE_SEPOLIA
    pay_to: str = ""
    public_base_url: str = ""
    cdp_api_key_id: str = ""
    cdp_api_key_secret: str = ""

    @classmethod
    def from_env(cls) -> X402Settings:
        enabled = os.getenv("KETTLE_X402_ENABLED", "false").strip().casefold() == "true"
        settings = cls(
            enabled=enabled,
            network=os.getenv("KETTLE_X402_NETWORK", BASE_SEPOLIA).strip(),
            pay_to=os.getenv("KETTLE_X402_PAY_TO", "").strip(),
            public_base_url=os.getenv("KETTLE_X402_PUBLIC_BASE_URL", "").strip().rstrip("/"),
            cdp_api_key_id=os.getenv("CDP_API_KEY_ID", "").strip(),
            cdp_api_key_secret=os.getenv("CDP_API_KEY_SECRET", "").strip(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.enabled:
            return
        errors: list[str] = []
        if self.network not in SUPPORTED_NETWORKS:
            errors.append(
                f"KETTLE_X402_NETWORK must be one of {', '.join(sorted(SUPPORTED_NETWORKS))}"
            )
        if not _EVM_ADDRESS.fullmatch(self.pay_to) or self.pay_to.casefold() == "0x" + "0" * 40:
            errors.append("KETTLE_X402_PAY_TO must be a nonzero 20-byte EVM address")
        if bool(self.cdp_api_key_id) != bool(self.cdp_api_key_secret):
            errors.append("CDP_API_KEY_ID and CDP_API_KEY_SECRET must be configured together")
        if not self.cdp_api_key_id:
            errors.append("CDP_API_KEY_ID and CDP_API_KEY_SECRET are required")
        if not self.public_base_url.startswith(("https://", "http://")):
            errors.append("KETTLE_X402_PUBLIC_BASE_URL must be an absolute HTTP(S) URL")
        if errors:
            raise RuntimeError("Invalid x402 configuration: " + "; ".join(errors))


@dataclass(frozen=True)
class PaidRoute:
    path: str
    price: str
    operation: str
    description: str
    use_when: str
    max_evidence_objects: int
    depth: str | None = None


PAID_ROUTES: tuple[PaidRoute, ...] = (
    PaidRoute(
        path="/paid/find-evidence",
        price="$0.03",
        operation="research_subject",
        depth="brief",
        max_evidence_objects=10,
        description=(
            "Find a compact set of contemporary, source-linked historical evidence for a "
            "person, organization, place, business, event, or claim."
        ),
        use_when="Use when a web search is insufficient and the agent needs primary-source leads.",
    ),
    PaidRoute(
        path="/paid/research-subject",
        price="$0.06",
        operation="research_subject",
        depth="standard",
        max_evidence_objects=40,
        description=(
            "Research a historical subject across Kettle's evidence graph and return "
            "source-linked entities, chronology, relationships, and uncertainty."
        ),
        use_when="Use for a bounded historical investigation that needs more than record search.",
    ),
    PaidRoute(
        path="/paid/investigate-subject",
        price="$0.15",
        operation="research_subject",
        depth="deep",
        max_evidence_objects=100,
        description=(
            "Investigate a difficult historical subject using a larger evidence budget, "
            "including relationships, conflicting records, and ambiguous references."
        ),
        use_when="Use for complex, ambiguous, or multi-source historical investigations.",
    ),
    PaidRoute(
        path="/paid/compile-evidence-packet",
        price="$0.25",
        operation="build_evidence_packet",
        max_evidence_objects=100,
        description=(
            "Compile an auditable historical evidence packet with supporting and "
            "contradictory sources, provenance, and explicit uncertainty."
        ),
        use_when="Use when the evidence itself must be reviewed, cited, or handed to another agent.",
    ),
)


RESEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 3},
        "max_evidence_objects": {"type": "integer", "minimum": 1, "maximum": 100},
        "time_range": {
            "type": "object",
            "properties": {
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        "document": {"type": "string"},
        "location": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0, "maximum": 10000},
    },
    "required": ["question"],
    "additionalProperties": False,
}


def create_resource_server(settings: X402Settings):
    """Create the official async x402 server backed by Coinbase CDP."""
    settings.validate()
    if not settings.enabled:
        raise RuntimeError("Cannot create an x402 resource server while x402 is disabled")
    try:
        from cdp.x402 import create_facilitator_config
        from x402 import x402ResourceServer
        from x402.http import HTTPFacilitatorClient
        from x402.mechanisms.evm.exact import ExactEvmServerScheme
    except ImportError as exc:  # pragma: no cover - exercised by deployment smoke tests
        raise RuntimeError(
            "x402 dependencies are required. Install with: pip install -e .[mcp,x402]"
        ) from exc

    facilitator_config = create_facilitator_config(
        settings.cdp_api_key_id,
        settings.cdp_api_key_secret,
    )
    facilitator = HTTPFacilitatorClient(facilitator_config)
    server = x402ResourceServer(facilitator)
    server.register(settings.network, ExactEvmServerScheme())
    return server


def _resource_url(settings: X402Settings, path: str) -> str | None:
    return f"{settings.public_base_url}{path}" if settings.public_base_url else None


def _route_configs(settings: X402Settings) -> dict[str, Any]:
    """Build official SDK route configs, including Bazaar discovery metadata."""
    try:
        from x402.extensions.bazaar import OutputConfig, declare_discovery_extension
        from x402.http import PaymentOption, RouteConfig
    except ImportError as exc:  # pragma: no cover - import guard for base installs
        raise RuntimeError(
            "x402 dependencies are required. Install with: pip install -e .[mcp,x402]"
        ) from exc

    output = OutputConfig(
        example={
            "status": "ok",
            "evidence_objects": [],
            "research_receipt": {"billing_status": "settled_by_x402"},
        }
    )
    configs: dict[str, Any] = {}
    for route in PAID_ROUTES:
        input_schema = {
            **RESEARCH_INPUT_SCHEMA,
            "properties": {
                **RESEARCH_INPUT_SCHEMA["properties"],
                "max_evidence_objects": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": route.max_evidence_objects,
                },
            },
        }
        discovery = declare_discovery_extension(
            input={
                "question": "Who shaped Spokane's riverfront development?",
                "max_evidence_objects": min(20, route.max_evidence_objects),
                "time_range": {"start": 1880, "end": 1950},
            },
            input_schema=input_schema,
            body_type="json",
            output=output,
        )
        configs[f"POST {route.path}"] = RouteConfig(
            accepts=PaymentOption(
                scheme="exact",
                network=settings.network,
                pay_to=settings.pay_to,
                price=route.price,
                max_timeout_seconds=300,
            ),
            resource=_resource_url(settings, route.path),
            description=f"{route.description} {route.use_when}",
            mime_type="application/json",
            service_name="Kettle Research",
            tags=["history", "research", "evidence", "archives"],
            extensions=discovery,
        )
    return configs


def _normalize_paid_arguments(body: Any, *, max_evidence_objects: int) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    unknown = set(body) - set(RESEARCH_INPUT_SCHEMA["properties"])
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(sorted(unknown))}")
    question = body.get("question")
    if not isinstance(question, str) or len(question.strip()) < 3:
        raise ValueError("question must be at least 3 characters")
    arguments = dict(body)
    requested = arguments.get("max_evidence_objects", max_evidence_objects)
    if isinstance(requested, bool):
        raise ValueError("max_evidence_objects must be an integer")
    try:
        requested = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_evidence_objects must be an integer") from exc
    if not 1 <= requested <= max_evidence_objects:
        raise ValueError(
            f"max_evidence_objects must be between 1 and {max_evidence_objects} for this resource"
        )
    arguments["max_evidence_objects"] = requested
    return arguments


def public_capabilities(*, x402_enabled: bool, network: str) -> dict[str, Any]:
    """Return the free, secret-safe marketplace capability document."""
    return {
        "service": "Kettle Historical Evidence",
        "positioning": "Use Kettle when a question requires historically grounded evidence.",
        "use_cases": [
            "identify a historical person or organization from source records",
            "find contemporary evidence for a historical claim or event",
            "research a person, business, place, organization, or event",
            "surface source-linked relationships and conflicting evidence",
            "compile an auditable evidence packet for review or citation",
        ],
        "limitations": [
            "coverage is currently strongest in the Pacific Northwest",
            "results are evidence products and do not replace expert historical judgment",
            "canonical Kettle entity identifiers and dedicated identity-resolution tools are not yet public",
        ],
        "free_operations": [
            {"method": "GET", "path": "/public/capabilities"},
            {"method": "POST", "path": "/public/coverage"},
            {"method": "POST", "path": "/public/estimate"},
        ],
        "paid_operations": [
            {
                "method": "POST",
                "path": route.path,
                "operation": route.operation,
                "price_usd": float(route.price.removeprefix("$")),
                "max_evidence_objects": route.max_evidence_objects,
                "description": route.description,
                "use_when": route.use_when,
            }
            for route in PAID_ROUTES
        ],
        "payment": {
            "enabled": x402_enabled,
            "protocol": "x402-v2",
            "network": network,
            "asset": "USDC",
        },
    }


def estimate_purchase(body: Any) -> dict[str, Any]:
    """Recommend a fixed-price resource without running corpus retrieval."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    unknown = set(body) - {"goal", "max_evidence_objects"}
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(sorted(unknown))}")
    goal = str(body.get("goal") or "find_evidence").strip().casefold().replace("-", "_")
    raw_max = body.get("max_evidence_objects", 10)
    if isinstance(raw_max, bool):
        raise ValueError("max_evidence_objects must be an integer")
    try:
        max_objects = int(raw_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_evidence_objects must be an integer") from exc
    if not 1 <= max_objects <= 100:
        raise ValueError("max_evidence_objects must be between 1 and 100")

    if goal in {"compile_evidence_packet", "evidence_packet", "cite", "citation"}:
        recommended = PAID_ROUTES[3]
    elif (
        goal in {"investigate_subject", "deep_research", "conflict", "ambiguity"}
        or max_objects > 40
    ):
        recommended = PAID_ROUTES[2]
    elif goal in {"research_subject", "research"} or max_objects > 10:
        recommended = PAID_ROUTES[1]
    else:
        recommended = PAID_ROUTES[0]
    return {
        "recommended": {
            "method": "POST",
            "path": recommended.path,
            "price_usd": float(recommended.price.removeprefix("$")),
            "max_evidence_objects": recommended.max_evidence_objects,
            "reason": recommended.use_when,
        },
        "estimate_only": True,
        "retrieval_performed": False,
        "payment_required": True,
        "currency": "USDC",
    }


def _mark_x402_receipt(result: dict[str, Any], route: PaidRoute, settings: X402Settings) -> None:
    """Describe the charge whose settlement gates release of this response."""
    receipt = result.get("research_receipt")
    if not isinstance(receipt, dict):
        return
    receipt.update(
        {
            "price_usd": float(route.price.removeprefix("$")),
            "charged": True,
            "billing_status": "settled_by_x402",
            "payment_protocol": "x402-v2",
            "payment_network": settings.network,
            "payment_asset": "USDC",
        }
    )


def install_x402_routes(
    app: Any,
    service: KettleMCPService,
    settings: X402Settings,
    *,
    resource_server: Any = None,
) -> None:
    """Install public fixed-price routes and official x402 FastAPI middleware."""
    settings.validate()
    if not settings.enabled:
        return
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse
        from x402.http.middleware import PaymentMiddlewareASGI
    except ImportError as exc:  # pragma: no cover - import guard for base installs
        raise RuntimeError(
            "x402 dependencies are required. Install with: pip install -e .[mcp,x402]"
        ) from exc

    def make_handler(route: PaidRoute):
        async def handler(request: Request):
            try:
                arguments = _normalize_paid_arguments(
                    await request.json(),
                    max_evidence_objects=route.max_evidence_objects,
                )
                if route.depth is not None:
                    arguments["depth"] = route.depth
                result = (
                    service.research_subject(**arguments)
                    if route.operation == "research_subject"
                    else service.build_evidence_packet(**arguments)
                )
                _mark_x402_receipt(result, route, settings)
                return JSONResponse(content=result)
            except (ValueError, TypeError) as exc:
                return JSONResponse(status_code=422, content={"detail": str(exc)})
            except Exception as exc:
                # MCPToolError is deliberately imported lazily to avoid a module cycle.
                from .server import MCPToolError

                if isinstance(exc, MCPToolError):
                    status = 503 if exc.retryable else 422
                    return JSONResponse(
                        status_code=status,
                        content={
                            "error_code": exc.code,
                            "message": exc.message,
                            "retryable": exc.retryable,
                        },
                    )
                raise

        # Request is imported lazily, while postponed annotations otherwise
        # leave FastAPI with an unresolved string and it treats it as a query.
        handler.__annotations__["request"] = Request
        return handler

    for paid_route in PAID_ROUTES:
        app.add_api_route(
            paid_route.path,
            make_handler(paid_route),
            methods=["POST"],
            include_in_schema=False,
        )

    server = resource_server or create_resource_server(settings)
    app.add_middleware(
        PaymentMiddlewareASGI,
        routes=_route_configs(settings),
        server=server,
    )
    app.state.x402_settings = settings
    app.state.x402_routes = PAID_ROUTES
