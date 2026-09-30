"""Machine-facing MCP tools for the Kettle research service.

The package deliberately keeps the MCP transport thin.  The tool service can
be used directly in tests, mounted into the existing FastAPI application, or
run as a small standalone HTTP endpoint.
"""

__all__ = ["KettleMCPService", "create_mcp_app"]


def __getattr__(name: str):
    if name in __all__:
        from .server import KettleMCPService, create_mcp_app

        return {"KettleMCPService": KettleMCPService, "create_mcp_app": create_mcp_app}[name]
    raise AttributeError(name)
