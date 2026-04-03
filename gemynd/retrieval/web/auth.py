"""Bearer-token authentication for the GraphRAG retrieval API.

This module is now a thin shim over ``gemynd.auth``.
All call sites in app.py that use ``from .auth import UserContext, require_user``
continue to work without modification.

Legacy bearer-token configuration (GRAPHRAG_API_TOKENS env var) is still
supported for API clients.  Browser access now uses JWT cookie auth provided
by the new auth package.

Roles and their permitted access levels (kept here for reference):
    public          → ["public"]
    staff           → ["public", "staff_only"]
    restricted      → ["public", "staff_only", "restricted"]
    indigenous_admin→ ["public", "staff_only", "restricted", "indigenous_restricted"]
    archivist       → ["public", "staff_only", "restricted"]
    admin           → ["public", "staff_only", "restricted", "indigenous_restricted"]
"""
from __future__ import annotations

# Re-export UserContext and ROLE_PERMITTED_LEVELS from the auth package so that
# existing imports in app.py (``from .auth import UserContext, require_user``)
# continue to resolve without any changes to app.py.
from gemynd.auth.models import ROLE_PERMITTED_LEVELS, UserContext  # noqa: F401

# require_user is an alias for require_login.  Both names work as FastAPI
# Depends() targets.
from gemynd.auth.dependencies import require_login as require_user  # noqa: F401

__all__ = ["UserContext", "ROLE_PERMITTED_LEVELS", "require_user"]
