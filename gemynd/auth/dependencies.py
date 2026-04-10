"""FastAPI dependencies for authentication and authorisation.

Usage
-----
In any FastAPI route:

    from gemynd.auth.dependencies import (
        require_login, require_admin, require_archivist_or_admin
    )

    @app.get("/something")
    def my_endpoint(user: UserContext = Depends(require_login)):
        ...

Browser requests without a valid session cookie are redirected to
``/auth/login?next=<original-path>`` via a NeedsLoginException that must
be registered on the application:

    from gemynd.auth.dependencies import NeedsLoginException

    @app.exception_handler(NeedsLoginException)
    async def _handler(request, exc):
        return RedirectResponse(url=exc.redirect_url, status_code=303)
"""
from __future__ import annotations

import json
import logging
import os
import time

from fastapi import Cookie, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from .jwt_utils import decode_access_token
from .models import ROLE_PERMITTED_LEVELS, UserContext
from .store import UserStore, verify_password

_LOG = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# Roles that may appear in a valid JWT.  Reject tokens claiming unknown roles
# rather than silently degrading to "public".
_VALID_ROLES: frozenset[str] = frozenset(ROLE_PERMITTED_LEVELS.keys())

# Module-level store singleton — initialised lazily from USERS_DB env var.
_store: UserStore | None = None


def _get_store() -> UserStore:
    global _store
    if _store is None:
        db_path = os.environ.get("USERS_DB", "data/users.db")
        _store = UserStore(db_path)
    return _store


# ---------------------------------------------------------------------------
# Legacy bearer-token support (backward compatibility)
# ---------------------------------------------------------------------------

def _load_token_store() -> dict[str, dict]:
    """Load the legacy GRAPHRAG_API_TOKENS env-var token store."""
    raw = os.environ.get("GRAPHRAG_API_TOKENS", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Custom exception for browser redirect
# ---------------------------------------------------------------------------

class NeedsLoginException(Exception):
    """Raised when a browser request arrives without a valid session.

    Register an exception handler on your FastAPI app:

        @app.exception_handler(NeedsLoginException)
        async def _handler(request, exc):
            return RedirectResponse(url=exc.redirect_url, status_code=303)
    """

    def __init__(self, redirect_url: str) -> None:
        self.redirect_url = redirect_url
        super().__init__(redirect_url)


def _is_browser_request(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    has_auth_header = "authorization" in {k.lower() for k in request.headers}
    return "text/html" in accept and not has_auth_header


def _reject_session(request: Request, detail: str) -> None:
    """Raise the appropriate error for a revoked/invalid session."""
    if _is_browser_request(request):
        raise NeedsLoginException("/auth/login")
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


# ---------------------------------------------------------------------------
# Core dependency
# ---------------------------------------------------------------------------

def require_login(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    access_token: str | None = Cookie(default=None),
) -> UserContext:
    """Return the authenticated UserContext for this request.

    Resolution order:
    1. JWT cookie  →  full UserContext with user_id, email, and DB-verified claims
    2. Bearer token  →  legacy UserContext (is_api_client=True)
    3. Neither  →  redirect (browser) or 401 (API)
    """
    # 1. JWT cookie
    if access_token:
        decoded: dict | None = None
        try:
            decoded = decode_access_token(access_token)
        except (JWTError, Exception) as exc:
            _LOG.warning("JWT cookie decode failed: %s", exc)
            # Fall through to bearer token check.

        if decoded is not None:
            role = decoded.get("role", "")
            if role not in _VALID_ROLES:
                _LOG.warning("JWT token contains unknown role %r — rejecting", role)
                decoded = None

        if decoded is not None:
            user_id = decoded.get("sub", "")
            db_user = _get_store().get_by_id(user_id) if user_id else None

            if db_user is None or not db_user.is_active:
                _reject_session(request, "Session invalidated")

            if db_user.token_version != decoded.get("tkv", -1):
                _reject_session(request, "Session invalidated")

            # Use DB values for role and institution — never trust JWT claims alone.
            return UserContext(
                role=db_user.role,
                institution_id=db_user.institution_id,
                permitted_levels=ROLE_PERMITTED_LEVELS.get(db_user.role, ["public"]),
                user_id=db_user.user_id,
                email=db_user.email,
                is_api_client=False,
            )

    # 2. Bearer token (legacy API clients)
    if credentials is not None:
        token_store = _load_token_store()
        entry = token_store.get(credentials.credentials)
        if entry is not None:
            # Check expiry if the token store entry carries an expires_at timestamp.
            expires_at = entry.get("expires_at")
            if expires_at is not None and expires_at < time.time():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API token has expired",
                )
            role = entry.get("role", "public")
            institution_id = entry.get("institution_id", "")
            client_id = entry.get("client_id")
            return UserContext.from_token_entry(role, institution_id, client_id=client_id)
        # Token present but not found in store.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired API token",
        )

    # 3. Not authenticated
    if _is_browser_request(request):
        next_path = str(request.url.path)
        if request.url.query:
            next_path = f"{next_path}?{request.url.query}"
        raise NeedsLoginException(f"/auth/login?next={next_path}")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Role-gating helpers
# ---------------------------------------------------------------------------

def require_admin(
    user: UserContext = Depends(require_login),
) -> UserContext:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return user


def require_archivist_or_admin(
    user: UserContext = Depends(require_login),
) -> UserContext:
    if user.role not in ("admin", "archivist"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Archivist or Admin role required",
        )
    return user
