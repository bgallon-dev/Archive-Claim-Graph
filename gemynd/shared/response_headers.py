"""Response-header middlewares shared across the FastAPI apps."""
from __future__ import annotations

from starlette.requests import Request


async def authenticated_no_store(request: Request, call_next):
    """Set no-cache headers on responses to authenticated requests.

    Closes two vectors at once:
      - Back-button-after-logout: browsers won't show the cached page.
      - Future reverse-proxy caches: Vary: Cookie, Authorization tells any
        intermediary that the response varies by credential.

    Anonymous responses pass through untouched so any future public
    endpoint remains CDN-cacheable.
    """
    response = await call_next(request)
    has_cookie_auth = "access_token" in request.cookies
    has_bearer = request.headers.get("authorization", "").lower().startswith("bearer ")
    if has_cookie_auth or has_bearer:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Vary"] = "Cookie, Authorization"
    return response
