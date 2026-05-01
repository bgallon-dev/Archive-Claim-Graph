"""Self-contained CSRF middleware with double-submit-cookie pattern.

Why not starlette-csrf: starlette-csrf sets the csrftoken cookie via
Set-Cookie on the safe-method response, but does NOT inject the new
token into the request scope before calling the inner handler.  As a
result, on first visit the template renders the hidden input as empty
(`request.cookies.get("csrftoken")` returns None), the browser stores
the cookie via Set-Cookie, and the very next form submission fails
because the form's csrftoken is empty while the cookie is populated.
This middleware injects the new token into scope's Cookie header so the
handler sees it on the same request that issued it.

Behaviour:
- Bearer-only requests (Authorization: Bearer ... AND no access_token
  cookie) are exempted entirely.  Hybrid auth in this codebase makes
  route-level exemption infeasible.
- Safe methods (GET/HEAD/OPTIONS/TRACE): if no csrftoken cookie, generate
  one, inject into request scope, and set Set-Cookie on response.
- Unsafe methods (POST/PUT/DELETE/PATCH): validate the submitted token
  matches the cookie.  Token may be sent as the X-CSRFToken header or
  as a `csrftoken` form field.

For form-field validation we buffer the request body and replay it to
the inner handler.  Memory cost is the body size — fine for forms and
small uploads, but bear in mind for multi-hundred-MB ingestion uploads.
For those routes the JS-driven path (header-based) is preferable.
"""
from __future__ import annotations

import secrets
from typing import Callable

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


class GemyndCSRFMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        secret: str = "",
        cookie_name: str = "csrftoken",
        cookie_samesite: str = "strict",
        cookie_secure: bool = True,
        cookie_httponly: bool = False,
        header_name: str = "x-csrftoken",
        cookie_path: str = "/",
    ) -> None:
        self.app = app
        self.cookie_name = cookie_name
        self.cookie_samesite = cookie_samesite
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.header_name = header_name.lower()
        self.cookie_path = cookie_path
        # `secret` is accepted for API parity with starlette-csrf but not
        # used.  Random-token double-submit doesn't need HMAC signing for
        # single-domain deployments; the cookie value IS the secret.

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer ") and "access_token" not in request.cookies:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()
        cookie_token = request.cookies.get(self.cookie_name)

        if method in _SAFE_METHODS:
            if cookie_token:
                await self.app(scope, receive, send)
                return
            new_token = secrets.token_urlsafe(32)
            self._inject_cookie_into_scope(scope, new_token)
            await self._send_with_set_cookie(scope, receive, send, new_token)
            return

        await self._validate_unsafe(scope, receive, send, request, cookie_token)

    async def _validate_unsafe(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        request: Request,
        cookie_token: str | None,
    ) -> None:
        # Header path: no body consumed.
        submitted = request.headers.get(self.header_name)
        if submitted is not None:
            if self._tokens_match(cookie_token, submitted):
                await self.app(scope, receive, send)
            else:
                await self._reject(scope, receive, send)
            return

        # Form-field fallback: buffer the body, parse form, replay to handler.
        body = await self._read_body(receive)

        async def parse_receive() -> Message:
            return {"type": "http.request", "body": body, "more_body": False}

        try:
            tmp_request = Request(scope, receive=parse_receive)
            form = await tmp_request.form()
            submitted = form.get(self.cookie_name)
        except Exception:
            submitted = None

        if not self._tokens_match(cookie_token, submitted):
            await self._reject(scope, receive, send)
            return

        await self.app(scope, self._make_replay_receive(body), send)

    @staticmethod
    def _tokens_match(cookie_token: str | None, submitted: object) -> bool:
        if not cookie_token or not submitted:
            return False
        return secrets.compare_digest(str(cookie_token), str(submitted))

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = PlainTextResponse("CSRF token verification failed", status_code=403)
        await response(scope, receive, send)

    @staticmethod
    async def _read_body(receive: Receive) -> bytes:
        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                more_body = message.get("more_body", False)
            elif message["type"] == "http.disconnect":
                more_body = False
        return b"".join(chunks)

    @staticmethod
    def _make_replay_receive(body: bytes) -> Callable[[], "object"]:
        sent = False

        async def replay() -> Message:
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        return replay

    def _inject_cookie_into_scope(self, scope: Scope, token: str) -> None:
        """Append `cookie_name=token` to scope's Cookie header so the inner
        handler's `Request(scope).cookies` sees it on the SAME request.
        Without this, templates render an empty hidden input on first visit
        and the next POST fails because the form value doesn't match the
        cookie the browser just received."""
        headers = list(scope.get("headers", []))
        addition = f"{self.cookie_name}={token}".encode("ascii")
        cookie_idx = None
        for i, (name, _value) in enumerate(headers):
            if name == b"cookie":
                cookie_idx = i
                break
        if cookie_idx is None:
            headers.append((b"cookie", addition))
        else:
            existing = headers[cookie_idx][1]
            headers[cookie_idx] = (b"cookie", existing + b"; " + addition)
        scope["headers"] = headers

    async def _send_with_set_cookie(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        token: str,
    ) -> None:
        cookie_value = self._format_cookie(token)

        async def wrapped_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append("set-cookie", cookie_value)
            await send(message)

        await self.app(scope, receive, wrapped_send)

    def _format_cookie(self, token: str) -> str:
        parts = [f"{self.cookie_name}={token}", f"Path={self.cookie_path}"]
        parts.append(f"SameSite={self.cookie_samesite.capitalize()}")
        if self.cookie_secure:
            parts.append("Secure")
        if self.cookie_httponly:
            parts.append("HttpOnly")
        return "; ".join(parts)
