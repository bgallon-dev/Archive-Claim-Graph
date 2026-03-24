"""Auth router: login, logout, /me, and admin user-management endpoints.

Mount this on both the retrieval and review apps:

    from graphrag_pipeline.auth.router import create_auth_router
    app.include_router(create_auth_router(), prefix="/auth")

Login page will be available at ``/auth/login``.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr, Field
from typing import Literal

from .dependencies import NeedsLoginException, require_admin, require_login
from .jwt_utils import create_access_token, _expire_hours
from .models import UserContext
from .setup import generate_and_write_secret, is_setup_needed, mark_setup_done
from .store import UserStore, verify_password

# ---------------------------------------------------------------------------
# Login page HTML (inline, following the pattern of the existing apps)
# ---------------------------------------------------------------------------

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sign in</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: #f5f5f5;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
    }}
    .card {{
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 2px 12px rgba(0,0,0,.1);
      padding: 2rem 2.5rem;
      width: 100%;
      max-width: 380px;
    }}
    h1 {{ font-size: 1.25rem; margin: 0 0 1.5rem; color: #111; }}
    label {{ display: block; font-size: .875rem; color: #444; margin-bottom: .25rem; }}
    input[type=email], input[type=password] {{
      width: 100%;
      padding: .5rem .75rem;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 1rem;
      margin-bottom: 1rem;
    }}
    input:focus {{ outline: 2px solid #3b82f6; border-color: transparent; }}
    button {{
      width: 100%;
      padding: .6rem;
      background: #1d4ed8;
      color: #fff;
      border: none;
      border-radius: 4px;
      font-size: 1rem;
      cursor: pointer;
    }}
    button:hover {{ background: #1e40af; }}
    .error {{
      background: #fee2e2;
      color: #b91c1c;
      border-radius: 4px;
      padding: .5rem .75rem;
      font-size: .875rem;
      margin-bottom: 1rem;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>GraphRAG &mdash; Sign in</h1>
    {error_block}
    <form method="post" action="/auth/login">
      <input type="hidden" name="next" value="{next}">
      <label for="email">Email</label>
      <input type="email" id="email" name="email" required autofocus
             autocomplete="username" value="{email_prefill}">
      <label for="password">Password</label>
      <input type="password" id="password" name="password" required
             autocomplete="current-password">
      <button type="submit">Sign in</button>
    </form>
  </div>
</body>
</html>"""

_ERROR_BLOCK = '<div class="error">{message}</div>'


def _login_page(
    error: str = "",
    next_url: str = "/",
    email_prefill: str = "",
) -> str:
    error_block = _ERROR_BLOCK.format(message=error) if error else ""
    return _LOGIN_HTML.format(
        error_block=error_block,
        next=next_url,
        email_prefill=email_prefill,
    )


def _safe_next(next_url: str) -> str:
    """Prevent open-redirect: only allow relative paths."""
    if next_url and next_url.startswith("/") and "://" not in next_url:
        return next_url
    return "/"


# ---------------------------------------------------------------------------
# First-run setup page HTML
# ---------------------------------------------------------------------------

_SETUP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GraphRAG &mdash; First-time Setup</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: #f0f4ff;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
    }}
    .card {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 4px 20px rgba(0,0,0,.12);
      padding: 2.5rem 3rem;
      width: 100%;
      max-width: 440px;
    }}
    .badge {{
      display: inline-block;
      background: #dbeafe;
      color: #1d4ed8;
      font-size: .75rem;
      font-weight: 600;
      letter-spacing: .05em;
      text-transform: uppercase;
      padding: .2rem .6rem;
      border-radius: 99px;
      margin-bottom: 1rem;
    }}
    h1 {{ font-size: 1.4rem; margin: 0 0 .4rem; color: #111; }}
    p.subtitle {{ color: #6b7280; font-size: .9rem; margin: 0 0 1.5rem; }}
    label {{ display: block; font-size: .875rem; color: #374151; margin-bottom: .25rem; }}
    input[type=email], input[type=password] {{
      width: 100%;
      padding: .5rem .75rem;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      font-size: 1rem;
      margin-bottom: 1rem;
    }}
    input:focus {{ outline: 2px solid #3b82f6; border-color: transparent; }}
    .hint {{ font-size: .78rem; color: #9ca3af; margin-top: -.75rem; margin-bottom: 1rem; }}
    button {{
      width: 100%;
      padding: .65rem;
      background: #1d4ed8;
      color: #fff;
      border: none;
      border-radius: 6px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      margin-top: .25rem;
    }}
    button:hover {{ background: #1e40af; }}
    .error {{
      background: #fee2e2;
      color: #b91c1c;
      border-radius: 6px;
      padding: .6rem .85rem;
      font-size: .875rem;
      margin-bottom: 1rem;
    }}
    .divider {{ border: none; border-top: 1px solid #f3f4f6; margin: 1.5rem 0 1rem; }}
    .note {{ font-size: .8rem; color: #6b7280; }}
    .note code {{ background: #f3f4f6; padding: .1rem .3rem; border-radius: 3px; font-size: .78rem; }}
  </style>
</head>
<body>
  <div class="card">
    <span class="badge">First-time setup</span>
    <h1>Create your admin account</h1>
    <p class="subtitle">
      This runs once. A secret key will be generated automatically
      and saved to your <code>.env</code> file.
    </p>
    {error_block}
    <form method="post" action="/auth/setup">
      <label for="email">Admin email</label>
      <input type="email" id="email" name="email" required autofocus
             autocomplete="username" value="{email_prefill}">
      <label for="password">Password</label>
      <input type="password" id="password" name="password" required
             autocomplete="new-password" minlength="12">
      <p class="hint">Minimum 12 characters</p>
      <label for="confirm">Confirm password</label>
      <input type="password" id="confirm" name="confirm" required
             autocomplete="new-password" minlength="12">
      <button type="submit">Create admin account &amp; continue</button>
    </form>
    <hr class="divider">
    <p class="note">
      You can create additional users (Archivists, Read-only) from
      <code>Admin &rarr; Users</code> after signing in.
    </p>
  </div>
</body>
</html>"""

_SETUP_DONE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="3;url=/auth/login">
  <title>Setup complete</title>
  <style>
    body {{ font-family: system-ui, sans-serif; display: flex; align-items: center;
           justify-content: center; min-height: 100vh; margin: 0; background: #f0fdf4; }}
    .card {{ background: #fff; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,.1);
             padding: 2.5rem 3rem; max-width: 400px; text-align: center; }}
    h1 {{ color: #15803d; font-size: 1.4rem; }}
    p {{ color: #6b7280; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Setup complete!</h1>
    <p>Your admin account has been created.<br>Redirecting to sign in&hellip;</p>
  </div>
</body>
</html>"""


def _setup_page(error: str = "", email_prefill: str = "") -> str:
    error_block = _ERROR_BLOCK.format(message=error) if error else ""
    return _SETUP_HTML.format(error_block=error_block, email_prefill=email_prefill)


# ---------------------------------------------------------------------------
# Pydantic models for user management API
# ---------------------------------------------------------------------------

class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=12)
    role: Literal["admin", "archivist", "readonly"]
    institution_id: str = "turnbull"


class UserResponse(BaseModel):
    user_id: str
    email: str
    role: str
    institution_id: str
    created_at: str
    is_active: bool


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=12)


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_auth_router(users_db_path: str = "") -> APIRouter:
    """Return a configured APIRouter.  Mount at prefix ``/auth``."""
    router = APIRouter(tags=["auth"])
    db_path = users_db_path or os.environ.get("USERS_DB", "data/users.db")
    store = UserStore(db_path)

    # ------------------------------------------------------------------
    # First-run setup
    # ------------------------------------------------------------------

    @router.get("/setup", response_class=HTMLResponse, include_in_schema=False)
    def get_setup() -> HTMLResponse:
        if not is_setup_needed(db_path):
            return RedirectResponse(url="/auth/login", status_code=303)
        return HTMLResponse(_setup_page())

    @router.post("/setup", include_in_schema=False)
    async def post_setup(
        email: str = Form(...),
        password: str = Form(...),
        confirm: str = Form(...),
    ) -> Response:
        if not is_setup_needed(db_path):
            return RedirectResponse(url="/auth/login", status_code=303)

        email = email.strip().lower()

        if len(password) < 12:
            return HTMLResponse(
                _setup_page(error="Password must be at least 12 characters.", email_prefill=email),
                status_code=400,
            )
        if password != confirm:
            return HTMLResponse(
                _setup_page(error="Passwords do not match.", email_prefill=email),
                status_code=400,
            )

        # Generate and persist the JWT secret key before creating the user.
        if not __import__("os").environ.get("JWT_SECRET_KEY"):
            generate_and_write_secret()

        try:
            store.create_user(email, password, role="admin")
        except ValueError as e:
            return HTMLResponse(_setup_page(error=str(e), email_prefill=email), status_code=400)

        mark_setup_done()
        return HTMLResponse(_SETUP_DONE_HTML)

    # ------------------------------------------------------------------
    # Login / logout
    # ------------------------------------------------------------------

    @router.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def get_login(
        request: Request,
        next: str = "/",
        access_token: str | None = Cookie(default=None),
    ) -> HTMLResponse:
        # Already logged in → redirect immediately
        if access_token:
            from .jwt_utils import decode_access_token
            from jose import JWTError
            try:
                decode_access_token(access_token)
                return RedirectResponse(url=_safe_next(next), status_code=303)
            except JWTError:
                pass
        return HTMLResponse(_login_page(next_url=_safe_next(next)))

    @router.post("/login", include_in_schema=False)
    async def post_login(
        email: str = Form(...),
        password: str = Form(...),
        next: str = Form(default="/"),
    ) -> Response:
        user = store.get_by_email(email)
        if user is None or not user.is_active or not verify_password(password, user.hashed_password):
            return HTMLResponse(
                _login_page(error="Invalid email or password.", next_url=_safe_next(next), email_prefill=email),
                status_code=401,
            )
        token = create_access_token(user.user_id, user.email, user.role, user.institution_id)
        resp = RedirectResponse(url=_safe_next(next), status_code=303)
        cookie_secure = os.environ.get("COOKIE_SECURE", "true").lower() != "false"
        resp.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=cookie_secure,
            samesite="lax",
            max_age=_expire_hours() * 3600,
        )
        return resp

    @router.post("/logout", include_in_schema=False)
    async def post_logout() -> Response:
        resp = RedirectResponse(url="/auth/login", status_code=303)
        resp.delete_cookie("access_token")
        return resp

    # ------------------------------------------------------------------
    # Current user info
    # ------------------------------------------------------------------

    @router.get("/me")
    def me(user: UserContext = Depends(require_login)) -> dict:
        return {
            "user_id": user.user_id,
            "email": user.email,
            "role": user.role,
            "institution_id": user.institution_id,
            "is_api_client": user.is_api_client,
        }

    # ------------------------------------------------------------------
    # Admin user management
    # ------------------------------------------------------------------

    @router.post("/users", response_model=UserResponse, status_code=201)
    def create_user(
        body: CreateUserRequest,
        _admin: UserContext = Depends(require_admin),
    ) -> UserResponse:
        try:
            user = store.create_user(body.email, body.password, body.role, body.institution_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return UserResponse(
            user_id=user.user_id,
            email=user.email,
            role=user.role,
            institution_id=user.institution_id,
            created_at=user.created_at,
            is_active=user.is_active,
        )

    @router.get("/users", response_model=list[UserResponse])
    def list_users(
        _admin: UserContext = Depends(require_admin),
    ) -> list[UserResponse]:
        users = store.list_users()
        return [
            UserResponse(
                user_id=u.user_id,
                email=u.email,
                role=u.role,
                institution_id=u.institution_id,
                created_at=u.created_at,
                is_active=u.is_active,
            )
            for u in users
        ]

    @router.post("/users/{user_id}/deactivate", response_model=UserResponse)
    def deactivate_user(
        user_id: str,
        admin: UserContext = Depends(require_admin),
    ) -> UserResponse:
        if admin.user_id and admin.user_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account.")
        if not store.deactivate_user(user_id):
            raise HTTPException(status_code=404, detail="User not found.")
        user = store.get_by_id(user_id)
        return UserResponse(
            user_id=user.user_id, email=user.email, role=user.role,
            institution_id=user.institution_id, created_at=user.created_at,
            is_active=user.is_active,
        )

    @router.post("/users/{user_id}/activate", response_model=UserResponse)
    def activate_user(
        user_id: str,
        _admin: UserContext = Depends(require_admin),
    ) -> UserResponse:
        if not store.activate_user(user_id):
            raise HTTPException(status_code=404, detail="User not found.")
        user = store.get_by_id(user_id)
        return UserResponse(
            user_id=user.user_id, email=user.email, role=user.role,
            institution_id=user.institution_id, created_at=user.created_at,
            is_active=user.is_active,
        )

    @router.post("/users/{user_id}/reset-password", status_code=204)
    def reset_password(
        user_id: str,
        body: ResetPasswordRequest,
        _admin: UserContext = Depends(require_admin),
    ) -> None:
        if not store.get_by_id(user_id):
            raise HTTPException(status_code=404, detail="User not found.")
        store.change_password(user_id, body.new_password)

    return router
