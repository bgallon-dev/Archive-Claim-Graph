"""Auth router: login, logout, /me, and admin user-management endpoints.

Mount this on both the retrieval and review apps:

    from graphrag_pipeline.auth.router import create_auth_router
    app.include_router(create_auth_router(), prefix="/auth")

Login page will be available at ``/auth/login``.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, Field
from typing import Literal

from .dependencies import NeedsLoginException, require_admin, require_login
from .jwt_utils import create_access_token, _expire_hours
from .models import UserContext
from .setup import ensure_setup_token_printed, get_setup_token, is_setup_needed, mark_setup_done
from .store import UserStore, verify_password

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_SHARED_TEMPLATES_DIR = Path(__file__).parent.parent / "shared_templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
_templates.env.loader = __import__("jinja2").FileSystemLoader(
    [str(_TEMPLATES_DIR), str(_SHARED_TEMPLATES_DIR)]
)


def _safe_next(next_url: str) -> str:
    """Prevent open-redirect: only allow relative paths."""
    if next_url and next_url.startswith("/") and "://" not in next_url:
        return next_url
    return "/"


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
    def get_setup(request: Request) -> Response:
        if not is_setup_needed(db_path):
            return RedirectResponse(url="/auth/login", status_code=303)
        ensure_setup_token_printed()
        return _templates.TemplateResponse("setup.html", {"request": request, "error": "", "email_prefill": ""})

    @router.post("/setup", include_in_schema=False)
    async def post_setup(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        confirm: str = Form(...),
        setup_token: str = Form(...),
    ) -> Response:
        # Always re-check the database — never rely on in-process cache here.
        if not is_setup_needed(db_path):
            return RedirectResponse(url="/auth/login", status_code=303)

        import secrets as _sec
        if not _sec.compare_digest(
            setup_token.encode("utf-8"),
            get_setup_token().encode("utf-8"),
        ):
            return _templates.TemplateResponse(
                "setup.html",
                {"request": request, "error": "Invalid setup token. Check the server logs.", "email_prefill": ""},
                status_code=403,
            )

        email = email.strip().lower()

        if len(password) < 12:
            return _templates.TemplateResponse(
                "setup.html",
                {"request": request, "error": "Password must be at least 12 characters.", "email_prefill": email},
                status_code=400,
            )
        if password != confirm:
            return _templates.TemplateResponse(
                "setup.html",
                {"request": request, "error": "Passwords do not match.", "email_prefill": email},
                status_code=400,
            )

        try:
            store.create_user(email, password, role="admin")
        except ValueError as e:
            return _templates.TemplateResponse(
                "setup.html",
                {"request": request, "error": str(e), "email_prefill": email},
                status_code=400,
            )

        mark_setup_done()
        return _templates.TemplateResponse("setup_done.html", {"request": request})

    # ------------------------------------------------------------------
    # Login / logout
    # ------------------------------------------------------------------

    @router.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def get_login(
        request: Request,
        next: str = "/",
        access_token: str | None = Cookie(default=None),
    ) -> Response:
        # Already logged in → redirect immediately
        if access_token:
            from .jwt_utils import decode_access_token
            from jose import JWTError
            try:
                decode_access_token(access_token)
                return RedirectResponse(url=_safe_next(next), status_code=303)
            except JWTError:
                pass
        return _templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "", "next": _safe_next(next), "email_prefill": ""},
        )

    @router.post("/login", include_in_schema=False)
    async def post_login(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        next: str = Form(default="/"),
    ) -> Response:
        user = store.get_by_email(email)
        if user is None or not user.is_active or not verify_password(password, user.hashed_password):
            return _templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid email or password.", "next": _safe_next(next), "email_prefill": email},
                status_code=401,
            )
        token = create_access_token(
            user.user_id, user.email, user.role, user.institution_id,
            token_version=user.token_version,
        )
        resp = RedirectResponse(url=_safe_next(next), status_code=303)
        # Secure cookies are ON by default (the safe production choice).
        # Set COOKIE_SECURE=false only in local dev/test over plain HTTP.
        # The != "false" pattern reads: "secure unless explicitly disabled".
        cookie_secure = os.environ.get("COOKIE_SECURE", "true").lower() != "false"
        resp.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=cookie_secure,
            samesite="strict",
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
        except ValueError:
            # Return a generic message to avoid confirming whether an email
            # address is already registered (information disclosure).
            raise HTTPException(
                status_code=400,
                detail="Could not create user. Verify that the email address is valid and the role is correct.",
            )
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
