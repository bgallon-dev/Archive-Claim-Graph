"""Shared data models for the authentication system."""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Role → permitted document access levels
# ---------------------------------------------------------------------------

ROLE_PERMITTED_LEVELS: dict[str, list[str]] = {
    # UI roles (username/password login)
    "admin":     ["public", "staff_only", "restricted", "indigenous_restricted"],
    "archivist": ["public", "staff_only", "restricted"],
    "readonly":  ["public"],
    # Legacy bearer-token roles (backward compatibility for API clients)
    "public":           ["public"],
    "staff":            ["public", "staff_only"],
    "restricted":       ["public", "staff_only", "restricted"],
    "indigenous_admin": ["public", "staff_only", "restricted", "indigenous_restricted"],
}

_DEFAULT_INSTITUTION = ""


# ---------------------------------------------------------------------------
# User — row from users.db
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class User:
    user_id: str
    email: str
    hashed_password: str
    role: str
    institution_id: str
    created_at: str
    is_active: bool
    token_version: int = 0


# ---------------------------------------------------------------------------
# UserContext — attached to every authenticated request
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class UserContext:
    # Fields that existed before this auth package — all call sites in
    # retrieval/web/app.py depend on these exact names.
    role: str
    institution_id: str
    permitted_levels: list[str]

    # New fields: None for legacy bearer-token callers.
    user_id: str | None = None
    email: str | None = None
    is_api_client: bool = False
    # Per-client identifier for API token holders (improves audit trail granularity).
    client_id: str | None = None

    @classmethod
    def from_user(cls, user: User) -> "UserContext":
        return cls(
            role=user.role,
            institution_id=user.institution_id,
            permitted_levels=ROLE_PERMITTED_LEVELS.get(user.role, ["public"]),
            user_id=user.user_id,
            email=user.email,
            is_api_client=False,
        )

    @classmethod
    def from_token_entry(
        cls, role: str, institution_id: str, *, client_id: str | None = None
    ) -> "UserContext":
        """Build a UserContext from a legacy bearer-token store entry."""
        return cls(
            role=role,
            institution_id=institution_id,
            permitted_levels=ROLE_PERMITTED_LEVELS.get(role, ["public"]),
            user_id=None,
            email=None,
            is_api_client=True,
            client_id=client_id,
        )

    @property
    def identity(self) -> str:
        """Human-readable identity string for audit logs and reviewer fields."""
        if self.email:
            return self.email
        if self.is_api_client:
            actor = self.client_id or self.role
            return f"{actor}@api"
        return self.role
