"""JWT creation and verification for the GraphRAG auth system.

Reads configuration from environment variables:

    JWT_SECRET_KEY   — required, min 32-char hex/base64 string.
                       Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    JWT_EXPIRE_HOURS — optional, defaults to 24.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

ALGORITHM = "HS256"


def _get_secret() -> str:
    secret = os.environ.get("JWT_SECRET_KEY", "")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set. Generate with:\n"
            "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
            "Then set it as an environment variable before starting the server."
        )
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters.")
    return secret


def _expire_hours() -> int:
    try:
        return int(os.environ.get("JWT_EXPIRE_HOURS", "24"))
    except ValueError:
        return 24


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    institution_id: str,
    token_version: int = 0,
) -> str:
    """Return a signed JWT string."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub":   user_id,
        "email": email,
        "role":  role,
        "inst":  institution_id,
        "tkv":   token_version,
        "iat":   now,
        "exp":   now + timedelta(hours=_expire_hours()),
    }
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT.

    Raises
    ------
    jose.JWTError
        If the token is invalid, expired, or the signature does not match.
    """
    return jwt.decode(
        token,
        _get_secret(),
        algorithms=["HS256"],
        options={
            "verify_exp": True,
            "verify_iat": True,
            "verify_nbf": True,
            "leeway": 0,
        },
    )
