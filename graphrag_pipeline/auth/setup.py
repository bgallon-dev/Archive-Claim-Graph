"""First-run setup detection and per-process setup token.

JWT_SECRET_KEY MUST be set as an environment variable before starting the
server.  The application will not issue tokens without it.  Generate one with:

    python -c "import secrets; print(secrets.token_hex(32))"

This module handles only:
  - detecting whether an admin user exists (is_setup_needed)
  - generating a per-process token printed to stderr so only the operator
    with log access can complete the first-run web setup form
"""
from __future__ import annotations

import os
import secrets
import sys

# Cached once setup is confirmed done — avoids DB round-trips on every request.
_setup_done: bool = False

# One-time setup token generated at import.  Printed to stderr on first visit
# to GET /setup so that only the operator (who has log access) can complete
# the first-run form.
_SETUP_TOKEN: str = secrets.token_urlsafe(32)
_token_printed: bool = False


def ensure_setup_token_printed() -> None:
    """Print the setup token to stderr once, on first call."""
    global _token_printed
    if _token_printed:
        return
    _token_printed = True
    print(
        f"\n[GraphRAG] First-run setup required.\n"
        f"[GraphRAG] Setup token: {_SETUP_TOKEN}\n"
        f"[GraphRAG] Enter this token in the setup form to create your admin account.\n",
        file=sys.stderr,
        flush=True,
    )


def get_setup_token() -> str:
    """Return the per-process setup token."""
    return _SETUP_TOKEN


def is_setup_needed(users_db_path: str = "data/users.db") -> bool:
    """Return True if no users exist in the database yet.

    Once setup is complete the result is cached in-process so subsequent
    calls are O(1) without a DB round-trip.
    """
    global _setup_done
    if _setup_done:
        return False

    try:
        from .store import UserStore
        store = UserStore(users_db_path)
        if store.list_users():
            _setup_done = True
            return False
    except Exception:
        pass

    return True


def mark_setup_done() -> None:
    """Call after a successful setup POST to flip the in-process cache."""
    global _setup_done
    _setup_done = True
