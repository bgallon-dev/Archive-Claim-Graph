"""First-run setup detection and file-backed setup token.

JWT_SECRET_KEY MUST be set as an environment variable before starting the
server.  The application will not issue tokens without it.  Generate one with:

    python -c "import secrets; print(secrets.token_hex(32))"

This module handles only:
  - detecting whether an admin user exists (is_setup_needed)
  - issuing a one-time setup token persisted to disk so it is shared across
    uvicorn workers and survives restarts; printed to stderr on creation so
    only the operator with log access can complete the first-run web form
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path


def _token_path() -> Path:
    return Path(os.environ.get("SETUP_TOKEN_PATH", "data/setup-token"))


def _print_banner(token: str) -> None:
    print(
        f"\n[GraphRAG] First-run setup required.\n"
        f"[GraphRAG] Setup token: {token}\n"
        f"[GraphRAG] Enter this token in the setup form to create your admin account.\n",
        file=sys.stderr,
        flush=True,
    )


def get_setup_token() -> str:
    """Return the on-disk setup token, generating it on first call.

    Race-safe across uvicorn workers: the first worker to win the
    O_CREAT|O_EXCL race writes the token and prints the banner; the loser
    reads the same file and returns the same value.
    """
    path = _token_path()
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    new_token = secrets.token_urlsafe(32)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, new_token.encode("utf-8"))
        finally:
            os.close(fd)
        _print_banner(new_token)
        return new_token
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()


def ensure_setup_token_printed() -> None:
    """Backward-compatible shim: ensures the token exists (and is printed
    on first creation).  Token printing is now driven by file creation in
    get_setup_token(), so this is just a touch."""
    get_setup_token()


def is_setup_needed(users_db_path: str = "data/users.db") -> bool:
    """Return True if no users exist in the database yet.

    Always queries the DB — no in-process cache, so workers can never
    disagree about setup state.  The cost is one tiny SELECT per call,
    well below the bug surface of cached state across workers.
    """
    try:
        from .store import UserStore
        store = UserStore(users_db_path)
        if store.list_users():
            return False
    except Exception:
        pass
    return True


def mark_setup_done() -> None:
    """Call after a successful setup POST to delete the on-disk token."""
    try:
        _token_path().unlink(missing_ok=True)
    except OSError:
        # Best-effort: if the file is locked or unlinkable, the next worker
        # restart picks up is_setup_needed=False from the DB and skips setup
        # anyway.  We don't want to crash the response over cleanup.
        pass
