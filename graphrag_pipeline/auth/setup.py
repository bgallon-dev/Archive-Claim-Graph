"""First-run setup detection and secret-key provisioning.

Provides:
  is_setup_needed(users_db_path)  — cheap check called on every request by the guard middleware
  generate_and_write_secret(env_path)  — called once during the setup POST handler

The check is cached after setup is confirmed complete so it adds no measurable
overhead to normal operation.
"""
from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

# Cached once setup is confirmed done — avoids DB round-trips on every request.
_setup_done: bool = False


def is_setup_needed(users_db_path: str = "data/users.db") -> bool:
    """Return True if the app has not been set up yet.

    Setup is needed when either:
      • JWT_SECRET_KEY env var is empty/missing, OR
      • The users database contains no active users

    Once setup is complete the result is cached in-process so subsequent
    calls are O(1) env-var lookups.
    """
    global _setup_done
    if _setup_done:
        return False

    if not os.environ.get("JWT_SECRET_KEY"):
        return True

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


def generate_and_write_secret(env_path: str = ".env") -> str:
    """Generate a 64-char hex JWT secret, persist it to *env_path*, and set os.environ.

    Updates the existing ``JWT_SECRET_KEY=`` line in the file. If the file does
    not exist or the key is not present, the key=value pair is appended.

    Returns the generated secret string.
    """
    secret = secrets.token_hex(32)  # 256 bits of entropy
    _write_env_key(env_path, "JWT_SECRET_KEY", secret)
    os.environ["JWT_SECRET_KEY"] = secret
    return secret


def _write_env_key(env_path: str, key: str, value: str) -> None:
    """Update or append *key*=*value* in an .env file."""
    path = Path(env_path)
    if path.exists():
        content = path.read_text(encoding="utf-8")
        pattern = rf"^{re.escape(key)}=.*$"
        replacement = f"{key}={value}"
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if new_content == content:
            # Key not found — append
            new_content = content.rstrip("\n") + f"\n{key}={value}\n"
        path.write_text(new_content, encoding="utf-8")
    else:
        path.write_text(f"{key}={value}\n", encoding="utf-8")
