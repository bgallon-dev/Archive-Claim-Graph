"""SQLite-backed user store with bcrypt password hashing.

Follows the same pattern as WriteAuditLogger: synchronous sqlite3,
check_same_thread=False, context-manager connections, uuid4 primary keys.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import bcrypt as _bcrypt

from .models import User

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user (
    user_id         TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    hashed_password TEXT NOT NULL,
    role            TEXT NOT NULL CHECK(role IN ('admin','archivist','readonly')),
    institution_id  TEXT NOT NULL DEFAULT 'turnbull',
    created_at      TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1,
    token_version   INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS user_email_lower ON user (lower(email));
CREATE INDEX IF NOT EXISTS user_role ON user (role);
"""

# Migration for databases created before token_version was added.
_MIGRATE_TOKEN_VERSION = (
    "ALTER TABLE user ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0"
)

_VALID_ROLES = {"admin", "archivist", "readonly"}


# Bcrypt silently discards bytes beyond position 72.  We truncate explicitly
# so the behaviour is documented and consistent rather than surprising.
_BCRYPT_MAX_BYTES = 72


def _prep(plain: str) -> bytes:
    return plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(_prep(plain), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(_prep(plain), hashed.encode("utf-8"))


class UserStore:
    """CRUD operations against the users SQLite database."""

    def __init__(self, db_path: str | Path = "data/users.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Migration: add token_version to databases created before this column existed.
            try:
                conn.execute(_MIGRATE_TOKEN_VERSION)
            except sqlite3.OperationalError:
                pass  # Column already exists

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_user(self, row: sqlite3.Row) -> User:
        return User(
            user_id=row["user_id"],
            email=row["email"],
            hashed_password=row["hashed_password"],
            role=row["role"],
            institution_id=row["institution_id"],
            created_at=row["created_at"],
            is_active=bool(row["is_active"]),
            token_version=int(row["token_version"]),
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create_user(
        self,
        email: str,
        password: str,
        role: str,
        institution_id: str = "turnbull",
    ) -> User:
        """Hash *password* and insert a new user row.

        Raises
        ------
        ValueError
            If *email* already exists or *role* is not valid.
        """
        email = email.strip().lower()
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}. Must be one of {sorted(_VALID_ROLES)}")
        user_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        hashed = hash_password(password)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO user
                        (user_id, email, hashed_password, role, institution_id, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (user_id, email, hashed, role, institution_id, created_at),
                )
        except sqlite3.IntegrityError:
            raise ValueError(f"A user with email {email!r} already exists.")
        return User(user_id, email, hashed, role, institution_id, created_at, True)

    def deactivate_user(self, user_id: str) -> bool:
        # Bump token_version atomically so any existing JWTs are invalidated immediately.
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE user SET is_active = 0, token_version = token_version + 1 WHERE user_id = ?",
                (user_id,),
            )
            return cur.rowcount > 0

    def activate_user(self, user_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE user SET is_active = 1 WHERE user_id = ?", (user_id,)
            )
            return cur.rowcount > 0

    def change_password(self, user_id: str, new_password: str) -> None:
        hashed = hash_password(new_password)
        # Bump token_version so all existing sessions are invalidated on password change.
        with self._connect() as conn:
            conn.execute(
                "UPDATE user SET hashed_password = ?, token_version = token_version + 1 WHERE user_id = ?",
                (hashed, user_id),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_by_email(self, email: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user WHERE lower(email) = lower(?)", (email.strip(),)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_by_id(self, user_id: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self, institution_id: str | None = None) -> list[User]:
        with self._connect() as conn:
            if institution_id:
                rows = conn.execute(
                    "SELECT * FROM user WHERE institution_id = ? ORDER BY created_at",
                    (institution_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM user ORDER BY created_at"
                ).fetchall()
        return [self._row_to_user(r) for r in rows]
