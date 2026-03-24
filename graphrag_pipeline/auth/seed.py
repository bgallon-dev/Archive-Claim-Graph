"""Bootstrap script to create the first admin user.

Usage
-----
Interactive (recommended — password not visible in shell history):

    python -m graphrag_pipeline.auth.seed --email admin@example.com

Non-interactive (CI / Docker entrypoint):

    python -m graphrag_pipeline.auth.seed --email admin@example.com --password "s3cur3P4ssw0rd!"

Also registered as a CLI entry point ``graphrag-create-admin`` in pyproject.toml.
"""
from __future__ import annotations

import sys


def seed_admin(
    email: str,
    password: str,
    institution_id: str = "turnbull",
    db_path: str = "data/users.db",
):
    """Create the first admin user. Idempotent if the email already exists as admin.

    Raises
    ------
    ValueError
        If the email already exists with a non-admin role, or if the password
        is shorter than 12 characters.
    """
    from .store import UserStore

    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters.")

    store = UserStore(db_path)
    existing = store.get_by_email(email)
    if existing:
        if existing.role == "admin":
            print(f"Admin user {email!r} already exists — no changes made.")
            return existing
        raise ValueError(
            f"User {email!r} already exists with role={existing.role!r}. "
            "Use the admin UI to change their role."
        )

    user = store.create_user(email, password, role="admin", institution_id=institution_id)
    print(f"Created admin user: {user.email} (id={user.user_id})")
    return user


def main() -> int:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(
        description="Seed the first admin user into the GraphRAG users database."
    )
    parser.add_argument("--email", required=True, help="Admin user email address")
    parser.add_argument(
        "--password",
        default=None,
        help="Password (min 12 chars). If omitted, prompted interactively.",
    )
    parser.add_argument(
        "--institution-id",
        default="turnbull",
        help="Institution ID (default: turnbull)",
    )
    parser.add_argument(
        "--db",
        default="data/users.db",
        help="Path to users SQLite database (default: data/users.db)",
    )
    args = parser.parse_args()

    password = args.password
    if not password:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Error: passwords do not match.", file=sys.stderr)
            return 1

    try:
        seed_admin(args.email, password, args.institution_id, args.db)
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
