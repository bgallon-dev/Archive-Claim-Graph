"""Create or reset the read-only account used for the museum demo.

The demo account is deliberately ``readonly``:

* ``ROLE_PERMITTED_LEVELS['readonly'] == ['public']``, so the Cypher access
  gate (``c.access_level IN $permitted_levels``) filters out ``staff_only``,
  ``restricted``, and ``indigenous_restricted`` claims. A kiosk anyone can
  walk up to must never hold a credential that clears tribal-consultation
  material.
* Corpus breadth is a *separate* axis. To show all registered corpora, run the
  server with ``DEMO_ALL_CORPORA=true`` (see ``demo_serve.ps1``) — that widens
  which corpora are readable without touching the sensitivity tier.

Idempotent: re-running rotates the password on the existing account rather
than failing, so an operator can reprint the card at any time.

    python scripts/demo_user.py                       # generate a password
    python scripts/demo_user.py --password "..."      # pin one
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

# Readable words — an operator may have to read this off a card and type it on
# a tablet. Avoids look-alike glyph pairs entirely by using whole words.
_WORDS = (
    "archive", "meadow", "lantern", "harvest", "prairie", "compass",
    "thicket", "granary", "kestrel", "juniper", "cedar", "willow",
)


def _generate_password() -> str:
    return "{}-{}-{:04d}".format(
        secrets.choice(_WORDS), secrets.choice(_WORDS), secrets.randbelow(10000)
    )


def _default_corpus() -> str:
    """Home corpus for the account. Only used for display/tenancy bookkeeping
    when DEMO_ALL_CORPORA is on, but must be a real corpus id."""
    try:
        from gemynd.core.corpus_registry import get_default_corpus
        return get_default_corpus().corpus_id
    except Exception:
        return "turnbull"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default="demo@museum.local")
    parser.add_argument("--password", default=None, help="Defaults to a generated passphrase.")
    parser.add_argument("--corpus", default=None, help="Home corpus id (default: registry default).")
    parser.add_argument("--db", default="data/users.db")
    args = parser.parse_args()

    from gemynd.auth.store import UserStore

    password = args.password or _generate_password()
    corpus = args.corpus or _default_corpus()
    email = args.email.strip().lower()

    store = UserStore(args.db)
    existing = store.get_by_email(email)

    if existing is None:
        store.create_user(email, password, role="readonly", institution_id=corpus)
        action = "created"
    elif existing.role != "readonly":
        print(
            f"[error] {email} already exists with role={existing.role!r}.\n"
            "        Refusing to reuse a privileged account as the demo login.\n"
            "        Pick a different --email.",
            file=sys.stderr,
        )
        return 1
    else:
        # change_password bumps token_version, invalidating any JWT still held
        # by a browser left logged in after a previous demo.
        store.change_password(existing.user_id, password)
        store.activate_user(existing.user_id)
        action = "reset"

    print(f"Demo account {action}:")
    print(f"  email:    {email}")
    print(f"  password: {password}")
    print(f"  role:     readonly  (public-tier claims only)")
    print(f"  corpus:   {corpus}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
