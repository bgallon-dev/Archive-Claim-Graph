# auth

JWT cookie authentication, user management, and role-based access control.

## Key invariants

- `token_version` on User: incrementing it invalidates all outstanding JWTs without a blacklist. Password changes and user deactivation MUST bump `token_version`.
- `ROLE_PERMITTED_LEVELS` in `models.py` is the single source of truth for role-to-access-level mapping. Three UI roles (`admin`, `archivist`, `readonly`) plus legacy bearer-token roles. Never add a role without updating this dict.
- `_prep()` in `store.py` SHA-256-then-base64-encodes the plaintext before bcrypt, eliminating the 72-byte truncation collision class. Existing pre-cutover hashes (built from raw-truncated plaintext) will not verify after this change — wipe `data/users.db` and re-run setup, or add a verify-fallback shim if a real user base must be preserved.
- JWT algorithm is HS256 only. `JWT_SECRET_KEY` must be >= 32 chars.

## Construction paths

- `UserContext.from_user()` — cookie auth path
- `UserContext.from_token_entry()` — bearer token path
- The `identity` property provides audit-log-friendly strings.

## Cross-module contract

`NeedsLoginException` must be registered as an exception handler on any FastAPI app that uses auth dependencies. Without it, unauthenticated requests will 500 instead of redirecting to login:

```python
from gemynd.auth.dependencies import NeedsLoginException

@app.exception_handler(NeedsLoginException)
async def _redirect_login(request, exc): ...
```

## First-run setup

`seed.py` and `setup.py` handle initial admin creation via a one-time setup token. The token is persisted to `${SETUP_TOKEN_PATH:-data/setup-token}` (mode 0600) on first call to `get_setup_token()`, printed to stderr at creation, and deleted by `mark_setup_done()`. File-backed so it is shared across uvicorn workers and survives restarts; the O_CREAT|O_EXCL race means only one worker prints the banner.

`is_setup_needed()` always queries the database (no in-process cache) — workers can never disagree about setup state.

## Testing

Auth tests need `USERS_DB` pointed at a temp SQLite. Mock `_get_secret()` for JWT tests.
