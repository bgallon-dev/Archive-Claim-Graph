# auth

JWT cookie authentication, user management, and role-based access control.

## Key invariants

- `token_version` on User: incrementing it invalidates all outstanding JWTs without a blacklist. Password changes and user deactivation MUST bump `token_version`.
- `ROLE_PERMITTED_LEVELS` in `models.py` is the single source of truth for role-to-access-level mapping. Three UI roles (`admin`, `archivist`, `readonly`) plus legacy bearer-token roles. Never add a role without updating this dict.
- Bcrypt truncates at 72 bytes (`_BCRYPT_MAX_BYTES` in `store.py`). The `_prep()` function enforces this. Do not change this behavior.
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

`seed.py` and `setup.py` handle initial admin creation via a one-time setup token printed to stderr.

## Testing

Auth tests need `USERS_DB` pointed at a temp SQLite. Mock `_get_secret()` for JWT tests.
