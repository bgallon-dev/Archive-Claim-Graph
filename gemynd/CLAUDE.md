# gemynd

Claim-centric archival knowledge graph engine. Five pipeline layers (parse, extract, load, retrieve, synthesize) plus review and auth systems.

## Conventions

- `from __future__ import annotations` at the top of every module.
- All record types are `@dataclass(slots=True)` subclasses of `_BaseRecord` in `core/models.py`. New dataclasses must use `slots=True`. Declare `_EDGE_FKS` (frozenset) for fields that become graph edges rather than node properties.
- `from_dict()` silently drops unknown keys. `SemanticBundle.from_dict()` must use `.get()` with defaults for backward compat with older serialized bundles.
- `ClaimRecord.certainty` is the Python field; it serializes as `epistemic_status` in `to_dict()`. Both names handled in `from_dict()`. Do not add a second field for the same concept.
- Deterministic IDs via `core/ids.py` using `stable_hash()` (SHA-1 prefix). IDs are type-prefixed (`doc_`, `run_`, `year_`). Never use `uuid4` except for SQLite primary keys in stores.
- `DomainConfig` (from `core/domain_config.py`) is the single entry point for domain configuration. Never call `resource_loader` functions directly from pipeline code.
- All Anthropic API calls must go through `MeteredAnthropicClient` (`shared/token_tracker.py`).
- SQLite stores use WAL mode, `check_same_thread=False`, `foreign_keys=ON`. All databases registered in `shared/database_manager.py`.
- Use `logging.getLogger(__name__)` in all modules.

## Access control

Four tiers: `public`, `staff_only`, `restricted`, `indigenous_restricted`. The `indigenous_restricted` tier requires tribal consultation for clearance — this is an ethical obligation, not a feature flag.

## Safety invariants

- Sensitivity gate fail-safe: if the gate raises, ALL claims are quarantined. Never catch exceptions in a way that allows unscreened claims through.
- PII redaction runs on synthesis output BEFORE it reaches the client.

## Module dependency order

`shared` -> `core` -> `ingest` -> `retrieval` / `review` / `auth`

`core` must never import from `ingest`, `retrieval`, `review`, or `auth`.

## Testing

`pytest`. The in-memory graph backend (`InMemoryGraphWriter`) enables tests without Neo4j. Sample data in `tests/fixtures/`.
