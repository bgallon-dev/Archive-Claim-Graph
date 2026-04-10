# core

Domain models, claim/entity contracts, entity resolution, and graph abstractions. Zero service dependencies — never import from `ingest`, `retrieval`, `review`, or `auth`.

## Models (`models.py`)

- `_BaseRecord` provides `to_dict()`, `from_dict()`, `node_props()`. Every new Record with FK fields must declare `_EDGE_FKS` as a class-level frozenset.
- `from_dict()` silently drops unknown keys for forward compatibility.
- `ClaimRecord.certainty` serializes as `epistemic_status` via `to_dict()`. Both names handled in `from_dict()`.

## Claim contract (`claim_contract.py`)

- Contract version is `v2`. `ALLOWED_CLAIM_TYPES` is the canonical set.
- `validate_claim_type()` coerces unknown types to `UNCLASSIFIED_TYPE` ("unclassified_assertion") — it never rejects. The pipeline never refuses a claim type; it coerces.
- `_LEGACY_RENAMES` maps old type names to current canonical names. When removing a claim type, move it to `_LEGACY_RENAMES` rather than deleting.
- `CLAIM_ENTITY_RELATION_PRECEDENCE` tuple defines priority order; lower index = higher priority.
- `get_relation_compatibility()` defaults to `"weak"` for unlisted pairs. Never default to `"strong"` or `"forbidden"`.

## Entity resolution (`resolver.py`)

- `REFERS_TO` threshold: >= 0.85 with uniqueness gap >= 0.05.
- `POSSIBLY_REFERS_TO`: 0.65–0.84.
- Two-pass algorithm when resolution contexts are provided.
- These thresholds are in `ResolutionPolicy` — changing them affects graph quality downstream.

## Graph abstractions (`graph/cypher.py`)

- `ID_CONSTRAINTS` defines uniqueness constraints per Neo4j label. Every label must have one.
- `SCHEMA_STATEMENTS` and `INDEX_STATEMENTS` for graph setup.

## Testing

Test `from_dict()` backward compatibility when adding fields. Test `validate_claim_type()` coercion. Test resolution thresholds via `DictionaryFuzzyResolver`.
