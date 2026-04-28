# core

Domain models, claim/entity contracts, entity resolution, and graph abstractions. Zero service dependencies — never import from `ingest`, `retrieval`, `review`, or `auth`.

## Models (`models.py`)

- `_BaseRecord` provides `to_dict()`, `from_dict()`, `node_props()`. Every new Record with FK fields must declare `_EDGE_FKS` as a class-level frozenset.
- `from_dict()` silently drops unknown keys for forward compatibility.
- `ClaimRecord.certainty` serializes as `epistemic_status` via `to_dict()`. Both names handled in `from_dict()`.
- Domain-agnostic anchor records: `DocumentAnchorLinkRecord` (doc → anchor entity) and `EntityHierarchyLinkRecord` (child → parent entity) replaced the old wildlife-specific `DocumentRefugeLinkRecord` / `PlaceRefugeLinkRecord`.
- `ObservationRecord` and `EventRecord` carry domain-specific bindings in a `role_entities: dict[str, str]` field, not as typed FKs. Legacy wildlife keys (`species_id`, `refuge_id`, `habitat_id`, `survey_method_id`) are migrated into `role_entities` by the `from_dict()` overrides for round-tripping older bundles. `place_id`, `period_id`, `year_id` remain first-class.

## Domain config (`domain_config.py`)

- `DomainConfig` is the single entry point. `load_domain_config(resources_dir)` loads a single corpus; `load_composite_domain_config([...])` unions multiple corpora into a `CompositeDomainConfig` for multi-institution deployments.
- Composite merges are field-specific. Notably, `preferred_entity_types` uses **anchor-relative insertion** (a new type is placed before the next shared anchor in its source list) so a later corpus can raise a newly-introduced type without it landing in the least-preferred slot. Append-at-end was silently lossy and demoted Refuge below REFERS_TO threshold for `turnbull_prehistory`; do not regress to that behavior.
- Role-driven derivation: `role_resolution`, `observation_role_edges`, and `event_role_edges` are parsed from the `derivation_roles` block in `domain_schema.yaml`. Observation and event edges are kept as separate maps because the writer emits asymmetric types (e.g. `OF_SPECIES` vs `INVOLVED_SPECIES`).
- Anchor properties (`anchor_entity_id`, `anchor_entity_type`, `extraction_stopwords`) are derived from `domain_anchor` + `seed_entities`; do not hardcode collection-specific defaults in retrieval or ingest code.

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
