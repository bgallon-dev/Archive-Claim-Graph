# resources

YAML and CSV domain configuration files. These files define what the pipeline knows about a specific archival domain.

## Key files

- `domain_profile.yaml` — collection-level config (institution_id, date range, synthesis_context, document_anchor, expected_claim_shares). The `resources` section can remap default filenames for other config files.
- `seed_entities.csv` — long-format CSV (one row per entity-property pair). Columns: `entity_type`, `name`, `prop_key`, `prop_value`. Rows sharing `(entity_type, name)` merge into one `EntityRecord`.
- `claim_type_patterns.yaml` — scored regex patterns. Each entry: `claim_type`, `regex`, `weight`. Compiled with `re.IGNORECASE`.
- `claim_relation_compatibility.yaml` — `compatibility` (claim_type x relation_type -> strong/weak/forbidden) and `preferred_entity_types`.
- `derivation_registry.yaml` — maps claim_type -> observation_type, event_type, required/optional entities.
- `domain_schema.yaml` — graph vocabulary: `claim_entity_relations` (ordered), `entity_labels`, `legacy_renames`.
- `sensitivity_config.yaml` + `indigenous_cultural_terms.yaml` — sensitivity detection rules.

## Adding a new claim type

Requires entries in ALL of:
1. `claim_type_patterns.yaml` (extraction patterns)
2. `derivation_registry.yaml` (observation/event derivation)
3. `claim_relation_compatibility.yaml` (entity compatibility)
4. Optionally `query_intent.yaml` (retrieval classification)

`_validate_config()` in `core/domain_config.py` warns about cross-resource inconsistencies at load time.

## Invariants

- All YAML regex patterns are validated by `_safe_compile()` in `shared/resource_loader.py`. Invalid regex raises `ValueError`, not `re.error`.
- `seed_entities.csv` entity types must match `entity_labels` in `domain_schema.yaml`. Cross-validation runs at `load_domain_config()` time.
- `token_pricing.yaml` can be overridden per-model via environment variables.

## Domain portability

To adapt to a new collection, replace these files. Use `scripts/bootstrap_domain.py` for automated generation from corpus samples, then `gemynd validate-domain` to verify extraction coverage.
