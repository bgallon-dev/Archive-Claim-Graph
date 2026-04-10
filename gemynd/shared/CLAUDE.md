# shared

Cross-cutting infrastructure: configuration, database management, file I/O, logging, resource loading, and token tracking. Imported by all other gemynd packages.

## Configuration (`settings.py`, `env.py`)

- `Settings.from_env()` is the single entry point for all configuration. Never read `os.environ` directly in other modules for settings that have a `Settings` field.
- `env.py` is a custom `.env` loader (no `python-dotenv` dependency). Supports `export` prefix, single/double quoting. Does NOT override existing env vars unless `override=True`.

## Database management (`database_manager.py`)

- `DatabaseManager` centralizes all SQLite connections. All connections use WAL mode, `check_same_thread=False`, `foreign_keys=ON`.
- Parent directories are created on first access.
- Registered databases: `users`, `review`, `ingest`, `conv_log`, `token_usage`, `write_audit`, optionally `annotation`.

## File I/O (`io_utils.py`)

- Bundle serialization: `save_structure_bundle`, `load_structure_bundle`, `save_semantic_bundle`, `load_semantic_bundle`.
- JSON saved with `indent=2, ensure_ascii=True`.
- Parent directories auto-created via `mkdir(parents=True, exist_ok=True)`.

## Resource loading (`resource_loader.py`)

- All loaders accept optional `resources_dir` argument defaulting to `gemynd/resources/`. This enables tests to use fixture directories.
- `_safe_compile()` wraps `re.compile()` to prevent malformed YAML regex from crashing the pipeline.
- `_resource_path()` resolves filenames through `domain_profile.yaml`, allowing resource remapping.

## Token tracking (`token_tracker.py`)

- `MeteredAnthropicClient` wraps the Anthropic SDK. Daemon-thread writer pattern (same as conversation logger — never blocks the calling thread).
- Pricing loaded from `token_pricing.yaml` with env var overrides (`TOKEN_PRICE_<MODEL_KEY>_INPUT/OUTPUT`).

## Logging (`logging_config.py`)

Use `logging.getLogger(__name__)` in all modules. Centralized setup in this module.
