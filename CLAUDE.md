# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.
Keep this file terse and factual — it is meant to be read at the start of
every session. For the long-form product description, see `README.md`.

## Project summary

**Gemynd** is a claim-centric archival knowledge-graph pipeline. It ingests
OCR JSON from digitized archival documents, extracts typed claims (with
entity, measurement, and mention resolution against a domain-specific seed
vocabulary), loads them into Neo4j (or an in-memory backend), and answers
natural-language questions via an Anthropic Claude synthesis step with
full sentence-level provenance. Three FastAPI web UIs sit on top: query,
ingest, and review.

The package is named `gemynd` (installed from `pyproject.toml`) and exposes
two console scripts: `gemynd` (the main CLI) and `gemynd-create-admin`.

## Repository layout

```
Archive-Claim-Graph/
├── gemynd/              # Installable package (all application code)
├── scripts/             # One-off maintenance / bootstrap scripts (not installed)
├── tools/               # Operator utilities: pdf_to_json.py, run_diagnostics.py, dump_entity_names.py
├── tests/               # Pytest suite
│   ├── conftest.py      # Auto-loads .env; provides populated_writer / populated_executor fixtures
│   └── fixtures/        # report1.json, report2.json, report3.json — canonical tiny corpus
├── pyproject.toml
├── .env.example         # Copy to .env and fill in
├── README.md
└── CLAUDE.md            # (this file)
```

Gitignored scratch / state directories — **never commit anything placed in
them**: `data/`, `out/`, `input/`, `.claude/`, `plan/`, `build/`, `dist/`,
`*.egg-info/`, `.pytest_cache/`, `__pycache__/`, and `RUNBOOK.md`.

## Package architecture (`gemynd/`)

The pipeline is five layers (Source Parser → Extraction → Graph Load →
Retrieval → Synthesis) plus supporting subsystems. See `README.md`
"Architecture" for the full write-up; below is the file map.

- **`gemynd/core/`** — domain types and contracts: `ids.py`, `models.py`,
  `claim_contract.py`, `claim_validator.py`, `resolver.py`,
  `domain_config.py`, `query_contracts.py`, `graph/` (schema).
- **`gemynd/ingest/`** — Layers 0–2 (parse → extract → load).
  - Main entry points: `pipeline.run_e2e`, `pipeline.parse_source`,
    `pipeline.extract_semantic`, `pipeline.quality_report`,
    `pipeline.build_spelling_review_queue`.
  - Subpackages: `extractors/`, `annotation/`, `export/`, `graph/`,
    `web/` (FastAPI ingest UI with HTMX progress polling).
  - Other files: `source_parser.py`, `observation_builder.py`,
    `event_builder.py`, `derivation_context.py`, `phases.py`,
    `sensitivity_gate.py`, `spelling_review.py`, `checkpoint.py`,
    `pdf_converter.py`.
- **`gemynd/retrieval/`** — Layers 3–4 (retrieval + synthesis).
  - `classifier.py`, `entity_gateway.py`, `query_builder.py`,
    `executor.py`, `in_memory_executor.py`, `context_assembler.py`,
    `synthesis.py`, `conversation_log.py`, `web/` (FastAPI query UI).
- **`gemynd/review/`** — anti-pattern detection and review workflow.
  - `detect.py`, `detectors/`, `patch_spec.py`, `store.py`, `actions.py`,
    `monitor.py`, `export.py`, `web/` (FastAPI review UI).
- **`gemynd/auth/`** — JWT cookie auth, bcrypt hashing, RBAC, first-run
  setup token. Router, dependencies, store, seed, templates.
- **`gemynd/shared/`** — cross-cutting utilities.
  - `settings.py` (single `Settings` dataclass, see below),
    `env.py`, `database_manager.py`, `resource_loader.py`,
    `token_tracker.py`, `io_utils.py`, `logging_config.py`.
- **`gemynd/resources/`** — YAML/CSV domain configuration shipped as package
  data (`[tool.setuptools.package-data]` in `pyproject.toml`). Includes
  `claim_type_patterns.yaml`, `claim_relation_compatibility.yaml`,
  `claim_role_policy.yaml`, `derivation_registry.yaml`, `domain_profile.yaml`,
  `domain_schema.yaml`, `concept_rules.yaml`, `query_intent.yaml`,
  `measurement_*.yaml`, `sensitivity_config.yaml`,
  `indigenous_cultural_terms.yaml`, `negative_entities.yaml`,
  `ocr_corrections.yaml`, `token_pricing.yaml`, `seed_entities.csv`.
- **`gemynd/shared_templates/`** — `gemynd_base.html` Jinja base layout.
- **`gemynd/cli.py`** — argparse entry point. Every command documented in
  `README.md` is registered here via `subparsers.add_parser(...)`. When
  adding a new operational command, add it here rather than creating a
  standalone script.

### Legacy shim packages

`gemynd.extractors`, `gemynd.graph`, `gemynd.annotation`, `gemynd.export`,
and `gemynd.queries` are kept importable as re-export stubs for backward
compatibility (see `pyproject.toml` packages list and the comment there).
**Do not add new code to them.** Land new modules under the canonical
`gemynd/ingest/...`, `gemynd/retrieval/...`, or `gemynd/review/...`
paths, and let the shim re-export if older call sites still reference the
old import path.

### Top-level public API

`gemynd/__init__.py` re-exports the service-layer entry points:

- `IngestPipeline` (from `gemynd.ingest`)
- `RetrievalService` (from `gemynd.retrieval`)
- `ReviewService` (from `gemynd.review`)
- Legacy function API: `parse_source`, `extract_semantic`,
  `build_spelling_review_queue`, `quality_report`, `run_e2e`.

## Configuration & environment

All runtime configuration flows through one dataclass:
`gemynd/shared/settings.py` → `Settings.from_env()`. When introducing a new
environment variable, **extend `Settings` and read from it** rather than
calling `os.environ` directly elsewhere in the codebase.

Key variables (see `.env.example` for the complete list):

- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`,
  `NEO4J_TRUST`, `NEO4J_CA_CERT`
- `ANTHROPIC_API_KEY` — the legacy alias `Anthropic_API_Key` is still
  accepted.
- `SYNTHESIS_MODEL` — defaults to `claude-sonnet-4-6`.
- `JWT_SECRET_KEY`, `JWT_EXPIRE_HOURS`, `USERS_DB`, `COOKIE_SECURE`
- `GRAPHRAG_API_TOKENS` — JSON object mapping bearer tokens to
  `{role, institution_id}`. Roles: `public`, `staff`, `restricted`,
  `indigenous_admin`, `admin`.
- SQLite paths default under `data/`: `CONV_LOG_DB`, `WRITE_AUDIT_DB`,
  `REVIEW_DB`, `ANNOTATION_DB`, `TOKEN_USAGE_DB`, `INGEST_DB`. All are
  created on first use.
- OCR metadata: `OCR_ENGINE`, `OCR_VERSION`.

`tests/conftest.py` parses `.env` and injects any missing keys into
`os.environ` before tests run, so local credentials apply to test runs
automatically.

## Install extras (`pyproject.toml`)

- `pip install -e .` — core only (`pyyaml`).
- `pip install -e .[dev]` — pytest + hypothesis.
- `pip install -e .[neo4j]` — Neo4j driver.
- `pip install -e .[retrieval]` — Neo4j + Anthropic + FastAPI + auth
  deps. Required for `gemynd query-serve` and `gemynd review-serve`.
- `pip install -e .[ingest]` — retrieval deps + PyMuPDF + pytesseract +
  Pillow. Required for `gemynd ingest-serve` and PDF conversion.
- `pip install -e .[tools]` — PyMuPDF + pytesseract + anthropic, for
  scripts under `scripts/` and `tools/`.

## Running things

- **Tests:** `pytest` (pyproject sets `pythonpath=["."]`, `addopts="-q"`).
  Integration tests that need a live Neo4j are marked with the
  `integration` marker — run `pytest -m "not integration"` to skip them.
- **In-memory end-to-end smoke test** (no external services):
  `gemynd run-e2e --inputs tests/fixtures --out-dir out --backend memory`.
- **Full ingest → graph flow** (Neo4j required):
  ```
  gemynd ingest-structure --input input/<corpus> --output out/
  gemynd extract-semantic --structure-dir out/ --output-dir out/
  gemynd load-graph --input-dir out/ --backend neo4j
  ```
- **Servers:** `gemynd query-serve`, `gemynd ingest-serve`,
  `gemynd review-serve`.
- The CLI (`gemynd/cli.py`) is the canonical operational surface. Prefer
  adding a new `subparsers.add_parser(...)` branch there over writing a
  standalone script in `scripts/` unless the task is truly a one-off.

## Coding conventions

- **Python 3.10+.** `from __future__ import annotations` is used
  throughout; keep that style consistent in new files.
- **Dataclasses** for configuration and DTOs (`Settings`, extraction
  results, etc.).
- Load YAML resources via `gemynd.shared.resource_loader` rather than
  opening them directly — this is how package-data paths resolve correctly
  both in development and when installed.
- Persistent state goes under `data/` (gitignored), with the path surfaced
  through `Settings`. Do not hard-code SQLite paths elsewhere.
- **Preserve decision trails.** Extractors, resolvers, and the claim
  contract append to `decision_trace`, `extraction_confidence`, and
  `epistemic_status` fields. Downstream synthesis reasons about these —
  don't drop them in refactors.
- Do not add emojis to files unless explicitly asked.
- Do not create new `*.md` documentation files unless explicitly asked.
- Do not commit anything in gitignored scratch directories.

## Testing patterns

- `tests/conftest.py` provides:
  - `fixtures_dir` — path to `tests/fixtures/`.
  - `populated_writer` — **session-scoped**; runs the full pipeline
    (`run_e2e`) against `report1.json`, `report2.json`, `report3.json`
    into an `InMemoryGraphWriter`.
  - `populated_executor` — an `InMemoryQueryExecutor` over that writer.
- Reuse these fixtures rather than re-running the pipeline inside
  individual tests.
- Hypothesis is used for property-based tests (see
  `tests/test_measurements_pbt.py`).
- When adding a feature, colocate a `tests/test_<feature>.py` and
  exercise it through the in-memory backend where possible so CI runs
  without Neo4j.

## Git & PR workflow for AI sessions

- The working branch for the current session is specified in the session
  instructions (e.g. `claude/add-claude-documentation-<id>`). Develop,
  commit, and push to that branch only.
- **Commit only when the user explicitly asks.**
- **Create pull requests only when the user explicitly asks.**
- Never force-push, never push to `main` / `master`, never skip hooks
  (`--no-verify`, `--no-gpg-sign`, etc.) unless the user asks in clear
  terms. Prefer new commits over `--amend`.
- GitHub access is via the `mcp__github__*` MCP tools and is restricted
  to `bgallon-dev/archive-claim-graph`. There is no `gh` CLI available in
  this environment.
- Use `git push -u origin <branch>`; retry on transient network errors
  with exponential backoff (2s, 4s, 8s, 16s) up to four attempts.

## Quick reference: where things live

| Need                                | Path                                                    |
| ----------------------------------- | ------------------------------------------------------- |
| Add a CLI subcommand                | `gemynd/cli.py` (`subparsers.add_parser(...)`)          |
| Add an env var                      | `gemynd/shared/settings.py` (`Settings` dataclass)      |
| Ingest pipeline entry point         | `gemynd/ingest/pipeline.py` (`run_e2e`, line 440)       |
| Claim extraction logic              | `gemynd/ingest/extractors/`                             |
| Retrieval query templates           | `gemynd/retrieval/query_builder.py`                     |
| Synthesis prompt + Anthropic call   | `gemynd/retrieval/synthesis.py`                         |
| Anti-pattern detectors              | `gemynd/review/detectors/`                              |
| Domain config (YAML/CSV)            | `gemynd/resources/`                                     |
| Auth routes / JWT                   | `gemynd/auth/`                                          |
| Shared resource loading             | `gemynd/shared/resource_loader.py`                      |
| Test fixtures (3-report corpus)     | `tests/fixtures/report{1,2,3}.json`                     |
| Session-scoped in-memory pipeline   | `populated_writer` / `populated_executor` in conftest   |
