# Gemynd Maintainability Roadmap

*Generated 2026-07-20 by a 10-dimension multi-agent audit (46 findings confirmed, 8 rejected after adversarial verification against the real code). Three latent bugs independently re-verified by hand — see the callout below.*

## Verified latent bugs (fix regardless of the refactor plan)

1. **`_apply_compatibility_penalty` — latent `NameError`.** Called at `ingest/phases.py:337` in the reachable document-level claim-link fallback, but defined nowhere in the source tree (only a stale `.pyc` contains the name). Fires whenever that fallback resolves a match. This is the strongest argument for landing ruff (F821 catches it).
2. **`Any` used but never imported** in `ingest/extractors/measurement_extractor.py` (lines 22, 44-45, 85-86) and `ingest/extractors/claim_extractor.py` (622, 676, 679). Dormant only because every use sits in an annotation position that `from __future__ import annotations` stringifies. Add `Any` to the `typing` imports.
3. **`_canonicalize_source_path` defined but never called** (`ingest/source_parser.py:42`). `ingest/CLAUDE.md` lists "Path traversal guards prevent directory escape in server contexts" as an *invariant* — but the guard is not wired into `parse_source_payload`/the upload path, so that documented security invariant is currently false. **Do not blindly delete it** (the audit's default suggestion): first confirm whether traversal is guarded elsewhere on the server upload path; if not, wire it in. Deleting an advertised security control is the wrong default.

---

## Executive summary

Gemynd is a well-factored-in-the-small system carrying its complexity in a handful of **god files** and a **missing tooling floor**. Roughly 80% of the maintainability debt concentrates in four places: `retrieval/web/app.py` (4409 lines, one `create_app` closure holding 49 routes), `retrieval/context_assembler.py` (1890 lines, five responsibilities), `core/graph/cypher.py` (1653 lines of query constants), and `retrieval/in_memory_executor.py` (a 980-line parallel reimplementation of the production Cypher that the tests actually validate instead of the real queries). The three highest-leverage moves are: **(1) land the tooling floor** — ruff/mypy/pre-commit/CI do not exist yet, ruff is already installed and its default ruleset surfaces two genuine latent bugs (an undefined `_apply_compatibility_penalty` call and an unimported `Any`); **(2) extract the shared web-app bootstrap and per-file loggers** — the same startup block is copy-pasted across three FastAPI apps, and the two biggest apps have no module-level logger, which is the mechanical root cause of every silent-swallow finding; **(3) decompose the four god files behind seams that also unlock testing** — the retrieval `/query` orchestrator (~500 lines) and the context-assembler cascade cannot be tested today without standing up the whole app. Do tooling + characterization tests **before** the structural refactors so the moves are diffable and green.

---

## Quick wins (high impact / low effort)

| Item | Where | Fix | Impact / Effort |
|---|---|---|---|
| Configure ruff (already installed, surfaces 2 real bugs) | `pyproject.toml` (ends at :84, no `[tool.ruff]`); bugs at `ingest/phases.py:337`, `ingest/extractors/measurement_extractor.py:22,44-45` | Add `[tool.ruff.lint]` block; `ruff check --fix` clears 41 auto-fixable; hand-fix the undefined-name `_apply_compatibility_penalty` and unimported `Any` | high / S |
| Add CI (none exists) | `.github/` absent | `ci.yml` gating on `ruff check gemynd` + `pytest -m "not integration"` (the whole non-integration suite runs Neo4j-free) | high / S |
| Add module-level loggers to the two big apps | `retrieval/web/app.py:1055` (logger bound inside `create_app`), `review/web/app.py:965` (no top-level `import logging` at all) | Bind `_log = logging.getLogger(__name__)` at module top; unblocks logging in every module-level helper; keystone for the error-handling fixes | high / S |
| Strip unused imports | epicenter `ingest/extractors/claim_extractor.py:3,11,19,22,31` (+ ~7 files) | `ruff check --fix` (F401); hand-clean the claim_extractor header (refactor residue) | medium / S |
| Kill dead `Settings.cookie_secure` + unify `COOKIE_SECURE` | `shared/settings.py:52,109` (field, default `False`, no reader) vs 4x raw `os.environ` reads defaulting `True` | Add `cookie_secure_web=True`, read Settings at all four entrypoints | medium / S |
| Log the two documented-intentional swallows | `cli.py:428-429` (audit-log write), `core/domain_config.py:940-941` (sensitivity-config validation) | Keep non-fatal behavior; emit `_log.warning(..., exc_info=True)` / call existing in-scope `_warn(...)` | medium / S |
| Delete `scripts/_archive/` | `scripts/_archive/README.md:1` ("Safe to delete this whole folder") | Delete wholesale (7 files, zero external importers) | low / S |
| Add pre-commit (tree is a live git repo) | `.pre-commit-config.yaml` absent, `.git/` present | `ruff` + `ruff-format` hooks pinned to `v0.15.12` | medium / S |

---

## Structural investments (higher effort, high payoff)

1. **Shared web-app bootstrap → `gemynd/shared/webapp.py`** [#7]. The Jinja shared-templates loader (4x), CSRF middleware block, generic-exception handler, `_no_store` middleware, and `_setup_guard` are copy-pasted verbatim across `ingest/web/app.py:50-134`, `review/web/app.py:933-1035`, `retrieval/web/app.py:1544-1708`, `auth/router.py:52-57`. Seam: `build_shared_templates()`, `install_common_exception_handlers(app)`, `install_security_middleware(app, *, users_db_path)`, and a **parametrized** `make_health_route(app, db_probe, db_key)` (health routes genuinely differ per app). This is the enabler for the retrieval god-file split.

2. **Decompose `retrieval/web/app.py`** [#1, #42/#35, #45]. Introduce an `AppState`/dependencies object provided via `Depends` so routes stop closing over `create_app` locals (`_require_executor:1158`, `db_mgr:1307`, `_settings`). Then: (a) extract the `/query` body (`1740-2237`, ~500 lines) into `retrieval/query_service.py::run_query(req, *, executor, synthesiser, config)`, leaving the route a thin adapter — this alone unlocks DB-free tests via the `populated_executor` fixture; (b) peel routes into `retrieval/web/routes/{query,explore,documents,history,ingest,users}.py`, one router per commit; (c) stage the 358-line `lifespan()` (`1169-1527`) into `_load_domain_state` / `_init_executor_and_caches` / `_init_stores` / `_init_pipeline_components`.

3. **Break up `retrieval/context_assembler.py`** [#3, #43]. Move sanitization/OCR helpers → `retrieval/text_sanitize.py`, the planning trio (incl. the 333-line `_select_retrieval_plan:246`) → `retrieval/retrieval_plan.py`, row/block/structural fns → `retrieval/blocks.py`. Then split the 282-line `assemble()` (`1357-1639`) into `_execute_cascade` / `_run_llm_cypher_fallback` / `_finalize_context`, factoring the duplicated claim_id dedup (`1457-1461` and `1541-1546` — but preserve the `traversal_rel_type` accumulation the primary loop alone performs). Promote the six app-consumed underscore helpers to public names during the move.

4. **`core/graph/cypher.py` → `core/graph/cypher/` package by family** [#4]. `schema.py`, `entity.py`, `claims.py` (the 358-875 browse family), `documents.py`, `paths.py`, `sensitivity.py`, `stats.py`, `gaps.py`, with `__init__.py` re-exporting every name so the ~dozen importers (`cli.py`, `ingest/pipeline.py`, `retrieval/web/app.py:56-83` = 26 names) are untouched. Pure move + re-export.

5. **Typed collaborator contracts** [#21, #22, #23]. Add `class QueryExecutor(Protocol)` to `retrieval/executor.py` (the doc claims one exists; it doesn't) so tests' `InMemoryQueryExecutor` substitution typechecks against the ~8 sites annotated with the concrete `Neo4jQueryExecutor`. Add `LLMClaimRow`/`ClaimVerdict` TypedDicts for the untyped extractor→pipeline seam, and a `ClaimRow` TypedDict linking cypher `RETURN` aliases to `_row_to_block`.

6. **Pin the two query backends together** [#33, #5]. `InMemoryQueryExecutor.run()` dispatches on `cypher is <CONSTANT>` object identity and never executes the query text, so the largest retrieval test file validates Python, not Cypher. Add `tests/test_cypher_strings.py` (golden snapshots of every `build_*` + constant, zero DB, reusing the `test_golden_parity.py:42-56` pattern) and an integration-marked Neo4j parity test. Extract the graph-walk primitives (`in_memory_executor.py:787-960`, all depend only on `self._w`) into `retrieval/in_memory_graph.py`.

7. **Decompose `ingest/pipeline.py::run_e2e()`** (329 lines) [#41]. Extract `_preflight_neo4j_dedup` / `_fetch_supplementary_entities`, promote the `_write_doc_to_graph` closure (`nonlocal writer` + ~11 captured names) to a `_DocGraphWriter` class, split `_run_single_worker` / `_run_multi_worker`, and factor the verbatim-duplicated frequency-prior warm-up (`744-746` == `777-779`).

8. **Fold the `CLAIM_*`/`OPENAI_*` extraction env family into `Settings`** [#28]. ~18 env vars are read straight from `os.environ` across four adapters + the factory, violating `shared/CLAUDE.md` ("never read os.environ directly… for settings that have a Settings field") and undocumented in `.env.example`. Add an `ExtractionSettings` group; thread via constructor args.

---

## Per-theme breakdown

### Tooling & CI

No linter, formatter, type-checker, CI, or pre-commit exists, yet ruff 0.15.12 and mypy 1.20.2 are both installed. `ruff check gemynd` (defaults) returns 61 errors; `ruff format --check` wants to reformat 110/122 files. Land these in order: config → mechanical reformat commit → CI.

- **ruff lint** [#15] — `pyproject.toml` ends at :84 with no lint config. Two default-rule hits are real bugs, not style: `ingest/phases.py:337` calls `_apply_compatibility_penalty(...)` in a reachable branch but the name is defined nowhere (NameError latent); `ingest/extractors/measurement_extractor.py:22,44-45` annotate `dict[str, Any]` while only importing `Protocol, TYPE_CHECKING` (dormant only via `from __future__ import annotations`). `B008` is safe to ignore — all 81 hits are the intentional FastAPI `Depends()`/`Query()` idiom.

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "B", "C4", "UP", "SIM"]
ignore = ["E501", "B008"]  # E501 line-length; B008 = intentional FastAPI Depends()/Query() defaults

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]     # re-exports
"tests/*" = ["F811", "F841"]

[tool.ruff.format]
quote-style = "double"
docstring-code-format = true
```

- **Mechanical reformat as ONE commit** [#18] — after the config lands, run `ruff format gemynd && ruff check --fix --select I,UP gemynd` (110-file reformat, 87 unsorted-import blocks, 96 legacy-quote annotations) with no logic changes, so `git blame` stays useful and the churn never contaminates a feature diff.

- **mypy, lax then tighten core/shared first** [#19] — many deps are untyped/optional (neo4j, fastapi, jose, bcrypt, pymupdf, anthropic, openai), so start permissive and harden the bottom of the import DAG.

```toml
[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
follow_imports = "silent"
warn_redundant_casts = true
warn_unused_ignores = true
exclude = "(^scripts/|^tools/|^notebooks/|kettle_eval)"

[[tool.mypy.overrides]]
module = ["gemynd.core.*", "gemynd.shared.*"]
disallow_untyped_defs = true
```

- **pre-commit** [#17] — the tree IS a git repo (`git status --short` shows modified tracked files), so hooks install today. Land the reformat commit first so the first hook run doesn't rewrite 110 files.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.12          # match the installed ruff + CI
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
```

- **CI** [#16] — no `.github/`. Gate on `ruff check` + tests first (the config/reformat/mypy baselines must be green before adding those checks, or the first run is red on 61 pre-existing errors + 110 unformatted files).

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev,retrieval,ingest]" ruff mypy
      - run: ruff check gemynd
      - run: pytest -m "not integration"
      # add once baselines go green:
      # - run: ruff format --check gemynd
      # - run: mypy gemynd/core gemynd/shared
```

- **dev bootstrap** [#20] — `dev = ["pytest>=8.0.0", "hypothesis>=6.0.0"]` (`pyproject.toml:18`) omits ruff/mypy/pre-commit; no Makefile, no CONTRIBUTING. Extend the extra, add a `Makefile` (`install`/`lint`/`fmt`/`typecheck`/`test`/`check`), and a `CONTRIBUTING.md` cross-linked from `README.md:98-101`.

### God files & decomposition (complexity hotspots folded in)

- **`retrieval/web/app.py` (4409 lines)** — see Structural #2. `create_app` is one closure from :1036 to `return app` at :4409 holding all 49 routes + ~66-74 nested defs. `/query` (`1740-2237`) rebuilds a 36-line `_tier1_caveat_by_op` dict per request (`1842-1877`, hoist to module constant) and builds a ~22-field `LogRecord` twice (`1912-1933`, `2200-2235` — unify into `_log_query_turn`) [#42]. Also embeds a **second copy of the review and ingest route surfaces** (`review_page:3781`, `review_accept:3949`…, `ingest_page:4147`, and re-implements the whole background job pipeline inline at `_run_ingest_job:4218-4248`) [#39] — extract a shared proposal-action `APIRouter` both apps `include_router()`; note auth (`require_admin` vs `require_archivist_or_admin`) and response shapes differ, so parametrize.
- **`retrieval/context_assembler.py` (1890 lines)** — Structural #3 [#3, #43].
- **`core/graph/cypher.py` (1653 lines)** — Structural #4 [#4].
- **`retrieval/in_memory_executor.py` (980 lines)** — Structural #6 [#5, #33]. Caveat: `run()` identity-compares two per-instance cyphers built in `__init__`, so a module-level dispatch dict can't hold them — build per-instance or keep as special cases.
- **`review/web/app.py` (1648 lines)** — ~830 lines of module-level card/HTML builders before `create_app:916` [#2]. Extract the pure `render_*`/`_key_info_html:758` helpers to `review/web/cards.py`; delete the two cross-app reach-ins `from gemynd.review.web.app import _key_info_html` in retrieval (`3829`, `4075`).
- **`ingest/pipeline.py` (1092 lines)** — Structural #7 [#41].
- **`cli.py` (988 lines)** [#6] — `build_parser` (`46-350`, ~20 subparsers) and an 18-branch, 606-line `main` (`382-988`) where every branch returns an int. Split into `cli/commands/cmd_<name>(args)->int` + `cli/parser.py`; `main` becomes `{command: handler}[args.command](args)`. Behavior-preserving, one command per commit.
- **`retrieval/synthesis.py::synthesise()` (196 lines)** [#44] — the `_resolve_citation_tokens → _repair_claim_ids → _validate_response` block (incl. its comment) is verbatim at `753-757` and `785-789`. Extract `_resolve_repair_validate`, `_build_user_message` (`716-744`), `_assemble_caveats` (`833-855`).
- **`ingest/phases.py::resolve_claim_links()` (71 lines)** [#46] — three nested loops (paragraph→claim→claim_link) with the whole 58-line body in the innermost. Extract `_process_claim_link(claim, claim_link, ctx, seen_claim_entity, seen_claim_location)`.

### Duplication

- **FastAPI bootstrap** [#7] — Structural #1 (the highest-value dedup).
- **Neo4j executor construction** [#10] — `shared/neo4j_helpers.build_neo4j_executor` (the documented "single source") is called **once** (`retrieval/__init__.py:37`) while ~11 sites re-inline the constructor. Drift already happened: `cli.py:782` omits the `trust_mode or "system"` fallback; three `scripts/backfill_*.py` omit `ca_cert_path` and each re-derive `.env` differently. `status.py:122-129` is a byte-identical drop-in — route it through the helper; add `build_neo4j_executor_from_env()` + `scripts/_bootstrap.py`.
- **Dead OpenAI SDK adapter** [#8] — `openai_claim_adapter.py` ships two full adapters for one capability; `OpenAICompatClaimAdapter` (`382`) + `create_claim_llm_adapter` + `_build_response_format`/`_normalize_claims`/`_parse_claims` are imported nowhere outside `tests/test_openai_claim_adapter.py`. Delete the SDK path, keep the stdlib `OpenAICompatibleClaimAdapter`, retarget the test.
- **SQLite connection bootstrap** [#31] — the `connect(check_same_thread=False)` + `WAL` + `busy_timeout=5000` + `foreign_keys=ON` + `row_factory` block is hand-rolled in ~10 stores despite `DatabaseManager.get_connection()`. `review/store.py:162-166` == `conversation_log.py:354-358` byte-for-byte; `auth/store.py:76-80` has **drifted** (omits WAL, foreign_keys, check_same_thread). Add `shared/database_manager.configure_sqlite(conn)` + a `BUSY_TIMEOUT_MS` constant; the fix also repairs the auth drift.
- **chat-completions POST transport** [#9] — the `Request(...Bearer...)` + `urlopen(timeout=)` block is written 3x (`openai_claim_adapter._post`, `claim_validator_adapter._validate_batch`, `openai_compat._OpenAIMessages._request`). Add `post_chat_completion(...) -> dict | None` for the two claim adapters only; do **not** fold in the streaming `openai_compat` path. Low impact (~8 lines).
- **JSON-snapshot parse idiom** [#26] — `try: json.loads(evidence_snapshot_json) except Exception: pass` across ~12 sites (`review/web/app.py:1583-1586` verbatim at `1604-1607`, `1630-1633`). Add `shared/io_utils.safe_json_obj(raw, *, context) -> dict` catching only `(json.JSONDecodeError, TypeError)` and logging. Note: the dual-output display sites (`retrieval/web/app.py:4060-4071`, `review 1256/1268`) return dict+string and need per-site care.

### Error handling & silent fallbacks

Root cause is the **missing module-level loggers** (Quick win [#24]) — module-level helpers in the two biggest apps have no logger in scope, so failures can't be logged even where the author wanted to. (This also violates `gemynd/CLAUDE.md`: "Use `logging.getLogger(__name__)` in all modules.")

- **Broad-except → silent empty data on dashboards** [#25] — `_read_query_signal_gaps` (`retrieval/web/app.py:449`) returns `[]` on any failure (locked/corrupt DB looks like "no gaps"); the stats quarantine count (`2022-2034`) does `except Exception: pass` leaving `0` (silently defeats the transparency the block exists for, per its own `2019-2021` comment); `_load_structure_json` (`review/web/app.py:85-94`) returns `{}` conflating missing/permission/corrupt. Narrow to `sqlite3.Error`/`OSError`/`json.JSONDecodeError` and log at warning with `exc_info=True`. In-file precedent already exists: `retrieval/web/app.py:2036-2037` "logs and swallows."
- **Documented-intentional swallows with no trace** [#27] — Quick win; `cli.py:428` (audit-log write; `cli.py` has no logger, uses `print`) and `domain_config.py:940` (narrow to `yaml.YAMLError`, call in-scope `_warn` for shape errors).

### Typing & contracts

- **No `QueryExecutor` Protocol** [#21] — Structural #5. `retrieval/CLAUDE.md` claims `executor.py` defines it; it defines zero Protocols. The two executor classes share only a duck-typed `run()`, so every test substitution would fail a checker. Conversely `annotation_store`/`llm_cypher_generator` are under-typed as `object | None` while concrete methods are called on them (`context_assembler.py:1508,1623`).
- **Untyped LLM claim seam** [#22] — `list[dict[str, object]]` from the `ClaimLLMAdapter` Protocol + 4 implementations; `Any` is referenced at `claim_extractor.py:622/676/679` but **never imported** (masked only by `from __future__ import annotations`) — add the import regardless of the TypedDict work.
- **Neo4j-row → block contract** [#23] — `_row_to_block(row: dict)` (`context_assembler.py:626`) hand-plucks `c/d/pg/para/obs/sp/y/measurements` that must match every synthesis `RETURN` clause with nothing linking them. Add a `ClaimRow` TypedDict in `retrieval/models.py` (same pass folds in the comment-only view-model shapes at `models.py:157,233-239`). Note: TypedDicts are erased at runtime with no checker configured — the near-term benefit is documentation + IDE nav + one greppable contract, not runtime enforcement.

### Testing

- **In-memory executor validates Python, not Cypher** [#33] — Structural #6. A Cypher-only refactor (rename a property, change a `WHERE`) passes the whole suite green because the real query bodies run only under integration tests that `pytest.skip` when `NEO4J_URI` is unset (every default run).
- **Three web surfaces have zero tests** [#34] — no test imports `gemynd.review.web`, `gemynd.ingest.web`, or `gemynd.auth.router`. Untested includes the auth **open-redirect guard** `_safe_next:60-64`, `post_login:196`, `create_user:261`, and reviewer state-mutation routes `do_accept:1336`/`do_reject`/`do_batch_accept:1461`. Unit-test the pure `_safe_next`/`_render_*` helpers (no app needed) + `TestClient` smoke tests mirroring `test_startup_resilience.py:78`. Lead with `test_auth_router.py`.
- **`/query` orchestrator untested** [#35] — folded into Structural #2 (extract → `run_query`, inject `populated_executor`).
- **Integration tests double-gated + no coverage** [#36] — the `integration` marker is applied once (`test_retrieval_improvements.py:876`) AND skips on unset env; add a Neo4j-service-container CI job + `pytest-cov` floor.
- **Skip-hides-regression on the largest test file** [#37] — 18 conditional `pytest.skip` in `test_retrieval_in_memory.py`; ~7 are repeated existence invariants (`91/102/116/139/214/248/375/512`) over a deterministic fixture, so a pipeline regression skips green. Add one session-scoped `assert_populated(populated_writer)` and delete the recurring existence checks; **keep** the data-shape guards (`267/287/333`).

### Config & constants

- **Extraction env family not in Settings** [#28] — Structural #8.
- **Dead `cookie_secure`** [#29] — Quick win.
- **Auth rate-limit knobs frozen at import** [#30] — `auth/router.py:35-50` builds four `TokenBucketLimiter`s at module import from `os.environ` with 8 inline defaults, while the query limiter is a proper Settings field (`settings.py:55-56`). Add `login_*`/`setup_*` Settings fields; because auth is an `APIRouter` (no `create_app`), build the limiters in a factory/lazy accessor, not at module scope.
- **Re-hardcoded `data/*.db` defaults** [#32] — canonical paths live in `Settings` (`settings.py:51-65`) yet `'data/users.db'` is re-typed as a signature default at `auth/store.py:63`, `auth/setup.py:70`, `auth/seed.py:24`, `review/web/app.py:916`, `ingest/web/app.py:30`; `'data/claim_cache.db'` is re-typed in `status.py:195` vs `claim_cache.DEFAULT_CACHE_PATH:17`. Default to `None`/resolve from Settings; import `DEFAULT_CACHE_PATH` in `status.py`. (Note: the argparse defaults at `cli.py:216`/`auth/seed.py:75` DO honor the env var — only the plain signature defaults diverge.)

### Consistency

- **sync-vs-async handler convention** [#38] — `retrieval/web/app.py:2912-2916,2936-2940` documents *why* blocking-DB handlers are plain `def` (async stalled the event loop), but `review/web/app.py` declares every handler `async def` and calls synchronous `sqlite3` inside (`do_accept:1336`→`accept_proposal`, `index:1037`→`store.list_proposals`), as do ingest `job_status_page:266`/`job_status_fragment:303`. Convert those to sync `def` and copy retrieval's one-line rationale. Keep genuinely-awaiting handlers async (ingest `create_ingest_job:251` awaits `upload.read()`).
- **Duplicated review/ingest route surfaces** [#39] — folded into the `retrieval/web/app.py` god-file entry above.
- **Singular vs plural URLs** [#40] — `/api/entities/search` next to `/api/entity/{id}/neighborhood`; `/documents` vs `/document/{id}/...`; conversations routes are consistently plural. Standardize on plural collections + id segment, updating templates in one atomic edit. Low impact (all `include_in_schema=False` HTMX routes).

### Dead code

- **Six orphaned top-level functions** [#11] — one is a **security guard the docs falsely advertise**: `_canonicalize_source_path` (`source_parser.py:42`) promises path-traversal protection but `parse_source_payload` never calls it (`ingest/CLAUDE.md` lists it as an invariant). **Investigate before deleting** — if the server upload path is not otherwise guarded, wire it in rather than removing an advertised control. Also delete `claim_is_active`, `load_saved_pair`, `default_lexicon_hints`, `detect_corrupt_numerals`; expose-or-delete `split_proposal` (+ drop "split" from `review/CLAUDE.md`).
- **Five dead Cypher constants + two false comments** [#12] — `AUDIT_VIEW_QUERY:142`, `RESTORE_CLAIM_QUERY:1305`, `RESTRICT_CLAIM_PERMANENTLY_QUERY:1313`, `QUARANTINE_DOCUMENT_QUERY:1319`, `GAP_TEMPORAL_DENSITY_BY_INSTITUTION_QUERY:1458` are referenced nowhere; comments at `cypher.py:1456-1457` and `patch_spec.py:206` falsely imply liveness. Delete-or-wire (the restore/restrict/quarantine-document proposal types are documented, so wiring into `review/monitor.py` is plausible).
- **Unused imports** [#13] — Quick win (ruff F401).
- **`scripts/_archive/`** [#14] — Quick win (self-authorizing deletion).

---

## Suggested sequencing

1. **Land the tooling floor, config-only, in dependency order** [#15 → #18 → #19 → #17 → #16, #20]: ruff `[tool.ruff]` block → the single mechanical `ruff format` + import-sort commit (isolated, no logic) → lax mypy → pre-commit → CI gating on `ruff check` + `pytest -m "not integration"`. Do this **first** because it makes every subsequent refactor diffable and catches regressions the missing CI currently hides — and ruff already surfaces two real latent bugs to fix by hand (`phases.py:337`, unimported `Any`).

2. **Sweep the free wins that ride on the floor**: strip unused imports [#13], delete `scripts/_archive/` [#14] and the dead Cypher constants + orphaned functions [#11, #12] (but investigate the path guard rather than deleting), and kill the dead `cookie_secure` field [#29]. Small, mechanical, and they shrink the surface before the structural work.

3. **Fix observability, then narrow the swallows** [#24 → #25, #26, #27]: add module-level loggers to the two big apps (the keystone), then narrow the broad excepts and log the intentional ones. Cheap, high-value, and required before touching the god files so refactor-induced failures are visible.

4. **Write characterization tests BEFORE the big refactors** [#33, #34, #35, #37]: golden Cypher-string snapshots (zero DB) pin the query text; the extracted `run_query` service + `populated_executor` injection covers the orchestrator; auth/review/ingest route smoke tests lock the web surfaces; the `assert_populated` guard stops skip-hides-regression. These are the safety net for step 5.

5. **Execute the structural decomposition, one seam per commit, behind the tests**: shared `webapp.py` [#7] first (unblocks the rest), then the retrieval god-file split [#1/#42/#45/#39], `context_assembler` [#3/#43], `cypher` package [#4], `in_memory_executor` primitives [#5], `run_e2e` [#41], `cli` [#6], and the typed contracts [#21/#22/#23]. Fold the extraction-config-into-Settings move [#28] in alongside the ingest work. Do the higher-effort L items (`retrieval/web/app.py`, `run_e2e`) last, when the net is densest.

**Effort honesty:** Steps 1-3 are days, not weeks (mostly S). Step 4 is the real investment (the golden/parity tests and the `run_query` extraction are M-L). Step 5 is weeks of careful, incremental work — the `retrieval/web/app.py` DI conversion (#1) and `run_e2e` (#41) are genuinely L and should not be attempted before the tooling + tests exist.
