# Gemynd Pipeline — System Audit (2026-07-08)

**Scope:** `gemynd/` (auth, core, ingest, retrieval, review, shared), `scripts/`, `tools/`, `tests/`, and the repo-root scratch scripts.
**Method:** 7 parallel finder lenses → adversarial verification of every finding against the real code → synthesis. 32 findings raised, 31 confirmed; the retrieval-memory lens was re-run separately after an output failure. File:line refs below were verified by re-reading the code.

## Update — fixes applied 2026-07-08

Low-risk, high-value items below were implemented and verified (full suite: 1064 passed,
1 skipped, 0 regressions; the 4 failures are the pre-existing CSRF-template render tests —
see note at end). **Applied:** §2.1 system-prompt hash centralization, §2.3 `.env` loaders,
§2.4 `_normalize_endpoint`, §2.5 `normalize_for_match`, §2.6 `save_rows_csv`, §2.7 nested
`_parse_year` removal, §2.2 `build_neo4j_executor` `trust_mode` default (behavioral seam
closed; direct call sites left as-is), §5.1/§5.2 test cleanups (mallard → positive `xfail`;
duplicate 1938 test removed), §6 dead-script archive (6 files → `scripts/_archive/`).
**Bonus:** fixed a pre-existing f-string-backslash `SyntaxError` in `review/web/app.py:890`
that broke `review-serve` on Python 3.10/3.11.

**Deferred** (need focused review or a running server to verify): §4.1 shared web-app
bootstrap extraction (security-critical CSRF/setup-guard wiring, thin test coverage of the
exact wiring), §3.5/§3.3 PDF-module extraction + ingest-job de-dup, §3.1 PDF subprocess
isolation, §3.2 durable ingest queue, and the low-value validation-guard helpers (the web
app's `TYPE_CHECKING`-guarded fastapi import makes a clean helper awkward).

**Environment note:** the declared optional deps `bcrypt`, `python-jose[cryptography]`,
`hypothesis`, `python-multipart` were installed into the local interpreter so the web/auth
stack and full suite could run.

**Now-fixed template failures:** the 4 `test_gaps_context` / `test_history_review_ui`
tests that failed at `shared_templates/gemynd_base.html:1538` were resolved by making the
CSRF `hx-headers` token expression request-null-safe (`... if request else ""`). Every
template test renders full pages with `request=None` (the codebase convention); the base
template's CSRF line was the only `request.` reference that didn't tolerate it. Production
always renders through `Jinja2Templates` with a real `request`, so this is a no-op there.
**Full suite is now green: 1068 tests, 0 failures, 1 skipped, 1 xfail (the mallard gap).**
Note: the same `request.cookies.get('csrftoken','')` pattern appears in 7 other templates
(`login.html`, `setup.html`, `upload.html`, `admin_ingest.html`, `batch.html`,
`history.html`) — not test-covered and production-only, left unguarded for now.

## Bottom line up front

- **There is no memory-safety crisis to fix.** The heaviest parts of the pipeline are already well-bounded. You cannot "get back" much memory because very little is being wasted in a way that matters — no leaks, no unbounded per-request growth on the live path.
- The retrieval **`in_memory_executor` is a red herring** — it's test-only and holds a *reference*, not the graph. Production uses the streaming Neo4j executor.
- The only genuine **process-lifetime** memory item worth watching is the boot-time `entity_degree` whole-graph dict (no `LIMIT`) in the retrieval app — matters only on a very large graph.
- The **real substance** of this audit is elsewhere: (a) untrusted PDF parsing runs in-process in the web server, (b) ingest is glued onto FastAPI `BackgroundTasks` with no durable queue, and (c) a broad centralization backlog where shared helpers already exist but callers bypass them.

## Highest-leverage actions

1. **Centralize the claim-cache system-prompt hash** — `sha256(...).hexdigest()[:16]` is hand-copied in `anthropic_claim_adapter.py:239`, `openai_claim_adapter.py:161`, `scripts/batch_prewarm_claims.py:190`. One-char drift silently invalidates all batch-prewarm LLM spend, with no test to catch it. Add `system_prompt_hash()` beside `claim_cache.py:29`.
2. **Sandbox PDF/OCR parsing out of the web process** — `ingest/web/app.py:140` and `retrieval/web/app.py:3944` parse attacker-supplied PDFs via `asyncio.to_thread → fitz.open` in-process; reuse the existing `_process_single_document` ProcessPool worker (`ingest/pipeline.py:759`).
3. **Take ingest off `BackgroundTasks`** — `ingest/web/app.py:247` schedules minutes-to-hours jobs in-process; a restart leaves the SQLite status row stuck at `converting`/`extracting` forever.
4. **Extract the triplicated web-app bootstrap** — CSRF, no-store, setup-guard, error handlers, Jinja loader are copy-pasted across `ingest/web/app.py:104`, `review/web/app.py:983`, `retrieval/web/app.py:1440`. Add `gemynd/shared/web_app.py`.
5. **Collapse the duplicated ingest-job body** — `retrieval/web/app.py:3932` is a near-verbatim (already-drifting) copy of `ingest/web/app.py:130`.
6. **Retire the dead newspaper-ingest chain** — delete `scripts/ingest_newspapers.py` (points at a non-existent dir) and retarget or archive `scripts/batch_prewarm_claims.py` (both fail-fast on the missing `Newspapers/` path); remove the three repo-root `_*.py` scratch scripts.
7. **Route Neo4j executor construction through the existing helper** — `shared/neo4j_helpers.build_neo4j_executor` is used exactly once while ~11 sites duplicate the constructor with inconsistent `trust_mode`/`ca_cert_path` handling.

---

## 1. Memory safety in the heaviest pipeline parts

### 1a. Retrieval (heaviest — 12k LOC)

**Verdict: memory-safe and well-bounded. No high-severity finding.** Per-request work is transient and hard-capped (≤250 rows per Cypher, ≤250 KB serialized context, ≤1000 blocks). All pipeline components are built once at startup and shared. Long-lived structures are bounded drop-oldest queues or SQLite-on-disk.

- **`in_memory_executor.py` does NOT hold the graph in production (debunk).** `InMemoryQueryExecutor.__init__` (`in_memory_executor.py:47`) stores a *reference* to a caller-supplied writer; it never loads/copies a graph. Grep confirms it's referenced only in tests + `cypher.py` + CLAUDE.md — never in `web/app.py`. Production wires `Neo4jQueryExecutor` (`retrieval/web/app.py:1117`), which streams rows and materializes only the current result set (`executor.py:81`). (Test-only caveat: the in-memory executor does quadratic linear scans of `rel_store`, e.g. `in_memory_executor.py:317`, `:367`, `:579` — a test-fixture CPU concern, not production memory.)
- **[medium at scale, else low] Boot-time `entity_degree` dict = unbounded whole-graph load, pinned for process lifetime.** `ENTITY_DEGREE_QUERY` (`core/graph/cypher.py:1105`) has **no `LIMIT`** — one row per `:Entity` in the whole graph — materialized into a process-lifetime dict at `retrieval/web/app.py:1141`. Loaded once, never rebuilt per request, grows only with the graph (not traffic). On a large multi-corpus graph (100k–1M+ entities) easily 100s of MB, with a ~2x transient spike during the dict comprehension. **Fix:** the path-finder only needs the degree-cap threshold + the set of above-cap hub ids (`web/path_finder.py:214`); precompute those at startup and discard the full map, or push hub-exclusion / inverse-frequency weighting into Cypher.
- **[low] `TokenBucketLimiter` never evicts idle keys.** `shared/rate_limit.py:37` prunes each bucket's timestamp *list* but never deletes the *key*, so a key seen once lingers forever (often as an empty list). Singleton (`app.py:1045`). Cardinality is bounded by distinct authenticated users (key = `user.user_id or institution_id`, `app.py:1530`), not attacker-amplifiable. Docstring's "memory stays bounded" is overstated. **Fix:** `del self._buckets[key]` when the pruned list is empty, or a TTL sweep.
- **Checked and fine:** assembler→ranker→synthesis block copies are transient and capped (`context_assembler.py:1548`, `_CONTEXT_CHAR_BUDGET` `:74`); Cypher fetch hard-capped at `_MAX_CYPHER_FETCH_LIMIT = 250` (`context_assembler.py:80`); synthesis buffers streamed chunks then joins once, bounded by `max_tokens` (`synthesis.py:765`); `ConversationLogger`/`TokenUsageLogger` are `maxsize=200` drop-oldest queues draining to SQLite (`conversation_log.py:256`, `token_tracker.py:308`) with a 90-day disk purge daemon (`conversation_log.py:589`); `SqliteCypherCache` is on-disk; `EntityResolutionGateway` caches are bounded by the seed vocabulary (`entity_gateway.py:56`).

### 1b. Ingest (9k LOC)

**Verdict: no leaks or unbounded growth. Every item here is transient, GC'd, redundant work — real inefficiency, but "memory-safety" overstates it. All low severity.**

- **[low] Full-document OCR text duplicated ~2x at the document level.** `source_parser.py:198` joins all page text into one giant `raw_ocr_text` string, then `:199` makes a second full copy `clean_text = normalize_text(...)`; both stored on `DocumentRecord` and persisted via `to_dict()` (`graph/writer.py:68`). Same text already lives per-page and per-paragraph. Only the doc-level ~2x is genuinely redundant. **Not free to remove:** `cypher.py:1000` `GET_DOCUMENT_SOURCE_TEXT_QUERY` reads `d.clean_text`/`d.raw_ocr_text` for the source-preview modal (there's a paragraph-reconstruction fallback at `cypher.py:1015`). **Fix:** drop the doc-level fields and switch the modal to the paragraph fallback, or compute the concat lazily.
- **[low, efficiency not memory] Resolver / seed gazetteer / domain-config rebuilt once per document.** `_write_doc_to_graph` builds a fresh `DictionaryFuzzyResolver` per doc after the first (`pipeline.py:661`) via `default_seed_entities` (re-reads `seed_entities.csv`, no cache) and the ctor reloads `claim_relation_compatibility.yaml` (`resolver.py:157`); `extract_document` reloads domain config per doc (`pipeline.py:402`) despite `run_e2e` holding `_run_config` (`pipeline.py:530`); `load_domain_config` does no caching by design (`domain_config.py:582`). Low payoff — per-doc LLM extraction dwarfs this. **Fix:** build the resolver once before the loop, pass `document_entity_counts` into `resolve(...)`; thread `_run_config` into `extract_document`.
- **[low, CPU/IO, neo4j path only] Neo4j pre-flight re-reads + re-serializes every input the workers re-hash.** The dup-check loop does `read_text → json.loads → json.dumps → sha1` per input (`pipeline.py:546`), then `parse_source_payload` computes the identical hash again (`source_parser.py:200`). Peak memory does not double (one file at a time). **Fix:** compute `file_hash` once in the parser and pass it (with the parsed payload) to the worker.
- **[low] Checkpoint file read + split in full twice per run.** `run_e2e` calls `load_checkpoint` then `load_checkpoint_inputs` back-to-back (`pipeline.py:625`); each does `read_text().splitlines()` with per-line `json.loads` (`checkpoint.py:44`, `:68`). Sequential, so peak is one file-worth. **Fix:** one loader returning both the doc_id set and input set.
- **[low] `cross_corpus_ner` keeps one dict per match for the whole run.** `all_report_rows` appended unconditionally (`enrichment/cross_corpus_ner.py:552`). Bounded by corpus size (~tens of MB), one-shot batch CLI. The sole caller *does* consume the rows for summary stats (`scripts/enrich_cross_corpus_persons.py:135`), so the naive "replace with an int counter" fix would break it. **Fix (optional):** stream rows to a file handle + keep separate counters.

---

## 2. Function duplication to centralize

Shared modules already exist for most of this; the issue is callers bypass them.

- **[medium] System-prompt fingerprint computed by hand in 3 places.** `_cache_key` (`claim_cache.py:29`) consumes a `system_prompt_hash` but doesn't produce it; the producer `sha256(system_prompt.encode('utf-8')).hexdigest()[:16]` is triplicated at `anthropic_claim_adapter.py:239`, `openai_claim_adapter.py:161`, `scripts/batch_prewarm_claims.py:190`, aligned only by a comment. Divergence → the entire batch-prewarmed cache becomes invisible to production. **Fix:** add `system_prompt_hash(prompt)` to `claim_cache.py` and call it from all three.
- **[low/medium] `Neo4jQueryExecutor` construction duplicated ~11 times.** `build_neo4j_executor` (`shared/neo4j_helpers.py:12`) exists as "the single source of construction" but is used exactly once (`retrieval/__init__.py:37`). Sites diverge on `trust_mode` (`status.py:122`, `pipeline.py:357`/`:559`, `html_report.py:92` append `or 'system'`) and `ca_cert_path` (omitted by `diag_composite_corpus.py:28` + backfill scripts). Only the three settings-based sites (`status.py:122`, `diag_composite_corpus.py:28`, `retrieval/web/app.py:1117`) are drop-in candidates. **Fix:** route those three through the helper; give it a `trust_mode='system'` default.
- **[low] Manual `.env` parsing reimplemented in 3 backfill scripts** (`backfill_loc_fields.py:31`, `backfill_about_refuge.py:7`, `backfill_about_spokane.py:14`) while `shared/env.load_dotenv` (`env.py:7`, used by 15 files) already handles comments/`export`/quotes. **Fix:** call `load_dotenv(REPO_ROOT / '.env')`.
- **[low] `_normalize_endpoint` duplicated verbatim** between `shared/openai_compat.py:56` and `ingest/extractors/openai_claim_adapter.py:40`; the adapter also re-implements the whole OpenAI chat POST that `openai_compat._OpenAIMessages.create` (`:168`) provides. **Fix:** import from shared; longer term build on `OpenAICompatibleMessagesClient`.
- **[low] `normalize_for_match` reimplemented inline** at `mention_extractor.py:113` and `:425` (both are exactly `re.sub(r'\s+',' ', surface.strip().lower())` = `shared/text_utils.py:12`); `resolver.normalize_name` (`resolver.py:109`) is a punctuation-stripping superset that should also delegate. **Fix:** use `normalize_for_match`; refactor `normalize_name` to strip then call `collapse_whitespace`.
- **[low] `save_rows_csv` duplicated** by `scripts/json_to_csv.py:36` (identical union-of-keys `DictWriter`) vs `shared/io_utils.py:21`. **Fix:** import and call `save_rows_csv`.
- **[low] `_parse_year` defined twice in one file** — module-level `retrieval/web/app.py:105` vs nested `:2465` (already diverge on exception handling). **Fix:** delete the nested copy. (`ingest/derivation_context.py:54` is a genuinely different regex extractor — leave it.)

---

## 3. Containerization — over- vs. under-isolating

There are **no Dockerfiles** in this repo — if you meant OS/Docker containerization, none currently exists. Interpreting "containerize" as isolation/boundary decisions: there is **no over-isolation** worth flagging (the three-app split is defensible). The problems are all **under-isolation**, plus one place the split is undercut by duplication.

- **[medium] Untrusted PDFs parsed by native C code in-process, bypassing the existing ProcessPool.** Uploaded `.pdf` bytes → `asyncio.to_thread(convert_pdf_to_json, ...)` → `fitz.open`/`get_pixmap(dpi=300)` (PyMuPDF/MuPDF native, long CVE history) + pytesseract, in the FastAPI worker thread — same process holding auth state, CSRF secret, DB connections (`ingest/web/app.py:140`, `pdf_converter.py:27`/`:77`, `retrieval/web/app.py:3944`). A native crash/hang takes the service down. The pipeline already has a pickle-safe `_process_single_document` ProcessPool worker (`pipeline.py:759`) the web path ignores. **Auth note:** uploads require `require_archivist_or_admin` (ingest) / `require_admin` (retrieval), so this is an insider / malicious-document *availability* risk, not unauthenticated RCE. **Fix:** run conversion+extraction in a short-lived subprocess with a wall-clock timeout and (Linux) `RLIMIT_AS`/`RLIMIT_CPU`.
- **[medium] Ingest runs as in-process `BackgroundTasks` with no durable queue.** `create_ingest_job` fires the whole convert+extract pipeline via `background_tasks.add_task(_run_job, ...)` (`ingest/web/app.py:247`). No queue/worker/persistence (no celery/rq/redis anywhere). On restart/crash mid-job the task is gone but the SQLite row stays at `converting`/`extracting` forever; no concurrency cap. **Fix:** move to a subprocess runner using the existing SQLite job table as the queue, with a concurrency cap; keep the web layer to enqueue + poll.
- **[medium] The retrieval app re-implements the ingest job inline.** `_run_ingest_job` (`retrieval/web/app.py:3932`) is a near-verbatim copy of `_run_job` (`ingest/web/app.py:130`) — same validation, staging, `to_thread` calls, status transitions — and has **already drifted** (ingest has a dedicated `except ConversionError` at `:165`; the retrieval copy collapses to `except Exception` at `:3958`). Any fix to the two items above must currently land twice. **Fix:** extract one shared ingest-job function; then decide deliberately whether ingest belongs in the retrieval service at all.
- **[low] `PersistResult` typed dataclass downgraded to a loose dict at the process boundary.** A pickle-safe typed `PersistResult` exists (`extraction_result.py:41`) but `_process_single_document` returns `to_summary_dict()` (`pipeline.py:500`) across the ProcessPool boundary, consumed by string key downstream (`quality.get('claim_count')`, `ingest/web/app.py:161`, `retrieval/web/app.py:3954`). A renamed key silently becomes `None`. **Fix:** return `PersistResult` (or a small typed summary); keep `to_summary_dict` only at the JSON edge.
- **[low] PDF/OCR extraction copy-pasted** between `ingest/pdf_converter.py` (docstring: "Refactored from tools/pdf_to_json.py") and `tools/pdf_to_json.py`; only `_ocr_page_image` is byte-identical, others share logic with differing signatures; already drifted (tool has `--meta-claude`/`--meta-prompt` the converter lacks). **Fix:** promote primitives into one module (`gemynd/ingest/pdf_extract.py`) — also the natural home for the §3.1 sandboxing.

---

## 4. Duplicated code blocks

Distinct from §2 — pattern duplications with no existing shared home.

- **[medium] Web-app factory bootstrap triplicated across all three FastAPI apps** (`ingest/web/app.py:104`, `review/web/app.py:983`, `retrieval/web/app.py:1440`, plus Jinja loaders / error handlers / `_no_store` / `_setup_guard` nearby). ~50–65 lines of security-critical wiring (CSRF, cache-control, setup-redirect, 500-swallowing) hand-copied — the CSRF comment even cross-references retrieval, confirming copy-paste. Minor variants exist (secret source, `/health` "starting" branch, an extra `_request_id_middleware` in retrieval) but they're logically equivalent. **Fix:** `gemynd/shared/web_app.py` with `install_common_middleware(app, users_db_path)`, `build_templates(templates_dir)`, `make_health_response(...)`.
- **[low] Hybrid rules+model merge algorithm duplicated** between `claim_extractor.py:530` and `mention_extractor.py:470` (docstrings say they mirror each other). Already diverged (claim has extra `label_changed`/`links_inherited` + different rules-only survivor logic), so a single generic helper is more awkward than it looks — weigh before extracting.
- **[low] Overlap-suppressing term matcher duplicated** — `mention_extractor.py:318` (`_scan_terms`) vs `claim_extractor.py:171` (`_find_term_matches`), identical span-collision body. **Fix:** move to `span_utils.py`/`text_utils.py`.
- **[low] `RuleBasedMeasurementExtractor.__init__` duplicates its config-vs-resources branches** (`measurement_extractor.py:144` and `:159`; identical trailing regex-build block). **Fix:** normalize both inputs to dicts, then one shared assign+compile block.
- **[low] Identical config-fallback preamble** in `claim_extractor.py:220` and `mention_extractor.py:176`. **Fix:** `resolve_domain_config(config, resources_dir)` in `domain_config.py`.
- **[low] Repeated entity_id/doc_id validation guards** across retrieval routes (`retrieval/web/app.py:2265`, `:2319`, `:2363`, `:2697`, `:2711`, `:2733`) — already drift in wording. **Fix:** `_validate_entity_id`/`_validate_doc_id` helpers or a FastAPI dependency with `pattern=`.

---

## 5. Obsolete / low-value tests

Small — only two items, both low, both in `tests/test_e2e_integration.py`. Nothing systemic; the suite is healthy.

- **[low] Inverted test asserts a known bug still exists** — `test_mallard_extraction_gap_documented` (`test_e2e_integration.py:225`) computes `gap_ids = resolved_ids - in_writer` and asserts `assert gap_ids`, i.e. requires the coverage gap to *remain*; its own message says "gap appears to be CLOSED ... Remove this test." Also `pytest.skip()`s if `'mallards'` fails to resolve. **Fix:** delete (surrounding `TestGatewayGraphConsistency` already pins the real round-trip via `'coots'`) or convert to a positive assertion.
- **[low] Duplicate e2e test** — `test_temporal_query_1938_returns_nonzero_blocks` (`:268`) and `test_no_entity_context_assembler_uses_refuge_anchor` (`:346`) make the identical `assemble("wildlife activity in 1938", EntityContext(), year_min=1938, year_max=1938)` call + bare `assert blocks`; only the failure message differs. **Fix:** delete one (keep the anchor-named one). `test_blocks_carry_correct_year_value` (`:281`) is distinct — keep it.

---

## 6. Dead scratch & one-off scripts

Largest count, all low-to-medium housekeeping.

- **[medium] `scripts/batch_prewarm_claims.py`** — substantive Batches-API infra (submit/poll/fetch, resumable manifest), but dead as written: hardcodes `NEWSPAPERS_ROOT` and `SystemExit`s when the missing `Newspapers/` dir isn't found; cache keys pinned to the Newspapers `domain_schema.yaml` prompt hash; its named consumer (`ingest_newspapers.py`) is itself dead. **Fix:** retarget input enumeration to the live `ingest_tutorstructure` corpus (or make it a CLI arg), or archive — don't delete blindly.
- **[low] Three repo-root scratch scripts** (`_audit_smoke.py`, `_corpus_stats.py`, `_resmoke_minting.py` at `Desktop/Gemynd/`) — outside the package, underscore-prefixed, unreferenced, hardcoding disposable `out/` smoke paths. `_audit_smoke.py` is a spent Turnbull-leak-check harness for the domain-agnosticism refactor. **Fix:** delete or move to a git-ignored `scratch/`. If the Turnbull-leak assertions are still wanted, port to a parametrized regression test in `tests/`.
- **[low] `scripts/ingest_newspapers.py`** — `INPUT_DIR`/`DOMAIN_DIR` resolve under `<Gemynd>/Newspapers`, which doesn't exist → `SystemExit`; superseded by `ingest_tutorstructure.py` (the only recently-modified script, targets `Desktop/Tutorstructure/corpus`). Referenced only in stale docstrings. **Fix:** delete; keep `ingest_tutorstructure.py` and parametrize its corpus path.
- **[low] `scripts/fix_entity_labels.py`** — spent one-time migration repairing an old `:Entity`/`:Place` split that no longer exists; hardcodes Turnbull labels + Refuge→Place coercion now handled by config (`domain_config.py:236`, `writer.py:144`). **Fix:** delete or move to `docs/migrations`.
- **[low] `scripts/diag_composite_corpus.py`** — one-off "does the composite include Spokane?" debug script hardcoding `['spokane_newspaper']`/`['turnbull']`. **Fix:** delete or fold into `tools/run_diagnostics.py` with `institution_ids` as an argument.
- **[low] `scripts/audit_false_positives.py`** — generic CSV analyzer with `SUSPICIOUS_PATTERNS` (`:25`) hardcoded to Turnbull refuge claim-types/species. **Fix:** externalize the pattern table to a corpus-resources YAML, or treat as Turnbull-only and archive.

---

## Suggested sequencing (best safety/effort ratio first)

1. **Centralize the system-prompt hash** (§2) — small, prevents silently wasting all batch-prewarm LLM spend.
2. **Delete/retarget the dead scripts** (§6) — pure cleanup, zero runtime risk, removes stale-path confusion before anyone builds on it.
3. **Extract the shared web-app bootstrap** (§4) into `gemynd/shared/web_app.py` — prerequisite that single-sources the CSRF/setup-guard/error wiring before touching ingest paths.
4. **Collapse the duplicated ingest-job body** (§3) — must precede the sandboxing so the fix lands once.
5. **Sandbox PDF parsing out of the web process** (§3), building on a shared `pdf_extract` module and reusing `_process_single_document`; add a wall-clock timeout even before full resource limits.
6. **Move ingest off `BackgroundTasks` to a durable subprocess runner with a concurrency cap** (§3) — largest effort; the SQLite job table already exists to serve as the queue.
7. **Opportunistic low-cost centralizations** as you touch files: `.env` loader, `_normalize_endpoint`, `normalize_for_match`, Neo4j-executor helper (3 sites), nested `_parse_year`, validation-guard helpers, the two test cleanups.
8. **Defer** the memory-lens items — the retrieval `entity_degree` map only matters at very large graph scale (§1a); everything else is low-payoff transient work. **Do not spend effort chasing memory safety — it is not where the risk is.**
