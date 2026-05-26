# kettle_eval

Active evaluation queue. ProEval-style failure discovery over the gemynd
pipeline: every extracted claim, mention, entity-resolution decision, page,
and claim-entity link becomes an `EvaluationCase`, scored by

    priority = 0.40*predicted_failure + 0.25*uncertainty + 0.25*impact + 0.10*topic_novelty

and persisted to its own SQLite store.

## Dependency tier

Sibling of `gemynd/`. Imports from `gemynd.core` (data models) and
`gemynd.review.detectors` / `gemynd.review.models` (read-only contributor
adapters). **Never imports from `gemynd.ingest`** — `pipeline.run_e2e` calls
into us, not the other way around.

Module dependency order: `shared -> core -> ingest -> review / kettle_eval`.

## Conventions

- `from __future__ import annotations` at the top of every module.
- `EvaluationCase` is a `@dataclass(slots=True)` subclass of `_BaseRecord`,
  same pattern as `gemynd.core.models`.
- IDs via `kettle_eval.ids`: `make_case_id` for per-run identity,
  `make_case_family_id` for the same evaluable object across runs.
- SQLite store mirrors `gemynd.shared.database_manager` conventions (WAL,
  `check_same_thread=False`, `foreign_keys=ON`, `busy_timeout=5000`) but is
  **standalone — intentionally not registered with the central manager**.
  Reason: the eval store opens per-`run_e2e` call inside the ingest process,
  unlike the web-server stores (`users`, `review`, `ingest`, …) that the
  manager owns for the lifetime of the FastAPI server. Registering would
  add Settings plumbing for an opt-in feature. See `store.py` module
  docstring for the full rationale and the v1 migration path.
- Upserts are idempotent on `case_id` — safe to re-run on the same
  `extraction_run_id` without doubling rows.

## Identity model

Three IDs per case:

- `case_id = stable_hash(case_type, object_id, extraction_run_id)` —
  per-run output. Re-running the pipeline produces a fresh row per run.
- `case_family_id = stable_hash(case_type, document_id, source_span_id,
  object_id)` — the underlying evaluable object across runs. Lets a future
  reviewer answer "this claim was risky in run A, fixed in run B" without
  the full event-tape infrastructure.
- `decision_class_id = stable_hash(case_type, …signature…)` — collapses
  cases that encode the *same decision* across runs and documents. Used
  by `top_k_per_decision_class` so the reviewer adjudicates each
  decision class once rather than once per occurrence. Signature is
  per-type:
  - `entity_resolution`: (normalized_surface_form, entity_id, relation_type)
  - `claim_extraction`: (normalized_sentence, claim_type)
  - `mention`: (normalized_form)
  - `relation`: (normalized_sentence, entity_id, relation_type)
  - `page_ocr`: (page_id) — pages are inherently per-class.

  Cases with an empty `decision_class_id` (pre-v0.2 rows) are excluded
  from `top_k_per_decision_class` to avoid lumping them into one fake
  cluster. The dedup view is purely additive — the audit view via
  `top_k` still sees every row.

For `entity_resolution`, `object_id` is **decision-level** —
`stable_hash("entity_resolution", mention_id, candidate_entity_id,
relation_type, run_id)`. Two mentions resolving to the same entity must
produce two cases. The evaluated unit is the decision, not the entity.

## Provenance gaps are risk, not absence

`risk_features.py` deliberately fires `missing_source_span`,
`missing_page_anchor`, and `missing_evidence_offsets` to 1.0 when the field
is `None`. A claim without inspectable source anchoring is not neutral —
it is review-worthy. The `claim_extraction` weight table reflects this
(provenance-gap weights > stylistic-flag weights).

## Topic novelty

`topic_novelty = 1 / sqrt(1 + count_seen_for_case_type_in_batch)`. A cheap
diversity proxy that prevents the queue from collapsing into the most
numerous case type (mentions dominate raw counts). Batch scope is per-call:
pass per-document case lists for per-document novelty; pass a corpus-wide
list for corpus-level novelty. The default wiring runs per-document inside
`run_e2e`.

Not a topic model. No embeddings, no BERTopic, no learned model. A real
topic taxonomy is v1+ work.

## Non-fatal error boundary

The pipeline-side wiring in `gemynd.ingest.pipeline.run_e2e` calls into
`kettle_eval` inside a `try/except`. By default, eval failures are logged
and the run continues. `--fail-on-kettle-eval` on the CLI flips this to
strict mode. Reason: kettle_eval observes `(structure, semantic)` — if
extraction succeeded but eval fails, the extraction work is still valid
and should be checkpointed.

## Worker pickling caveat

If anyone moves case generation into `extract_document` (the
`ProcessPoolExecutor` worker function), every detector adapter import
must be pickle-safe on Windows. Currently the eval pass runs in the
parent process after `_write_doc_to_graph`, so this is not a concern —
but stating it explicitly here for the next person.

## v0.3 — review labels + adaptive weights

Severity 0–4 scale is now live via `ReviewLabel`. Reviewers run
`label_cases.py` at the repo root; each label is validated against the
controlled `FAILURE_MODES` vocab and persisted to the `review_label`
SQLite table.

Every label triggers a small perceptron-style nudge on the per-case-type
risk-feature weight tables:

    observed = severity / 4.0
    prior    = case.predicted_failure_mean
    error    = observed - prior
    delta    = 0.05 * error * fired_value         # only fired features
    new_w    = clip(current + delta, 0.05, 3.0)

Weights persist in the `feature_weight` table. `score_cases(cases,
store=...)` overlays them on top of the module-level defaults via
`load_weights(store)`. Without a store, scoring stays purely
deterministic (v0.2 behavior). The pipeline wiring in
`gemynd/ingest/pipeline.py` passes the open store handle so the next
ingestion run benefits from accumulated labels.

The `_WEIGHT_TABLES` constants in `selector.py` remain the priors. Reset
adaptive learning with `python label_cases.py --eval-db <path>
--reset-weights`.

## Out of scope for v0.3 (deferred to v0.4+)

Learned `predicted_failure` model (replacing the linear blend), Bayesian
quadrature for `predicted_failure_uncertainty`, dashboard / new UI,
multi-reviewer agreement / disagreement detection, severity calibration
beyond `severity / 4.0`, `KettleEvent` event log, asynchronous syncer,
`RetrievalCase` / `AnswerCase` / `CitationCase`, bbox-dependent features
(handwriting, table/caption detection, newspaper column segmentation),
cross-store FK to review proposals, proper topic taxonomy.

## Testing

Tests live in `tests/kettle_eval/`. Use `tests/kettle_eval/conftest.py`'s
`bundle_factory` fixture to construct synthetic `(StructureBundle,
SemanticBundle)` pairs — no Neo4j, no LLM. The store tests use
`tmp_path`.

Critical tests guarding the v0 philosophy:

- `test_priority_formula_weights_sum_to_one` — guards against formula drift.
- `test_missing_source_span_raises_claim_priority` — provenance-gap-as-risk.
- `test_ambiguous_entity_resolution_has_high_impact` — corpus-wide blast radius. Threshold ambiguity (not actual cross-document evidence) is what the v0 feature observes; a separate `cross_document_entity_merge` feature can be added later when per-entity document tracking lands.
- `test_entity_resolution_decisions_not_collapsed` — decision-level identity.
- `test_store_upsert_idempotent` — re-run safety.
