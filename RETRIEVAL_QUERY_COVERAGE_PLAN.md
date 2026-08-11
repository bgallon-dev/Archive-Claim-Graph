# Retrieval query coverage plan

## Objective

Improve retrieval so that broad archival questions continue to return diverse,
useful evidence while specific questions reliably preserve their named entity,
event, date, relationship, corpus, and conversational scope.

This plan is organized as independently verifiable checkpoints. A checkpoint is
complete only when its required artifacts exist and its measurable gate passes.
Passing unit tests alone is not sufficient.

## Current evidence

The starting evidence is the 105 conversations currently recorded in
`data/conversation_log.db`, including these recurring failure shapes:

- A named Event resolves, direct claim anchoring returns zero, and the cascade
  drops the anchor.
- An exact day is reduced to a year-wide temporal scan.
- A correctly resolved entity has mention edges but no direct Claim-to-Entity
  edge.
- Multiword names are searched as a broad OR expression.
- Follow-up turns lose the entity/topic established by the previous turn.
- List, count, participant, and relationship questions are sent through
  evidence synthesis instead of a deterministic structural query.
- Compound questions are sometimes captured by a narrow Tier-1 intent.

The six cases in `data/benchmarks/newspaper_demo_queries.json` form the initial
broad-question regression set. They are not sufficient coverage for the
specific-query failures above.

## Verification model

### Two separate quality layers

1. **Retrieval gate — deterministic and required**

   Verify route selection, scope preservation, evidence facets, candidate
   bounds, access-control parameters, and latency. This gate must not depend on
   model prose.

2. **Answer gate — end-to-end and required before release**

   Verify citations, answer topic terms, caveats, and evidence-supported
   statements using the existing live benchmark. Because synthesis can vary,
   this is run three times and judged on aggregate results.

### Coverage manifest

Create `data/benchmarks/retrieval_query_coverage.json` with exactly 40 cases:

| Family | Cases | Purpose |
|---|---:|---|
| Broad regression controls | 6 | Existing newspaper demo questions |
| Named event and exact date | 6 | Event anchors, day/month/year, event-only chronology |
| Mention-only entity | 6 | Entity present through Paragraph/Mention rather than Claim edge |
| Conversational follow-up | 5 | Pronouns, omitted subjects, relative temporal scopes |
| Phrase and historical geography | 5 | Multiword names, settlements, aliases, containment |
| Structural list/aggregation | 5 | People, streets, entities, counts, rankings |
| Relationship/cross-corpus | 5 | Strict intersections, paths, ownership chains |
| Compound Tier-1 guards | 2 | A structural sub-question plus a narrative request |
| **Total** | **40** | |

Each case must contain:

```json
{
  "id": "stable_case_id",
  "family": "mention_only_entity",
  "turns": [{"text": "What are the mentions of the IWW?"}],
  "expected_routes": ["mention_anchored"],
  "forbidden_routes": ["entity_anchored:drop_entity_anchor"],
  "scope": {
    "year_min": null,
    "year_max": null,
    "exact_date": null,
    "entity_names": ["Industrial Workers of the World"],
    "institution_ids": ["spokane_newspaper"]
  },
  "required_evidence_facets": [
    {
      "claim_types": ["labor_action", "free_speech_event"],
      "source_terms_any": ["IWW", "Industrial Workers"]
    }
  ],
  "max_single_query_candidates": 250
}
```

Use evidence matchers such as document date, series, claim type, entity name,
and source-text terms. Do not make live acceptance depend exclusively on
`claim_id`, because re-ingestion can change run-qualified IDs.

### Benchmark artifacts

Create `scripts/benchmark_retrieval_plans.py`. It must support:

```powershell
python scripts/benchmark_retrieval_plans.py `
  --cases data/benchmarks/retrieval_query_coverage.json `
  --output out/benchmarks/query-coverage/<checkpoint>.json
```

Each result must record:

- case ID and family;
- classified intent and resolved/maybe/unresolved entities;
- every attempted rung, template name, row count, and duration;
- selected rung;
- candidate and context counts;
- matched/missing evidence facets;
- document dates, claim types, and institutions represented;
- whether requested scope was preserved;
- per-template candidate count;
- overall pass/fail and failed checks;
- source commit SHA plus a dirty-worktree flag.

The benchmark must redact credentials and must not copy full conversation
responses or user identifiers into committed fixtures.

### Global gates applied at every checkpoint

A checkpoint passes only when all of the following are true:

- All checkpoint-specific route and scope assertions pass.
- At least 90% of the checkpoint's required evidence facets are present in the
  top 50 retrieved claims, and every case has at least one matching facet.
- No single Cypher execution returns more than 250 candidates.
- All retrieval templates carry `permitted_levels` and `institution_ids`.
- The six broad controls lose no more than 0.05 relevant-claim ratio compared
  with the frozen baseline.
- Broad-control median retrieval latency regresses by no more than 15%.
- Focused-query median retrieval latency improves or stays below the frozen
  baseline; no focused case may exceed twice its baseline without a documented
  query-plan explanation.
- Targeted unit tests, `git diff --check`, and the non-integration retrieval
  suite pass.

For live synthesis, run the 40-case suite three times. Release requires:

- at least 38/40 cases passing in each run;
- every case passing at least two of three runs;
- zero unsupported citations;
- zero access-scope violations.

## Checkpoint 0 — Freeze the baseline and build the harness

**Status:** harness complete; live baseline pending

### Checkpoint record — 2026-07-30

- Added the exact 40-case manifest at
  `data/benchmarks/retrieval_query_coverage.json`, with the required
  `6/6/6/5/5/5/5/2` family distribution.
- Added `scripts/benchmark_retrieval_plans.py`. It records classifier output,
  entity-resolution partitions, route attempts, selected rung, safe
  per-template parameters, row counts, timings, evidence matches, represented
  dates/types/institutions, source SHA, and dirty-worktree state.
- Added manifest, evidence-matcher, multi-turn serialization, redaction, and
  forbidden-fallback tests in `tests/test_benchmark_retrieval_plans.py`.
- The targeted benchmark tests and adjacent non-integration retrieval suite
  pass.
- A live baseline attempt exceeded the 120-second execution window before an
  artifact was produced. A bounded offline route-capture artifact was saved at
  `out/benchmarks/query-coverage/checkpoint-0-baseline.json`; it is explicitly
  marked `connection_attempted=false`, `neo4j_available=false`, and must not be
  used as the latency or evidence-quality comparison baseline.
- Checkpoint 0 remains open until the same runner completes against a reachable
  configured Neo4j instance and replaces the offline artifact with live
  candidate, evidence, and timing measurements.

### Work

1. Curate the 40-case manifest from sanitized history examples plus synthetic
   coverage cases.
2. Add the retrieval-only benchmark runner.
3. Capture the current route, evidence, timing, and candidate metrics before
   additional retrieval changes.
4. Record the current event/date implementation as part of the baseline,
   including the fact that it has not yet executed against the configured live
   Neo4j instance.

### Required tests

- Manifest schema validation.
- Evidence matcher tests.
- Multi-turn case serialization tests.
- Benchmark output redaction test.
- Failing-case fixture proving that a forbidden fallback fails the benchmark.

### Pass gate

- Exactly 40 valid cases with the family counts above.
- The runner produces a machine-readable report even when individual cases
  fail.
- Every case has at least one deterministic scope or evidence assertion.
- Baseline report saved as
  `out/benchmarks/query-coverage/checkpoint-0-baseline.json`.

### Verifiable outcome

We can name every known query family, reproduce its current route, and compare
future checkpoints against a frozen result rather than memory.

## Checkpoint 1 — Validate and finish event/date retrieval

**Status:** implementation present; live validation pending

### Work

1. Run `EXPLAIN` and then execute `EXACT_DATE_CLAIMS_QUERY`.
2. Run `EXPLAIN` and then execute the event-aware
   `TEMPORAL_CLAIMS_QUERY`.
3. Replay:
   - “What happened in Spokane on November 3rd, 1909?”
   - “Who were the people named in the 1909 Free Speech Fight?”
   - “Can you give me a timeline of the free speech fights?”
4. Confirm broad year questions still include Observation- and Event-derived
   claims.
5. Decide strict no-evidence behavior: an exact-day miss must be reported as an
   exact-day miss; broad fallback evidence may be offered as context but may not
   be represented as evidence from that date.

### Required tests

- Date parsing variants and invalid-date rejection.
- Exact-day route precedes year-wide temporal retrieval.
- Event-only claims appear in temporal results.
- Named Event full text precedes event-anchor dropping.
- Exact-date no-result trace remains distinguishable from a broad fallback.

### Pass gate

- All 6 event/date cases pass route and scope assertions.
- The November 3 case returns only documents whose date interval covers
  `1909-11-03`.
- Named Event cases never select `entity_anchored:drop_entity_anchor` while a
  focused event route has results.
- Live `EXPLAIN` shows bounded document/claim expansion and no unrestricted
  all-node scan.
- Broad temporal controls retain both event and observation evidence.

### Evidence artifact

`out/benchmarks/query-coverage/checkpoint-1-event-date.json`

### Rollback boundary

Event/date template, date parser, event-routing rung, in-memory parity, and
their tests form one reversible unit.

## Checkpoint 2 — Add mention-anchored entity retrieval

**Status:** pending

### Work

Add a bounded template that traverses:

```text
Document → Page → Section → Paragraph → Claim
                              └→ Mention → Entity
```

Use confirmed `REFERS_TO` first and allow `POSSIBLY_REFERS_TO` only in a
separate, labeled relaxation rung. Deduplicate claims before optional
provenance expansion.

Run it after a direct Claim-to-Entity lookup returns zero, but before dropping
the entity anchor.

### Required tests

- Direct-claim edge still wins when present.
- Mention-only entity returns its containing claims.
- Confirmed mention edges outrank possible mention edges.
- Duplicate mentions in one paragraph produce one claim row.
- Access controls and latest-run filtering are preserved.

### Pass gate

- All 6 mention-only cases select `mention_anchored` or
  `mention_anchored:include_maybe`.
- No passing mention case reaches `drop_entity_anchor`.
- IWW history example returns evidence containing IWW/Industrial Workers terms.
- Broad entity-control relevant-claim ratio does not regress beyond the global
  threshold.

### Evidence artifact

`out/benchmarks/query-coverage/checkpoint-2-mention-anchor.json`

### Rollback boundary

New Cypher constant, one cascade rung, in-memory handler, and tests.

## Checkpoint 3 — Make retrieval conversation-aware

**Status:** pending

### Work

Add a deterministic contextual-query layer before classification and
resolution. It must carry forward only explicitly established fields:

- resolved entity IDs and surface names;
- event/topic claim types;
- document type, series, and institution scope;
- prior absolute year/date scope.

Relative expressions such as “after 1909” may replace the prior temporal scope,
but must retain the prior subject. The rewritten retrieval query and inherited
fields must be visible in the plan trace. Synthesis still receives the original
user wording.

Do not begin with an LLM rewrite. Start with deterministic inheritance so the
behavior is testable and auditable.

### Required tests

- “What about after 1909?” inherits the previous topic/entity and changes the
  lower year bound.
- “Are there other streets mentioned?” inherits the previous protest/event
  scope.
- A new explicit entity replaces, rather than merges with, the old subject.
- A new session inherits nothing.
- Access scope can never be inherited from another user/session.

### Pass gate

- All 5 follow-up cases resolve the expected inherited subject and scope.
- Zero follow-up cases route as unscoped full text when a prior subject exists.
- Trace output contains original text, effective retrieval text, and inherited
  fields.
- Identical single-turn questions retain their baseline routes.

### Evidence artifact

`out/benchmarks/query-coverage/checkpoint-3-conversation-context.json`

### Rollback boundary

Contextual-query object, rewrite function, web-to-assembler plumbing, and
multi-turn tests.

## Checkpoint 4 — Add phrase-first retrieval and entity-resolution recovery

**Status:** pending

### Work

For multiword names and unresolved proper nouns, use this relaxation ladder:

1. canonical entity resolution;
2. exact escaped phrase in claim full text;
3. all meaningful terms required;
4. bounded OR fallback.

Add a live graph entity-name/alias probe for unresolved capitalized phrases,
then cache successful resolutions. Keep Lucene sanitization and the existing
length cap.

### Required tests

- “Grand Coulee Dam” phrase search does not match claims containing only
  “Grand”.
- “Valleyford, Washington” preserves the two-word geographic intent.
- Punctuation and Lucene special characters remain escaped.
- A phrase miss relaxes to AND, then OR, with each attempt visible in trace.
- Entity alias recovery cannot introduce a label outside the domain allowlist.

### Pass gate

- All 5 phrase/geography cases have a relevant-claim ratio of at least 0.60 in
  the top 50.
- Phrase cases do not select OR fallback when phrase or AND results exist.
- Candidate count and latency satisfy the global gates.
- Existing broad topical full text remains unchanged.

### Evidence artifact

`out/benchmarks/query-coverage/checkpoint-4-phrase-recovery.json`

### Rollback boundary

Phrase/AND templates, alias probe, ladder ordering, and tests.

## Checkpoint 5 — Add deterministic structural list and aggregation templates

**Status:** pending

### Work

Add structural operations for:

- people/organizations/places participating in a focused event or claim set;
- locations or streets associated with a focused topic;
- entity co-occurrence within a focused claim set;
- grouped counts and rankings;
- date range for a document series.

Return stable rows with provenance counts. Use synthesis only when the user also
asks for interpretation.

### Required tests

- Participant lists contain only the requested entity label.
- Counts deduplicate entities and claims.
- A street/location query excludes organizations and people.
- Series date range uses document dates without claim expansion.
- A compound “list and explain” request preserves both structured rows and
  evidence context.

### Pass gate

- All 5 structural cases select an explicit structural operation.
- List/count results are deterministic across three executions.
- Returned totals match independent verification Cypher on the fixture graph.
- Pure structural questions make zero synthesis calls.

### Evidence artifact

`out/benchmarks/query-coverage/checkpoint-5-structural.json`

### Rollback boundary

Tier-1 intent additions, query-builder methods, response models, and tests.

## Checkpoint 6 — Require strict entity intersections before broad paths

**Status:** pending

### Work

For “link,” “connect,” “both,” and “between” questions:

1. require a single claim to match every requested entity;
2. if empty, try a bounded two-claim bridge through a shared entity;
3. if empty, try the existing bounded path templates;
4. only then degrade to ordinary multi-entity retrieval.

Keep cross-corpus overlap as a distinct route with one evidence facet reserved
per corpus.

### Required tests

- A strict two-entity case excludes claims matching only one entity.
- Two-claim bridges return both claim provenance chains.
- Path length and intermediate-node counts are bounded.
- Cross-corpus results contain evidence from every requested corpus.
- Duplicate aliases resolving to one ID do not trigger relationship routing.

### Pass gate

- All 5 relationship cases satisfy every requested entity/corpus facet.
- No strict-link case selects ordinary `multi_entity` while an intersection or
  bounded bridge exists.
- No query exceeds configured path depth or candidate caps.
- Relationship-query median latency improves over baseline.

### Evidence artifact

`out/benchmarks/query-coverage/checkpoint-6-relationships.json`

### Rollback boundary

Relationship intent, intersection/bridge queries, planner rungs, and tests.

## Checkpoint 7 — Model historical geography explicitly

**Status:** pending

### Work

Define reviewed geographic relationships and their temporal meaning, initially:

- `PART_OF`
- `NEAR`
- `LATER_INCORPORATED_IN`
- `FORMER_NAME_OF`

Add aliases for the known history cases—Spokane Valley, Millwood, Otis
Orchards, Greenacres, Valleyford, and Mica Peak—only after verifying that their
canonical nodes or source mentions exist.

Add a bounded place-neighborhood query. Do not use unrestricted variable-length
paths.

### Required tests

- Direct place matches rank before neighboring/contained places.
- Historical containment edges are traversed only when the query signals area,
  vicinity, community, former name, or later incorporation.
- Every inferred geographic expansion is shown in trace.
- Temporal properties on geographic edges are respected where present.

### Pass gate

- Geographic cases return at least one facet for each explicitly named place
  that exists in the archive.
- No query silently substitutes Spokane city for Spokane Valley.
- Expanded geography never crosses institution or access scope.
- Broad Spokane controls do not automatically expand to every neighboring
  community.

### Evidence artifact

`out/benchmarks/query-coverage/checkpoint-7-geography.json`

### Rollback boundary

Reviewed alias data, relationship backfill, neighborhood query, routing signal,
and tests. Keep the backfill separately reversible.

## Checkpoint 8 — Guard compound Tier-1 and no-evidence behavior

**Status:** pending

### Work

Require Tier-1 patterns to cover the complete user request. If a question
contains both a structural request and narrative/temporal analysis, either:

- return structured data plus normal retrieval context; or
- split it into two explicit sub-plans.

Add a scope-aware no-evidence result that distinguishes:

- no documents in requested scope;
- documents exist but no extracted claims match;
- claims exist only after relaxing scope;
- claims are excluded by access/quarantine.

### Required tests

- “When was Turnbull created? Give me a timeline” is not reduced to a year
  range alone.
- Exact-date misses do not claim the whole year was searched as if it were the
  requested day.
- Relaxed evidence is labeled as contextual, not in-scope evidence.
- Access-excluded evidence is never counted or described.

### Pass gate

- Both compound Tier-1 cases return all requested result components.
- All 40 cases expose an explicit scope outcome.
- No-evidence answers cite zero out-of-scope claims.
- Existing pure Tier-1 cases remain deterministic and make zero synthesis calls.

### Evidence artifact

`out/benchmarks/query-coverage/checkpoint-8-scope-outcomes.json`

### Rollback boundary

Tier-1 completeness guard, scope-outcome model, synthesis contract update, and
tests.

## Checkpoint 9 — Consolidated release gate

**Status:** pending

### Work

1. Run the complete unit and non-integration test suite.
2. Run live Neo4j `EXPLAIN` checks for every new template.
3. Run the retrieval-only 40-case benchmark.
4. Run the end-to-end 40-case benchmark three times.
5. Compare latency, candidate counts, evidence facets, and token use with
   Checkpoint 0.
6. Document migrations or backfills, rollback commands, and operational
   dashboards.

### Pass gate

- Every global retrieval gate passes.
- End-to-end benchmark meets the three-run release thresholds.
- No new Cypher query has an unbounded expansion.
- No access-control regression is present.
- All required migrations/backfills have a tested dry-run and rollback.
- A final report identifies remaining failed cases rather than averaging them
  away.

### Evidence artifacts

- `out/benchmarks/query-coverage/checkpoint-9-retrieval.json`
- `out/benchmarks/query-coverage/checkpoint-9-e2e-run-1.json`
- `out/benchmarks/query-coverage/checkpoint-9-e2e-run-2.json`
- `out/benchmarks/query-coverage/checkpoint-9-e2e-run-3.json`
- `plans/retrieval-query-coverage-release-report.md`

## Working protocol

For each checkpoint:

1. Record the starting SHA and dirty-worktree state.
2. Run and save the pre-change checkpoint cases.
3. Implement only that checkpoint's bounded unit.
4. Run targeted tests.
5. Run the checkpoint cases and six broad controls.
6. Inspect failures individually; do not accept an aggregate pass that hides a
   known history regression.
7. Save the evidence artifact.
8. Mark the checkpoint complete only after its gate passes.
9. Commit or otherwise snapshot the checkpoint before beginning the next one,
   so rollback never requires unwinding later work.

## Immediate next action

Finish Checkpoint 0 by running the retrieval-plan benchmark against a reachable
configured Neo4j instance and replacing the explicitly offline artifact:

```powershell
python scripts/benchmark_retrieval_plans.py `
  --cases data/benchmarks/retrieval_query_coverage.json `
  --output out/benchmarks/query-coverage/checkpoint-0-baseline.json
```

Do not use the offline timings as frozen comparisons and do not add another
retrieval template yet. Once the live baseline is captured, validate the
already-implemented event/date work as Checkpoint 1 before starting
mention-anchored retrieval.
