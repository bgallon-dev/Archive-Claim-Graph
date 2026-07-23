# Deep-research mode — scaffold

Status: **scaffold built and unit-tested; not wired into the live query path.**

## Why

On 2026-07-23 (demo day 2) we noticed the same question returning *different*
claims on repeated runs — none wrong, but each run missed useful material another
run found. That variance is the signal: a single retrieval pass samples only a
slice of the relevant evidence. Deep-research mode turns that into coverage by
running several angles on one question and synthesising over the **union**.

## What exists now

`gemynd/retrieval/deep_research.py` — self-contained, imports nothing from the
web app, so it cannot disturb the running server.

```
probe(question)            one cheap classify+resolve, to seed planning
  planner.plan(...)        question -> N research angles (sub-queries)
    gather(angle) × N      each angle runs the normal retrieval path
      merge + dedup        union blocks by claim_id; record who found each
        synthesise(union)  one answer over the combined evidence
```

Pieces:

| Piece | Role | State |
|---|---|---|
| `DeepResearchOrchestrator` | Runs probe → plan → gather → merge → synthesise | **Working** |
| `HeuristicPlanner` | Deterministic angles: original + per-entity + broadened. No extra LLM call. | **Working (default)** |
| `RetrievalBackend` (Protocol) | The one seam onto the existing engine | **Working** |
| `build_backend(...)` | Wires the backend from the components the web app already builds | **Working, untested against a live graph** |
| `DeepResearchResult.coverage_gain` | Claims found only past the first angle — the metric that justifies the mode | **Working** |

Tested in `tests/test_deep_research.py` (10 tests, no Neo4j / no Anthropic):
merge/dedup, coverage accounting, highest-confidence-wins on overlap, one-angle
failure tolerated, synthesis over the union, and a signature-pinning test for
`build_backend`.

Run a coverage measurement without paying for synthesis:

```python
res = orchestrator.research(question, scope, synthesise=False)
print(res.unique_claims, "claims;", res.coverage_gain, "beyond a single pass")
print(res.new_claims_by_angle)   # marginal contribution per angle
```

## What is deliberately NOT built (the next steps)

1. **Endpoint + UI toggle.** No change was made to `web/app.py` — the live demo
   is running and template/endpoint edits load on restart, so this stays out
   until after the event. To wire it, inside `create_app` where `state["gateway"]`
   etc. already exist:

   ```python
   from gemynd.retrieval.deep_research import (
       DeepResearchOrchestrator, PermissionScope, build_backend,
   )
   from gemynd.retrieval.context_assembler import (
       _serialise_block, _apply_context_char_budget, _infer_claim_types,
   )
   from gemynd.retrieval.classifier import classify_query

   def _classify(q):                       # bind the app's fixed params
       return classify_query(
           text=q, year_range=None, entity_hints=[],
           extra_stopwords=_dc.extraction_stopwords if _dc else None,
           year_bounds=state.get("classifier_year_bounds"),
           seed_vocab=state.get("classifier_seed_vocab"),
       )
   _intent_map = getattr(state.get("composite_config"), "query_intent_to_claim_types", None)

   backend = build_backend(
       gateway=state["gateway"], assembler=state["assembler"], engine=state["synthesis"],
       classify_fn=_classify,
       infer_claim_types_fn=lambda q: _infer_claim_types(q, intent_map=_intent_map),
   )
   orchestrator = DeepResearchOrchestrator(
       backend,
       char_budget_fn=_apply_context_char_budget,   # reuse the real char cap
       serialise_fn=_serialise_block,                # byte-identical context
   )
   ```

   Then add a `POST /query/deep` route (copy the auth/rate-limit guards from
   `/query`) that builds `PermissionScope(user.permitted_levels,
   user.permitted_institution_ids)` and returns `orchestrator.research(...)`.
   In the UI, a "Deep research" toggle next to "Stream as written". Expect
   **N× the latency** of a normal query (one retrieval pass per angle plus one
   synthesis) — this is a "tap and wait a few minutes" affordance, not the
   default. Rate-limit it harder than `/query`.

2. **LLM decomposition planner.** `HeuristicPlanner` is deterministic and free
   but blunt. A planner that asks the model to break the question into genuine
   sub-questions ("what were the causes / the consequences / the people / the
   dates") will produce sharper angles. Same `ResearchPlanner` interface — drop
   it in, no orchestrator change. Keep the heuristic as the offline/no-cost
   fallback.

3. **Completeness-critic loop.** After the first round, ask a critic "what is
   still missing?" and spend another angle on it, stopping when a round adds
   ~no new claims. The orchestrator already reports `new_claims_by_angle`, which
   is exactly the stop signal; the loop is the unbuilt part.

4. **Streaming.** `/query/stream` streams a single synthesis. Deep research adds
   a long retrieval phase first, so the honest UX is a staged progress list
   ("searching angle 2 of 4…") during gather, then stream the final synthesis.

## Guardrails preserved

Deep research **cannot widen access**. `PermissionScope` carries the same
`permitted_levels` / `institution_ids` the retrieval layer already enforces, and
every angle runs through the same `assemble()` gate — the mode re-runs permitted
retrieval more times, it never bypasses it.
