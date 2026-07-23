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

## Cost — where it balloons and what holds it down

Deep research spends more than a single query on two axes. Both are now
controlled; the controls default safe.

**Axis 1 — retrieval-side LLM-Cypher fan-out.** The assembler calls an LLM to
generate Cypher only when a pass finds nothing rule-based (confidence < 0.7).
Vague reformulated angles are the phrasings most likely to find nothing, so a
naive 4-angle run could fire 4 LLM-Cypher calls instead of one.
*Control:* the orchestrator passes `allow_llm_cypher=True` only on the original
angle (`assemble(..., allow_llm_cypher=False)` on the rest). A fan-out costs at
most one LLM-Cypher call. Flip `llm_cypher_on_reformulations=True` to opt out.

**Axis 2 — synthesis over a denser union (the dominant cost).** One synthesis
call, but the union fills the context budget with more unique claims than a
single pass, so it sends far more input tokens.
*Control:* the union is ranked **centrality-first** (claims independently found
by multiple angles lead, confidence breaks ties), so a deliberately *smaller*
synthesis char budget than the retrieval maximum keeps the most-corroborated
evidence and trims the single-angle tail. Pass a tighter `char_budget_fn`:

```python
from functools import partial
from gemynd.retrieval.context_assembler import _apply_context_char_budget
DeepResearchOrchestrator(
    backend,
    char_budget_fn=partial(_apply_context_char_budget, char_budget=120_000),  # vs 250k default
    serialise_fn=_serialise_block,
)
```

### Rough numbers (order-of-magnitude, Sonnet 4.6 at $3/1M in, $15/1M out)

| | context in | synthesis | + LLM-Cypher | ≈ per query |
|---|---|---|---|---|
| Single pass | ~30K tok | ~$0.16 | ≤ 1 call | **~$0.16** |
| Deep, uncontrolled (4 angles) | ~62K tok | ~$0.26 | up to 4 calls | **~$0.36** |
| Deep, both controls on | ~30K tok | ~$0.16 | 1 call | **~$0.18** |

So the controls bring a 4-angle run from ~2.3× down to ~1.1× the cost of a
single query, while still *finding* the full union (coverage is reported from
all angles; only the synthesis input is trimmed). These are estimates — chars/
token and context sizes vary. **The real figures are already logged**:
`MeteredAnthropicClient` writes per-call token usage to `data/token_usage.db`
(`caller="synthesis"`), and `/api/token-stats` surfaces it. Measure there rather
than trust this table. `DeepResearchResult.estimated_context_tokens` is a free
pre-call gauge that the synthesis budget is biting.

### Structural options (not built)

* **Batch API — 50% off.** Deep research is already "tap and wait minutes,"
  which is the Batch API's sweet spot. It trades away the live streaming we
  built (async submit + poll), so it fits a "queue a deep report, come back for
  it" UX, not the interactive toggle. Biggest single lever if volume grows.
* **Prompt caching the synthesis system prompt.** One-line `cache_control` on
  the stable system prompt. Modest here — the system prompt is small next to the
  per-query context, which differs every call and can't be cached across
  queries. Helps most on the validation-retry path (same prefix re-sent). Touches
  the shared synthesis path, so after the demo.
* **Cheaper model / lower effort for the LLM planner** (when built) — decomposition
  is a small, structured task; run it on Haiku, keep synthesis on Sonnet.
* **Zero-gain telemetry.** When `coverage_gain == 0`, the extra angles added
  nothing — surface it so the UI can say "no additional evidence beyond a
  standard query," and so a future policy could skip the fan-out for
  question shapes that never benefit.

## Guardrails preserved

Deep research **cannot widen access**. `PermissionScope` carries the same
`permitted_levels` / `institution_ids` the retrieval layer already enforces, and
every angle runs through the same `assemble()` gate — the mode re-runs permitted
retrieval more times, it never bypasses it.
