# Deep-research mode

Status: **bounded investigation loop implemented, unit-tested, and exposed at
`POST /query/deep` and `POST /v1/query/deep`.** Browser UI and streaming are not
part of this iteration.

## Pipeline

Deep research is an opt-in, synchronous investigation over the normal retrieval
path:

```text
probe -> seed original + heuristic fallbacks
      -> gather one angle -> merge/deduplicate claims
      -> ask the optional evidence-gap adviser for follow-up searches
      -> validate and prioritize accepted proposals
      -> repeat under deterministic budgets
      -> synthesize once over the ranked union
```

The model is advisory. It cannot stop a run, widen access, alter budgets, enable
LLM-Cypher, or select final evidence. Malformed output, unknown fields, duplicate
queries, replacement of the original query, and queries longer than 500
characters are rejected. If the adviser fails once, the run disables it and
continues with deterministic heuristic fallbacks.

## Deterministic controls

Defaults:

| Control | Default |
|---|---:|
| Total gather attempts, including original | 4 |
| Consecutive successful zero-gain attempts | 2 |
| Gather failures | 2 |
| Adviser calls | 3 |
| Accepted proposals per adviser call | 2 |

Stop conditions are evaluated in this order:

1. `angle_budget_exhausted`
2. `failure_budget_exhausted`
3. `no_new_evidence`
4. `no_candidate_gaps`

A successful attempt that adds evidence resets the zero-gain streak. Failed
gathers consume attempt and failure budgets but never masquerade as evidence
saturation. There is no wall-clock or model-confidence termination in the core.

## API behavior

The deep endpoint accepts the same `QueryRequest` as `/query`, including year
bounds, entity hints, document/series filters, provider choice, and conversation
history. It applies the authenticated user's institution and access-level scope
to every angle.

The normal `QueryResponse` fields are returned with an additional
`research_trace` containing attempts, marginal gains, the last validated gap
assessment, stop reason, provider use, and timings. Final synthesis failure
returns HTTP 200 with the retrieved evidence and `synthesis_available=false`.
Only failures that prevent the initial probe/investigation return HTTP 503.

Non-admin users are limited separately from ordinary queries. Defaults are four
deep requests per 60 seconds and can be changed with:

```text
DEEP_RESEARCH_RATE_LIMIT_MAX=4
DEEP_RESEARCH_RATE_LIMIT_PERIOD=60
```

## Offline verification

```powershell
python -m pytest -q tests\test_deep_research.py tests\test_deep_research_loop.py tests\test_deep_research_api.py
```

These tests use fake planners, advisers, retrieval backends, blocks, and model
responses. They require no Neo4j, model provider, or network access.

## Cost and provenance

At most the original angle may invoke the LLM-Cypher fallback unless an operator
explicitly opts reformulations in. The merged evidence is ranked by the number
of angles surfacing it, extraction confidence, and a stable lexical key before
the synthesis context is capped. Access filtering occurs independently on every
gather, so deep research cannot widen the user's permission scope.
