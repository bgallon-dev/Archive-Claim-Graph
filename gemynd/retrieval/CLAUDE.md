# retrieval

Natural-language query to synthesized answer. Four-step architecture: classify, resolve entities, build Cypher + assemble context, synthesize.

## Pipeline stages

1. `classifier.py` — routes queries to analytical vs. conversational paths BEFORE graph traversal. Intent classification drives template selection.
2. `entity_gateway.py` — maps surface forms to graph node IDs using the same fuzzy resolution logic as ingestion.
3. `query_builder.py` — six parameterized Cypher templates (temporal, entity-anchored, multi-entity, fulltext, claim-type-scoped, anchor-entity). Template selected based on resolved entities, year bounds, inferred claim types. Constructed with `institution_id` (no hardcoded collection defaults).
4. `context_assembler.py` — provenance context assembly, budget-capped (default 20 blocks). Candidate ordering is pluggable via `context_ranker.py` (`CONTEXT_RANKER` env: `auto` = graph-proximity BFS from resolved anchors, plus KGR temporal rerank when `KGR_ROOT` points at a kettle-graph-reasoner checkout; `confidence` restores the legacy static sort). Rankers only reorder — they never add or drop candidates — and every failure mode degrades to the next-simpler strategy. Ordering decides what survives truncation; the confidence sort measured at the random floor for query relevance (see `out/kgr_ab_eval_2026-07-07/`). Multi-corpus deployments use `_stratified_budget_cap()` to give each corpus an equal quota (≥5 blocks threshold, remainder to largest), and `_temporally_diversified_take()` to proportionally sample across year buckets within each corpus's quota. When truncation occurs, `_build_retrieval_summary()` is prepended to `context_text` so the model can see what was dropped.
5. `synthesis.py` — single-turn LLM call, provider-dispatched via `shared/openai_compat.py`: Anthropic by default, or any OpenAI-compatible server (vLLM / Ollama / LM Studio / llama.cpp) when `SYNTHESIS_LLM_BASE_URL` or `SYNTHESIS_LLM_PROVIDER` is set (`SYNTHESIS_LLM_MODEL` names the served model). The same provider backs the LLM-Cypher fallback; both stay wrapped in `MeteredAnthropicClient` for token tracking. Returns typed JSON with `answer`, `confidence_assessment`, `supporting_claim_ids`, `caveats`. The system prompt's `RETRIEVED_VIA semantics` paragraph clarifies that `LOCATION_FOCUS` / `OCCURRED_AT` mark spatial setting, not topic — without this, the model conflates the two and over-cites colocated documents.

## Key invariants

- Fulltext queries MUST pass through `_sanitize_fulltext()` in `context_assembler.py` — escapes Lucene special characters and caps length at 500 chars. This prevents query injection.
- PII redaction via `_redact_pii()` runs AFTER synthesis, BEFORE client response. `_PII_PATTERNS` are ordered most-specific to least-specific; preserve this ordering when adding patterns.
- Archivist notes, when present, take precedence over extraction confidence in synthesis context.
- `synthesis_context` from `DomainConfig` is injected into the system prompt. Empty context degrades to a generic "historical archival documents" fallback.
- The retrieval layer is fully config-driven: `institution_id`, `anchor_entity_id`, `anchor_entity_type`, `extraction_stopwords`, and `expected_claim_shares` are threaded from `DomainConfig` at app construction. Do not reintroduce collection-specific constants (Turnbull, Refuge, etc.) into retrieval code.

## Conversation logging

`conversation_log.py` uses a fire-and-forget daemon-thread SQLite writer. It must never block the response path.

## Web app

`web/app.py` is the FastAPI query UI. Rate limiting is per-user with admin exemption (uses `gemynd.shared.rate_limit.TokenBucketLimiter`). Templates use HTMX fragments in `web/templates/fragments/`.

## Executor protocol

`executor.py` defines the Protocol. `Neo4jQueryExecutor` for production, `InMemoryQueryExecutor` for tests.
