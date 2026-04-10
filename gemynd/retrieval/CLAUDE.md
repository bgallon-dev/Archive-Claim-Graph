# retrieval

Natural-language query to synthesized answer. Four-step architecture: classify, resolve entities, build Cypher + assemble context, synthesize.

## Pipeline stages

1. `classifier.py` — routes queries to analytical vs. conversational paths BEFORE graph traversal. Intent classification drives template selection.
2. `entity_gateway.py` — maps surface forms to graph node IDs using the same fuzzy resolution logic as ingestion.
3. `query_builder.py` — six parameterized Cypher templates (temporal, entity-anchored, multi-entity, fulltext, claim-type-scoped, anchor-entity). Template selected based on resolved entities, year bounds, inferred claim types.
4. `context_assembler.py` — provenance context assembly, budget-capped (default 20 blocks).
5. `synthesis.py` — single-turn Anthropic API call. Returns typed JSON with `answer`, `confidence_assessment`, `supporting_claim_ids`, `caveats`.

## Key invariants

- Fulltext queries MUST pass through `_sanitize_fulltext()` in `context_assembler.py` — escapes Lucene special characters and caps length at 500 chars. This prevents query injection.
- PII redaction via `_redact_pii()` runs AFTER synthesis, BEFORE client response. `_PII_PATTERNS` are ordered most-specific to least-specific; preserve this ordering when adding patterns.
- Archivist notes, when present, take precedence over extraction confidence in synthesis context.
- `synthesis_context` from `DomainConfig` is injected into the system prompt. Empty context degrades to generic fallback.

## Conversation logging

`conversation_log.py` uses a fire-and-forget daemon-thread SQLite writer. It must never block the response path.

## Web app

`web/app.py` is the FastAPI query UI. Rate limiting is per-user with admin exemption (in `web/rate_limit.py`). Templates use HTMX fragments in `web/templates/fragments/`.

## Executor protocol

`executor.py` defines the Protocol. `Neo4jQueryExecutor` for production, `InMemoryQueryExecutor` for tests.
