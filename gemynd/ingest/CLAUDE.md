# ingest

Five-layer ingestion pipeline: source parsing (L0), semantic extraction (L1), graph loading (L2), derivation (L1/L2), and export (L4).

## Architecture

- `pipeline.py` is the orchestrator; `phases.py` breaks extraction into composable typed phase functions.
- Pure/effectful split: `extract_document()` is pure (no I/O side effects), `persist_document()` is effectful (writes files, runs sensitivity gate). This split is intentional for testability.
- `ExtractionState` (in `extraction_state.py`) accumulates intermediate results. Phases mutate this state object.

## Key invariants

- `source_parser.py`: Path traversal guards prevent directory escape in server contexts. OCR corrections applied via `ocr_correction_map` from domain config.
- `HybridClaimExtractor`: rule-based extraction first; LLM fallback only for sentences below threshold. All claims carry `decision_trace` fields. `is_valid_claim_sentence()` from `core/claim_validator.py` gates entry.
- `DerivationContext` pre-computes entity bindings and year extraction so observation and event builders share state rather than re-deriving. Year source tracked as `"claim_date"`, `"document_primary_year"`, or `"unknown"`.
- Sensitivity gate fail-safe: on exception, quarantines ALL active claims. `NullSensitivityGate` exists for testing only.
- Checkpoint: JSONL at `{out_dir}/.ingest_checkpoint.jsonl`. Corrupt lines silently skipped. Doc IDs are deterministic — metadata corrections produce new IDs (intentional fresh ingest).

## Sub-packages

- `extractors/`: Claim, mention, measurement extractors. `claim_cache.py` for LLM response caching.
- `graph/`: `writer.py` — `GraphWriter` Protocol with `Neo4jGraphWriter` (batched UNWIND MERGE) and `InMemoryGraphWriter`.
- `annotation/`: Archivist note storage.
- `export/`: CSV, HTML report, EAD XML. EAD exports silently exclude access-restricted documents.
- `web/`: Ingest UI (HTMX drag-and-drop upload with real-time progress).

## Mention extraction

`ResolutionContext` carries paragraph-level context (claim types, resolved entity types). Two-pass resolution: high-confidence first, then remaining with co-mention boost.

## Testing

Use `InMemoryGraphWriter` and `NullSensitivityGate`. `no_llm=True` disables LLM calls. Pipeline integration tests in `test_pipeline_e2e.py`.
