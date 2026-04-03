# Gemynd

**A claim-centric archival knowledge graph engine with natural-language retrieval.**

Neo4j · Anthropic API · FastAPI · Python 3.11+

---

## What Gemynd does

Digitized archival collections are structurally opaque. A wildlife refuge annual report from 1952, a labor union correspondence archive from 1909, a municipal land-use record series spanning forty years — each encodes decades of institutional knowledge in narrative prose that was never designed for machine reading. OCR introduces noise, formatting varies by decade, terminology shifts across authors, and no two document series organize their information the same way. The result is material that is technically digitized but practically unsearchable at any analytical scale.

Generic extraction tools fail here in predictable ways. A general-purpose NER model does not know that "the south unit" refers to a specific management zone, that "pintail" and "northern pintail" resolve to the same entity, or that a sentence describing "25 mallard hens observed in Pool 2" is a population count claim with a specific evidential status — not a casual observation to be chunked and embedded alongside everything else. Tools that do not understand what the documents are _saying_ produce embeddings of noise.

Gemynd's answer is **domain configuration as a first-class architectural concern**. Every sentence in a corpus is evaluated as a potential structured claim — typed against a pattern library built from actual knowledge of what the documents contain, resolved against a seed entity vocabulary derived from archival research, and linked through a compatibility matrix that enforces which claim types can validly reference which entity types. The result is a Neo4j property graph queryable in natural language with full sentence-level provenance tracing every answer back to its source page and document.

The system is designed to be handed to real institutions operating under real data governance obligations — not demonstrated at a conference and abandoned.

---

## Architecture

Gemynd is organized into five pipeline layers, three web applications, and several supporting subsystems. Each pipeline layer produces durable intermediate outputs, enabling partial reruns after configuration changes.

### Ingestion pipeline

**Layer 0 — Source Parser.** Ingests raw OCR JSON, normalizes text against a correction table, and infers document structure (pages, sections, paragraphs, annotations). Year inference from filename patterns handles multi-decade variance in metadata quality. Path traversal guards prevent directory escape in server contexts.

**Layer 1 — Semantic Extraction.** Rule-based claim extraction uses scored pattern matching against a domain-specific claim type taxonomy, with a `HybridClaimExtractor` providing LLM fallback for sentences that score below threshold. Measurement extraction and mention detection run in parallel. Entity resolution matches surface forms against a seed vocabulary via fuzzy matching, producing tiered confidence resolutions. Claim-entity links are typed by a compatibility matrix that penalizes weak pairings and drops forbidden ones entirely. All extraction decisions are recorded in `decision_trace` fields — every claim carries an audit trail.

**Layer 2 — Graph Loading.** Writes to Neo4j via batched `UNWIND MERGE` Cypher with duplicate suppression at the database level. An in-memory backend supports development and testing without a running Neo4j instance. `Observation` and `Event` nodes are derived from claims at load time via a derivation registry, providing a semantically richer analytical surface. A shared `DerivationContext` object consolidates entity-binding and year-extraction logic so observations and events are built from pre-computed state rather than re-deriving independently.

**Layer 3 — Retrieval.** A four-component layer. The query intent classifier routes incoming queries to analytical or conversational paths before any graph traversal occurs. The entity resolution gateway maps query surface forms to graph node IDs using the same fuzzy resolution logic as ingestion. The Cypher query builder selects from six parameterized analytical templates — temporal, entity-anchored, multi-entity, fulltext, claim-type-scoped, and anchor-entity traversal — based on resolved entities, year bounds, and inferred claim types. The provenance context assembler selects and serializes claim blocks for synthesis, preserving `extraction_confidence` and `epistemic_status` fields so the model can reason about evidential quality.

**Layer 4 — Synthesis.** Single-turn Anthropic API call with a structured system prompt that instructs the model to treat confidence tiers and epistemic status as meaningful signals, express uncertainty where the source record is ambiguous, cite claim IDs in responses, and return typed JSON. PII redaction runs on synthesis output before it reaches the client. Archivist notes — when attached to source documents — are injected into the synthesis context as authoritative annotations that take precedence over extraction confidence signals.

### Web applications

**Query UI (Gemynd).** Chat-based natural-language query interface with multi-turn conversation history, adjustable retrieval mode (auto/conversational/analytical), year range filtering, and entity hint injection. Inline source citation with per-claim confidence badges, analytical data tables, and archivist note panels. Rate-limited per user with admin exemption.

**Review UI.** Anti-pattern detection across four detector families: OCR/entity cleanup (duplicate and corruption variants), junk mention suppression (header contamination, boilerplate, OCR garbage), builder repair (missing species focus, missing location links, method over-triggering), and sensitivity monitoring (PII, Indigenous cultural material, living person references). Proposals are validated, stored in SQLite with full revision history, and surfaced through a priority-ranked review queue. Archivists can accept, reject, defer, edit, or split proposals. Batch review mode supports high-throughput triage of low-consequence proposals. An error classification taxonomy captures reviewer feedback for model training export.

**Ingest UI.** Drag-and-drop PDF/JSON upload with real-time per-document progress tracking via HTMX polling. PDF-to-JSON conversion uses PyMuPDF with pytesseract fallback for image-only pages. Background processing with job status persistence in SQLite.

### Supporting subsystems

**Authentication.** JWT cookie authentication with bcrypt password hashing and token versioning: deactivating a user or changing a password immediately invalidates all outstanding sessions without requiring a token blacklist. Role-based access control operates across four permission levels — `public`, `staff_only`, `restricted`, `indigenous_restricted` — enforced at the route layer. First-run setup uses a per-process setup token printed to stderr.

**Sensitivity gate.** At ingestion time, claims are screened for PII, Indigenous cultural material, and living person references before reaching the graph. High-confidence flags are auto-quarantined; all flags produce review proposals. The `indigenous_restricted` access tier and the tribal consultation requirement for quarantine clearance reflect archival ethics obligations embedded in the architecture. A background monitor re-scans existing claims when detection rules are updated.

**Conversation logging.** Fire-and-forget SQLite logger captures the full causal chain per query — classification, entity resolution, retrieval strategy, per-claim interaction data (traversal path, citation status, confidence). Bounded daemon-thread writer never blocks the response path. Query history browsing and saved-search management for archivists.

**Token usage tracking.** Per-request metering of Anthropic API calls via a transparent `MeteredAnthropicClient` wrapper. Daily aggregates, per-caller breakdowns, configurable budget alerts with institution-level isolation. Pricing loaded from YAML with environment variable overrides.

**Archivist annotations.** Free-text notes attached to documents by archivists, surfaced in synthesis context as authoritative knowledge. Note history with audit trail.

**Collection analytics.** Statistics dashboard (document counts, claim type distribution, entity categories, temporal coverage, extraction confidence tiers), gap analysis (temporal coverage gaps, entity depth gaps, geographic coverage gaps, topical balance against expected claim-type profiles, query signal gaps from conversation log analysis), and relationship mapping explorer (entity neighborhood graph via Cytoscape.js with co-occurrence evidence).

**Corpus export.** Three export formats: semantic-aware CSV (claims, entities, relationships), standalone HTML statistics report, and EAD 2002 XML finding aid. Access-restricted documents are silently excluded from EAD exports.

**Checkpoint and resume.** JSONL checkpoint for ingestion pipeline resume support. Pre-ingest duplicate detection via file hash comparison against the graph. Deterministic document IDs ensure metadata corrections trigger fresh ingestion rather than silent skips.

---

## Domain portability

Gemynd is not specific to any single collection. Three files define what the pipeline knows about a domain:

- **`seed_entities.csv`** — the named entity vocabulary: canonical names, aliases, entity types, and metadata used for resolution and display.
- **`claim_type_patterns.yaml`** — scored regex patterns that map sentence structures to typed claims, with weights controlling how aggressively the rule-based extractor fires before falling back to the LLM.
- **`domain_profile.yaml`** — collection-level configuration: domain name, date range, subject focus, sensitivity flags, document anchor, synthesis context, and expected claim-type distribution for gap analysis.

Additional configuration files (derivation registry, compatibility matrix, domain schema, concept rules, query intent mapping, measurement vocabularies, sensitivity vocabulary) are versioned alongside the pipeline code and adapt extraction, retrieval, and review behavior to the domain.

To adapt the pipeline to a new collection:

```bash
python scripts/bootstrap_domain.py \
    --samples /path/to/sample/documents/*.json \
    --out-dir config/new_domain/ \
    --max-iterations 5

gemynd validate-domain \
    --samples /path/to/sample/documents/*.json \
    --domain-dir config/new_domain/
```

`bootstrap_domain.py` uses the Anthropic API to generate draft configuration from corpus samples, then runs an automated refinement loop — validating, identifying gaps, and asking the LLM to fix them — until the unclassified-claim rate drops below threshold or convergence is detected. `validate-domain` measures extraction coverage and cross-resource consistency before any full-corpus run is committed.

---

## Installation and quickstart

**Prerequisites:** Python 3.11+, Neo4j 5.x (optional for development), Anthropic API key (required for retrieval and LLM extraction).

```bash
# Core pipeline
pip install -e .

# Add retrieval layer (Neo4j driver, Anthropic, FastAPI, auth dependencies)
pip install -e .[retrieval]
```

Create `.env` in the project root:

```bash
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
Anthropic_API_Key=your_anthropic_api_key

# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=your_generated_secret
JWT_EXPIRE_HOURS=24
USERS_DB=data/users.db
```

**Ingest a corpus:**

```bash
gemynd ingest-structure --input input/1950s --output out/
gemynd extract-semantic --structure-dir out/ --output-dir out/
gemynd load-graph --input-dir out/ --backend neo4j
```

**Start the query server:**

```bash
gemynd query-serve --port 8788
```

Test without Neo4j using the in-memory backend and included fixtures:

```bash
gemynd run-e2e --inputs tests/fixtures --out-dir out --backend memory
```

---

## CLI reference

| Command                           | Description                                                |
| --------------------------------- | ---------------------------------------------------------- |
| `gemynd ingest-structure`       | Parse OCR JSON into StructureBundle                        |
| `gemynd extract-semantic`       | Extract claims, entities, mentions, measurements           |
| `gemynd load-graph`             | Load bundles into Neo4j or in-memory backend               |
| `gemynd run-e2e`                | Full pipeline: parse, extract, load, review detect         |
| `gemynd quality-report`         | Spot-check extraction output with per-bundle statistics    |
| `gemynd spelling-review-report` | Generate OCR correction review queue                       |
| `gemynd review-detect`          | Run anti-pattern detectors; write proposals to SQLite      |
| `gemynd review-serve`           | Launch review web UI                                       |
| `gemynd review-export`          | Export proposals, patches, or revision history             |
| `gemynd query-serve`            | Start the Gemynd query API and web UI                      |
| `gemynd ingest-serve`           | Launch the document ingestion web UI                       |
| `gemynd resolve-mentions`       | Re-run entity resolution after seed entity changes         |
| `gemynd verify-integrity`       | Check graph consistency via file hash verification         |
| `gemynd sensitivity-scan`       | Background scan for PII, cultural material, living persons |
| `gemynd validate-domain`        | Measure extraction quality before committing to a full run |
| `gemynd export-corpus`          | Export as CSV, standalone HTML report, or EAD 2002 XML     |

---

## Project status

Active development.

---

## License

Copyright © 2025 Kettle Systems LLC. All rights reserved.

This source code is made available for viewing, personal use, and academic research. You may fork and modify the code for non-commercial personal or academic purposes. Commercial use, redistribution, and derivative works intended for commercial deployment require explicit written permission from Kettle Systems LLC.

---

## Acknowledgments

Developed in conjunction with research at Spokane Valley Heritage Museum. Tribal consultation protocols for `indigenous_restricted` content are managed in coordination with relevant tribal cultural offices.
