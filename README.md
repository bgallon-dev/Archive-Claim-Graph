# graphrag_pipeline

**A claim-centric archival knowledge graph pipeline with natural-language retrieval.**

Neo4j · Anthropic API · FastAPI · Python 3.11+

---

## Why this exists

Historical documents are structurally opaque. A wildlife refuge annual report from 1952 contains decades of ecological data — waterfowl census figures, habitat acreage, management interventions, species observations — encoded in narrative prose that was never designed for machine reading. Across 109 such reports spanning 1938 to 2000, that opacity compounds: OCR introduces noise, formatting varies by decade, terminology shifts, and no two field managers organized their notes the same way. The result is a corpus that is technically digitized but practically unsearchable at any analytical scale.

The harder problem is that generic extraction tools fail here in predictable ways. A general-purpose NER model does not know that "the south unit" refers to a specific management zone, that "pintail" and "northern pintail" resolve to the same entity, or that a sentence describing "25 mallard hens observed in Pool 2" is an ecological count claim with a specific evidential status — not a casual observation to be chunked and embedded alongside everything else. Tools that don't understand what the documents are *saying* produce embeddings of noise.

This pipeline's answer is domain configuration as a first-class concern. Every sentence in the corpus is evaluated as a potential structured claim — typed against a pattern library built from actual knowledge of what wildlife refuge annual reports contain, resolved against a seed entity vocabulary derived from archival research, and linked through a compatibility matrix that enforces which claim types can validly reference which entity types. The result is a Neo4j property graph of approximately 300,000 nodes and 700,000 relationships, queryable in natural language with full sentence-level provenance tracing every answer back to its source page and document.

This is applied digital humanities and computational source criticism. The developer is a historian finishing a History BA (Honors) while serving as Digital Humanities Project Lead at Spokane Valley Heritage Museum — which is to say, someone with direct institutional obligations to the questions this tool addresses. The immediate context is the Turnbull National Wildlife Refuge corpus, but the broader research context includes a Métis cultural recovery project at the museum and scholarly work on IWW labor history using the Enacted Legitimacy framework. Both projects require tools that can surface pattern across primary sources at scale. This pipeline was built because waiting for someone else to build it was not an option.

---

## Architecture

The pipeline is organized into five layers plus two subsystems. Each layer produces durable intermediate outputs, enabling partial reruns after configuration changes.

### Layer 0 — Source Parser ([source_parser.py](graphrag_pipeline/source_parser.py))

Ingests raw OCR JSON, normalizes text against a correction table (`ocr_corrections.yaml`), and infers document structure — pages, sections, paragraphs, annotations — producing a `StructureBundle`. Year inference from filename patterns handles the multi-decade variance in metadata quality; path traversal guards on `source_file` fields prevent directory escape in server contexts. The parser is intentionally conservative: it preserves OCR artifacts rather than silently correcting them, so downstream decisions about cleanup are auditable.

### Layer 1 — Semantic Extraction ([pipeline.py](graphrag_pipeline/pipeline.py) → [extractors/](graphrag_pipeline/extractors/))

Rule-based claim extraction uses scored pattern matching against `claim_type_patterns.yaml`, with a `HybridClaimExtractor` providing LLM fallback for sentences that score below threshold. Measurement extraction and mention detection run in parallel. Entity resolution matches surface forms against `seed_entities.csv` via fuzzy matching, producing `REFERS_TO` or `POSSIBLY_REFERS_TO` confidence tiers. Claim-entity links are typed by `claim_relation_compatibility.yaml`, which penalizes weak pairings and drops forbidden ones entirely. All extraction decisions are recorded in `decision_trace` fields — every claim carries an audit trail explaining why it was typed and linked the way it was.

### Layer 2 — Graph Loading ([graph/writer.py](graphrag_pipeline/graph/writer.py))

Writes to Neo4j via batched `UNWIND MERGE` Cypher, which handles duplicate suppression at the database level rather than in application logic. An in-memory backend supports development and testing without a running Neo4j instance. Schema migrations enforce uniqueness constraints and indexes across all node types at startup. `Observation` and `Event` nodes are derived from claims at load time, providing a semantically richer analytical surface: querying for waterfowl population trends operates at the Observation layer, not against raw claim text.

### Layer 3 — Retrieval ([retrieval/](graphrag_pipeline/retrieval/))

A four-component layer. The query intent classifier ([classifier.py](graphrag_pipeline/retrieval/classifier.py)) routes incoming queries to analytical or conversational paths before any graph traversal occurs. The entity resolution gateway ([entity_gateway.py](graphrag_pipeline/retrieval/entity_gateway.py)) maps surface forms from the query to graph node IDs, applying the same fuzzy resolution logic used during ingestion. The Cypher query builder ([query_builder.py](graphrag_pipeline/retrieval/query_builder.py)) selects from six parameterized analytical templates — covering temporal, entity-anchored, multi-entity, fulltext, claim-type-scoped, and refuge-anchored traversal paths — based on resolved entities, year bounds, and inferred claim types. The provenance context assembler ([context_assembler.py](graphrag_pipeline/retrieval/context_assembler.py)) selects and serializes claim blocks for synthesis, preserving `extraction_confidence` and `epistemic_status` fields so the model can reason about evidential quality rather than treating all retrieved text as equivalent.

### Layer 4 — Synthesis ([retrieval/synthesis.py](graphrag_pipeline/retrieval/synthesis.py))

Single-turn Anthropic API call with a structured system prompt that instructs the model to treat `extraction_confidence` and `epistemic_status` as meaningful signals, express uncertainty where the source record is ambiguous, cite `claim_id`s in responses, and return typed JSON. PII redaction runs on synthesis output before it reaches the client. Synthesis does not have access to the full graph — it receives only the assembled provenance context, which bounds both cost and hallucination surface area.

### Review Subsystem ([review/](graphrag_pipeline/review/))

Anti-pattern detection across four detector families: OCR/entity cleanup (duplicate and corruption variants), junk mention suppression (header contamination, boilerplate, OCR garbage), builder repair (missing species focus, missing location links, method over-triggering), and sensitivity monitoring. Proposals are validated, stored in SQLite with full revision history, and surfaced through a FastAPI + HTMX review UI. Archivists can accept, reject, defer, edit, or split proposals. Accepted patches are exported as typed `patch_spec` payloads. Every proposal action is logged with user identity and timestamp; soft-delete preserves the record even when proposals are dismissed.

### Auth Subsystem ([auth/](graphrag_pipeline/auth/))

JWT cookie authentication with bcrypt password hashing and token versioning: deactivating a user or changing a password immediately invalidates all outstanding sessions without requiring a token blacklist. Role-based access control operates across four permission levels — `public`, `staff_only`, `restricted`, `indigenous_restricted` — enforced at the route layer. First-run setup uses a per-process setup token printed to stderr, avoiding the need to ship default credentials. Full admin user management API is included. The `indigenous_restricted` access tier and the sensitivity detection gate — which screens for PII, Indigenous cultural material, and living person references and quarantines flagged content before graph write, requiring tribal consultation for clearance — reflect genuine archival ethics obligations rather than compliance theater.

---

## What this project proves

- **Domain knowledge as a force multiplier.** The claim type taxonomy, entity seed vocabulary, and compatibility matrix are not generic NLP components — they encode what wildlife refuge annual reports are actually saying. A compatibility matrix entry that forbids a `WEATHER_EVENT` claim from linking to a `SPECIES` entity via `CAUSED_BY` exists because the developer knows when that link is meaningful and when it is OCR noise producing a false syntactic pattern. Generic tools cannot make that distinction.

- **Systems thinking at scale.** 109 documents ingested through a pipeline with a sensitivity gate that fires before data reaches the graph, a feedback loop designed to teach the retrieval layer to prefer shorter traversal paths via SQLite interaction logging, and a review subsystem that closes the loop between archivist judgment and graph state. The architecture anticipates operational use, not just extraction.

- **Production-grade infrastructure built by one person.** Token versioning for immediate session invalidation, soft-delete across all review records, multi-tenancy access control, audit logging on every proposal action, `UNWIND MERGE` batching for idempotent graph writes, `patch_spec` validation before export — none of this was required for a proof-of-concept. It was built because the system was designed to be handed to real institutions operating under real data governance obligations.

- **Archival ethics embedded in the architecture.** The `indigenous_restricted` access tier, the tribal consultation requirement for quarantine clearance, donor restriction notices surfaced in synthesis output, and PII redaction at the synthesis boundary are first-class design constraints, not post-hoc additions. The sensitivity configuration (`sensitivity_config.yaml`, `indigenous_cultural_terms.yaml`) is versioned alongside the pipeline code.

- **Portability.** The pipeline is not Turnbull-specific. `seed_entities.csv`, `claim_type_patterns.yaml`, and `domain_profile.yaml` are the only files that need to change for a new collection. `bootstrap_domain.py` uses the Anthropic API to generate draft versions of all three from corpus samples. `validate-domain` measures extraction quality against a configurable threshold before committing to a full run. The Métis cultural recovery project at Spokane Valley Heritage Museum is an active second use case in development.

---

## Installation and quickstart

**Prerequisites:** Python 3.11+, Neo4j 5.x (optional for development), Anthropic API key (required for retrieval).

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
graphrag ingest-structure --input input/1950s --output out/
graphrag extract-semantic --structure-dir out/ --output-dir out/
graphrag load-graph --input-dir out/ --backend neo4j
```

**Start the query server:**

```bash
graphrag query-serve --port 8788
```

Test without Neo4j using the in-memory backend and included fixtures:

```bash
graphrag run-e2e --inputs tests/fixtures --out-dir out --backend memory
```

---

## CLI reference

| Command | Description |
|---|---|
| `graphrag ingest-structure` | Parse OCR JSON into StructureBundle (pages, sections, paragraphs) |
| `graphrag extract-semantic` | Extract claims, entities, mentions, measurements |
| `graphrag load-graph` | Load semantic bundles into Neo4j or in-memory backend |
| `graphrag run-e2e` | Full pipeline in one command; accepts `--backend memory` for development |
| `graphrag quality-report` | Spot-check extraction output; prints per-bundle statistics |
| `graphrag spelling-review-report` | Generate OCR correction review queue tied to source pages |
| `graphrag review-detect` | Run anti-pattern detectors; write proposals to SQLite |
| `graphrag review-serve` | Launch FastAPI + HTMX review UI |
| `graphrag review-export` | Export proposals or accepted `patch_spec` payloads |
| `graphrag query-serve` | Start natural-language query API (requires `.[retrieval]`) |
| `graphrag resolve-mentions` | Re-run entity resolution after `seed_entities.csv` changes |
| `graphrag verify-integrity` | Check graph consistency; report orphaned nodes and broken links |
| `graphrag sensitivity-scan` | Scan bundles for PII, cultural material, and living person references |
| `graphrag validate-domain` | Measure extraction quality against threshold before full run |

---

## Domain portability

Three files define what the pipeline knows about a collection:

- **`seed_entities.csv`** — the named entity vocabulary: canonical names, aliases, entity types, and metadata the graph will use for resolution and display.
- **`claim_type_patterns.yaml`** — scored regex patterns that map sentence structures to typed claims. The scoring weights control how aggressively the rule-based extractor fires before falling back to the LLM.
- **`domain_profile.yaml`** — collection-level configuration: domain name, date range, subject focus, sensitivity flags, and thresholds that govern extraction behavior.

To adapt the pipeline to a new collection, run:

```bash
python graphrag_pipeline/bootstrap_domain.py --corpus-dir /path/to/samples --output-dir config/
graphrag validate-domain --config-dir config/ --test-corpus /path/to/samples
```

`bootstrap_domain.py` uses the Anthropic API to generate draft versions of all three files from corpus samples. `validate-domain` measures extraction coverage and precision against a configurable threshold, producing a report before any full-corpus run is committed. The Métis cultural recovery project at Spokane Valley Heritage Museum is actively in development as a second domain configuration, with distinct sensitivity requirements and a different entity vocabulary.

---

## Project status

Active development. Proof-of-concept targeting June 2026.

**Known active bugs:**
- Entity resolution results from the review subsystem are not persisting to the conversation log; resolved entities are applied to the graph but not reflected in subsequent query session context.
- Some retrieval paths return empty `relationship_types` arrays in traversal metadata when the query resolves to a refuge-anchored template; the Cypher is correct but the metadata serialization drops the field.

**Partially implemented:**
- Salience-aware Cypher query rewrite: the feedback loop that teaches the retrieval layer to prefer shorter traversal paths by logging interaction outcomes to SQLite and adjusting template selection weights is designed and the logging infrastructure is in place. The weight update and template re-ranking steps are not yet wired.

The specificity of these notes is intentional. A system this size has known issues; the question is whether they are tracked.

---

## License and acknowledgments

License: TBD.

Developed in conjunction with research at Spokane Valley Heritage Museum. Turnbull National Wildlife Refuge corpus processed under fair use for scholarly research. Tribal consultation protocols for `indigenous_restricted` content are managed in coordination with relevant tribal cultural offices.
