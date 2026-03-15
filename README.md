# GraphRAG Claim-Centric Pipeline

Python implementation of a claim-centered ingestion and graph loading pipeline for narrative reports.

## Features

- Source + structure parsing into `Document/Page/Section/Paragraph/Annotation`.
- Semantic extraction into `Claim/Measurement/Mention`.
- Entity resolution with `REFERS_TO` vs `POSSIBLY_REFERS_TO` policy.
- Provenance via `ExtractionRun`.
- Claim-linked spelling review queue with PDF/page provenance.
- Neo4j AuraDB-compatible Cypher constraints, indexes, and upsert patterns.
- CLI commands:
  - `ingest-structure`
  - `extract-semantic`
  - `load-graph`
  - `run-e2e`
  - `quality-report`
  - `spelling-review-report`

## Quick Start

```bash
python -m pip install -e .[dev]
python -m graphrag_pipeline.cli run-e2e --inputs report1.json report2.json report3.json --out-dir out --backend memory
python -m graphrag_pipeline.cli spelling-review-report --structure out/report3.structure.json --semantic out/report3.semantic.json --output out/report3.spelling_review.json
python -m pytest -q
```

## Neo4j

Install optional dependency and configure environment variables:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (optional, default: `neo4j`)
- `NEO4J_TRUST` (optional: `system|all|custom`, used for `bolt://` or `neo4j://`)
- `NEO4J_CA_CERT` (required when `NEO4J_TRUST=custom`)

The CLI auto-loads a local `.env` file if present. A template is included at `.env.example`.

If you hit certificate-chain errors with self-signed certs in dev, use:

```bash
NEO4J_URI=bolt+ssc://host:7687
```

For single-instance servers, prefer `bolt://...`/`bolt+s://...`/`bolt+ssc://...` over routed `neo4j://...` schemes.

Then:

```bash
python -m pip install -e .[neo4j]
graphrag load-graph --structure out/report1.structure.json --semantic out/report1.semantic.json --backend neo4j
```
