# GraphRAG Claim-Centric Pipeline

`Python 3.10+` &nbsp; `Neo4j optional` &nbsp; `FastAPI` &nbsp; `SQLite`

Turn OCR'd narrative reports into structured, searchable data.

This project reads report text that has already been OCR'd and stored as JSON, breaks it into pages and sections, extracts useful facts, and can optionally load the results into Neo4j as a knowledge graph.

It is aimed at researchers, archivists, historians, and analysts who want something more useful than raw OCR text but do not want to build an extraction pipeline from scratch.

---

## Contents

- [What It Does](#what-it-does)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Pipeline Flow](#pipeline-flow)
- [Input Format](#input-format)
- [Output Files](#output-files)
- [Command Reference](#command-reference)
  - [Full Pipeline](#full-pipeline)
  - [Step-by-Step](#step-by-step)
  - [Entity Resolution](#entity-resolution)
  - [CSV Export](#csv-export)
  - [Testing](#testing)
- [Neo4j Setup](#neo4j-setup)
- [Anti-Pattern Review](#anti-pattern-review)
- [Natural-Language Query Server](#natural-language-query-server)
- [Optional Dependencies](#optional-dependencies)
- [Project Layout](#project-layout)
- [Notes](#notes)

---

## What It Does

- Reads OCR-based report JSON files.
- Organizes each report into a document, pages, sections, paragraphs, and annotations.
- Extracts claims, measurements, mentions, and entities from the text.
- Keeps page-level provenance so extracted facts can be traced back to the source.
- Writes outputs as JSON files, with optional CSV export and optional Neo4j loading.
- Detects and queues anti-patterns (OCR errors, junk mentions, missing links) for human review.
- Serves a natural-language query API over the loaded graph.

---

## Prerequisites

- Python 3.10 or newer
- Neo4j only if you want a graph database output

---

## Quick Start

Install the project:

```bash
python -m pip install -e .[dev]
```

Run the pipeline on the included sample fixtures:

```bash
graphrag run-e2e --inputs tests/fixtures --out-dir out --backend memory
```

Run it on the archive inputs included in this repository:

```bash
graphrag run-e2e --inputs input/1930s --out-dir out --backend memory
```

> **No `graphrag` in your PATH?** Use the module form instead:
> ```bash
> python -m graphrag_pipeline.cli run-e2e --inputs tests/fixtures --out-dir out --backend memory
> ```

---

## Pipeline Flow

```
OCR JSON input
     |
     v
[ ingest-structure ]  -->  *.structure.json
     |                      (pages, sections, paragraphs, annotations)
     v
[ extract-semantic ]  -->  *.semantic.json
     |                      (claims, entities, mentions, measurements)
     |
     +-----> [ quality-report ]           (spot-checking)
     |
     +-----> [ spelling-review-report ]   (OCR review queue)
     |
     +-----> [ review-detect ]            (anti-pattern queue --> review.db)
     |
     v
[ load-graph ]        -->  Neo4j graph  (optional)
     |
     v
[ query-serve ]       -->  FastAPI NL query endpoint  (optional)
```

---

## Input Format

The pipeline expects JSON created from OCR output. A minimal example looks like this:

```json
{
  "metadata": {
    "title": "Narrative Report",
    "date_start": "1938-01-01",
    "date_end": "1938-12-31",
    "report_year": 1938
  },
  "pages": [
    {
      "page_number": 1,
      "raw_ocr_text": "Full OCR text from the page..."
    }
  ]
}
```

Use `raw_ocr_text` for page text. The older `raw_text` field is still accepted, but `raw_ocr_text` is the current format.

---

## Output Files

After a run, the `out/` folder will contain:

| Output | Description |
|--------|-------------|
| `*.structure.json` | Report split into pages, sections, paragraphs, and annotations |
| `*.semantic.json` | Extracted claims, entities, mentions, and measurements |
| CSV exports | For spreadsheets or bulk import |
| Quality reports | For spot-checking |
| Spelling review queues | Tied back to source pages |
| Neo4j graph | For graph search and visualization |

---

## Command Reference

Quick-reference table of all available commands:

| Command | Purpose |
|---------|---------|
| `graphrag run-e2e` | Full end-to-end pipeline run |
| `graphrag ingest-structure` | Parse OCR JSON into structure bundle |
| `graphrag extract-semantic` | Extract claims, entities, measurements |
| `graphrag quality-report` | Spot-check extracted bundles |
| `graphrag spelling-review-report` | Generate OCR spelling review queue |
| `graphrag resolve-mentions` | Re-run entity resolution |
| `graphrag load-graph` | Load bundles into Neo4j |
| `graphrag review-detect` | Run anti-pattern detectors |
| `graphrag review-serve` | Launch review web UI |
| `graphrag review-export` | Export review proposals or patch specs |
| `graphrag query-serve` | Start NL query API server |
| `python scripts/json_to_csv.py` | Export JSON outputs to CSV |

### Full Pipeline

Run the full pipeline (in-memory, no Neo4j needed):

```bash
graphrag run-e2e --inputs input/1930s --out-dir out --backend memory
```

Run with parallel workers to speed up large batches:

```bash
graphrag run-e2e --inputs input/1930s --out-dir out --backend memory --workers 4
```

### Step-by-Step

Run each stage individually:

```bash
graphrag ingest-structure --input tests/fixtures/report1.json --output out/report1.structure.json
graphrag extract-semantic --structure out/report1.structure.json --output out/report1.semantic.json
graphrag quality-report --structure out/report1.structure.json --semantic out/report1.semantic.json
graphrag spelling-review-report --structure out/report1.structure.json --semantic out/report1.semantic.json --output out/report1.spelling_review.json
```

### Entity Resolution

Re-run entity resolution on existing semantic bundles (useful after updating `seed_entities.csv`):

```bash
graphrag resolve-mentions --semantic-dir out --dry-run
graphrag resolve-mentions --semantic-dir out
```

### CSV Export

Export JSON outputs to CSV:

```bash
python scripts/json_to_csv.py --src-dir out --out-dir csv_out
```

### Testing

Run the test suite:

```bash
python -m pytest -q
```

---

## Neo4j Setup

You do not need Neo4j to use this project. The in-memory backend is enough if you only want JSON outputs.

If you do want graph output:

1. Install the optional Neo4j dependency:

```bash
python -m pip install -e .[neo4j]
```

2. Create a local `.env` file from `.env.example` and set your connection details:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (optional, defaults to `neo4j`)

3. Load the generated files into Neo4j:

```bash
graphrag load-graph --input-dir out --backend neo4j
```

---

## Anti-Pattern Review

The review subsystem detects data-quality issues in extracted bundles and queues them for human review.

### Detectors

It runs three detectors automatically:

- **OCR/entity queue** — flags duplicate or OCR-corrupted entity variants for merging or aliasing.
- **Junk mention queue** — flags header contamination, boilerplate, short generic tokens, and OCR garbage for suppression.
- **Builder repair queue** — flags claims with missing species focus, missing location links, or method over-triggering for correction.

Reviewed decisions are stored in a local SQLite database with full revision history. Accepted proposals produce typed patch specs ready for a later patch engine.

### Commands

Run detection on a bundle pair:

```bash
graphrag review-detect \
  --structure out/report1.structure.json \
  --semantic out/report1.semantic.json \
  --review-db review.db
```

Populate the review store as part of an end-to-end run:

```bash
graphrag run-e2e --inputs input/1930s --out-dir out --backend memory --review-db review.db
```

Launch the local review web application (requires `uvicorn`):

```bash
graphrag review-serve --review-db review.db
```

Export proposals or accepted patch specs:

```bash
graphrag review-export --review-db review.db --output proposals.json
graphrag review-export --review-db review.db --output patches.json --mode patches
graphrag review-export --review-db review.db --output proposals.csv --mode proposals --status accepted_pending_apply
```

---

## Natural-Language Query Server

The retrieval subsystem serves a FastAPI query endpoint over a loaded Neo4j graph. It accepts natural-language questions, builds Cypher queries, retrieves context, and synthesizes answers using an LLM.

Install the retrieval dependencies:

```bash
python -m pip install -e .[retrieval]
```

Start the server:

```bash
graphrag query-serve --port 8788
```

> **Required:** Set `ANTHROPIC_API_KEY` in your `.env` or environment before starting the server.

---

## Optional Dependencies

Install extras with `pip install -e .[extra]`. Available extras:

| Extra | Installs | Use for |
|-------|----------|---------|
| `.[dev]` | pytest | running tests |
| `.[neo4j]` | neo4j driver | graph database output |
| `.[tools]` | pymupdf, pytesseract, anthropic | PDF/OCR tools |
| `.[retrieval]` | neo4j, anthropic, fastapi, uvicorn | natural-language query server |

---

## Project Layout

```
Archive-Claim-Graph/
├── input/                    OCR-derived source reports (organized by decade)
├── out/                      Generated JSON outputs
├── csv_out/                  Optional CSV exports
├── graphrag_pipeline/        Pipeline source code
│   ├── extractors/           Claim, mention, and measurement extractors
│   ├── graph/                Neo4j writer and Cypher helpers
│   ├── review/               Anti-pattern detection, review store, and web app
│   ├── retrieval/            Natural-language query engine and API
│   └── queries/              Query contract definitions
├── tests/                    Automated tests and sample fixtures
├── scripts/                  Utility scripts (e.g. json_to_csv.py)
├── RUNBOOK.md                Detailed operating notes
└── pyproject.toml            Package configuration and dependency extras
```

---

## Notes

- This project starts after OCR. It does not convert PDFs to text as part of the main pipeline.
- Every extracted item is designed to keep provenance so you can trace it back to the original report text.
- For deeper operational details, advanced review commands, and Neo4j connection notes, see [RUNBOOK.md](RUNBOOK.md).
