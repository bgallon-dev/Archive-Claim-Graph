# GraphRAG Claim-Centric Pipeline

Turn OCR'd narrative reports into structured, searchable data.

This project reads report text that has already been OCR'd and stored as JSON, breaks it into pages and sections, extracts useful facts, and can optionally load the results into Neo4j as a knowledge graph.

It is aimed at researchers, archivists, historians, and analysts who want something more useful than raw OCR text but do not want to build an extraction pipeline from scratch.

## What It Does

- Reads OCR-based report JSON files.
- Organizes each report into a document, pages, sections, paragraphs, and annotations.
- Extracts claims, measurements, mentions, and entities from the text.
- Keeps page-level provenance so extracted facts can be traced back to the source.
- Writes outputs as JSON files, with optional CSV export and optional Neo4j loading.

## What You Need

- Python 3.10 or newer
- Neo4j only if you want a graph database output

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

If `graphrag` is not available in your shell, use:

```bash
python -m graphrag_pipeline.cli run-e2e --inputs tests/fixtures --out-dir out --backend memory
```

## What You Get

After a run, the `out/` folder will contain:

- `*.structure.json`: the report split into pages, sections, paragraphs, and annotations
- `*.semantic.json`: extracted claims, entities, mentions, and measurements

You can also generate:

- CSV exports for spreadsheets or bulk import
- quality reports for spot-checking
- spelling review queues tied back to source pages
- a Neo4j graph for graph search and visualization

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

## Common Commands

Run the full pipeline:

```bash
graphrag run-e2e --inputs input/1930s --out-dir out --backend memory
```

Run it step by step:

```bash
graphrag ingest-structure --input tests/fixtures/report1.json --output out/report1.structure.json
graphrag extract-semantic --structure out/report1.structure.json --output out/report1.semantic.json
graphrag quality-report --structure out/report1.structure.json --semantic out/report1.semantic.json
graphrag spelling-review-report --structure out/report1.structure.json --semantic out/report1.semantic.json --output out/report1.spelling_review.json
```

Export JSON outputs to CSV:

```bash
python scripts/json_to_csv.py --src-dir out --out-dir csv_out
```

Run the test suite:

```bash
python -m pytest -q
```

## Neo4j Output

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

## Project Layout

- `input/`: OCR-derived source reports
- `out/`: generated JSON outputs
- `csv_out/`: optional CSV exports
- `graphrag_pipeline/`: pipeline code
- `tests/`: automated tests and sample fixtures
- `RUNBOOK.md`: more detailed operating notes

## Notes

- This project starts after OCR. It does not convert PDFs to text as part of the main pipeline.
- Every extracted item is designed to keep provenance so you can trace it back to the original report text.
- For deeper operational details, advanced review commands, and Neo4j connection notes, see [RUNBOOK.md](RUNBOOK.md).
