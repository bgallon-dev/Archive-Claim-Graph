"""JSONL checkpoint for ingestion pipeline resume support.

The checkpoint file lives at {out_dir}/.ingest_checkpoint.jsonl.  Each line
is a JSON object recording one successfully-written document:

    {"doc_id": "...", "input": "...", "completed_at": "2026-03-24T12:00:00Z"}

The file is human-readable: open it in any text editor to see which documents
have been written to Neo4j.  To force a re-run of specific documents, delete
the corresponding lines.

## doc_id determinism and re-ingestion

doc_ids are generated deterministically from document content via make_doc_id.
This means that re-running a document with *corrected metadata* (e.g. a fixed
publication year) will produce a *different* doc_id and will bypass the
checkpoint, triggering a fresh ingest.  This is intentional — a metadata
correction should result in a new graph node, not a skip.  If you want to
replace an existing node instead, soft-delete the old document first.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_FILENAME = ".ingest_checkpoint.jsonl"


def checkpoint_path(out_dir: Path) -> Path:
    return out_dir / CHECKPOINT_FILENAME


def load_checkpoint(out_dir: Path) -> set[str]:
    """Return the set of doc_ids already successfully written to the graph.

    Skips corrupt lines silently so a partial file write never blocks a run.
    """
    cp = checkpoint_path(out_dir)
    if not cp.exists():
        return set()
    completed: set[str] = set()
    for line in cp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            doc_id = entry.get("doc_id")
            if doc_id:
                completed.add(doc_id)
        except json.JSONDecodeError:
            pass  # corrupt line — skip, never crash
    return completed


def append_checkpoint(out_dir: Path, doc_id: str, input_path: str) -> None:
    """Append one successfully-written document to the checkpoint file."""
    entry = {
        "doc_id": doc_id,
        "input": input_path,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    cp = checkpoint_path(out_dir)
    with cp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
