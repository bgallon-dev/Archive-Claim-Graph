"""JSONL bundle input for ingestion.

One JSONL file per newspaper title (LCCN); each line is one issue payload in the
same ``{metadata, pages}`` shape a loose ``.json`` document uses. This module
*stages* a JSONL file into one loose ``.json`` per line so the rest of the
pipeline — which assumes one file per document (the file's stem names its output
bundles, keys the resume checkpoint, and keys the Neo4j dedup hash) — runs
completely unchanged.

Staging is ingest-time scratch, not the stored artifact: callers stage one title
at a time and delete it after, so peak disk/inode use is bounded to a single
title regardless of corpus size, and memory stays bounded to one line at a time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

_log = logging.getLogger(__name__)


class JsonlLineError(Exception):
    """A single malformed JSONL line (carries the 1-based line number)."""

    def __init__(self, line_number: int, message: str) -> None:
        super().__init__(f"line {line_number}: {message}")
        self.line_number = line_number
        self.message = message


def iter_jsonl_payloads(path: str | Path) -> Iterator[tuple[int, str, dict]]:
    """Stream ``(line_number, raw_line, payload)`` for each non-blank line.

    Memory-bounded: reads one line at a time, never the whole file. Blank lines
    are skipped silently. A line that is not a JSON *object* raises
    :class:`JsonlLineError` — the caller decides whether to skip or abort (we
    do not silently drop documents, unlike the resume checkpoint).
    """
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise JsonlLineError(i, f"invalid JSON ({exc})") from exc
            if not isinstance(payload, dict):
                raise JsonlLineError(i, f"expected a JSON object, got {type(payload).__name__}")
            yield i, stripped, payload


def stage_jsonl(
    jsonl_path: str | Path,
    dest_dir: str | Path,
    *,
    skip_bad_lines: bool = True,
) -> tuple[list[Path], list[JsonlLineError]]:
    """Expand *jsonl_path* into one ``<stem>__NNNNNN.json`` file per line under
    *dest_dir*, preserving each line's exact bytes so parsing is byte-identical
    to a hand-authored loose document. Returns ``(staged_paths, errors)``.

    Filenames embed the title stem + zero-padded line index, so they are stable
    across re-runs (resume-safe) and unique across titles in a flat output dir.
    When *skip_bad_lines* is True a malformed line is logged and skipped;
    otherwise the first bad line raises.
    """
    src = Path(jsonl_path)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    stem = src.stem  # e.g. "wa_sn88085187" (the title/LCCN)

    staged: list[Path] = []
    errors: list[JsonlLineError] = []
    # Iterate lines directly (not via iter_jsonl_payloads): a raising generator
    # would terminate on the first bad line and drop every subsequent document.
    with src.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise JsonlLineError(i, f"expected a JSON object, got {type(payload).__name__}")
            except json.JSONDecodeError as exc:
                err = JsonlLineError(i, f"invalid JSON ({exc})")
            except JsonlLineError as exc:
                err = exc
            else:
                out_path = dest / f"{stem}__{i:06d}.json"
                out_path.write_text(stripped, encoding="utf-8")
                staged.append(out_path)
                continue
            if not skip_bad_lines:
                raise err
            _log.warning("stage_jsonl: skipping %s %s", src.name, err)
            errors.append(err)

    _log.info(
        "stage_jsonl: %s -> %d document(s) in %s (%d skipped)",
        src.name, len(staged), dest, len(errors),
    )
    return staged, errors
