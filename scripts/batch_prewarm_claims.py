"""Pre-warm the gemynd claim cache via the Anthropic Message Batches API.

Enumerates every paragraph that the sync ingest pipeline would send to the
Anthropic claim-extraction API, submits them as chunked Message Batches,
and writes each response directly into ``data/claim_cache.db`` using the
same cache key formula as the sync path.  After this script finishes, a
subsequent ``python scripts/ingest_newspapers.py --full`` run sees ~100%
cache hits and its LLM path becomes free local SQLite lookups.

The tool is strictly additive — it does not modify the ingest pipeline,
``HybridClaimExtractor``, or any runtime configuration.  Safe to abort
with Ctrl-C; state lives in a JSONL manifest so you can ``--resume``.

Usage:
    python scripts/batch_prewarm_claims.py                       # dry-run, print plan + exit
    python scripts/batch_prewarm_claims.py --submit              # submit + poll + write
    python scripts/batch_prewarm_claims.py --submit --smoke 5    # same but first 5 docs only
    python scripts/batch_prewarm_claims.py --resume              # poll/fetch pending batches

NOTE: cache keys are derived from the claim-extraction system prompt, which
is built from the Newspapers ``domain_schema.yaml`` / seed entities / claim
type patterns.  If you edit any of those files after pre-warming, the
prompt hash changes and the pre-warmed entries become dead weight.  Re-run
the pre-warm after any domain-resource edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_log = logging.getLogger("batch_prewarm_claims")


REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPHRAG_ROOT = REPO_ROOT.parent
NEWSPAPERS_ROOT = GRAPHRAG_ROOT / "Newspapers"
INPUT_DIR = NEWSPAPERS_ROOT / "gemynd_input"
DOMAIN_DIR = NEWSPAPERS_ROOT
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "claim_cache.db"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "batch_prewarm_manifest.jsonl"

MAX_BATCH_SIZE = 10_000
DEFAULT_POLL_INTERVAL_SEC = 60
BATCH_DISCOUNT = 0.5

# Base per-MTok rates from https://www.anthropic.com/pricing.
# Only current (non-deprecated) models; update this table if the default
# changes in gemynd/ingest/extractors/anthropic_claim_adapter.py.
MODEL_RATES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":   {"input": 5.00, "output": 25.00},
    "claude-opus-4-5":   {"input": 5.00, "output": 25.00},
}

DEFAULT_MAX_TOKENS = 2048  # mirrors AnthropicClaimAdapter(max_tokens=2048) default
OUTPUT_TOKEN_GUESS = 400   # rough midpoint for claim-JSON responses


@dataclass
class Candidate:
    doc_id: str
    paragraph_id: str
    text: str
    cache_key: str = ""


@dataclass
class BatchRecord:
    batch_id: str
    submitted_at: str
    request_count: int
    status: str
    results_written: bool = False
    written_count: int = 0
    error_count: int = 0
    custom_ids_head: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "batch_id": self.batch_id,
            "submitted_at": self.submitted_at,
            "request_count": self.request_count,
            "status": self.status,
            "results_written": self.results_written,
            "written_count": self.written_count,
            "error_count": self.error_count,
            "custom_ids_head": self.custom_ids_head,
        })

    @classmethod
    def from_json(cls, line: str) -> BatchRecord:
        d = json.loads(line)
        return cls(
            batch_id=d["batch_id"],
            submitted_at=d["submitted_at"],
            request_count=d["request_count"],
            status=d["status"],
            results_written=d.get("results_written", False),
            written_count=d.get("written_count", 0),
            error_count=d.get("error_count", 0),
            custom_ids_head=d.get("custom_ids_head", []),
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-warm the gemynd claim cache via Anthropic Message Batches.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Actually submit batches. Without this flag, runs dry-run and exits.")
    mode.add_argument("--resume", action="store_true", help="Poll and fetch pending batches from the existing manifest. No new submissions.")
    p.add_argument("--smoke", type=int, default=None, metavar="N", help="Limit enumeration to first N docs (sorted). Useful for small test runs.")
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH, help=f"Override claim cache DB path. Default: {DEFAULT_CACHE_PATH}")
    p.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH, help=f"Override manifest path. Default: {DEFAULT_MANIFEST_PATH}")
    p.add_argument("--max-batch-size", type=int, default=MAX_BATCH_SIZE, help=f"Max requests per batch submission. Default: {MAX_BATCH_SIZE}")
    p.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SEC, help=f"Seconds between batch status polls. Default: {DEFAULT_POLL_INTERVAL_SEC}")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = p.parse_args()
    if args.resume and args.smoke is not None:
        p.error("--resume is mutually exclusive with --smoke")
    return args


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _require_sdk_batches() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"anthropic SDK not installed: {exc}")
    from anthropic import Anthropic
    probe = Anthropic(api_key="x")
    if not hasattr(probe.messages, "batches"):
        import anthropic as _a
        raise SystemExit(
            f"anthropic SDK {_a.__version__} does not expose messages.batches; "
            "upgrade to a version that supports the Batches API."
        )


def _resolve_model() -> str:
    return (
        os.environ.get("CLAIM_EXTRACTION_MODEL")
        or os.environ.get("SYNTHESIS_MODEL")
        or "claude-haiku-4-5"
    )


def _resolve_min_length() -> int:
    return int(os.environ.get("CLAIM_MIN_PARAGRAPH_LENGTH", "40"))


# ----------------------------------------------------------------------
# Phase 1: load config + fingerprint
# ----------------------------------------------------------------------

def phase1_load_fingerprint() -> tuple[str, str, str, int]:
    if not DOMAIN_DIR.is_dir():
        raise SystemExit(f"Domain dir not found: {DOMAIN_DIR}")
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Input dir not found: {INPUT_DIR}")

    from gemynd.core.domain_config import load_domain_config
    from gemynd.ingest.extractors.anthropic_claim_adapter import _build_system_prompt

    _log.info("Loading domain config from %s", DOMAIN_DIR)
    config = load_domain_config(DOMAIN_DIR)

    system_prompt = _build_system_prompt(config)
    # Must match anthropic_claim_adapter.try_create_anthropic_adapter which
    # truncates the prompt hash to 16 chars before feeding it into cache keys.
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    model = _resolve_model()
    min_length = _resolve_min_length()

    _log.info("model              = %s", model)
    _log.info("prompt_hash[:12]   = %s", prompt_hash[:12])
    _log.info("min_paragraph_len  = %d", min_length)
    _log.info("system prompt len  = %d chars", len(system_prompt))
    return model, system_prompt, prompt_hash, min_length


# ----------------------------------------------------------------------
# Phase 2: parse + enumerate candidates
# ----------------------------------------------------------------------

def phase2_enumerate(min_length: int, smoke_n: int | None) -> tuple[list[Candidate], int, int]:
    from gemynd.ingest.source_parser import parse_source_file

    paths = sorted(INPUT_DIR.glob("*.json"))
    if not paths:
        raise SystemExit(f"No .json files found in {INPUT_DIR}")
    if smoke_n is not None:
        paths = paths[:smoke_n]
    _log.info("parsing %d doc(s) from %s", len(paths), INPUT_DIR)

    candidates: list[Candidate] = []
    paragraphs_seen = 0
    skipped_short = 0

    for i, path in enumerate(paths, start=1):
        try:
            bundle = parse_source_file(path)
        except Exception as exc:
            _log.warning("parse failed for %s: %s", path.name, exc)
            continue
        doc_id = bundle.document.doc_id
        for paragraph in bundle.paragraphs:
            paragraphs_seen += 1
            text = (paragraph.clean_text or paragraph.raw_ocr_text or "").strip()
            if len(text) < min_length:
                skipped_short += 1
                continue
            candidates.append(Candidate(
                doc_id=doc_id,
                paragraph_id=paragraph.paragraph_id,
                text=text,
            ))
        if i % 500 == 0:
            _log.info("  parsed %d/%d docs, %d paragraph candidates so far", i, len(paths), len(candidates))

    _log.info("enumeration done: %d docs, %d paragraphs seen, %d skipped (<%d chars), %d candidates",
              len(paths), paragraphs_seen, skipped_short, min_length, len(candidates))
    return candidates, paragraphs_seen, skipped_short


# ----------------------------------------------------------------------
# Phase 3: cache probe, dedupe, cost estimate
# ----------------------------------------------------------------------

def phase3_probe_and_report(
    candidates: list[Candidate],
    model: str,
    system_prompt: str,
    prompt_hash: str,
    cache_path: Path,
    paragraphs_seen: int,
    skipped_short: int,
    parsed_docs: int,
) -> list[Candidate]:
    from gemynd.ingest.extractors.claim_cache import _cache_key

    for c in candidates:
        c.cache_key = _cache_key(model, prompt_hash, c.text)

    unique: dict[str, Candidate] = {}
    duplicates = 0
    for c in candidates:
        if c.cache_key in unique:
            duplicates += 1
            continue
        unique[c.cache_key] = c

    already_cached: set[str] = set()
    if cache_path.exists():
        conn = sqlite3.connect(str(cache_path))
        try:
            rows = conn.execute(
                "SELECT cache_key FROM claim_cache WHERE model = ?",
                (model,),
            ).fetchall()
            already_cached = {r[0] for r in rows}
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    submission_queue: list[Candidate] = [c for key, c in unique.items() if key not in already_cached]
    prefiltered_hits = len(unique) - len(submission_queue)

    system_prompt_chars = len(system_prompt)
    if submission_queue:
        mean_para_chars = sum(len(c.text) for c in submission_queue) // len(submission_queue)
    else:
        mean_para_chars = 0

    # Rough char→token estimate: 4 chars per token is the standard guess.
    sys_tok = max(1, system_prompt_chars // 4)
    para_tok = max(1, mean_para_chars // 4)
    est_input_per_req = sys_tok + para_tok + 20  # +20 envelope for message structure
    est_input_total = len(submission_queue) * est_input_per_req
    est_output_total = len(submission_queue) * OUTPUT_TOKEN_GUESS

    rates = MODEL_RATES.get(model)
    if rates is None:
        _log.warning("cost estimate unavailable for model=%s; update MODEL_RATES if needed", model)
        est_sync = None
        est_batch = None
    else:
        est_sync = (est_input_total / 1_000_000) * rates["input"] + (est_output_total / 1_000_000) * rates["output"]
        est_batch = est_sync * BATCH_DISCOUNT

    # --- chunking preview ---
    batch_count = (len(submission_queue) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE if submission_queue else 0

    # --- report ---
    print()
    print("=" * 66)
    print(f"[prewarm] parsed docs          = {parsed_docs}")
    print(f"[prewarm] paragraphs seen      = {paragraphs_seen}")
    print(f"[prewarm] < MIN_LENGTH skipped = {skipped_short}")
    print(f"[prewarm] candidates           = {len(candidates)}")
    print(f"[prewarm] duplicate paragraphs = {duplicates} (-{duplicates})")
    print(f"[prewarm] already in cache     = {prefiltered_hits}")
    print(f"[prewarm] to submit            = {len(submission_queue)}")
    print(f"[prewarm] model                = {model}")
    print(f"[prewarm] system prompt chars  = {system_prompt_chars}  (~{sys_tok} tok)")
    print(f"[prewarm] mean para chars      = {mean_para_chars}  (~{para_tok} tok)")
    print(f"[prewarm] est input tokens     = {est_input_total:,} ({len(submission_queue)} x {est_input_per_req})")
    print(f"[prewarm] est output tokens    = {est_output_total:,} ({len(submission_queue)} x {OUTPUT_TOKEN_GUESS}, rough)")
    if rates is not None:
        print(f"[prewarm] est sync cost        = ${est_sync:7.2f}  (${rates['input']:.2f}/M in + ${rates['output']:.2f}/M out)")
        print(f"[prewarm] est batch cost       = ${est_batch:7.2f}  (50% discount)")
    else:
        print(f"[prewarm] est cost             = unavailable for model={model}")
    print(f"[prewarm] batches              = {batch_count} x up to {MAX_BATCH_SIZE} requests")
    print("=" * 66)
    print()

    return submission_queue


# ----------------------------------------------------------------------
# Phase 5: submit batches
# ----------------------------------------------------------------------

def phase5_submit(
    submission_queue: list[Candidate],
    model: str,
    system_prompt: str,
    max_batch_size: int,
    manifest_path: Path,
) -> list[BatchRecord]:
    from anthropic import Anthropic
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    api_key = os.environ.get("Anthropic_API_Key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — cannot submit batches.")
    client = Anthropic(api_key=api_key)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[BatchRecord] = []

    chunks = [
        submission_queue[i : i + max_batch_size]
        for i in range(0, len(submission_queue), max_batch_size)
    ]
    _log.info("submitting %d chunk(s) of up to %d requests each", len(chunks), max_batch_size)

    for idx, chunk in enumerate(chunks, start=1):
        requests = [
            Request(
                custom_id=c.cache_key,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": c.text}],
                ),
            )
            for c in chunk
        ]
        _log.info("[submit %d/%d] creating batch with %d requests...", idx, len(chunks), len(requests))
        try:
            batch = client.messages.batches.create(requests=requests)
        except Exception as exc:
            _log.error("[submit %d/%d] batch creation failed: %s", idx, len(chunks), exc)
            _append_manifest(manifest_path, records)
            raise SystemExit(1) from exc
        record = BatchRecord(
            batch_id=batch.id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            request_count=len(chunk),
            status=getattr(batch, "processing_status", "in_progress"),
            custom_ids_head=[c.cache_key for c in chunk[:5]],
        )
        records.append(record)
        _log.info("[submit %d/%d] submitted batch %s (status=%s)", idx, len(chunks), batch.id, record.status)

    _append_manifest(manifest_path, records)
    return records


def _append_manifest(manifest_path: Path, records: list[BatchRecord]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")


def _load_manifest(manifest_path: Path) -> list[BatchRecord]:
    if not manifest_path.exists():
        return []
    records: list[BatchRecord] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(BatchRecord.from_json(line))
    return records


# ----------------------------------------------------------------------
# Phase 6: poll
# ----------------------------------------------------------------------

def phase6_poll(records: list[BatchRecord], manifest_path: Path, poll_interval: int) -> None:
    from anthropic import Anthropic
    api_key = os.environ.get("Anthropic_API_Key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — cannot poll batches.")
    client = Anthropic(api_key=api_key)

    pending = [r for r in records if r.status != "ended"]
    if not pending:
        _log.info("all batches already ended; skipping poll")
        return

    _log.info("polling %d pending batch(es) every %ds", len(pending), poll_interval)
    try:
        while pending:
            for r in list(pending):
                try:
                    batch = client.messages.batches.retrieve(r.batch_id)
                except Exception as exc:
                    _log.warning("retrieve failed for %s: %s", r.batch_id, exc)
                    continue
                r.status = getattr(batch, "processing_status", r.status)
                counts = getattr(batch, "request_counts", None)
                if counts is not None:
                    _log.info(
                        "  batch %s: %s (processing=%d succeeded=%d errored=%d canceled=%d expired=%d)",
                        r.batch_id[:16], r.status,
                        getattr(counts, "processing", 0),
                        getattr(counts, "succeeded", 0),
                        getattr(counts, "errored", 0),
                        getattr(counts, "canceled", 0),
                        getattr(counts, "expired", 0),
                    )
                else:
                    _log.info("  batch %s: %s", r.batch_id[:16], r.status)
                if r.status == "ended":
                    pending.remove(r)
            _append_manifest(manifest_path, records)
            if pending:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        _append_manifest(manifest_path, records)
        _log.warning("interrupted during poll; rerun with --resume to continue")
        raise SystemExit(0)


# ----------------------------------------------------------------------
# Phase 7: fetch + write to claim_cache.db
# ----------------------------------------------------------------------

_CACHE_INIT_SQL = """\
CREATE TABLE IF NOT EXISTS claim_cache (
    cache_key   TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def phase7_fetch_and_write(
    records: list[BatchRecord],
    manifest_path: Path,
    cache_path: Path,
    model: str,
) -> None:
    from anthropic import Anthropic
    from gemynd.ingest.extractors.anthropic_claim_adapter import _parse_response

    api_key = os.environ.get("Anthropic_API_Key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — cannot fetch batch results.")
    client = Anthropic(api_key=api_key)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    conn.execute(_CACHE_INIT_SQL)
    conn.commit()

    try:
        for r in records:
            if r.status != "ended" or r.results_written:
                continue
            _log.info("fetching results for batch %s (%d requests)", r.batch_id[:16], r.request_count)
            written = 0
            errored = 0
            pending_rows: list[tuple[str, str, str]] = []
            try:
                for result in client.messages.batches.results(r.batch_id):
                    custom_id = getattr(result, "custom_id", None)
                    res = getattr(result, "result", None)
                    res_type = getattr(res, "type", None)
                    if res_type == "succeeded":
                        try:
                            message = res.message
                            raw = message.content[0].text.strip()
                            parsed = _parse_response(raw)
                            pending_rows.append((custom_id, model, json.dumps(parsed)))
                            written += 1
                        except Exception as exc:
                            _log.warning("result parse failed for %s: %s", (custom_id or "?")[:12], exc)
                            errored += 1
                    else:
                        errored += 1
                        _log.debug("non-success result %s: type=%s", (custom_id or "?")[:12], res_type)
                    if len(pending_rows) >= 1000:
                        conn.executemany(
                            "INSERT OR REPLACE INTO claim_cache (cache_key, model, response) VALUES (?, ?, ?)",
                            pending_rows,
                        )
                        conn.commit()
                        pending_rows.clear()
            except KeyboardInterrupt:
                if pending_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO claim_cache (cache_key, model, response) VALUES (?, ?, ?)",
                        pending_rows,
                    )
                    conn.commit()
                r.written_count += written
                r.error_count += errored
                _append_manifest(manifest_path, records)
                _log.warning("interrupted during fetch of %s; rerun with --resume", r.batch_id[:16])
                raise SystemExit(0)

            if pending_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO claim_cache (cache_key, model, response) VALUES (?, ?, ?)",
                    pending_rows,
                )
                conn.commit()

            r.written_count = written
            r.error_count = errored
            r.results_written = True
            _append_manifest(manifest_path, records)
            _log.info("  batch %s: wrote %d, errored %d", r.batch_id[:16], written, errored)
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    from gemynd.shared.env import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    if args.submit or args.resume:
        _require_sdk_batches()

    if args.resume:
        records = _load_manifest(args.manifest_path)
        if not records:
            raise SystemExit(f"No manifest at {args.manifest_path} — nothing to resume.")
        _log.info("resuming %d batch record(s) from %s", len(records), args.manifest_path)
        phase6_poll(records, args.manifest_path, args.poll_interval)
        # We need model for writing — recover from the first record's cache_key prefix is
        # impossible, so recompute from env (must match the submission environment).
        model = _resolve_model()
        phase7_fetch_and_write(records, args.manifest_path, args.cache_path, model)
        _log.info("resume complete")
        return 0

    # Normal path: fingerprint → enumerate → probe → (submit → poll → fetch)
    model, system_prompt, prompt_hash, min_length = phase1_load_fingerprint()

    parsed_docs_count = len(sorted(INPUT_DIR.glob("*.json")))
    if args.smoke is not None:
        parsed_docs_count = min(parsed_docs_count, args.smoke)

    candidates, paragraphs_seen, skipped_short = phase2_enumerate(min_length, args.smoke)

    submission_queue = phase3_probe_and_report(
        candidates=candidates,
        model=model,
        system_prompt=system_prompt,
        prompt_hash=prompt_hash,
        cache_path=args.cache_path,
        paragraphs_seen=paragraphs_seen,
        skipped_short=skipped_short,
        parsed_docs=parsed_docs_count,
    )

    if not args.submit:
        _log.info("dry-run complete; pass --submit to actually send batches")
        return 0

    if not submission_queue:
        _log.info("nothing to submit; cache is fully warm")
        return 0

    records = phase5_submit(
        submission_queue=submission_queue,
        model=model,
        system_prompt=system_prompt,
        max_batch_size=args.max_batch_size,
        manifest_path=args.manifest_path,
    )

    phase6_poll(records, args.manifest_path, args.poll_interval)
    phase7_fetch_and_write(records, args.manifest_path, args.cache_path, model)

    total_written = sum(r.written_count for r in records)
    total_errored = sum(r.error_count for r in records)
    _log.info("pre-warm complete: %d written, %d errored across %d batch(es)",
              total_written, total_errored, len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
