"""Run the gemynd ingest pipeline against the Spokane newspapers corpus.

Thin wrapper around `gemynd run-e2e` that hardcodes the paths for the
Newspapers corpus living outside this repo:

    <repo_root>/../Newspapers/gemynd_input/   (4,942 article JSONs)
    <repo_root>/../Newspapers/                (domain resources dir)
    <repo_root>/out/newspapers/               (structure/semantic bundles)

Usage:
    python scripts/ingest_newspapers.py                  # smoke test, first 3 files, LLM on
    python scripts/ingest_newspapers.py --smoke 10       # smoke test, first 10 files
    python scripts/ingest_newspapers.py --full           # full 4,942-doc run
    python scripts/ingest_newspapers.py --full --no-llm  # full run, rule-based only
    python scripts/ingest_newspapers.py --smoke 3 --workers 4
    python scripts/ingest_newspapers.py --out-dir out/newspapers_rerun
    python scripts/ingest_newspapers.py --full --backend neo4j --workers 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPHRAG_ROOT = REPO_ROOT.parent
NEWSPAPERS_ROOT = GRAPHRAG_ROOT / "Newspapers"
INPUT_DIR = NEWSPAPERS_ROOT / "gemynd_input"
DOMAIN_DIR = NEWSPAPERS_ROOT
DEFAULT_OUT_DIR = REPO_ROOT / "out" / "newspapers"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gemynd run-e2e against the Spokane newspapers corpus.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        type=int,
        default=3,
        metavar="N",
        help="Run on only the first N newspaper JSONs (sorted). Default: 3.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run on the entire corpus (~4,942 docs). Mutually exclusive with --smoke.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based claim extraction (zero API calls, rule-based only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for extraction. Default: 1. Use 0 to auto-detect.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for structure/semantic bundles. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--backend",
        choices=("memory", "neo4j"),
        default="memory",
        help="Graph backend. Default: memory (volatile). Use neo4j to persist to the running server.",
    )
    return parser.parse_args()


def _resolve_inputs(full: bool, smoke_n: int) -> list[str]:
    if full:
        return [str(INPUT_DIR)]
    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json files found in {INPUT_DIR}")
    if smoke_n > len(files):
        smoke_n = len(files)
    return [str(f) for f in files[:smoke_n]]


def main() -> int:
    args = _parse_args()

    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Input dir not found: {INPUT_DIR}")
    if not DOMAIN_DIR.is_dir():
        raise SystemExit(f"Domain dir not found: {DOMAIN_DIR}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs(full=args.full, smoke_n=args.smoke)

    mode_label = "FULL" if args.full else f"SMOKE ({len(inputs)} files)"
    print(f"[ingest_newspapers] mode        = {mode_label}")
    print(f"[ingest_newspapers] input_dir   = {INPUT_DIR}")
    print(f"[ingest_newspapers] domain_dir  = {DOMAIN_DIR}")
    print(f"[ingest_newspapers] out_dir     = {out_dir}")
    print(f"[ingest_newspapers] llm         = {'off' if args.no_llm else 'on'}")
    print(f"[ingest_newspapers] workers     = {args.workers}")
    print(f"[ingest_newspapers] backend     = {args.backend}")

    argv = [
        sys.executable,
        "-m",
        "gemynd.cli",
        "run-e2e",
        "--inputs",
        *inputs,
        "--out-dir",
        str(out_dir),
        "--domain-dir",
        str(DOMAIN_DIR),
        "--backend",
        args.backend,
        "--workers",
        str(args.workers),
    ]
    if args.no_llm:
        argv.append("--no-llm")

    print(f"[ingest_newspapers] invoking: {' '.join(argv[:6])} ... (+{len(argv) - 6} args)")
    result = subprocess.run(argv, cwd=str(REPO_ROOT))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
