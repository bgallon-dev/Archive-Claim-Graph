"""Bounded concurrency stress test for the live Gemynd demo.

The harness deliberately separates cheap public traffic, authenticated
database-backed pages, and expensive synthesis requests so a failure can be
attributed to the correct layer. Credentials are read from an environment
variable and are never printed.

Example:

    $env:GEMYND_DEMO_ACCESS_TOKEN = "..."
    python scripts/stress_demo.py --phase all --output stress.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx


PUBLIC_REQUEST_CAP = 1000
READ_REQUEST_CAP = 300
SYNTHESIS_REQUEST_CAP = 6
CONCURRENCY_CAP = 64

DEFAULT_READ_PATHS = ("/", "/stats", "/explore", "/documents")
DEFAULT_QUESTIONS = (
    "What infrastructure projects changed Spokane between 1900 and 1910, "
    "and what problems were they meant to solve?",
    "How were wheat prices, freight rates, and farmers' organizations "
    "connected in Spokane coverage from 1905 to 1915?",
    "Compare how Spokane newspapers covered mining labor disputes and "
    "railroad labor disputes between 1905 and 1915.",
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _summarize(samples: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    latencies = [float(sample["elapsed_seconds"]) for sample in samples]
    statuses = Counter(str(sample.get("status", "exception")) for sample in samples)
    failures = [
        sample for sample in samples
        if sample.get("error") or not 200 <= int(sample.get("status") or 0) < 400
    ]
    return {
        "requests": len(samples),
        "wall_seconds": round(elapsed, 3),
        "throughput_rps": round(len(samples) / elapsed, 3) if elapsed else 0.0,
        "status_counts": dict(sorted(statuses.items())),
        "failures": len(failures),
        "latency_seconds": {
            "min": round(min(latencies), 3) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "failure_examples": failures[:5],
    }


async def _run_bounded(
    count: int,
    concurrency: int,
    operation: Callable[[int], Awaitable[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(index: int) -> dict[str, Any]:
        async with semaphore:
            return await operation(index)

    started = time.perf_counter()
    samples = await asyncio.gather(*(guarded(index) for index in range(count)))
    return samples, time.perf_counter() - started


async def _request_sample(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.request(method, path, **kwargs)
        elapsed = time.perf_counter() - started
        return {
            "path": path,
            "status": response.status_code,
            "elapsed_seconds": round(elapsed, 3),
            "bytes": len(response.content),
        }
    except Exception as exc:
        return {
            "path": path,
            "status": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _prepare_authenticated_client(
    base_url: str,
    token: str,
    timeout: float,
    limits: httpx.Limits,
) -> httpx.AsyncClient:
    client = httpx.AsyncClient(
        base_url=base_url,
        follow_redirects=True,
        timeout=timeout,
        limits=limits,
    )
    response = await client.get("/auth/login")
    response.raise_for_status()
    client.cookies.set("access_token", token)
    response = await client.get("/auth/me")
    response.raise_for_status()
    return client


async def run_stress(args: argparse.Namespace) -> dict[str, Any]:
    phases = {args.phase} if args.phase != "all" else {"public", "read", "synthesis"}
    token = os.environ.get(args.access_token_env, "")
    if phases.intersection({"read", "synthesis"}) and not token:
        raise RuntimeError(
            f"credential environment variable {args.access_token_env!r} is empty"
        )

    result: dict[str, Any] = {
        "base_url": args.base_url,
        "configuration": {
            "phase": args.phase,
            "public_requests": args.public_requests,
            "read_requests": args.read_requests,
            "synthesis_requests": args.synthesis_requests,
            "concurrency": args.concurrency,
            "synthesis_concurrency": args.synthesis_concurrency,
        },
        "phases": {},
    }
    limits = httpx.Limits(
        max_connections=max(args.concurrency, args.synthesis_concurrency) + 4,
        max_keepalive_connections=max(args.concurrency, args.synthesis_concurrency),
    )

    if "public" in phases:
        async with httpx.AsyncClient(
            base_url=args.base_url,
            follow_redirects=True,
            timeout=args.timeout,
            limits=limits,
        ) as client:
            samples, elapsed = await _run_bounded(
                args.public_requests,
                args.concurrency,
                lambda _: _request_sample(client, "GET", "/auth/login"),
            )
        result["phases"]["public"] = _summarize(samples, elapsed)

    if "read" in phases:
        client = await _prepare_authenticated_client(
            args.base_url, token, args.timeout, limits
        )
        try:
            paths = tuple(args.read_path) or DEFAULT_READ_PATHS
            samples, elapsed = await _run_bounded(
                args.read_requests,
                args.concurrency,
                lambda index: _request_sample(
                    client, "GET", paths[index % len(paths)]
                ),
            )
        finally:
            await client.aclose()
        result["phases"]["read"] = _summarize(samples, elapsed)

    if "synthesis" in phases:
        client = await _prepare_authenticated_client(
            args.base_url, token, args.synthesis_timeout, limits
        )
        csrf = client.cookies.get("csrftoken") or ""
        try:
            async def query(index: int) -> dict[str, Any]:
                question = DEFAULT_QUESTIONS[index % len(DEFAULT_QUESTIONS)]
                sample = await _request_sample(
                    client,
                    "POST",
                    "/query",
                    headers={"X-CSRFToken": csrf},
                    json={
                        "text": question,
                        "mode": "auto",
                        "session_id": f"stress_{int(time.time())}_{index}",
                        "conversation_history": [],
                    },
                )
                # Request the response once more only from the object already in
                # memory is impossible here, so richer answer fields are added
                # by a dedicated path below.
                sample["question_index"] = index % len(DEFAULT_QUESTIONS)
                return sample

            # Synthesis samples need response metadata in addition to generic
            # HTTP metrics, so use a specialized operation.
            async def synthesis_sample(index: int) -> dict[str, Any]:
                question = DEFAULT_QUESTIONS[index % len(DEFAULT_QUESTIONS)]
                started = time.perf_counter()
                try:
                    response = await client.post(
                        "/query",
                        headers={"X-CSRFToken": csrf},
                        json={
                            "text": question,
                            "mode": "auto",
                            "session_id": f"stress_{int(time.time())}_{index}",
                            "conversation_history": [],
                        },
                    )
                    elapsed = time.perf_counter() - started
                    payload = response.json() if response.status_code == 200 else {}
                    return {
                        "path": "/query",
                        "question_index": index % len(DEFAULT_QUESTIONS),
                        "status": response.status_code,
                        "elapsed_seconds": round(elapsed, 3),
                        "bytes": len(response.content),
                        "answer_chars": len(str(payload.get("answer") or "")),
                        "citations": len(payload.get("supporting_claim_ids") or []),
                        "claims_in_context": int(
                            (payload.get("retrieval_stats") or {}).get(
                                "claims_in_context"
                            ) or 0
                        ),
                        "retrieval_plan": (
                            ((payload.get("retrieval_stats") or {}).get("plan_trace") or {})
                            .get("selected_label")
                        ),
                    }
                except Exception as exc:
                    return {
                        "path": "/query",
                        "question_index": index % len(DEFAULT_QUESTIONS),
                        "status": 0,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            samples, elapsed = await _run_bounded(
                args.synthesis_requests,
                args.synthesis_concurrency,
                synthesis_sample,
            )
        finally:
            await client.aclose()
        summary = _summarize(samples, elapsed)
        summary["responses"] = samples
        result["phases"]["synthesis"] = summary

    result["passed"] = all(
        phase["failures"] == 0 for phase in result["phases"].values()
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8788")
    parser.add_argument(
        "--phase",
        choices=("public", "read", "synthesis", "all"),
        default="all",
    )
    parser.add_argument(
        "--access-token-env",
        default="GEMYND_DEMO_ACCESS_TOKEN",
    )
    parser.add_argument("--public-requests", type=int, default=200)
    parser.add_argument("--read-requests", type=int, default=40)
    parser.add_argument("--synthesis-requests", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--synthesis-concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--synthesis-timeout", type=float, default=900.0)
    parser.add_argument("--read-path", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for name, value, cap in (
        ("public requests", args.public_requests, PUBLIC_REQUEST_CAP),
        ("read requests", args.read_requests, READ_REQUEST_CAP),
        ("synthesis requests", args.synthesis_requests, SYNTHESIS_REQUEST_CAP),
        ("concurrency", args.concurrency, CONCURRENCY_CAP),
        ("synthesis concurrency", args.synthesis_concurrency, SYNTHESIS_REQUEST_CAP),
    ):
        if value < 1 or value > cap:
            parser.error(f"{name} must be between 1 and {cap}")
    if args.synthesis_concurrency > args.synthesis_requests:
        parser.error("synthesis concurrency cannot exceed synthesis requests")

    result = asyncio.run(run_stress(args))
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
