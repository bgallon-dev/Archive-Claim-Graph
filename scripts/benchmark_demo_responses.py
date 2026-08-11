"""Run realistic response-quality checks against the live query demo.

Credentials are read from environment variables and are never printed. Use
either the normal demo password or a pre-issued read-only access token:

    $env:GEMYND_DEMO_PASSWORD = "..."
    python scripts/benchmark_demo_responses.py --case labor_free_speech_temporal

    $env:GEMYND_DEMO_ACCESS_TOKEN = "..."
    python scripts/benchmark_demo_responses.py --case labor_free_speech_temporal

Omit ``--case`` to run the complete benchmark manifest. Results are emitted as
JSON so runs can be compared or archived without scraping console prose.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO_ROOT / "data" / "benchmarks" / "newspaper_demo_queries.json"
_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")


def _evidence_years(response: dict[str, Any]) -> list[int]:
    years: list[int] = []
    for detail in response.get("supporting_claims_detail") or []:
        searchable = " ".join(
            str(detail.get(key) or "")
            for key in ("doc_title", "doc_title_raw", "source_sentence")
        )
        years.extend(int(value) for value in _YEAR_RE.findall(searchable))
    return sorted(set(years))


def evaluate_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Apply transparent evidence and answer gates to one API response."""
    stats = response.get("retrieval_stats") or {}
    details = response.get("supporting_claims_detail") or []
    citations = response.get("supporting_claim_ids") or []
    answer = str(response.get("answer") or "")
    answer_folded = answer.casefold()

    actual_types = sorted(
        {
            str(detail.get("claim_type"))
            for detail in details
            if detail.get("claim_type")
        }
    )
    expected_types = set(case.get("expected_claim_types") or [])
    relevant_count = sum(
        1 for detail in details if detail.get("claim_type") in expected_types
    )
    relevant_ratio = relevant_count / len(details) if details else 0.0

    term_groups = [
        [str(term) for term in group]
        for group in case.get("answer_term_groups") or []
    ]
    matched_term_groups = [
        [term for term in group if term.casefold() in answer_folded]
        for group in term_groups
    ]
    years = _evidence_years(response)
    year_min = case.get("year_min")
    year_max = case.get("year_max")
    years_in_range = [
        year
        for year in years
        if (year_min is None or year >= int(year_min))
        and (year_max is None or year <= int(year_max))
    ]

    checks = {
        "synthesis_available": bool(response.get("synthesis_available", True)),
        "answer_present": len(answer.strip()) >= 80,
        "context_claims": int(stats.get("claims_in_context") or 0)
        >= int(case.get("min_context_claims") or 1),
        "citations": len(citations) >= int(case.get("min_citations") or 1),
        "claim_type_relevance": bool(expected_types.intersection(actual_types))
        and relevant_ratio >= 0.60,
        "answer_topic_terms": bool(matched_term_groups)
        and all(bool(matches) for matches in matched_term_groups),
        "date_range_evidence": len(years_in_range)
        >= int(case.get("min_in_range_years") or 0),
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "confidence": response.get("confidence_assessment"),
            "candidates_retrieved": int(stats.get("candidates_retrieved") or 0),
            "claims_in_context": int(stats.get("claims_in_context") or 0),
            "documents_in_context": int(stats.get("documents_in_context") or 0),
            "citation_count": len(citations),
            "relevant_claim_count": relevant_count,
            "relevant_claim_ratio": round(relevant_ratio, 3),
            "claim_types": actual_types,
            "matched_answer_term_groups": matched_term_groups,
            "evidence_years": years,
            "evidence_years_in_requested_range": years_in_range,
            "retrieval_plan": (stats.get("plan_trace") or {}).get("selected_label"),
            "retrieval_attempts": (stats.get("plan_trace") or {}).get("attempts") or [],
        },
    }


def _load_cases(path: Path, selected: set[str]) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("benchmark manifest must contain a JSON list")
    known = {str(case.get("id")) for case in cases}
    missing = selected - known
    if missing:
        raise ValueError(f"unknown benchmark case(s): {', '.join(sorted(missing))}")
    return [case for case in cases if not selected or str(case.get("id")) in selected]


def _login(
    client: httpx.Client,
    *,
    email: str,
    password: str,
) -> None:
    response = client.get("/auth/login")
    response.raise_for_status()
    csrf = client.cookies.get("csrftoken")
    if not csrf:
        raise RuntimeError("login page did not issue a CSRF cookie")
    response = client.post(
        "/auth/login",
        data={
            "email": email,
            "password": password,
            "next": "/",
            "csrftoken": csrf,
        },
    )
    response.raise_for_status()
    if not client.cookies.get("access_token"):
        raise RuntimeError("login failed: access token cookie was not issued")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8788")
    parser.add_argument("--email", default="demo@museum.local")
    parser.add_argument("--password-env", default="GEMYND_DEMO_PASSWORD")
    parser.add_argument(
        "--access-token-env",
        default="GEMYND_DEMO_ACCESS_TOKEN",
        help="Optional access-token environment variable; skips form login.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this case id; repeat to select multiple cases.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    access_token = os.environ.get(args.access_token_env, "")
    password = os.environ.get(args.password_env, "")
    if not access_token and not password:
        parser.error(
            f"both credential environment variables {args.access_token_env!r} "
            f"and {args.password_env!r} are empty"
        )

    cases = _load_cases(args.cases, set(args.case))
    results: list[dict[str, Any]] = []
    with httpx.Client(
        base_url=args.base_url.rstrip("/"),
        follow_redirects=True,
        timeout=args.timeout,
    ) as client:
        if access_token:
            response = client.get("/auth/login")
            response.raise_for_status()
            client.cookies.set("access_token", access_token)
            response = client.get("/auth/me")
            response.raise_for_status()
        else:
            _login(client, email=args.email, password=password)
        for index, case in enumerate(cases, start=1):
            case_id = str(case["id"])
            print(
                f"[{index}/{len(cases)}] {case_id}: {case['question']}",
                file=sys.stderr,
                flush=True,
            )
            started = time.perf_counter()
            response = client.post(
                "/query",
                headers={"X-CSRFToken": client.cookies.get("csrftoken") or ""},
                json={
                    "text": case["question"],
                    "mode": "auto",
                    "session_id": f"benchmark_{case_id}",
                    "conversation_history": [],
                },
            )
            elapsed = time.perf_counter() - started
            response.raise_for_status()
            payload = response.json()
            evaluation = evaluate_response(case, payload)
            results.append(
                {
                    "case_id": case_id,
                    "category": case.get("category"),
                    "question": case["question"],
                    "elapsed_seconds": round(elapsed, 1),
                    "answer": payload.get("answer"),
                    "caveats": payload.get("caveats") or [],
                    **evaluation,
                }
            )
            print(
                f"  {'PASS' if evaluation['passed'] else 'FAIL'} "
                f"in {elapsed:.1f}s; "
                f"{evaluation['metrics']['citation_count']} citations, "
                f"{evaluation['metrics']['claims_in_context']} context claims",
                file=sys.stderr,
                flush=True,
            )

    report = {
        "base_url": args.base_url,
        "manifest": str(args.cases),
        "case_count": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "results": results,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
