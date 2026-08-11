"""Benchmark deterministic retrieval routing and evidence coverage.

This runner deliberately stops before answer synthesis.  It exercises the
same classifier, entity gateway, Tier-1 detector, retrieval router, Cypher
templates, and context assembler used by the web application, then emits a
machine-readable report suitable for checkpoint-to-checkpoint comparison.

Neo4j credentials are loaded from ``.env``/the environment but are never
written to the report.  If Neo4j is unavailable, the runner still classifies
all cases, records their planned routes, marks execution-dependent checks as
failed, and writes a complete report.
"""
from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO_ROOT / "data" / "benchmarks" / "retrieval_query_coverage.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "out"
    / "benchmarks"
    / "query-coverage"
    / "checkpoint-0-baseline.json"
)

EXPECTED_FAMILY_COUNTS = {
    "broad_regression": 6,
    "named_event_exact_date": 6,
    "mention_only_entity": 6,
    "conversational_follow_up": 5,
    "phrase_historical_geography": 5,
    "structural_list_aggregation": 5,
    "relationship_cross_corpus": 5,
    "compound_tier1_guard": 2,
}

_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|authorization|cookie)",
    re.IGNORECASE,
)
_URI_CREDENTIAL_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.I)


class ManifestValidationError(ValueError):
    """The benchmark manifest does not satisfy the frozen case contract."""


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def validate_manifest(cases: Any) -> list[dict[str, Any]]:
    """Validate and return a 40-case retrieval coverage manifest."""
    if not isinstance(cases, list):
        raise ManifestValidationError("manifest must contain a JSON list")

    errors: list[str] = []
    ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    facet_keys = {
        "claim_types",
        "source_terms_any",
        "source_terms_all",
        "doc_title_terms_any",
        "institution_ids",
        "series",
        "exact_date",
        "year_min",
        "year_max",
    }

    for index, case in enumerate(cases):
        prefix = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix}: must be an object")
            continue

        case_id = case.get("id")
        if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
            errors.append(f"{prefix}: id must be a stable snake_case identifier")
        elif case_id in ids:
            errors.append(f"{prefix}: duplicate id {case_id!r}")
        else:
            ids.add(case_id)

        family = case.get("family")
        if family not in EXPECTED_FAMILY_COUNTS:
            errors.append(f"{prefix}: unknown family {family!r}")
        else:
            family_counts[str(family)] += 1

        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            errors.append(f"{prefix}: turns must be a non-empty list")
        else:
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict) or not isinstance(turn.get("text"), str):
                    errors.append(
                        f"{prefix}.turns[{turn_index}]: text must be a string"
                    )
                elif not turn["text"].strip():
                    errors.append(f"{prefix}.turns[{turn_index}]: text is empty")
                role = turn.get("role", "user") if isinstance(turn, dict) else "user"
                if role not in {"user", "assistant"}:
                    errors.append(
                        f"{prefix}.turns[{turn_index}]: role must be user or assistant"
                    )

        for field in ("expected_routes", "forbidden_routes"):
            values = case.get(field, [])
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                errors.append(f"{prefix}: {field} must be a list of route patterns")

        scope = case.get("scope")
        if not isinstance(scope, dict):
            errors.append(f"{prefix}: scope must be an object")
            scope = {}
        exact_date = scope.get("exact_date")
        if exact_date is not None and (
            not isinstance(exact_date, str) or not _DATE_RE.fullmatch(exact_date)
        ):
            errors.append(f"{prefix}: scope.exact_date must use YYYY-MM-DD")

        facets = case.get("required_evidence_facets")
        if not isinstance(facets, list) or not facets:
            errors.append(f"{prefix}: required_evidence_facets must be non-empty")
            facets = []
        for facet_index, facet in enumerate(facets):
            if not isinstance(facet, dict) or not facet:
                errors.append(
                    f"{prefix}.required_evidence_facets[{facet_index}]: "
                    "must be a non-empty object"
                )
                continue
            unknown = set(facet) - facet_keys
            if unknown:
                errors.append(
                    f"{prefix}.required_evidence_facets[{facet_index}]: "
                    f"unknown keys {sorted(unknown)}"
                )
            if not any(_nonempty(value) for value in facet.values()):
                errors.append(
                    f"{prefix}.required_evidence_facets[{facet_index}]: "
                    "must contain an active matcher"
                )

        max_candidates = case.get("max_single_query_candidates")
        if (
            not isinstance(max_candidates, int)
            or isinstance(max_candidates, bool)
            or not 1 <= max_candidates <= 250
        ):
            errors.append(
                f"{prefix}: max_single_query_candidates must be an integer 1..250"
            )

        has_scope_assertion = any(_nonempty(value) for value in scope.values())
        if not has_scope_assertion and not facets:
            errors.append(
                f"{prefix}: requires at least one deterministic scope or evidence assertion"
            )

    if len(cases) != 40:
        errors.append(f"manifest must contain exactly 40 cases; found {len(cases)}")
    if family_counts != Counter(EXPECTED_FAMILY_COUNTS):
        errors.append(
            "family counts differ: "
            f"expected={dict(EXPECTED_FAMILY_COUNTS)} actual={dict(family_counts)}"
        )

    if errors:
        raise ManifestValidationError("\n".join(errors))
    return cases


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def _block_value(block: Any, field: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(field, default)
    return getattr(block, field, default)


def _block_years(block: Any) -> set[int]:
    years: set[int] = set()
    observation_year = _block_value(block, "year")
    if isinstance(observation_year, int):
        years.add(observation_year)
    for field in ("doc_date_start", "doc_date_end"):
        value = str(_block_value(block, field, "") or "")
        if re.match(r"^\d{4}", value):
            years.add(int(value[:4]))
    return years


def evidence_matches_facet(block: Any, facet: dict[str, Any]) -> bool:
    """Return whether one provenance block satisfies every active facet axis."""
    claim_types = set(str(value) for value in facet.get("claim_types") or [])
    if claim_types and str(_block_value(block, "claim_type", "")) not in claim_types:
        return False

    source_text = str(_block_value(block, "source_sentence", "") or "")
    title_text = " ".join(
        str(_block_value(block, field, "") or "")
        for field in ("doc_title", "doc_title_raw")
    )
    source_folded = source_text.casefold()
    title_folded = title_text.casefold()

    source_any = [str(value).casefold() for value in facet.get("source_terms_any") or []]
    if source_any and not any(term in source_folded for term in source_any):
        return False
    source_all = [str(value).casefold() for value in facet.get("source_terms_all") or []]
    if source_all and not all(term in source_folded for term in source_all):
        return False
    title_any = [
        str(value).casefold() for value in facet.get("doc_title_terms_any") or []
    ]
    if title_any and not any(term in title_folded for term in title_any):
        return False

    institutions = set(str(value) for value in facet.get("institution_ids") or [])
    if institutions and str(_block_value(block, "institution_id", "")) not in institutions:
        return False
    series = set(str(value) for value in facet.get("series") or [])
    if series and str(_block_value(block, "series", "")) not in series:
        return False

    exact_date = facet.get("exact_date")
    if exact_date:
        date_start = str(_block_value(block, "doc_date_start", "") or "")
        if date_start[:10] != str(exact_date):
            return False

    year_min = facet.get("year_min")
    year_max = facet.get("year_max")
    if year_min is not None or year_max is not None:
        years = _block_years(block)
        if not any(
            (year_min is None or year >= int(year_min))
            and (year_max is None or year <= int(year_max))
            for year in years
        ):
            return False

    return True


def evaluate_evidence(
    facets: list[dict[str, Any]],
    blocks: Iterable[Any],
    *,
    limit: int = 50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition evidence facets into matched and missing groups."""
    top_blocks = list(blocks)[:limit]
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, facet in enumerate(facets):
        match_index = next(
            (
                block_index
                for block_index, block in enumerate(top_blocks)
                if evidence_matches_facet(block, facet)
            ),
            None,
        )
        result = {"facet_index": index, "facet": facet}
        if match_index is None:
            missing.append(result)
        else:
            matched.append({**result, "matched_block_rank": match_index + 1})
    return matched, missing


def _route_match(pattern: str, label: str) -> bool:
    if any(char in pattern for char in "*?["):
        return fnmatch.fnmatchcase(label, pattern)
    return label == pattern or label.startswith(pattern + ":")


def evaluate_routes(
    expected: list[str],
    forbidden: list[str],
    attempted_labels: list[str],
    selected_label: str,
) -> dict[str, Any]:
    observed = list(dict.fromkeys([*attempted_labels, selected_label]))
    observed = [label for label in observed if label]
    matched_expected = [
        pattern
        for pattern in expected
        if any(_route_match(pattern, label) for label in observed)
    ]
    matched_forbidden = [
        {"pattern": pattern, "labels": [
            label for label in observed if _route_match(pattern, label)
        ]}
        for pattern in forbidden
        if any(_route_match(pattern, label) for label in observed)
    ]
    return {
        "expected_route_seen": not expected or bool(matched_expected),
        "forbidden_route_absent": not matched_forbidden,
        "matched_expected_routes": matched_expected,
        "matched_forbidden_routes": matched_forbidden,
    }


def _safe_scope_params(params: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "institution_ids",
        "permitted_levels",
        "year_min",
        "year_max",
        "target_date",
        "claim_types",
        "doc_types",
        "series",
        "corpus_a",
        "corpus_b",
        "limit",
        "candidate_limit",
    )
    return {key: params.get(key) for key in keys if key in params}


class RecordingExecutor:
    """Record safe per-template metrics while delegating read-only execution."""

    def __init__(self, delegate: Any | None, *, unavailable_reason: str = "") -> None:
        self.delegate = delegate
        self.unavailable_reason = unavailable_reason
        self.calls: list[dict[str, Any]] = []

    @property
    def connected(self) -> bool:
        return self.delegate is not None

    def clear(self) -> None:
        self.calls.clear()

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        safe_params = dict(params or {})
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        error_type: str | None = None
        if self.delegate is None:
            error_type = "Neo4jUnavailable"
        else:
            try:
                rows = self.delegate.run(cypher, safe_params)
            except Exception as exc:  # benchmark continuity is intentional
                error_type = type(exc).__name__
        self.calls.append(
            {
                "_cypher": cypher,
                "row_count": len(rows),
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "scope_params": _safe_scope_params(safe_params),
                "has_permitted_levels": "permitted_levels" in safe_params,
                "has_institution_ids": "institution_ids" in safe_params,
                "error_type": error_type,
            }
        )
        return rows

    def close(self) -> None:
        if self.delegate is not None:
            self.delegate.close()


def _source_state() -> dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    return {
        "commit_sha": _git("rev-parse", "HEAD") or None,
        "dirty_worktree": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def redact_sensitive(value: Any) -> Any:
    """Recursively remove credentials from report-shaped data."""
    secret_values = [
        env_value
        for env_key, env_value in os.environ.items()
        if _SENSITIVE_KEY_RE.search(env_key) and len(env_value) >= 4
    ]

    def _walk(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): (
                    "[REDACTED]" if _SENSITIVE_KEY_RE.search(str(key)) else _walk(child)
                )
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [_walk(child) for child in item]
        if isinstance(item, tuple):
            return [_walk(child) for child in item]
        if isinstance(item, str):
            text = _URI_CREDENTIAL_RE.sub(r"\g<scheme>[REDACTED]@", item)
            for secret in secret_values:
                text = text.replace(secret, "[REDACTED]")
            return text
        return item

    return _walk(value)


def _entity_report(
    entity_context: Any,
    canonical_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    canonical_names = canonical_names or {}

    def _one(entity: Any) -> dict[str, Any]:
        return {
            "surface_form": entity.surface_form,
            "canonical_name": canonical_names.get(entity.entity_id),
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "confidence": round(float(entity.resolution_confidence), 4),
            "relation": entity.resolution_relation,
        }

    return {
        "resolved": [_one(entity) for entity in entity_context.resolved],
        "maybe_resolved": [_one(entity) for entity in entity_context.maybe_resolved],
        "ambiguous": list(entity_context.ambiguous),
        "unresolved": list(entity_context.unresolved),
    }


def _scope_preserved(
    scope: dict[str, Any],
    entity_report: dict[str, Any],
    executions: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    requested_institutions = list(scope.get("institution_ids") or [])
    if requested_institutions:
        for execution in executions:
            actual = execution["scope_params"].get("institution_ids")
            if actual is None or not set(requested_institutions).issubset(set(actual)):
                failures.append("institution_ids")
                break

    exact_date = scope.get("exact_date")
    if exact_date and not any(
        execution["scope_params"].get("target_date") == exact_date
        for execution in executions
    ):
        failures.append("exact_date")

    for key in ("year_min", "year_max"):
        requested = scope.get(key)
        if requested is not None and not any(
            execution["scope_params"].get(key) == requested for execution in executions
        ):
            failures.append(key)

    requested_entities = [str(value) for value in scope.get("entity_names") or []]
    if requested_entities:
        observed = [
            str(value).casefold()
            for bucket in ("resolved", "maybe_resolved")
            for entity in entity_report.get(bucket, [])
            for value in (
                entity.get("surface_form"),
                entity.get("canonical_name"),
            )
            if value
        ]
        observed.extend(str(value).casefold() for value in entity_report.get("unresolved", []))
        for requested in requested_entities:
            folded = requested.casefold()
            if not any(folded in value or value in folded for value in observed if value):
                failures.append(f"entity:{requested}")

    return not failures, failures


def _template_report(
    calls: list[dict[str, Any]],
    template_names: dict[str, str],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for call in calls:
        report.append(
            {
                "template_name": template_names.get(call["_cypher"], "UNKNOWN"),
                "row_count": call["row_count"],
                "duration_ms": call["duration_ms"],
                "scope_params": call["scope_params"],
                "has_permitted_levels": call["has_permitted_levels"],
                "has_institution_ids": call["has_institution_ids"],
                "error_type": call["error_type"],
            }
        )
    return report


def _known_template_names() -> dict[str, str]:
    """Return names for every module-level Cypher query constant."""
    from gemynd.core.graph import cypher

    return {
        value: name.removesuffix("_QUERY")
        for name, value in vars(cypher).items()
        if name.endswith("_QUERY") and isinstance(value, str)
    }


def _trace_report(trace: Any) -> dict[str, Any]:
    if trace is None:
        return {"selected_label": "", "resolved_entity_ids": [], "attempts": []}
    return {
        "selected_label": trace.selected_label,
        "resolved_entity_ids": list(trace.resolved_entity_ids),
        "attempts": [
            {
                "label": attempt.label,
                "template_names": list(attempt.template_names),
                "row_count": attempt.row_count,
                "duration_ms": round(attempt.duration_ms, 3),
            }
            for attempt in trace.attempts
        ],
    }


@dataclasses.dataclass
class BenchmarkRuntime:
    composite: Any
    recorder: RecordingExecutor
    gateway: Any
    assembler: Any
    startup_error: str = ""

    def close(self) -> None:
        self.assembler.close()
        self.recorder.close()


def build_runtime(*, connect: bool = True) -> BenchmarkRuntime:
    """Build a production-equivalent retrieval stack without synthesis."""
    from gemynd.core.domain_config import (
        load_all_registered_corpora,
        merge_domain_configs,
    )
    from gemynd.core.resolver import DictionaryFuzzyResolver, ResolutionPolicy
    from gemynd.retrieval.context_assembler import ProvenanceContextAssembler
    from gemynd.retrieval.entity_gateway import EntityResolutionGateway
    from gemynd.shared.env import load_dotenv
    from gemynd.shared.neo4j_helpers import build_neo4j_executor
    from gemynd.shared.settings import Settings

    load_dotenv(REPO_ROOT / ".env")
    settings = Settings.from_env()
    members = load_all_registered_corpora()
    if not members:
        raise RuntimeError("no registered corpora could be loaded")
    composite = merge_domain_configs(members)

    delegate = None
    startup_error = ""
    if connect:
        try:
            delegate = build_neo4j_executor(
                settings,
                entity_labels=composite.entity_labels,
            )
        except Exception as exc:
            startup_error = f"{type(exc).__name__}: {exc}"
    else:
        startup_error = "Offline mode requested; Neo4j was not contacted."

    recorder = RecordingExecutor(delegate, unavailable_reason=startup_error)
    assembler = ProvenanceContextAssembler(
        recorder,
        budget_conversational=50,
        budget_hybrid=50,
        query_intent_map=composite.query_intent_to_claim_types,
        institution_ids=composite.institution_ids,
        anchors={
            institution_id: composite.anchor_for(institution_id)
            for institution_id in composite.institution_ids
        },
    )
    gateway = EntityResolutionGateway(
        resolver=DictionaryFuzzyResolver(
            config=composite,
            policy=ResolutionPolicy(),
        ),
        executor=recorder if recorder.connected else None,
    )
    recorder.clear()  # exclude assembler anchor probes from case metrics
    return BenchmarkRuntime(
        composite=composite,
        recorder=recorder,
        gateway=gateway,
        assembler=assembler,
        startup_error=startup_error,
    )


def run_case(case: dict[str, Any], runtime: BenchmarkRuntime) -> dict[str, Any]:
    """Run one case and always return a report, including on failure."""
    from gemynd.retrieval.classifier import classify_query, detect_tier1_intent
    from gemynd.retrieval.context_assembler import _infer_claim_types

    query_text = str(case["turns"][-1]["text"])
    scope = dict(case.get("scope") or {})
    default_member = runtime.composite.default_member
    year_validation = default_member.year_validation or {}
    year_bounds = None
    if (
        year_validation.get("min_year") is not None
        and year_validation.get("max_year") is not None
    ):
        year_bounds = (
            int(year_validation["min_year"]),
            int(year_validation["max_year"]),
        )
    seed_vocab = frozenset(
        entity.normalized_form.lower()
        for entity in runtime.composite.seed_entities
        if entity.normalized_form
    )
    intent = classify_query(
        query_text,
        extra_stopwords=runtime.composite.extraction_stopwords,
        year_bounds=year_bounds,
        seed_vocab=seed_vocab,
    )
    inferred_claim_types = _infer_claim_types(
        query_text,
        intent_map=runtime.composite.query_intent_to_claim_types,
    )
    entity_context = runtime.gateway.resolve(
        intent.entities,
        inferred_claim_types=inferred_claim_types,
    )
    entity_report = _entity_report(
        entity_context,
        {
            entity.entity_id: entity.name
            for entity in runtime.composite.seed_entities
        },
    )
    runtime.recorder.clear()  # exclude live-ID resolution probes

    blocks: list[Any] = []
    trace: Any = None
    tier1_result: Any = None
    case_error: str | None = None
    started = time.perf_counter()
    try:
        tier1 = detect_tier1_intent(
            query_text,
            entity_context,
            year_min=intent.year_min,
            year_max=intent.year_max,
        )
        if tier1 is not None and intent.bucket != "analytical":
            tier1_result = runtime.assembler.tier_1_lookup(
                tier1,
                institution_ids=list(scope.get("institution_ids") or []),
                permitted_levels=list(scope.get("permitted_levels") or ["public"]),
            )
            from gemynd.retrieval.models import (
                RetrievalPlanAttempt,
                RetrievalPlanTrace,
            )

            label = f"tier_1:{tier1.operation}"
            trace = RetrievalPlanTrace(
                attempts=[
                    RetrievalPlanAttempt(
                        label=label,
                        template_names=["TIER_1"],
                        row_count=sum(call["row_count"] for call in runtime.recorder.calls),
                        duration_ms=sum(
                            call["duration_ms"] for call in runtime.recorder.calls
                        ),
                    )
                ],
                resolved_entity_ids=[
                    entity.entity_id for entity in entity_context.resolved
                ],
                selected_label=label,
            )
        else:
            blocks, _ = runtime.assembler.assemble(
                query_text=query_text,
                entity_context=entity_context,
                year_min=intent.year_min,
                year_max=intent.year_max,
                is_hybrid=intent.bucket == "hybrid",
                permitted_levels=list(scope.get("permitted_levels") or ["public"]),
                institution_ids=list(scope.get("institution_ids") or []),
                intent=intent,
                doc_types=list(scope.get("doc_types") or []),
                series=list(scope.get("series") or []),
                allow_llm_cypher=False,
            )
            trace = runtime.assembler._last_plan_trace
    except Exception as exc:
        case_error = f"{type(exc).__name__}: {exc}"

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    template_names = _known_template_names()
    template_names.update(runtime.assembler._template_names)
    template_executions = _template_report(
        runtime.recorder.calls,
        template_names,
    )
    trace_data = _trace_report(trace)
    attempted_labels = [
        attempt["label"] for attempt in trace_data.get("attempts", [])
    ]
    route_eval = evaluate_routes(
        list(case.get("expected_routes") or []),
        list(case.get("forbidden_routes") or []),
        attempted_labels,
        str(trace_data.get("selected_label") or ""),
    )
    matched_facets, missing_facets = evaluate_evidence(
        list(case.get("required_evidence_facets") or []),
        blocks,
    )
    scope_ok, scope_failures = _scope_preserved(
        scope,
        entity_report,
        template_executions,
    )
    max_candidates = int(case["max_single_query_candidates"])
    candidate_bound = all(
        int(execution["row_count"]) <= max_candidates
        for execution in template_executions
    )
    access_scope = bool(template_executions) and all(
        execution["has_permitted_levels"] and execution["has_institution_ids"]
        for execution in template_executions
    )
    execution_available = runtime.recorder.connected and all(
        execution["error_type"] is None for execution in template_executions
    )
    evidence_coverage = (
        len(matched_facets)
        / len(case["required_evidence_facets"])
        if case["required_evidence_facets"]
        else 1.0
    )

    checks = {
        "execution_available": execution_available,
        "expected_route_seen": route_eval["expected_route_seen"],
        "forbidden_route_absent": route_eval["forbidden_route_absent"],
        "scope_preserved": scope_ok,
        "access_scope_parameters": access_scope,
        "candidate_bound": candidate_bound,
        "evidence_present": bool(matched_facets),
        "evidence_coverage_90pct": evidence_coverage >= 0.90,
        "case_completed_without_error": case_error is None,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    dates = sorted(
        {
            str(_block_value(block, "doc_date_start"))
            for block in blocks
            if _block_value(block, "doc_date_start")
        }
    )
    claim_types = sorted(
        {
            str(_block_value(block, "claim_type"))
            for block in blocks
            if _block_value(block, "claim_type")
        }
    )
    institutions = sorted(
        {
            str(_block_value(block, "institution_id"))
            for block in blocks
            if _block_value(block, "institution_id")
        }
    )
    candidate_count = (
        int(tier1_result.count)
        if tier1_result is not None
        else int(runtime.assembler._last_candidate_count)
    )

    return {
        "case_id": case["id"],
        "family": case["family"],
        "turn_count": len(case["turns"]),
        "intent": dataclasses.asdict(intent),
        "entities": entity_report,
        "retrieval_plan": trace_data,
        "template_executions": template_executions,
        "selected_rung": trace_data.get("selected_label") or "",
        "candidate_count": candidate_count,
        "attempted_candidate_count": sum(
            execution["row_count"] for execution in template_executions
        ),
        "context_count": len(blocks),
        "matched_evidence_facets": matched_facets,
        "missing_evidence_facets": missing_facets,
        "evidence_coverage": round(evidence_coverage, 4),
        "represented": {
            "document_dates": dates,
            "claim_types": claim_types,
            "institutions": institutions,
        },
        "scope_preserved": scope_ok,
        "scope_failures": scope_failures,
        "route_evaluation": route_eval,
        "elapsed_ms": elapsed_ms,
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": failed_checks,
        "error": case_error,
    }


def build_report(
    cases: list[dict[str, Any]],
    runtime: BenchmarkRuntime,
    *,
    manifest_path: Path,
    selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    selected_ids = selected_ids or set()
    chosen = [case for case in cases if not selected_ids or case["id"] in selected_ids]
    unknown = selected_ids - {case["id"] for case in cases}
    if unknown:
        raise ManifestValidationError(
            f"unknown benchmark case(s): {', '.join(sorted(unknown))}"
        )

    results: list[dict[str, Any]] = []
    for index, case in enumerate(chosen, start=1):
        print(
            f"[{index}/{len(chosen)}] {case['id']}",
            file=sys.stderr,
            flush=True,
        )
        result = run_case(case, runtime)
        results.append(result)
        print(
            f"  {'PASS' if result['passed'] else 'FAIL'} "
            f"route={result['selected_rung'] or 'none'} "
            f"context={result['context_count']}",
            file=sys.stderr,
            flush=True,
        )

    manifest_bytes = manifest_path.read_bytes()
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source": _source_state(),
        "execution": {
            "connection_attempted": not runtime.startup_error.startswith(
                "Offline mode requested"
            ),
            "neo4j_available": runtime.recorder.connected,
            "startup_error": runtime.startup_error or None,
            "synthesis_executed": False,
        },
        "summary": {
            "case_count": len(results),
            "passed": sum(1 for result in results if result["passed"]),
            "failed": sum(1 for result in results if not result["passed"]),
            "family_counts": dict(Counter(result["family"] for result in results)),
        },
        "results": results,
    }
    return redact_sensitive(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this case id; repeat to select multiple cases.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Do not contact Neo4j. Classify all cases and capture their planned "
            "routes with execution-dependent checks marked failed."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write the JSON artifact without also printing it to stdout.",
    )
    args = parser.parse_args()

    try:
        cases = load_manifest(args.cases)
        runtime = build_runtime(connect=not args.offline)
    except Exception as exc:
        parser.error(str(exc))

    try:
        report = build_report(
            cases,
            runtime,
            manifest_path=args.cases,
            selected_ids=set(args.case),
        )
    finally:
        runtime.close()

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
