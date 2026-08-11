"""Bounded, evidence-gap-driven research over archival claims.

The orchestrator owns every control-flow decision.  Planners and optional model
advisers may propose searches, but cannot change access scope, budgets, or stop
the investigation.  This keeps the loop observable and fully testable without
Neo4j or an LLM.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from gemynd.retrieval.models import ProvenanceBlock, SynthesisResult

_log = logging.getLogger(__name__)

_DEFAULT_MAX_ANGLES = 4
_DEFAULT_MAX_CONSECUTIVE_NO_GAIN = 2
_DEFAULT_MAX_FAILURES = 2
_DEFAULT_MAX_ADVISER_CALLS = 3
_DEFAULT_MAX_PROPOSALS = 2
_MAX_ANGLE_QUERY_CHARS = 500


class StopReason(str, Enum):
    ANGLE_BUDGET_EXHAUSTED = "angle_budget_exhausted"
    FAILURE_BUDGET_EXHAUSTED = "failure_budget_exhausted"
    NO_NEW_EVIDENCE = "no_new_evidence"
    NO_CANDIDATE_GAPS = "no_candidate_gaps"


@dataclass(slots=True)
class PermissionScope:
    permitted_levels: list[str]
    institution_ids: list[str]


@dataclass(slots=True)
class QueryProbe:
    entities: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None


@dataclass(slots=True)
class ResearchAngle:
    label: str
    query_text: str
    rationale: str


@dataclass(frozen=True, slots=True)
class ResearchObjective:
    question: str
    task_type: str = "historical_research"
    target_entities: tuple[str, ...] = ()
    year_min: int | None = None
    year_max: int | None = None
    answer_requirements: tuple[str, ...] = (
        "answer grounded in retrieved claims",
        "documentary provenance for material assertions",
    )


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    claim_id: str
    claim_type: str
    doc_title: str
    doc_date_start: str | None
    source_sentence: str


@dataclass(frozen=True, slots=True)
class AngleAttempt:
    label: str
    query_text: str
    rationale: str
    status: str
    blocks_retrieved: int = 0
    new_claims: int = 0
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class InvestigationSnapshot:
    objective: ResearchObjective
    attempts: tuple[AngleAttempt, ...]
    evidence: tuple[EvidenceSummary, ...]
    remaining_attempts: int


@dataclass(frozen=True, slots=True)
class EvidenceGapAssessment:
    proposed_angles: tuple[ResearchAngle, ...] = ()
    known: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    provider_used: str | None = None


@dataclass(slots=True)
class _MergedClaim:
    block: "ProvenanceBlock"
    surfaced_by: list[str]


@dataclass(slots=True)
class DeepResearchResult:
    original_query: str
    angles: list[ResearchAngle]
    merged_blocks: list["ProvenanceBlock"]
    context_text: str
    blocks_by_angle: dict[str, int]
    new_claims_by_angle: dict[str, int]
    synthesis: "SynthesisResult | None"
    timings_ms: dict[str, float]
    stop_reason: StopReason = StopReason.NO_CANDIDATE_GAPS
    attempts: list[AngleAttempt] = field(default_factory=list)
    objective: ResearchObjective | None = None
    assessment: EvidenceGapAssessment | None = None
    adviser_calls: int = 0
    adviser_failed: bool = False
    synthesis_error: str | None = None

    @property
    def unique_claims(self) -> int:
        return len(self.merged_blocks)

    @property
    def coverage_gain(self) -> int:
        first = self.angles[0].label if self.angles else None
        return sum(n for label, n in self.new_claims_by_angle.items() if label != first)

    @property
    def estimated_context_tokens(self) -> int:
        return len(self.context_text) // 4


class RetrievalBackend(Protocol):
    def probe(self, query: str) -> QueryProbe: ...

    def gather(
        self,
        angle_query: str,
        scope: PermissionScope,
        *,
        allow_llm_cypher: bool = True,
    ) -> list["ProvenanceBlock"]: ...

    def synthesise(
        self,
        query: str,
        blocks: list["ProvenanceBlock"],
        context_text: str,
        scope: PermissionScope,
        conversation_history: list[dict[str, str]] | None,
    ) -> "SynthesisResult": ...


class ResearchPlanner(Protocol):
    def plan(self, query: str, probe: QueryProbe, max_angles: int) -> list[ResearchAngle]: ...


class EvidenceGapAdviser(Protocol):
    def advise(self, snapshot: InvestigationSnapshot) -> EvidenceGapAssessment: ...


class HeuristicPlanner:
    """Free, deterministic fallbacks retained behind model-proposed angles."""

    def plan(self, query: str, probe: QueryProbe, max_angles: int) -> list[ResearchAngle]:
        angles = [ResearchAngle("original", query, "The question as asked.")]
        seen_labels = {"original"}
        for ent in probe.entities:
            label = f"entity:{ent}"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            angles.append(ResearchAngle(
                label,
                f"{query} — specifically regarding {ent}",
                f"Re-anchor retrieval on the resolved entity '{ent}'.",
            ))
        angles.append(ResearchAngle(
            "broadened",
            f"Background and surrounding context for: {query}",
            "Seek contextual claims omitted by focused retrieval.",
        ))
        return angles[:max_angles]


class ModelEvidenceGapAdviser:
    """Provider-neutral strict-JSON adviser.

    ``complete_json`` must return either a JSON object or ``(object,
    provider_id)``.  The surrounding orchestrator still validates and caps all
    proposed angles.
    """

    _SYSTEM_PROMPT = """You identify missing evidence in an archival investigation.
Return exactly one JSON object with keys gaps, known, missing, conflicts.
gaps is a list of objects with exactly label, query_text, rationale.
Do not declare the investigation complete and do not propose access or budget changes."""

    def __init__(self, complete_json: Callable[..., Any]) -> None:
        self._complete_json = complete_json

    def advise(self, snapshot: InvestigationSnapshot) -> EvidenceGapAssessment:
        payload = {
            "objective": {
                "question": snapshot.objective.question,
                "task_type": snapshot.objective.task_type,
                "target_entities": list(snapshot.objective.target_entities),
                "year_min": snapshot.objective.year_min,
                "year_max": snapshot.objective.year_max,
                "answer_requirements": list(snapshot.objective.answer_requirements),
            },
            "attempts": [
                {
                    "label": a.label,
                    "query_text": a.query_text,
                    "status": a.status,
                    "blocks_retrieved": a.blocks_retrieved,
                    "new_claims": a.new_claims,
                }
                for a in snapshot.attempts
            ],
            "evidence": [
                {
                    "claim_id": e.claim_id,
                    "claim_type": e.claim_type,
                    "doc_title": e.doc_title,
                    "doc_date_start": e.doc_date_start,
                    "source_sentence": e.source_sentence,
                }
                for e in snapshot.evidence
            ],
            "remaining_attempts": snapshot.remaining_attempts,
        }
        import json

        completed = self._complete_json(
            system_prompt=self._SYSTEM_PROMPT,
            user_message=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            max_tokens=900,
        )
        provider_used = None
        raw = completed
        if isinstance(completed, tuple) and len(completed) == 2:
            raw, provider_used = completed
        if not isinstance(raw, dict) or set(raw) != {"gaps", "known", "missing", "conflicts"}:
            raise ValueError("invalid evidence-gap response envelope")
        for key in ("gaps", "known", "missing", "conflicts"):
            if not isinstance(raw[key], list):
                raise ValueError(f"evidence-gap field {key!r} must be a list")
        proposed: list[ResearchAngle] = []
        for gap in raw["gaps"]:
            if not isinstance(gap, dict) or set(gap) != {"label", "query_text", "rationale"}:
                raise ValueError("invalid evidence-gap proposal")
            if not all(isinstance(gap[k], str) for k in gap):
                raise ValueError("evidence-gap proposal values must be strings")
            proposed.append(ResearchAngle(gap["label"], gap["query_text"], gap["rationale"]))
        text_fields: dict[str, tuple[str, ...]] = {}
        for key in ("known", "missing", "conflicts"):
            if not all(isinstance(value, str) for value in raw[key]):
                raise ValueError(f"evidence-gap field {key!r} must contain strings")
            text_fields[key] = tuple(value for value in raw[key] if value.strip())
        return EvidenceGapAssessment(
            proposed_angles=tuple(proposed),
            known=text_fields["known"],
            missing=text_fields["missing"],
            conflicts=text_fields["conflicts"],
            provider_used=str(provider_used) if provider_used else None,
        )


class DeepResearchOrchestrator:
    def __init__(
        self,
        backend: RetrievalBackend,
        planner: ResearchPlanner | None = None,
        *,
        adviser: EvidenceGapAdviser | None = None,
        max_angles: int = _DEFAULT_MAX_ANGLES,
        max_consecutive_no_gain: int = _DEFAULT_MAX_CONSECUTIVE_NO_GAIN,
        max_failures: int = _DEFAULT_MAX_FAILURES,
        max_adviser_calls: int = _DEFAULT_MAX_ADVISER_CALLS,
        max_proposals_per_call: int = _DEFAULT_MAX_PROPOSALS,
        char_budget_fn: Callable[[list["ProvenanceBlock"]], list["ProvenanceBlock"]] | None = None,
        serialise_fn: Callable[["ProvenanceBlock"], str] | None = None,
        llm_cypher_on_reformulations: bool = False,
    ) -> None:
        self.backend = backend
        self.planner = planner or HeuristicPlanner()
        self.adviser = adviser
        self.max_angles = max(1, max_angles)
        self.max_consecutive_no_gain = max(1, max_consecutive_no_gain)
        self.max_failures = max(1, max_failures)
        self.max_adviser_calls = max(0, max_adviser_calls)
        self.max_proposals_per_call = max(1, max_proposals_per_call)
        self.llm_cypher_on_reformulations = llm_cypher_on_reformulations
        self._char_budget_fn = char_budget_fn or (lambda blocks: blocks)
        self._serialise_fn = serialise_fn or (lambda b: str(getattr(b, "claim_id", "")))

    def research(
        self,
        query: str,
        scope: PermissionScope,
        *,
        conversation_history: list[dict[str, str]] | None = None,
        synthesise: bool = True,
    ) -> DeepResearchResult:
        timings: dict[str, float] = {}
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        probe = self.backend.probe(query)
        timings["probe_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        objective = ResearchObjective(
            question=query,
            target_entities=tuple(probe.entities),
            year_min=probe.year_min,
            year_max=probe.year_max,
        )

        try:
            planned = self.planner.plan(query, probe, self.max_angles)
        except Exception:  # noqa: BLE001 - deterministic fallback by design
            _log.exception("deep-research: planner failed; using original query")
            planned = []
        queue = self._seed_queue(query, planned)
        merged: dict[str, _MergedClaim] = {}
        blocks_by_angle: dict[str, int] = {}
        new_by_angle: dict[str, int] = {}
        attempts: list[AngleAttempt] = []
        attempted_angles: list[ResearchAngle] = []
        seen_labels = {_normalise(a.label) for a in queue}
        seen_queries = {_normalise(a.query_text) for a in queue}
        failures = 0
        no_gain_streak = 0
        adviser_calls = 0
        adviser_failed = False
        assessment: EvidenceGapAssessment | None = None
        stop_reason = StopReason.NO_CANDIDATE_GAPS

        t0 = time.perf_counter()
        while True:
            stop_reason = self._stop_reason(
                len(attempts), failures, no_gain_streak, bool(queue)
            )
            if stop_reason is not None:
                break

            angle = queue.pop(0)
            attempted_angles.append(angle)
            allow_llm = len(attempts) == 0 or self.llm_cypher_on_reformulations
            try:
                blocks = self.backend.gather(
                    angle.query_text, scope, allow_llm_cypher=allow_llm
                )
            except Exception:  # noqa: BLE001 - one angle cannot sink the run
                _log.exception("deep-research: angle %r failed", angle.label)
                failures += 1
                blocks_by_angle[angle.label] = 0
                new_by_angle[angle.label] = 0
                attempts.append(AngleAttempt(
                    angle.label, angle.query_text, angle.rationale,
                    "failed", error_code="gather_failed",
                ))
                continue

            blocks_by_angle[angle.label] = len(blocks)
            added = self._merge_blocks(merged, blocks, angle.label)
            new_by_angle[angle.label] = added
            attempts.append(AngleAttempt(
                angle.label, angle.query_text, angle.rationale,
                "succeeded", len(blocks), added,
            ))
            no_gain_streak = 0 if added else no_gain_streak + 1

            terminal = self._stop_reason(
                len(attempts), failures, no_gain_streak, bool(queue)
            )
            if terminal in {
                StopReason.ANGLE_BUDGET_EXHAUSTED,
                StopReason.FAILURE_BUDGET_EXHAUSTED,
                StopReason.NO_NEW_EVIDENCE,
            }:
                stop_reason = terminal
                break

            if (
                self.adviser is not None
                and not adviser_failed
                and adviser_calls < self.max_adviser_calls
            ):
                adviser_calls += 1
                snapshot = self._snapshot(
                    objective, attempts, merged, self.max_angles - len(attempts)
                )
                try:
                    proposed_assessment = self.adviser.advise(snapshot)
                    accepted = self._validate_proposals(
                        proposed_assessment.proposed_angles,
                        query,
                        seen_labels,
                        seen_queries,
                        self.max_angles - len(attempts),
                    )
                    assessment = EvidenceGapAssessment(
                        proposed_angles=tuple(accepted),
                        known=proposed_assessment.known,
                        missing=proposed_assessment.missing,
                        conflicts=proposed_assessment.conflicts,
                        provider_used=proposed_assessment.provider_used,
                    )
                    if accepted:
                        queue = accepted + queue
                except Exception:  # noqa: BLE001 - fallback queue remains safe
                    _log.exception("deep-research: adviser failed; using heuristic fallbacks")
                    adviser_failed = True

        timings["gather_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        ordered = sorted(
            merged.items(),
            key=lambda item: (
                -len(item[1].surfaced_by),
                -_confidence(item[1].block),
                item[0],
            ),
        )
        capped = self._char_budget_fn([entry.block for _, entry in ordered])
        context_text = "\n\n".join(self._serialise_fn(block) for block in capped)

        synthesis = None
        synthesis_error = None
        if synthesise and capped:
            t0 = time.perf_counter()
            try:
                synthesis = self.backend.synthesise(
                    query=query,
                    blocks=capped,
                    context_text=context_text,
                    scope=scope,
                    conversation_history=conversation_history,
                )
            except Exception:  # noqa: BLE001 - preserve retrieved evidence
                _log.exception("deep-research: final synthesis failed")
                synthesis_error = "synthesis_failed"
            timings["synthesise_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        timings["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
        return DeepResearchResult(
            original_query=query,
            angles=attempted_angles,
            merged_blocks=capped,
            context_text=context_text,
            blocks_by_angle=blocks_by_angle,
            new_claims_by_angle=new_by_angle,
            synthesis=synthesis,
            timings_ms=timings,
            stop_reason=stop_reason,
            attempts=attempts,
            objective=objective,
            assessment=assessment,
            adviser_calls=adviser_calls,
            adviser_failed=adviser_failed,
            synthesis_error=synthesis_error,
        )

    def _stop_reason(
        self,
        attempts: int,
        failures: int,
        no_gain_streak: int,
        has_candidates: bool,
    ) -> StopReason | None:
        if attempts >= self.max_angles:
            return StopReason.ANGLE_BUDGET_EXHAUSTED
        if failures >= self.max_failures:
            return StopReason.FAILURE_BUDGET_EXHAUSTED
        if no_gain_streak >= self.max_consecutive_no_gain:
            return StopReason.NO_NEW_EVIDENCE
        if not has_candidates:
            return StopReason.NO_CANDIDATE_GAPS
        return None

    def _seed_queue(self, query: str, planned: list[ResearchAngle]) -> list[ResearchAngle]:
        original = ResearchAngle("original", query, "The question as asked.")
        queue = [original]
        labels = {_normalise(original.label)}
        queries = {_normalise(original.query_text)}
        for angle in planned:
            accepted = self._validate_proposals(
                (angle,), query, labels, queries, self.max_angles - len(queue)
            )
            queue.extend(accepted)
            if len(queue) >= self.max_angles:
                break
        return queue

    def _validate_proposals(
        self,
        proposals: tuple[ResearchAngle, ...] | list[ResearchAngle],
        original_query: str,
        seen_labels: set[str],
        seen_queries: set[str],
        remaining: int,
    ) -> list[ResearchAngle]:
        accepted: list[ResearchAngle] = []
        cap = min(self.max_proposals_per_call, max(0, remaining))
        for angle in proposals:
            if len(accepted) >= cap or not isinstance(angle, ResearchAngle):
                break
            label = angle.label.strip()
            query_text = angle.query_text.strip()
            rationale = angle.rationale.strip()
            label_key = _normalise(label)
            query_key = _normalise(query_text)
            if (
                not label_key
                or not query_key
                or len(query_text) > _MAX_ANGLE_QUERY_CHARS
                or label_key == "original"
                or query_key == _normalise(original_query)
                or label_key in seen_labels
                or query_key in seen_queries
            ):
                continue
            clean = ResearchAngle(label, query_text, rationale)
            accepted.append(clean)
            seen_labels.add(label_key)
            seen_queries.add(query_key)
        return accepted

    @staticmethod
    def _merge_blocks(
        merged: dict[str, _MergedClaim],
        blocks: list["ProvenanceBlock"],
        label: str,
    ) -> int:
        added = 0
        for block in blocks:
            key = DeepResearchOrchestrator._dedup_key(block)
            existing = merged.get(key)
            if existing is None:
                merged[key] = _MergedClaim(block=block, surfaced_by=[label])
                added += 1
            else:
                if label not in existing.surfaced_by:
                    existing.surfaced_by.append(label)
                if _confidence(block) > _confidence(existing.block):
                    existing.block = block
        return added

    @staticmethod
    def _snapshot(
        objective: ResearchObjective,
        attempts: list[AngleAttempt],
        merged: dict[str, _MergedClaim],
        remaining: int,
    ) -> InvestigationSnapshot:
        ordered = sorted(
            merged.items(),
            key=lambda item: (-len(item[1].surfaced_by), -_confidence(item[1].block), item[0]),
        )
        evidence: list[EvidenceSummary] = []
        chars = 0
        for _, entry in ordered:
            block = entry.block
            sentence = str(getattr(block, "source_sentence", ""))
            if evidence and chars + len(sentence) > 20_000:
                break
            evidence.append(EvidenceSummary(
                claim_id=str(getattr(block, "claim_id", "")),
                claim_type=str(getattr(block, "claim_type", "")),
                doc_title=str(getattr(block, "doc_title", "")),
                doc_date_start=getattr(block, "doc_date_start", None),
                source_sentence=sentence,
            ))
            chars += len(sentence)
        return InvestigationSnapshot(
            objective=objective,
            attempts=tuple(attempts),
            evidence=tuple(evidence),
            remaining_attempts=max(0, remaining),
        )

    @staticmethod
    def _dedup_key(block: "ProvenanceBlock") -> str:
        claim_id = getattr(block, "claim_id", None)
        if claim_id:
            return f"claim:{claim_id}"
        identity = "\x1f".join((
            str(getattr(block, "doc_id", "")),
            str(getattr(block, "paragraph_id", "")),
            str(getattr(block, "source_sentence", "")),
        ))
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        return f"loc:{digest}"


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _confidence(block: "ProvenanceBlock") -> float:
    try:
        return float(getattr(block, "extraction_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_backend(
    *,
    gateway: Any,
    assembler: Any,
    engine: Any,
    classify_fn: Callable[[str], object],
    infer_claim_types_fn: Callable[[str], list[str]],
    entity_hints: list[str] | None = None,
    doc_types: list[str] | None = None,
    series: list[str] | None = None,
    synthesise_fn: Callable[..., "SynthesisResult"] | None = None,
) -> RetrievalBackend:
    """Bind the existing retrieval components into a request-local backend."""

    hints = list(entity_hints or [])

    class GemyndRetrievalBackend:
        def probe(self, query: str) -> QueryProbe:
            intent = classify_fn(query)
            context = gateway.resolve(
                surface_forms=getattr(intent, "entities", []) or [],
                entity_hints=hints,
                inferred_claim_types=infer_claim_types_fn(query),
            )
            names = [
                getattr(resolved, "surface_form", None)
                or getattr(resolved, "entity_id", "")
                for resolved in getattr(context, "resolved", [])
            ]
            return QueryProbe(
                entities=[name for name in names if name],
                year_min=getattr(intent, "year_min", None),
                year_max=getattr(intent, "year_max", None),
            )

        def gather(
            self,
            angle_query: str,
            scope: PermissionScope,
            *,
            allow_llm_cypher: bool = True,
        ) -> list["ProvenanceBlock"]:
            intent = classify_fn(angle_query)
            context = gateway.resolve(
                surface_forms=getattr(intent, "entities", []) or [],
                entity_hints=hints,
                inferred_claim_types=infer_claim_types_fn(angle_query),
            )
            assemble_kwargs: dict[str, Any] = dict(
                query_text=angle_query,
                entity_context=context,
                year_min=getattr(intent, "year_min", None),
                year_max=getattr(intent, "year_max", None),
                is_hybrid=getattr(intent, "bucket", "") == "hybrid",
                permitted_levels=scope.permitted_levels,
                institution_ids=scope.institution_ids,
                intent=intent,
                allow_llm_cypher=allow_llm_cypher,
            )
            if doc_types is not None:
                assemble_kwargs["doc_types"] = doc_types
            if series is not None:
                assemble_kwargs["series"] = series
            blocks, _ = assembler.assemble(**assemble_kwargs)
            return blocks

        def synthesise(
            self,
            query: str,
            blocks: list["ProvenanceBlock"],
            context_text: str,
            scope: PermissionScope,
            conversation_history: list[dict[str, str]] | None,
        ) -> "SynthesisResult":
            kwargs = dict(
                query=query,
                provenance_blocks=blocks,
                context_text=context_text,
                conversation_history=conversation_history,
                searched_institution_ids=list(scope.institution_ids or []),
            )
            if synthesise_fn is not None:
                return synthesise_fn(**kwargs)
            return engine.synthesise(**kwargs)

    return GemyndRetrievalBackend()
