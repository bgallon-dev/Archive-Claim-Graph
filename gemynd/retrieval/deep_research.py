"""Deep-research mode — multi-pass retrieval over a single question.

Motivation
----------
A single retrieval pass covers only a *slice* of the useful evidence. Re-asking
the same question surfaces different claims that are not wrong, just missed the
first time (observed live on 2026-07-23). That variance is the signal: each pass
is one sampling of a larger evidence set, so the fix is to run several angles on
the question, union what they each surface, and synthesise over the superset.

Pipeline
--------
    probe(question)                  one cheap classify+resolve, to seed planning
      -> planner.plan(...)           question -> N research angles (sub-queries)
        -> gather(angle) for each    reuse the normal retrieval path per angle
          -> merge + dedup           union blocks by claim_id, track who found each
            -> synthesise(union)     one answer over the combined evidence

The merge step records, per angle, how many *new* claims it contributed
(`new_claims_by_angle`). That number quantifies the coverage gap the extra passes
close — it is the metric that justifies the mode's cost.

Design seams (see DEEP_RESEARCH.md)
-----------------------------------
* ``ResearchPlanner`` is a Protocol. ``HeuristicPlanner`` ships as the working
  default (deterministic, no extra LLM call). An LLM-decomposition planner is the
  intended upgrade — same interface, better angles.
* ``RetrievalBackend`` is the single seam onto the existing engine. The web app
  wires ``GemyndRetrievalBackend`` from the components it already builds; tests
  pass a fake. Nothing here imports the web app, so there is no cycle and no way
  for this module to disturb the live query path.
* A completeness-critic loop (gather more angles until coverage stops growing) is
  a documented extension point, not yet built.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from gemynd.retrieval.models import ProvenanceBlock, SynthesisResult

_log = logging.getLogger(__name__)

# How many angles a single research run may spend, including the original
# question. Kept small: each angle is a full retrieval pass, and marginal
# coverage falls off fast. Raise deliberately, not by default.
_DEFAULT_MAX_ANGLES = 4


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PermissionScope:
    """The access scope a research run executes under. Mirrors the fields the
    retrieval layer already enforces so deep research cannot widen access — it
    only re-runs the same permitted retrieval more times."""
    permitted_levels: list[str]
    institution_ids: list[str]


@dataclass(slots=True)
class QueryProbe:
    """Cheap read of a question used to seed planning: what entities resolved
    and what year bounds were detected. Deliberately does not expose internal
    intent/entity types, so planners stay decoupled from retrieval internals."""
    entities: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None


@dataclass(slots=True)
class ResearchAngle:
    """One sub-query the question is decomposed into."""
    label: str        # short, unique within a run, e.g. "original" / "entity:Turnbull"
    query_text: str   # the sub-question actually sent through retrieval
    rationale: str    # why this angle exists (surfaced in the trace, not to end users)


@dataclass(slots=True)
class _MergedClaim:
    block: "ProvenanceBlock"
    surfaced_by: list[str]  # angle labels that independently retrieved this claim


@dataclass(slots=True)
class DeepResearchResult:
    original_query: str
    angles: list[ResearchAngle]
    merged_blocks: list["ProvenanceBlock"]
    context_text: str
    # angle label -> blocks that angle retrieved (before dedup)
    blocks_by_angle: dict[str, int]
    # angle label -> claims this angle added that no earlier angle had (the payoff)
    new_claims_by_angle: dict[str, int]
    synthesis: "SynthesisResult | None"
    timings_ms: dict[str, float]

    @property
    def unique_claims(self) -> int:
        return len(self.merged_blocks)

    @property
    def coverage_gain(self) -> int:
        """Claims found only because we went past the first angle. This is the
        number to show when justifying deep-research mode over a single pass."""
        first = self.angles[0].label if self.angles else None
        return sum(n for label, n in self.new_claims_by_angle.items() if label != first)

    @property
    def estimated_context_tokens(self) -> int:
        """Rough token count of the synthesis context (~4 chars/token). A free,
        pre-call gauge of synthesis input cost — the real figure is logged by
        MeteredAnthropicClient to token_usage.db after the call. Use this to see
        whether COST CONTROL 2's budget is doing its job without paying for a
        count_tokens round-trip."""
        return len(self.context_text) // 4


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------
class RetrievalBackend(Protocol):
    """The only coupling to the existing retrieval engine. Implemented for real
    by ``GemyndRetrievalBackend``; implemented trivially by tests."""

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


class HeuristicPlanner:
    """Deterministic angle generator — no extra model call.

    Angles: the original question, one per resolved entity (so retrieval
    re-anchors on each), and one broadened "background/context" reformulation
    that biases toward different fulltext hits. Enough distinct assemble() calls
    to produce a real superset; the LLM planner is the quality upgrade.
    """

    def plan(self, query: str, probe: QueryProbe, max_angles: int) -> list[ResearchAngle]:
        angles: list[ResearchAngle] = [
            ResearchAngle("original", query, "The question as asked."),
        ]
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
        # A broadened pass tends to surface context claims the focused passes
        # skip. Placed last so it only runs if the angle budget allows.
        angles.append(ResearchAngle(
            "broadened",
            f"Background and surrounding context for: {query}",
            "Bias toward contextual/background claims a focused pass drops.",
        ))
        return angles[:max_angles]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class DeepResearchOrchestrator:
    def __init__(
        self,
        backend: RetrievalBackend,
        planner: ResearchPlanner | None = None,
        *,
        max_angles: int = _DEFAULT_MAX_ANGLES,
        char_budget_fn: Callable[[list["ProvenanceBlock"]], list["ProvenanceBlock"]] | None = None,
        serialise_fn: Callable[["ProvenanceBlock"], str] | None = None,
        llm_cypher_on_reformulations: bool = False,
    ) -> None:
        self.backend = backend
        self.planner = planner or HeuristicPlanner()
        self.max_angles = max(1, max_angles)
        # COST CONTROL 1 (retrieval side). The LLM-Cypher fallback fires when a
        # pass finds nothing rule-based — which vague reformulated angles do
        # often. Left False, only the original angle may spend an LLM-Cypher
        # call, so a 4-angle run costs at most one, not four. See DEEP_RESEARCH.md.
        self.llm_cypher_on_reformulations = llm_cypher_on_reformulations
        # COST CONTROL 2 (synthesis side). char_budget_fn caps the union before
        # synthesis. Because the union is ranked centrality-first (claims found
        # by multiple angles lead), a deliberately SMALLER budget than the
        # retrieval max keeps the most-corroborated evidence while cutting the
        # dominant cost — synthesis input tokens. Reuse _apply_context_char_budget
        # bound to a tighter char count for deep research.
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
        """Run a multi-angle research pass.

        ``synthesise=False`` runs retrieval + merge only and skips the LLM call —
        use it to measure coverage gain cheaply, or in tests.
        """
        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        probe = self.backend.probe(query)
        timings["probe_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        angles = self.planner.plan(query, probe, self.max_angles)
        _log.info("deep-research: %d angle(s) for %r", len(angles), query[:80])

        merged: dict[str, _MergedClaim] = {}
        blocks_by_angle: dict[str, int] = {}
        new_by_angle: dict[str, int] = {}

        t0 = time.perf_counter()
        for idx, angle in enumerate(angles):
            # Only the original angle (idx 0) may spend an LLM-Cypher fallback,
            # unless the operator opts every angle in. Caps the retrieval-side
            # LLM cost of a fan-out at one call.
            allow_llm = idx == 0 or self.llm_cypher_on_reformulations
            try:
                blocks = self.backend.gather(
                    angle.query_text, scope, allow_llm_cypher=allow_llm
                )
            except Exception:
                # One angle failing must not sink the whole run — the point of
                # multiple passes is resilience. Record zero and continue.
                _log.exception("deep-research: angle %r failed", angle.label)
                blocks_by_angle[angle.label] = 0
                new_by_angle[angle.label] = 0
                continue
            blocks_by_angle[angle.label] = len(blocks)
            added = 0
            for b in blocks:
                key = self._dedup_key(b)
                existing = merged.get(key)
                if existing is None:
                    merged[key] = _MergedClaim(block=b, surfaced_by=[angle.label])
                    added += 1
                else:
                    if angle.label not in existing.surfaced_by:
                        existing.surfaced_by.append(angle.label)
                    # Keep the highest-confidence instance of a claim seen across passes.
                    if _confidence(b) > _confidence(existing.block):
                        existing.block = b
            new_by_angle[angle.label] = added
        timings["gather_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Centrality-first ranking: a claim independently surfaced by several
        # angles is more corroborated, so it should survive a tight synthesis
        # budget over a claim only one angle found. Confidence breaks ties.
        # This is what lets COST CONTROL 2 use a smaller budget without dropping
        # the evidence that matters — the trimmed tail is single-angle, low-conf.
        ordered = sorted(
            merged.values(),
            key=lambda m: (len(m.surfaced_by), _confidence(m.block)),
            reverse=True,
        )
        capped = self._char_budget_fn([m.block for m in ordered])
        context_text = "\n\n".join(self._serialise_fn(b) for b in capped)

        synthesis = None
        if synthesise and capped:
            t0 = time.perf_counter()
            synthesis = self.backend.synthesise(
                query=query,
                blocks=capped,
                context_text=context_text,
                scope=scope,
                conversation_history=conversation_history,
            )
            timings["synthesise_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        timings["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
        return DeepResearchResult(
            original_query=query,
            angles=angles,
            merged_blocks=capped,
            context_text=context_text,
            blocks_by_angle=blocks_by_angle,
            new_claims_by_angle=new_by_angle,
            synthesis=synthesis,
            timings_ms=timings,
        )

    @staticmethod
    def _dedup_key(block: "ProvenanceBlock") -> str:
        # claim_id is the natural key; fall back to a content key for blocks
        # that lack one (defensive — the retrieval path always sets claim_id).
        cid = getattr(block, "claim_id", None)
        if cid:
            return f"claim:{cid}"
        return "loc:{}|{}|{}".format(
            getattr(block, "doc_id", ""),
            getattr(block, "paragraph_id", ""),
            hash(getattr(block, "source_sentence", "")),
        )


def _confidence(block: "ProvenanceBlock") -> float:
    try:
        return float(getattr(block, "extraction_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Real backend — wires the orchestrator onto the components the web app already
# builds. Kept here (not in app.py) so the web module is untouched; the endpoint
# that calls this is the documented next step, not part of this scaffold.
# ---------------------------------------------------------------------------
def build_backend(
    *,
    gateway,
    assembler,
    engine,
    classify_fn: Callable[[str], object],
    infer_claim_types_fn: Callable[[str], list[str]],
) -> RetrievalBackend:
    """Construct a ``RetrievalBackend`` from live retrieval components.

    ``classify_fn`` is ``classify_query`` pre-bound with the app's stopwords,
    year bounds and seed vocab. ``infer_claim_types_fn`` is the app's
    ``_infer_claim_types`` pre-bound with the composite intent map. Binding them
    at the call site keeps this module free of app/config coupling.
    """

    class GemyndRetrievalBackend:
        def probe(self, query: str) -> QueryProbe:
            intent = classify_fn(query)
            ectx = gateway.resolve(
                surface_forms=getattr(intent, "entities", []) or [],
                entity_hints=[],
                inferred_claim_types=infer_claim_types_fn(query),
            )
            names = [
                getattr(r, "surface_form", None) or getattr(r, "entity_id", "")
                for r in getattr(ectx, "resolved", [])
            ]
            return QueryProbe(
                entities=[n for n in names if n],
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
            ectx = gateway.resolve(
                surface_forms=getattr(intent, "entities", []) or [],
                entity_hints=[],
                inferred_claim_types=infer_claim_types_fn(angle_query),
            )
            blocks, _context = assembler.assemble(
                query_text=angle_query,
                entity_context=ectx,
                year_min=getattr(intent, "year_min", None),
                year_max=getattr(intent, "year_max", None),
                is_hybrid=getattr(intent, "bucket", "") == "hybrid",
                permitted_levels=scope.permitted_levels,
                institution_ids=scope.institution_ids,
                intent=intent,
                allow_llm_cypher=allow_llm_cypher,
            )
            return blocks

        def synthesise(
            self,
            query: str,
            blocks: list["ProvenanceBlock"],
            context_text: str,
            scope: PermissionScope,
            conversation_history: list[dict[str, str]] | None,
        ) -> "SynthesisResult":
            return engine.synthesise(
                query=query,
                provenance_blocks=blocks,
                context_text=context_text,
                conversation_history=conversation_history,
                searched_institution_ids=list(scope.institution_ids or []),
            )

    return GemyndRetrievalBackend()
