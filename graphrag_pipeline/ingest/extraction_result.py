"""Typed result dataclasses for the extract/persist pipeline split."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from graphrag_pipeline.core.models import SemanticBundle, StructureBundle


@dataclass(slots=True)
class ExtractionResult:
    """Pure output of document extraction (parse + semantic).  Pickle-safe."""

    structure: StructureBundle
    semantic: SemanticBundle
    doc_id: str
    run_id: str
    input_path: str


@dataclass(slots=True)
class QuarantineSummary:
    """Describes quarantine actions taken by a sensitivity gate."""

    total_claims: int
    quarantined_count: int
    quarantined_ids: list[str]
    gate_error: bool = False
    gate_error_message: str = ""


@dataclass(slots=True)
class GateResult:
    """Return value of a :class:`SensitivityGate` invocation."""

    quarantine_summary: QuarantineSummary
    semantic: SemanticBundle  # with quarantine fields mutated in-place


@dataclass(slots=True)
class PersistResult:
    """Output of :func:`persist_document` — all side-effect results."""

    input_path: str
    doc_id: str
    run_id: str
    structure_output: str
    semantic_output: str
    quality: dict[str, Any]
    quarantine_summary: QuarantineSummary
    spelling_review_output: str | None = None
    spelling_review_issue_count: int = 0

    def to_summary_dict(self) -> dict[str, Any]:
        """Backward-compatible dict matching the old ``_process_single_document`` return."""
        d: dict[str, Any] = {
            "input": self.input_path,
            "doc_id": self.doc_id,
            "run_id": self.run_id,
            "structure_output": self.structure_output,
            "semantic_output": self.semantic_output,
            "quality": self.quality,
        }
        if self.spelling_review_output is not None:
            d["spelling_review_output"] = self.spelling_review_output
            d["spelling_review_issue_count"] = self.spelling_review_issue_count
        return d


@runtime_checkable
class SensitivityGate(Protocol):
    """Callable that screens extracted claims and quarantines sensitive ones."""

    def __call__(
        self,
        structure: StructureBundle,
        semantic: SemanticBundle,
    ) -> GateResult: ...
