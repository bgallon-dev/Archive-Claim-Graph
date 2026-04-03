"""Tests for the extract_document / persist_document split."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest

from graphrag_pipeline.core.models import SemanticBundle, StructureBundle
from graphrag_pipeline.ingest.extraction_result import (
    ExtractionResult,
    GateResult,
    PersistResult,
    QuarantineSummary,
    SensitivityGate,
)
from graphrag_pipeline.ingest.pipeline import extract_document, persist_document
from graphrag_pipeline.ingest.sensitivity_gate import (
    DefaultSensitivityGate,
    NullSensitivityGate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _QuarantineOneGate:
    """Test gate that quarantines the first claim it finds."""

    def __call__(
        self, structure: StructureBundle, semantic: SemanticBundle
    ) -> GateResult:
        quarantined_ids: list[str] = []
        if semantic.claims:
            first = semantic.claims[0]
            first.quarantine_status = "quarantined"
            first.quarantine_reason = "test_gate"
            quarantined_ids.append(first.claim_id)
        return GateResult(
            quarantine_summary=QuarantineSummary(
                total_claims=len(semantic.claims),
                quarantined_count=len(quarantined_ids),
                quarantined_ids=quarantined_ids,
            ),
            semantic=semantic,
        )


class _ExplodingGate:
    """Test gate that always raises."""

    def __call__(
        self, structure: StructureBundle, semantic: SemanticBundle
    ) -> GateResult:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# extract_document
# ---------------------------------------------------------------------------

def test_extract_document_returns_typed_result(fixtures_dir: Path) -> None:
    result = extract_document(str(fixtures_dir / "report1.json"))
    assert isinstance(result, ExtractionResult)
    assert result.doc_id == result.structure.document.doc_id
    assert result.run_id == result.semantic.extraction_run.run_id
    assert result.input_path == str(fixtures_dir / "report1.json")
    assert result.semantic.claims  # non-empty


def test_extraction_result_is_picklable(fixtures_dir: Path) -> None:
    result = extract_document(str(fixtures_dir / "report1.json"))
    roundtripped = pickle.loads(pickle.dumps(result))
    assert roundtripped.doc_id == result.doc_id
    assert len(roundtripped.semantic.claims) == len(result.semantic.claims)


# ---------------------------------------------------------------------------
# persist_document
# ---------------------------------------------------------------------------

def test_persist_document_writes_files(fixtures_dir: Path, tmp_path: Path) -> None:
    extraction = extract_document(str(fixtures_dir / "report1.json"))
    persist = persist_document(extraction, str(tmp_path))
    assert isinstance(persist, PersistResult)
    assert Path(persist.structure_output).exists()
    assert Path(persist.semantic_output).exists()
    assert persist.quality  # non-empty dict
    assert persist.doc_id == extraction.doc_id


def test_persist_document_runs_custom_gate(fixtures_dir: Path, tmp_path: Path) -> None:
    extraction = extract_document(str(fixtures_dir / "report1.json"))
    persist = persist_document(
        extraction, str(tmp_path), sensitivity_gate=_QuarantineOneGate()
    )
    assert persist.quarantine_summary.quarantined_count == 1
    assert persist.quarantine_summary.quarantined_ids
    assert not persist.quarantine_summary.gate_error


def test_null_sensitivity_gate(fixtures_dir: Path, tmp_path: Path) -> None:
    extraction = extract_document(str(fixtures_dir / "report1.json"))
    persist = persist_document(
        extraction, str(tmp_path), sensitivity_gate=NullSensitivityGate()
    )
    assert persist.quarantine_summary.quarantined_count == 0
    assert not persist.quarantine_summary.gate_error


def test_persist_document_spelling_review(fixtures_dir: Path, tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    extraction = extract_document(str(fixtures_dir / "report1.json"))
    persist = persist_document(
        extraction,
        str(tmp_path),
        review_out_dir=str(review_dir),
        sensitivity_gate=NullSensitivityGate(),
    )
    if persist.spelling_review_output is not None:
        assert Path(persist.spelling_review_output).exists()


# ---------------------------------------------------------------------------
# PersistResult.to_summary_dict backward compatibility
# ---------------------------------------------------------------------------

def test_persist_result_to_summary_dict(fixtures_dir: Path, tmp_path: Path) -> None:
    extraction = extract_document(str(fixtures_dir / "report1.json"))
    persist = persist_document(
        extraction, str(tmp_path), sensitivity_gate=NullSensitivityGate()
    )
    d = persist.to_summary_dict()
    assert d["input"] == str(fixtures_dir / "report1.json")
    assert d["doc_id"] == extraction.doc_id
    assert d["run_id"] == extraction.run_id
    assert "structure_output" in d
    assert "semantic_output" in d
    assert "quality" in d


# ---------------------------------------------------------------------------
# SensitivityGate protocol conformance
# ---------------------------------------------------------------------------

def test_gates_satisfy_protocol() -> None:
    assert isinstance(DefaultSensitivityGate(), SensitivityGate)
    assert isinstance(NullSensitivityGate(), SensitivityGate)
    assert isinstance(_QuarantineOneGate(), SensitivityGate)
