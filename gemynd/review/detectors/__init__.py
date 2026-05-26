"""Detector families for the anti-pattern review layer."""
from __future__ import annotations

from types import ModuleType
from typing import Protocol

from gemynd.core.models import SemanticBundle, StructureBundle

from ..models import DetectorProposal
from . import builder_repair, junk_mention, ocr_entity, sensitivity_monitor


class Detector(Protocol):
    def __call__(
        self,
        structure: StructureBundle,
        semantic: SemanticBundle,
        snapshot_id: str,
    ) -> list[DetectorProposal]: ...


# Detector modules in canonical execution order. Iteration uses attribute lookup
# (module.detect) so tests that monkeypatch `<module>.detect` see overrides.
DETECTOR_MODULES: tuple[ModuleType, ...] = (
    ocr_entity,
    junk_mention,
    builder_repair,
    sensitivity_monitor,
)
