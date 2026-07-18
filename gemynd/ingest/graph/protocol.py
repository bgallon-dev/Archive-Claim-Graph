from __future__ import annotations

from typing import Protocol

from gemynd.core.models import SemanticBundle, StructureBundle


class GraphWriter(Protocol):
    def create_schema(self) -> None:
        ...

    def load_structure(self, structure: StructureBundle) -> None:
        ...

    def load_semantic(self, structure: StructureBundle, semantic: SemanticBundle) -> None:
        ...
