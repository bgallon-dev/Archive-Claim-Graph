"""Graph write backends for the ingest layer.

Re-exports the GraphWriter Protocol plus the two writer implementations.
``build_driver_kwargs`` and ``format_connection_error`` are publicly
exported because ``retrieval/executor.py`` reuses them when wiring its
own Neo4j driver. Previously they lived as underscore-prefixed helpers
on the monolithic ``writer.py`` and were imported from outside the
module — promoting them surfaces the intent properly.
"""
from __future__ import annotations

from .in_memory import InMemoryGraphWriter
from .neo4j_writer import (
    Neo4jGraphWriter,
    build_driver_kwargs,
    format_connection_error,
)
from .protocol import GraphWriter

__all__ = [
    "GraphWriter",
    "InMemoryGraphWriter",
    "Neo4jGraphWriter",
    "build_driver_kwargs",
    "format_connection_error",
]
