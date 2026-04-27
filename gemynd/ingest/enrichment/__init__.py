"""Post-ingest enrichment passes.

These modules operate on an already-ingested graph and write additional nodes
and relationships without re-running the full pipeline.  They read from and
write to a live Neo4j instance via the driver passed by the caller.
"""
from __future__ import annotations
