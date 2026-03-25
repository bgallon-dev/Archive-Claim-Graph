"""Ingest sub-package: document parsing, semantic extraction, graph loading, and export."""
from __future__ import annotations

from .pipeline import (
    parse_source,
    extract_semantic,
    load_graph,
    quality_report,
    build_spelling_review_queue,
    run_e2e,
    resolve_mentions_targeted,
)

__all__ = [
    "IngestPipeline",
    "parse_source",
    "extract_semantic",
    "load_graph",
    "quality_report",
    "build_spelling_review_queue",
    "run_e2e",
    "resolve_mentions_targeted",
]


class IngestPipeline:
    """Facade that wires together the ingest pipeline steps.

    Typical usage::

        pipeline = IngestPipeline()
        structure = pipeline.parse_source(path)
        semantic  = pipeline.extract_semantic(structure)
        pipeline.load_graph(structure, semantic, graph_writer)
    """

    # ------------------------------------------------------------------
    # Source parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_source(path, **kwargs):
        """Parse a source JSON file into a StructureBundle.

        Delegates to :func:`graphrag_pipeline.ingest.pipeline.parse_source`.
        """
        return parse_source(path, **kwargs)

    # ------------------------------------------------------------------
    # Semantic extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_semantic(structure, **kwargs):
        """Extract claims, mentions, and measurements from a StructureBundle.

        Delegates to :func:`graphrag_pipeline.ingest.pipeline.extract_semantic`.
        """
        return extract_semantic(structure, **kwargs)

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_graph(structure, semantic, writer, **kwargs):
        """Write a (StructureBundle, SemanticBundle) pair to a graph backend.

        Delegates to :func:`graphrag_pipeline.ingest.pipeline.load_graph`.
        """
        return load_graph(structure, semantic, writer, **kwargs)

    # ------------------------------------------------------------------
    # Quality reporting
    # ------------------------------------------------------------------

    @staticmethod
    def quality_report(structure, semantic, **kwargs):
        """Generate a quality report dict for a processed document pair.

        Delegates to :func:`graphrag_pipeline.ingest.pipeline.quality_report`.
        """
        return quality_report(structure, semantic, **kwargs)

    # ------------------------------------------------------------------
    # Spelling review
    # ------------------------------------------------------------------

    @staticmethod
    def build_spelling_review_queue(structure, semantic, **kwargs):
        """Build the spelling-review queue for a processed document pair.

        Delegates to
        :func:`graphrag_pipeline.ingest.pipeline.build_spelling_review_queue`.
        """
        return build_spelling_review_queue(structure, semantic, **kwargs)

    # ------------------------------------------------------------------
    # End-to-end run
    # ------------------------------------------------------------------

    @staticmethod
    def run_e2e(*args, **kwargs):
        """Run the full end-to-end ingest pipeline over a directory of files.

        Delegates to :func:`graphrag_pipeline.ingest.pipeline.run_e2e`.
        """
        return run_e2e(*args, **kwargs)

    # ------------------------------------------------------------------
    # Targeted mention resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_mentions_targeted(*args, **kwargs):
        """Re-resolve entity mentions for a single document against a target set.

        Delegates to
        :func:`graphrag_pipeline.ingest.pipeline.resolve_mentions_targeted`.
        """
        return resolve_mentions_targeted(*args, **kwargs)
