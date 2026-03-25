"""Claim-centric narrative report graph pipeline."""

from graphrag_pipeline.ingest import IngestPipeline
from graphrag_pipeline.retrieval import RetrievalService
from graphrag_pipeline.review import ReviewService

# Legacy function-level API — kept for backward compatibility
from graphrag_pipeline.ingest.pipeline import (
    build_spelling_review_queue,
    extract_semantic,
    parse_source,
    quality_report,
    run_e2e,
)

__all__ = [
    "IngestPipeline",
    "RetrievalService",
    "ReviewService",
    "parse_source",
    "extract_semantic",
    "build_spelling_review_queue",
    "quality_report",
    "run_e2e",
]
