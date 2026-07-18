"""Claim-centric narrative report graph pipeline."""

from typing import TYPE_CHECKING

from gemynd.ingest import IngestPipeline

# Legacy function-level API — kept for backward compatibility
from gemynd.ingest.pipeline import (
    build_spelling_review_queue,
    extract_semantic,
    parse_source,
    quality_report,
    run_e2e,
)

# RetrievalService / ReviewService are imported LAZILY (PEP 562): the retrieval
# stack pulls torch (KGR reranker), which the ingest path never needs. Eagerly
# importing them here forced every `import gemynd.*` — including ingest worker
# processes — to load torch (~2GB/proc), which OOM'd memory-constrained hosts.
# They remain accessible as `gemynd.RetrievalService` / `gemynd.ReviewService`.
if TYPE_CHECKING:  # keep names resolvable for type checkers / IDEs
    from gemynd.retrieval import RetrievalService
    from gemynd.review import ReviewService

_LAZY = {
    "RetrievalService": ("gemynd.retrieval", "RetrievalService"),
    "ReviewService": ("gemynd.review", "ReviewService"),
}


def __getattr__(name: str):  # PEP 562 module-level lazy attribute access
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)


def __dir__():
    return sorted([*globals(), *_LAZY])

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
