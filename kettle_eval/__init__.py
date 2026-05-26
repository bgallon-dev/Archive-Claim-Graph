"""Active evaluation queue. ProEval-style failure discovery over the gemynd pipeline.

Wraps every extracted claim, mention, entity resolution, page, and claim-entity
link as an EvaluationCase. Scores each with priority = 0.40*predicted_failure
+ 0.25*uncertainty + 0.25*impact + 0.10*topic_novelty, then persists ranked
cases to its own SQLite store.

Top-level package, sibling of gemynd. Imports gemynd.core (data models) and
gemynd.review.detectors (read-only contributor adapters). Never imports from
gemynd.ingest.
"""
from __future__ import annotations

from .cases import cases_from_bundle
from .models import FAILURE_MODES, EvaluationCase, ReviewLabel
from .risk_features import compute_features
from .selector import (
    learn_from_label,
    load_weights,
    score_cases,
    top_k,
    top_k_balanced,
    top_k_per_decision_class,
)
from .store import KettleEvalStore

__version__ = "0.3.0"

__all__ = [
    "EvaluationCase",
    "FAILURE_MODES",
    "KettleEvalStore",
    "ReviewLabel",
    "__version__",
    "cases_from_bundle",
    "compute_features",
    "learn_from_label",
    "load_weights",
    "score_cases",
    "top_k",
    "top_k_balanced",
    "top_k_per_decision_class",
]
