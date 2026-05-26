"""Stable IDs for EvaluationCase records.

`case_id`        — this exact pipeline-run output. Hashes case_type + object_id
                   + extraction_run_id, so re-running the pipeline produces a
                   distinct case row per run.
`case_family_id` — the same evaluable object across runs. Hashes case_type +
                   document_id + source_span_id + object_id, deliberately
                   excluding the run identifier. Lets us answer "this claim
                   was risky in run A, fixed in run B" without an event log.
"""
from __future__ import annotations

from gemynd.core.ids import stable_hash


def make_case_id(case_type: str, object_id: str, extraction_run_id: str) -> str:
    return f"case_{stable_hash(case_type, object_id, extraction_run_id)}"


def make_case_family_id(
    case_type: str,
    document_id: str,
    source_span_id: str | None,
    object_id: str,
) -> str:
    return f"fam_{stable_hash(case_type, document_id, source_span_id or '', object_id)}"


def make_resolution_object_id(
    mention_id: str,
    entity_id: str,
    relation_type: str,
    extraction_run_id: str,
) -> str:
    """Decision-level identity for an entity-resolution case.

    The evaluated object is the resolution *decision*, not the entity. Two
    different mentions resolving to the same entity must produce two cases.
    """
    return f"resdec_{stable_hash(mention_id, entity_id, relation_type, extraction_run_id)}"


def make_relation_object_id(claim_id: str, entity_id: str, relation_type: str) -> str:
    return f"rel_{stable_hash(claim_id, entity_id, relation_type)}"


def make_decision_class_id(case_type: str, *signature_parts: object) -> str:
    """Stable identifier for a *class* of equivalent decisions.

    Two cases that represent the same underlying decision should share a
    `decision_class_id`, even when they live in different documents and
    different runs. Examples:

      - entity_resolution: ("entity_resolution", normalized_surface_form,
        entity_id, relation_type) — "Washington Water Power -> Washington State
        as POSSIBLY_REFERS_TO" is one class regardless of which article it
        appeared in.
      - claim_extraction: ("claim_extraction", normalized_sentence, claim_type)
        — boilerplate claims that repeat verbatim share a class.
      - mention: ("mention", normalized_form) — every extraction of "Spokane"
        belongs to one class.
      - relation: ("relation", normalized_sentence, entity_id, relation_type)
      - page_ocr: ("page_ocr", page_id) — pages are inherently unique.

    The class id is the dedup key for `top_k_per_decision_class()`. Individual
    case_ids remain per-run and unique; the class id only collapses identical
    decisions for *queue display*, not for storage or audit.
    """
    return f"dcls_{stable_hash(case_type, *signature_parts)}"
