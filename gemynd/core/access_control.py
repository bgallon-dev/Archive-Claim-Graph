"""Document- and claim-level access-control predicates.

The same logic is encoded as Cypher in ``core/graph/cypher.py`` (`_DOC_ACCESS`
and ``_CLAIM_ACTIVE``); this module is the Python mirror used by the
in-memory executor and any caller that needs to filter records before they
reach the graph.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def doc_is_valid(
    props: dict[str, Any],
    institution_ids: Iterable[str],
    permitted_levels: Iterable[str],
) -> bool:
    """Return True if the document satisfies all four access-control predicates.

    Mirrors the canonical Cypher block:
        d.institution_id IN $institution_ids
          AND d.access_level IN $permitted_levels
          AND d.deleted_at IS NULL
          AND (d.quarantine_status IS NULL OR d.quarantine_status = 'active')

    ``institution_ids`` is treated as "no filter" when empty (matches the
    in-memory executor's prior behavior; the Cypher templates require a
    non-empty list).
    """
    if props.get("deleted_at") is not None:
        return False
    if institution_ids and props.get("institution_id") not in institution_ids:
        return False
    if props.get("access_level") not in permitted_levels:
        return False
    qs = props.get("quarantine_status")
    if qs is not None and qs != "active":
        return False
    return True


def claim_is_active(props: dict[str, Any]) -> bool:
    """Return True if the claim is not quarantined.

    Mirrors the canonical Cypher predicate:
        (c.quarantine_status IS NULL OR c.quarantine_status = 'active')
    """
    qs = props.get("quarantine_status")
    return qs is None or qs == "active"
