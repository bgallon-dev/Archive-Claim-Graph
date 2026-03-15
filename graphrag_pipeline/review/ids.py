"""Deterministic ID generation for the review subsystem."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ..ids import stable_hash


SNAPSHOT_SCHEMA_VERSION = "v1"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def make_snapshot_id(
    doc_id: str,
    structure_bundle_sha256: str,
    semantic_bundle_sha256: str,
    snapshot_schema_version: str = SNAPSHOT_SCHEMA_VERSION,
    extraction_run_id: str = "",
) -> str:
    return stable_hash(
        doc_id,
        structure_bundle_sha256,
        semantic_bundle_sha256,
        snapshot_schema_version,
        extraction_run_id,
        size=32,
    )


def make_review_run_id(snapshot_id: str, created_at: str) -> str:
    return f"rr_{stable_hash(snapshot_id, created_at, size=16)}"


def make_link_target_id(kind: str, *parts: str) -> str:
    return f"link::{kind}::{'::'.join(parts)}"


def make_claim_entity_link_target_id(
    claim_id: str, relation_type: str, entity_id: str,
) -> str:
    return make_link_target_id("claim_entity", claim_id, relation_type, entity_id)


def make_claim_location_link_target_id(
    claim_id: str, entity_id: str,
) -> str:
    return make_link_target_id("claim_location", claim_id, "OCCURRED_AT", entity_id)


def _canonicalize_patch_spec(patch_spec: dict[str, Any]) -> str:
    """Produce a canonical JSON string for fingerprinting.

    Sorts all keys recursively and sorts list values that are lists of strings.
    Excludes reviewer-only metadata fields.
    """
    def _sort_value(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _sort_value(v2) for k, v2 in sorted(v.items())}
        if isinstance(v, list):
            sorted_items = [_sort_value(item) for item in v]
            if all(isinstance(item, str) for item in sorted_items):
                return sorted(sorted_items)
            return sorted_items
        return v

    canonical = _sort_value(patch_spec)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def patch_spec_fingerprint(patch_spec: dict[str, Any]) -> str:
    canonical = _canonicalize_patch_spec(patch_spec)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _target_fingerprint(target_kind: str, target_id: str, target_role: str) -> str:
    return f"{target_kind}|{target_id}|{target_role}"


def make_proposal_id(
    snapshot_id: str,
    issue_class: str,
    proposal_type: str,
    targets: list[tuple[str, str, str]],
    patch_spec: dict[str, Any],
) -> str:
    sorted_fingerprints = sorted(
        _target_fingerprint(kind, tid, role) for kind, tid, role in targets
    )
    ps_fp = patch_spec_fingerprint(patch_spec)
    return stable_hash(
        snapshot_id,
        issue_class,
        proposal_type,
        "|".join(sorted_fingerprints),
        ps_fp,
        size=32,
    )


def make_revision_id(proposal_id: str, revision_number: int) -> str:
    return f"rev_{stable_hash(proposal_id, str(revision_number), size=16)}"


def make_correction_event_id(proposal_id: str, action: str, created_at: str) -> str:
    return f"ce_{stable_hash(proposal_id, action, created_at, size=16)}"
