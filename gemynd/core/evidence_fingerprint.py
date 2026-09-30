"""Deterministic SHA-256 fingerprints for provenance-bearing claims."""

from __future__ import annotations

import hashlib
import json


FINGERPRINT_VERSION = "gemynd-evidence-v1"


def evidence_fingerprint_sha256(
    *,
    claim_id: str,
    doc_id: str = "",
    paragraph_id: str = "",
    source_sentence: str = "",
) -> str:
    """Return the canonical SHA-256 value rendered as an entropy barcode."""
    canonical = json.dumps(
        {
            "claim_id": str(claim_id or ""),
            "doc_id": str(doc_id or ""),
            "paragraph_id": str(paragraph_id or ""),
            "source_sentence": str(source_sentence or ""),
            "version": FINGERPRINT_VERSION,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
