"""Semantic-aware CSV export from pipeline output bundles.

Produces three CSV files from all *.semantic.json files in a directory:
  claims.csv        — one row per ClaimRecord
  entities.csv      — one row per unique EntityRecord (deduplicated by entity_id)
  relationships.csv — one row per ClaimEntityLinkRecord / ClaimLocationLinkRecord
"""
from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def export_semantic_csv(
    bundles_dir: Path,
    output_dir: Path,
    *,
    glob_pattern: str = "*.semantic.json",
) -> dict[str, int]:
    """Export claims, entities, and relationships to CSV from bundle files.

    Loads all files matching *glob_pattern* in *bundles_dir* and their
    sibling *.structure.json* files for document metadata.

    Returns row counts: ``{"claims": N, "entities": N, "relationships": N}``.
    """
    from graphrag_pipeline.io_utils import load_semantic_bundle, load_structure_bundle, save_rows_csv

    output_dir.mkdir(parents=True, exist_ok=True)

    claims_rows: list[dict] = []
    entities_seen: dict[str, dict] = {}   # entity_id → row (last writer wins)
    relationships_rows: list[dict] = []

    semantic_paths = sorted(bundles_dir.glob(glob_pattern))
    if not semantic_paths:
        _log.warning("export_semantic_csv: no files matching %r in %s", glob_pattern, bundles_dir)
        for name in ("claims.csv", "entities.csv", "relationships.csv"):
            save_rows_csv(output_dir / name, [])
        return {"claims": 0, "entities": 0, "relationships": 0}

    for sem_path in semantic_paths:
        try:
            sem = load_semantic_bundle(sem_path)
        except Exception as exc:
            _log.warning("Skipping %s: %s", sem_path.name, exc)
            continue

        # Load sibling structure bundle for document-level metadata.
        struct_path = sem_path.with_suffix("").with_suffix(".structure.json")
        doc_id = ""
        doc_title = ""
        report_year = ""
        if struct_path.exists():
            try:
                struct = load_structure_bundle(struct_path)
                doc_id = struct.document.doc_id
                doc_title = struct.document.title
                report_year = str(struct.document.report_year or "")
            except Exception as exc:
                _log.warning("Could not load structure for %s: %s", sem_path.name, exc)

        # Claims
        for claim in sem.claims:
            claims_rows.append({
                "claim_id": claim.claim_id,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "report_year": report_year,
                "claim_type": claim.claim_type,
                "source_sentence": claim.source_sentence,
                "certainty": claim.epistemic_status,
                "extraction_confidence": claim.extraction_confidence,
                "review_status": claim.review_status,
                "quarantine_status": claim.quarantine_status or "",
            })

        # Entities (deduplicated across bundles)
        for entity in sem.entities:
            if entity.entity_id not in entities_seen:
                entities_seen[entity.entity_id] = {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "name": entity.name,
                    "normalized_form": entity.normalized_form,
                }

        # Relationships: claim-entity and claim-location links
        for link in sem.claim_entity_links:
            relationships_rows.append({
                "claim_id": link.claim_id,
                "entity_id": link.entity_id,
                "relation_type": link.relation_type,
            })
        for link in sem.claim_location_links:
            relationships_rows.append({
                "claim_id": link.claim_id,
                "entity_id": link.entity_id,
                "relation_type": link.relation_type,
            })

    save_rows_csv(output_dir / "claims.csv", claims_rows)
    save_rows_csv(output_dir / "entities.csv", list(entities_seen.values()))
    save_rows_csv(output_dir / "relationships.csv", relationships_rows)

    counts = {
        "claims": len(claims_rows),
        "entities": len(entities_seen),
        "relationships": len(relationships_rows),
    }
    _log.info("export_semantic_csv: %s", counts)
    return counts
