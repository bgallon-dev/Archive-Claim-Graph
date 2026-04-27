from __future__ import annotations

from datetime import datetime
from typing import Any

from gemynd.core.ids import make_claim_id, make_entity_id, make_mention_id
from gemynd.core.models import (
    ClaimEntityLinkRecord,
    ClaimLocationLinkRecord,
    ClaimRecord,
    EntityRecord,
    EntityResolutionRecord,
    MentionRecord,
    SemanticBundle,
    StructureBundle,
)

from .phases import build_extraction_run


_PARCEL_PROPERTY_KEYS = ("township", "range", "section_number", "aliquot", "county", "meridian", "state")


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _grantee_entity(grantee: dict[str, Any]) -> EntityRecord:
    display = str(grantee.get("person_display") or grantee.get("name_raw") or "").strip()
    normalized = str(grantee.get("person_norm") or _normalize(display))
    entity_id = grantee.get("person_entity_id") or make_entity_id("Person", normalized)
    return EntityRecord(
        entity_id=entity_id,
        entity_type="Person",
        name=display or normalized,
        normalized_form=normalized,
        properties={"source": "json_metadata", "role": "grantee"},
    )


def _parcel_entity(parcel: dict[str, Any], metadata: dict[str, Any]) -> EntityRecord:
    name = str(parcel.get("name") or "").strip()
    normalized = _normalize(name)
    entity_id = parcel.get("entity_id") or make_entity_id("Parcel", normalized)
    properties: dict[str, Any] = {"source": "json_metadata"}
    for key in _PARCEL_PROPERTY_KEYS:
        value = parcel.get(key)
        if value is None:
            value = metadata.get(key)
        if value is not None:
            properties[key] = value
    return EntityRecord(
        entity_id=entity_id,
        entity_type="Parcel",
        name=name or normalized,
        normalized_form=normalized,
        properties=properties,
    )


def _mention(
    run_id: str,
    paragraph_id: str,
    surface: str,
    normalized: str,
    mention_id: str | None,
    ocr_suspect: bool = False,
) -> MentionRecord:
    surface = surface or normalized
    mid = mention_id or make_mention_id(run_id, paragraph_id, 0, len(surface), surface)
    return MentionRecord(
        mention_id=mid,
        run_id=run_id,
        paragraph_id=paragraph_id,
        surface_form=surface,
        normalized_form=normalized,
        start_offset=0,
        end_offset=len(surface),
        detection_confidence=1.0,
        ocr_suspect=ocr_suspect,
    )


def build_land_patent_semantic(
    structure: StructureBundle,
    run_overrides: dict[str, str] | None = None,
    now: datetime | None = None,
) -> SemanticBundle:
    """Translate structured land-patent metadata directly into a SemanticBundle.

    Bypasses rule/LLM extractors — the JSON already carries grantees and
    parcels as resolved entities, so there is nothing to discover. Each
    (grantee × parcel) pair becomes one synthetic ``land_grant`` claim with
    deterministic IDs, full entity resolution links, and claim→entity /
    claim→location links ready for the graph writer.
    """
    metadata = dict(structure.source_metadata or {})
    extraction_run = build_extraction_run(
        {**(run_overrides or {}), "ner_model": "json_metadata", "relation_model": "json_metadata"},
        now=now,
    )
    run_id = extraction_run.run_id

    paragraph_id = structure.paragraphs[0].paragraph_id if structure.paragraphs else f"{structure.document.doc_id}:paragraph:00001:p0001"

    date_start = metadata.get("date_start") or structure.document.date_start
    grantees = metadata.get("grantees") or []
    parcels = metadata.get("parcels") or []

    entities: list[EntityRecord] = []
    mentions: list[MentionRecord] = []
    resolutions: list[EntityResolutionRecord] = []
    seen_entity_ids: set[str] = set()

    grantee_bindings: list[tuple[EntityRecord, MentionRecord]] = []
    for grantee in grantees:
        entity = _grantee_entity(grantee)
        mention = _mention(
            run_id=run_id,
            paragraph_id=paragraph_id,
            surface=str(grantee.get("name_raw") or grantee.get("person_display") or ""),
            normalized=entity.normalized_form,
            mention_id=grantee.get("mention_id"),
        )
        if entity.entity_id not in seen_entity_ids:
            entities.append(entity)
            seen_entity_ids.add(entity.entity_id)
        mentions.append(mention)
        resolutions.append(EntityResolutionRecord(
            mention_id=mention.mention_id,
            entity_id=entity.entity_id,
            relation_type="RESOLVED_TO",
            match_score=1.0,
            confirmed_by="json_metadata",
        ))
        grantee_bindings.append((entity, mention))

    parcel_bindings: list[tuple[EntityRecord, MentionRecord]] = []
    for parcel in parcels:
        entity = _parcel_entity(parcel, metadata)
        mention = _mention(
            run_id=run_id,
            paragraph_id=paragraph_id,
            surface=entity.name,
            normalized=entity.normalized_form,
            mention_id=parcel.get("mention_id"),
        )
        if entity.entity_id not in seen_entity_ids:
            entities.append(entity)
            seen_entity_ids.add(entity.entity_id)
        mentions.append(mention)
        resolutions.append(EntityResolutionRecord(
            mention_id=mention.mention_id,
            entity_id=entity.entity_id,
            relation_type="RESOLVED_TO",
            match_score=1.0,
            confirmed_by="json_metadata",
        ))
        parcel_bindings.append((entity, mention))

    claims: list[ClaimRecord] = []
    claim_entity_links: list[ClaimEntityLinkRecord] = []
    claim_location_links: list[ClaimLocationLinkRecord] = []

    claim_index = 0
    for grantee_entity, _ in grantee_bindings:
        for parcel_entity, _ in parcel_bindings:
            claim_index += 1
            sentence = (
                f"{grantee_entity.name} was granted parcel {parcel_entity.name}"
                + (f" on {date_start}." if date_start else ".")
            )
            normalized_sentence = _normalize(sentence)
            claim_id = make_claim_id(run_id, paragraph_id, claim_index, normalized_sentence)
            claims.append(ClaimRecord(
                claim_id=claim_id,
                run_id=run_id,
                paragraph_id=paragraph_id,
                claim_type="land_grant",
                source_sentence=sentence,
                normalized_sentence=normalized_sentence,
                certainty="certain",
                extraction_confidence=1.0,
                claim_date=date_start,
                notes="source:json_metadata",
            ))
            claim_entity_links.append(ClaimEntityLinkRecord(
                claim_id=claim_id,
                entity_id=grantee_entity.entity_id,
                relation_type="SUBJECT_OF_CLAIM",
            ))
            claim_location_links.append(ClaimLocationLinkRecord(
                claim_id=claim_id,
                entity_id=parcel_entity.entity_id,
                relation_type="OCCURRED_AT",
            ))

    return SemanticBundle(
        extraction_run=extraction_run,
        claims=claims,
        measurements=[],
        mentions=mentions,
        entities=entities,
        entity_resolutions=resolutions,
        claim_entity_links=claim_entity_links,
        claim_link_diagnostics=[],
        claim_location_links=claim_location_links,
        claim_period_links=[],
        document_anchor_links=[],
        document_period_links=[],
        document_signed_by_links=[],
        person_affiliation_links=[],
    )
