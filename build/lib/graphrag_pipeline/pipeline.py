from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extractors import HybridClaimExtractor, RuleBasedMeasurementExtractor, RuleBasedMentionExtractor
from .extractors.claim_extractor import ClaimExtractor
from .extractors.measurement_extractor import MeasurementExtractor
from .extractors.mention_extractor import MentionExtractor
from .graph.writer import GraphWriter, InMemoryGraphWriter, Neo4jGraphWriter
from .ids import (
    make_claim_id,
    make_entity_id,
    make_measurement_id,
    make_mention_id,
    make_period_id,
    make_run_id,
)
from .io_utils import load_json, load_semantic_bundle, load_structure_bundle, save_semantic_bundle, save_structure_bundle
from .models import (
    ClaimEntityLinkRecord,
    ClaimLocationLinkRecord,
    ClaimPeriodLinkRecord,
    ClaimRecord,
    DocumentPeriodLinkRecord,
    DocumentRefugeLinkRecord,
    EntityRecord,
    ExtractionRunRecord,
    MeasurementRecord,
    MentionRecord,
    SemanticBundle,
    StructureBundle,
)
from .resolver import DictionaryFuzzyResolver, EntityResolver
from .source_parser import parse_source_file, parse_source_payload

MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}


def parse_source(input_path: str | Path | dict[str, Any]) -> StructureBundle:
    if isinstance(input_path, dict):
        return parse_source_payload(input_path, source_file=None)
    path = Path(input_path)
    if path.suffix.lower() == ".json":
        return parse_source_file(path)
    payload = load_json(path)
    return parse_source_payload(payload, source_file=str(path))


def extract_semantic(
    structure: StructureBundle,
    *,
    claim_extractor: ClaimExtractor | None = None,
    measurement_extractor: MeasurementExtractor | None = None,
    mention_extractor: MentionExtractor | None = None,
    resolver: EntityResolver | None = None,
    run_overrides: dict[str, str] | None = None,
) -> SemanticBundle:
    claim_extractor = claim_extractor or HybridClaimExtractor()
    measurement_extractor = measurement_extractor or RuleBasedMeasurementExtractor()
    mention_extractor = mention_extractor or RuleBasedMentionExtractor()
    resolver = resolver or DictionaryFuzzyResolver()

    now = datetime.now(timezone.utc)
    run_id = (run_overrides or {}).get("run_id", make_run_id((run_overrides or {}).get("ocr_engine", "unknown"), now))
    extraction_run = ExtractionRunRecord(
        run_id=run_id,
        ocr_engine=(run_overrides or {}).get("ocr_engine", "unknown"),
        ocr_version=(run_overrides or {}).get("ocr_version", "unknown"),
        normalizer_version=(run_overrides or {}).get("normalizer_version", "v1"),
        ner_model=(run_overrides or {}).get("ner_model", "hybrid-rules-llm"),
        relation_model=(run_overrides or {}).get("relation_model", "hybrid-rules-llm"),
        run_timestamp=(run_overrides or {}).get("run_timestamp", now.isoformat()),
    )

    claims: list[ClaimRecord] = []
    measurements: list[MeasurementRecord] = []
    mentions: list[MentionRecord] = []

    claim_counter = 0
    measurement_counter = 0

    mentions_by_paragraph: dict[str, list[MentionRecord]] = defaultdict(list)
    claims_by_paragraph: dict[str, list[ClaimRecord]] = defaultdict(list)

    for paragraph in structure.paragraphs:
        paragraph_text = paragraph.clean_text or paragraph.raw_text
        claim_drafts = claim_extractor.extract(paragraph_text)
        mention_drafts = mention_extractor.extract(paragraph_text)

        for draft in mention_drafts:
            mention_id = make_mention_id(
                extraction_run.run_id,
                paragraph.paragraph_id,
                draft.start_offset,
                draft.end_offset,
                draft.surface_form,
            )
            mention = MentionRecord(
                mention_id=mention_id,
                run_id=extraction_run.run_id,
                paragraph_id=paragraph.paragraph_id,
                surface_form=draft.surface_form,
                normalized_form=draft.normalized_form,
                start_offset=draft.start_offset,
                end_offset=draft.end_offset,
                confidence=draft.confidence,
                ocr_suspect=draft.ocr_suspect,
            )
            mentions.append(mention)
            mentions_by_paragraph[paragraph.paragraph_id].append(mention)

        for idx, draft in enumerate(claim_drafts, start=1):
            claim_counter += 1
            claim_id = make_claim_id(extraction_run.run_id, paragraph.paragraph_id, claim_counter + idx, draft.normalized_sentence)
            claim_date = draft.claim_date or _infer_claim_date(
                draft.raw_sentence,
                fallback_year=structure.document.report_year,
            )
            claim = ClaimRecord(
                claim_id=claim_id,
                run_id=extraction_run.run_id,
                paragraph_id=paragraph.paragraph_id,
                claim_type=draft.claim_type,
                raw_sentence=draft.raw_sentence,
                normalized_sentence=draft.normalized_sentence,
                certainty=draft.certainty,
                extraction_confidence=draft.extraction_confidence,
                evidence_start=draft.evidence_start,
                evidence_end=draft.evidence_end,
                claim_date=claim_date,
                notes=draft.notes,
            )
            claims.append(claim)
            claims_by_paragraph[paragraph.paragraph_id].append(claim)

            measurement_drafts = measurement_extractor.extract(draft)
            for measurement_idx, measurement_draft in enumerate(measurement_drafts, start=1):
                measurement_counter += 1
                measurement = MeasurementRecord(
                    measurement_id=make_measurement_id(
                        extraction_run.run_id,
                        claim_id,
                        measurement_counter + measurement_idx,
                        measurement_draft.name,
                        measurement_draft.raw_value,
                    ),
                    claim_id=claim_id,
                    run_id=extraction_run.run_id,
                    name=measurement_draft.name,
                    raw_value=measurement_draft.raw_value,
                    numeric_value=measurement_draft.numeric_value,
                    unit=measurement_draft.unit,
                    approximate=measurement_draft.approximate,
                    lower_bound=measurement_draft.lower_bound,
                    upper_bound=measurement_draft.upper_bound,
                    qualifier=measurement_draft.qualifier,
                )
                measurements.append(measurement)

    entities, entity_resolutions = resolver.resolve(mentions)
    entity_lookup = {entity.entity_id: entity for entity in entities}
    resolutions_by_mention = {resolution.mention_id: resolution for resolution in entity_resolutions}

    claim_entity_links: list[ClaimEntityLinkRecord] = []
    claim_location_links: list[ClaimLocationLinkRecord] = []
    claim_period_links: list[ClaimPeriodLinkRecord] = []
    document_refuge_links: list[DocumentRefugeLinkRecord] = []
    document_period_links: list[DocumentPeriodLinkRecord] = []

    seen_claim_entity: set[tuple[str, str]] = set()
    seen_claim_location: set[tuple[str, str]] = set()

    for paragraph_id, paragraph_claims in claims_by_paragraph.items():
        paragraph_mentions = mentions_by_paragraph.get(paragraph_id, [])
        for claim in paragraph_claims:
            for mention in paragraph_mentions:
                resolution = resolutions_by_mention.get(mention.mention_id)
                if not resolution:
                    continue
                entity = entity_lookup.get(resolution.entity_id)
                if not entity:
                    continue

                key = (claim.claim_id, entity.entity_id)
                if key not in seen_claim_entity:
                    claim_entity_links.append(ClaimEntityLinkRecord(claim_id=claim.claim_id, entity_id=entity.entity_id))
                    seen_claim_entity.add(key)

                if entity.label in {"Place", "Refuge"} and key not in seen_claim_location:
                    claim_location_links.append(ClaimLocationLinkRecord(claim_id=claim.claim_id, entity_id=entity.entity_id))
                    seen_claim_location.add(key)

    period_entity_id: str | None = None
    if structure.document.date_start or structure.document.date_end:
        period_entity_id = make_period_id(structure.document.date_start, structure.document.date_end, structure.document.title)
        period_entity = EntityRecord(
            entity_id=period_entity_id,
            label="Period",
            name=f"{structure.document.date_start or '?'} to {structure.document.date_end or '?'}",
            normalized_name=f"{structure.document.date_start or '?'} to {structure.document.date_end or '?'}",
            properties={
                "period_id": period_entity_id,
                "start_date": structure.document.date_start,
                "end_date": structure.document.date_end,
                "label": structure.document.title,
            },
        )
        if period_entity_id not in entity_lookup:
            entities.append(period_entity)
            entity_lookup[period_entity_id] = period_entity
        document_period_links.append(DocumentPeriodLinkRecord(doc_id=structure.document.doc_id, period_id=period_entity_id))
        for claim in claims:
            claim_period_links.append(ClaimPeriodLinkRecord(claim_id=claim.claim_id, period_id=period_entity_id))

    title_lc = structure.document.title.lower()
    if "turnbull" in title_lc:
        refuge_id = make_entity_id("Refuge", "turnbull refuge")
        if refuge_id not in entity_lookup:
            refuge = EntityRecord(
                entity_id=refuge_id,
                label="Refuge",
                name="Turnbull Refuge",
                normalized_name="turnbull refuge",
                properties={"refuge_id": "turnbull_refuge", "name": "Turnbull Refuge"},
            )
            entities.append(refuge)
            entity_lookup[refuge_id] = refuge
        document_refuge_links.append(DocumentRefugeLinkRecord(doc_id=structure.document.doc_id, refuge_id=refuge_id))

    return SemanticBundle(
        extraction_run=extraction_run,
        claims=claims,
        measurements=measurements,
        mentions=mentions,
        entities=entities,
        entity_resolutions=entity_resolutions,
        claim_entity_links=claim_entity_links,
        claim_location_links=claim_location_links,
        claim_period_links=claim_period_links,
        document_refuge_links=document_refuge_links,
        document_period_links=document_period_links,
        document_signed_by_links=[],
        person_affiliation_links=[],
    )


def load_graph(
    structure: StructureBundle,
    semantic: SemanticBundle,
    *,
    backend: str = "memory",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str = "neo4j",
    neo4j_trust_mode: str | None = None,
    neo4j_ca_cert: str | None = None,
) -> GraphWriter:
    writer: GraphWriter
    if backend == "neo4j":
        uri = neo4j_uri or os.environ.get("NEO4J_URI")
        user = neo4j_user or os.environ.get("NEO4J_USER")
        password = neo4j_password or os.environ.get("NEO4J_PASSWORD")
        database = os.environ.get("NEO4J_DATABASE", neo4j_database)
        trust_mode = neo4j_trust_mode or os.environ.get("NEO4J_TRUST", "system")
        ca_cert_path = neo4j_ca_cert or os.environ.get("NEO4J_CA_CERT")
        if not uri or not user or not password:
            raise ValueError("Neo4j backend requires URI/user/password via args or environment variables.")
        writer = Neo4jGraphWriter(
            uri=uri,
            user=user,
            password=password,
            database=database,
            trust_mode=trust_mode,
            ca_cert_path=ca_cert_path,
        )
    else:
        writer = InMemoryGraphWriter()

    writer.create_schema()
    writer.load_structure(structure)
    writer.load_semantic(structure, semantic)
    return writer


def quality_report(structure: StructureBundle, semantic: SemanticBundle) -> dict[str, Any]:
    paragraph_ids = {paragraph.paragraph_id for paragraph in structure.paragraphs}
    claim_ids = {claim.claim_id for claim in semantic.claims}

    claims_with_evidence = sum(1 for claim in semantic.claims if claim.paragraph_id in paragraph_ids)
    measurement_with_claim = sum(1 for item in semantic.measurements if item.claim_id in claim_ids)

    mention_valid_offsets = 0
    paragraph_texts = {paragraph.paragraph_id: (paragraph.clean_text or paragraph.raw_text) for paragraph in structure.paragraphs}
    for mention in semantic.mentions:
        text = paragraph_texts.get(mention.paragraph_id, "")
        if 0 <= mention.start_offset < mention.end_offset <= len(text):
            mention_valid_offsets += 1

    duplicate_counts = _duplicate_counts(
        {
            "doc_id": [structure.document.doc_id],
            "page_id": [row.page_id for row in structure.pages],
            "section_id": [row.section_id for row in structure.sections],
            "paragraph_id": [row.paragraph_id for row in structure.paragraphs],
            "claim_id": [row.claim_id for row in semantic.claims],
            "measurement_id": [row.measurement_id for row in semantic.measurements],
            "mention_id": [row.mention_id for row in semantic.mentions],
        }
    )

    claim_count = max(1, len(semantic.claims))
    measurement_count = max(1, len(semantic.measurements))
    mention_count = max(1, len(semantic.mentions))

    metrics = {
        "claim_count": len(semantic.claims),
        "measurement_count": len(semantic.measurements),
        "mention_count": len(semantic.mentions),
        "claims_with_evidence_pct": claims_with_evidence / claim_count,
        "measurements_linked_pct": measurement_with_claim / measurement_count,
        "mention_offset_valid_pct": mention_valid_offsets / mention_count,
        "duplicate_id_violations": duplicate_counts,
        "manual_review_precision_target": 0.85,
    }
    metrics["quality_gates"] = {
        "claims_have_evidence": metrics["claims_with_evidence_pct"] >= 1.0,
        "measurements_linked": metrics["measurements_linked_pct"] >= 1.0,
        "no_duplicate_ids": sum(duplicate_counts.values()) == 0,
        "mention_offsets_valid": metrics["mention_offset_valid_pct"] >= 0.95,
    }
    return metrics


def run_e2e(
    inputs: list[str | Path],
    out_dir: str | Path,
    *,
    backend: str = "memory",
) -> dict[str, Any]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer: GraphWriter | None = None
    per_doc: list[dict[str, Any]] = []

    for input_item in inputs:
        structure = parse_source(input_item)
        semantic = extract_semantic(structure)
        stem = Path(str(input_item)).stem
        structure_path = output_dir / f"{stem}.structure.json"
        semantic_path = output_dir / f"{stem}.semantic.json"
        save_structure_bundle(structure_path, structure)
        save_semantic_bundle(semantic_path, semantic)

        if writer is None:
            writer = load_graph(structure, semantic, backend=backend)
        else:
            writer.load_structure(structure)
            writer.load_semantic(structure, semantic)

        metrics = quality_report(structure, semantic)
        per_doc.append(
            {
                "input": str(input_item),
                "doc_id": structure.document.doc_id,
                "run_id": semantic.extraction_run.run_id,
                "structure_output": str(structure_path),
                "semantic_output": str(semantic_path),
                "quality": metrics,
            }
        )

    return {
        "documents_processed": len(per_doc),
        "outputs": per_doc,
        "backend": backend,
    }


def load_saved_pair(structure_path: str | Path, semantic_path: str | Path) -> tuple[StructureBundle, SemanticBundle]:
    return load_structure_bundle(structure_path), load_semantic_bundle(semantic_path)


def _duplicate_counts(values: dict[str, list[str]]) -> dict[str, int]:
    duplicates: dict[str, int] = {}
    for key, items in values.items():
        duplicates[key] = len(items) - len(set(items))
    return duplicates


def _infer_claim_date(sentence: str, fallback_year: int | None) -> str | None:
    match = re.search(
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:,\s*(\d{4}))?\b",
        sentence,
    )
    if not match:
        return None
    month_key = match.group(1).strip().lower()
    month = MONTHS.get(month_key)
    if not month:
        return None
    day = int(match.group(2))
    year = int(match.group(3)) if match.group(3) else fallback_year
    if not year:
        return None
    return f"{year:04d}-{month}-{day:02d}"
