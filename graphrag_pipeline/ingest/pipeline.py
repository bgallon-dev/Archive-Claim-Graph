from __future__ import annotations

import logging
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extractors import HybridClaimExtractor, RuleBasedMeasurementExtractor, RuleBasedMentionExtractor
from graphrag_pipeline.core.claim_validator import is_valid_claim_sentence
from .extractors.claim_extractor import ClaimExtractor, ClaimLinkDraft
from .extractors.measurement_extractor import MeasurementExtractor
from .extractors.mention_extractor import MentionExtractor
from .graph.writer import GraphWriter, InMemoryGraphWriter, Neo4jGraphWriter
from graphrag_pipeline.core.ids import (
    make_claim_id,
    make_entity_id,
    make_measurement_id,
    make_mention_id,
    make_period_id,
    make_run_id,
    make_year_id,
    stable_hash,
)
from graphrag_pipeline.shared.io_utils import load_json, load_semantic_bundle, load_structure_bundle, save_json, save_semantic_bundle, save_structure_bundle
from graphrag_pipeline.core.models import (
    ClaimConceptLinkRecord,
    ClaimEntityLinkRecord,
    ClaimLinkDiagnosticRecord,
    ClaimLocationLinkRecord,
    ClaimPeriodLinkRecord,
    ClaimRecord,
    DocumentPeriodLinkRecord,
    DocumentRefugeLinkRecord,
    DocumentYearLinkRecord,
    EntityRecord,
    ExtractionRunRecord,
    MeasurementRecord,
    MentionRecord,
    PlaceRefugeLinkRecord,
    SemanticBundle,
    StructureBundle,
    YearRecord,
)
from .concept_assigner import assign_concepts
from graphrag_pipeline.core.claim_contract import (
    CLAIM_ENTITY_RELATIONS,
    CLAIM_ENTITY_RELATION_PRECEDENCE,
    CLAIM_LOCATION_RELATION,
    entity_type_allowed_for_relation,
    get_preferred_entity_types,
    get_relation_compatibility,
    validate_claim_type,
)
from .event_builder import build_events
from .observation_builder import build_observations
from graphrag_pipeline.core.resolver import DictionaryFuzzyResolver, EntityResolver, default_seed_entities
from graphrag_pipeline.shared.resource_loader import load_domain_profile
from .spelling_review import build_spelling_review_queue as _build_spelling_review_queue
from .source_parser import parse_source_file, parse_source_payload

_log = logging.getLogger(__name__)

PERIOD_TYPE_PUBLICATION = "publication_period"
PERIOD_TYPE_EVENT = "event_period"
PERIOD_TYPE_SURVEY_SEASON = "survey_season"

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

CLAUSE_MARKERS = (",", ";", " and ", " or ")


def parse_source(input_path: str | Path | dict[str, Any]) -> StructureBundle:
    if isinstance(input_path, dict):
        return parse_source_payload(input_path, source_file=None)
    path = Path(input_path)
    if path.suffix.lower() == ".json":
        return parse_source_file(path)
    payload = load_json(path)
    return parse_source_payload(payload, source_file=str(path))


def _claim_span(claim: ClaimRecord, paragraph_text: str) -> tuple[int, int]:
    if claim.evidence_start is not None and claim.evidence_end is not None:
        return claim.evidence_start, claim.evidence_end
    return 0, len(paragraph_text)


def _is_short_simple_sentence(sentence: str) -> bool:
    normalized = f" {sentence.strip().lower()} "
    token_count = len(re.findall(r"\b\w+\b", sentence))
    return token_count < 25 and not any(marker in normalized for marker in CLAUSE_MARKERS)


def _diagnostic_record(
    claim_id: str,
    claim_link: ClaimLinkDraft,
    diagnostic_code: str,
    *,
    candidate_count: int = 0,
    detail: str = "",
) -> ClaimLinkDiagnosticRecord:
    return ClaimLinkDiagnosticRecord(
        claim_id=claim_id,
        relation_type=claim_link.relation_type,
        surface_form=claim_link.surface_form,
        normalized_form=claim_link.normalized_form,
        diagnostic_code=diagnostic_code,
        entity_type_hint=claim_link.entity_type_hint,
        candidate_count=candidate_count,
        detail=detail,
    )


_WEAK_CONFIDENCE_PENALTY: float = 0.10


def _narrow_link_candidates(
    candidates: list[tuple[MentionRecord, EntityRecord]],
    claim_link: ClaimLinkDraft,
    claim_type: str = "",
) -> tuple[tuple[MentionRecord, EntityRecord] | None, ClaimLinkDiagnosticRecord | None]:
    if not candidates:
        return None, None

    relation_filtered = [
        row for row in candidates if entity_type_allowed_for_relation(claim_link.relation_type, row[1].entity_type)
    ]
    if not relation_filtered:
        return None, _diagnostic_record(
            "",
            claim_link,
            "RELATION_TYPE_CONFLICT",
            candidate_count=len(candidates),
            detail="Resolved mentions in span did not match the relation's allowed entity types.",
        )

    # ── Compatibility check: claim_type × relation_type ───────────────────────
    if claim_type:
        tier = get_relation_compatibility(claim_type, claim_link.relation_type)
        if tier == "forbidden":
            return None, _diagnostic_record(
                "",
                claim_link,
                "RELATION_COMPATIBILITY_FORBIDDEN",
                candidate_count=len(relation_filtered),
                detail=(
                    f"Relation {claim_link.relation_type!r} is forbidden for claim type {claim_type!r}. "
                    "Link dropped."
                ),
            )
        # "weak" tier: keep candidates but note it; confidence will be penalised
        # in _resolve_claim_link after the winning entity is selected.

    # ── End-label preference: re-rank by preferred entity types ──────────────
    preferred = get_preferred_entity_types(claim_type) if claim_type else []
    if preferred and len(relation_filtered) > 1:
        def _preference_rank(row: tuple[MentionRecord, EntityRecord]) -> int:
            et = row[1].entity_type
            try:
                return preferred.index(et)
            except ValueError:
                return len(preferred)

        relation_filtered_sorted = sorted(relation_filtered, key=_preference_rank)
        # If the best preferred candidate differs from the first unranked one,
        # emit an informational diagnostic (non-blocking) to make it visible.
        if relation_filtered_sorted[0][1].entity_type != relation_filtered[0][1].entity_type:
            # We don't return a diagnostic here — we just reorder so the
            # preferred entity wins.  Callers see the reranking in the resolved link.
            pass
        relation_filtered = relation_filtered_sorted

    if len(relation_filtered) == 1 and not claim_link.entity_type_hint:
        return relation_filtered[0], None

    if claim_link.entity_type_hint:
        hinted = [row for row in relation_filtered if row[1].entity_type == claim_link.entity_type_hint]
        if len(hinted) == 1:
            return hinted[0], None
        if not hinted:
            return None, _diagnostic_record(
                "",
                claim_link,
                "TYPE_HINT_CONFLICT",
                candidate_count=len(relation_filtered),
                detail=f"No resolved mention matched entity_type_hint={claim_link.entity_type_hint!r}.",
            )
        return None, _diagnostic_record(
            "",
            claim_link,
            "AMBIGUOUS_FALLBACK",
            candidate_count=len(hinted),
            detail="Multiple resolved mentions remained after applying entity_type_hint.",
        )

    if len(relation_filtered) == 1:
        return relation_filtered[0], None
    return None, _diagnostic_record(
        "",
        claim_link,
        "AMBIGUOUS_FALLBACK",
        candidate_count=len(relation_filtered),
        detail="Multiple resolved mentions matched the normalized form within the claim span.",
    )


def _resolve_claim_link(
    claim: ClaimRecord,
    claim_link: ClaimLinkDraft,
    paragraph_mentions: list[MentionRecord],
    resolutions_by_mention: dict[str, Any],
    entity_lookup: dict[str, EntityRecord],
    paragraph_text: str,
) -> tuple[EntityRecord | None, ClaimLinkDiagnosticRecord | None]:
    claim_type = claim.claim_type
    span_start, span_end = _claim_span(claim, paragraph_text)
    resolved_in_span: list[tuple[MentionRecord, EntityRecord]] = []
    for mention in paragraph_mentions:
        if mention.start_offset < span_start or mention.end_offset > span_end:
            continue
        resolution = resolutions_by_mention.get(mention.mention_id)
        if not resolution:
            continue
        entity = entity_lookup.get(resolution.entity_id)
        if not entity:
            continue
        if entity.entity_type == "SurveyMethod" and resolution.relation_type != "REFERS_TO":
            continue
        resolved_in_span.append((mention, entity))

    exact_candidates = [
        row
        for row in resolved_in_span
        if claim_link.start_offset is not None
        and claim_link.end_offset is not None
        and row[0].start_offset == claim_link.start_offset
        and row[0].end_offset == claim_link.end_offset
    ]
    exact_match, exact_diagnostic = _narrow_link_candidates(exact_candidates, claim_link, claim_type)
    if exact_match:
        return _apply_compatibility_penalty(exact_match[1], claim_type, claim_link.relation_type), None

    normalized_candidates = [
        row for row in resolved_in_span if row[0].normalized_form == claim_link.normalized_form
    ]
    normalized_match, normalized_diagnostic = _narrow_link_candidates(normalized_candidates, claim_link, claim_type)
    if normalized_match:
        return _apply_compatibility_penalty(normalized_match[1], claim_type, claim_link.relation_type), None

    # Fallback: if span matching found nothing, try matching
    # any resolved mention in the paragraph whose normalized_form
    # appears in the claim's normalized_sentence
    if not exact_match and not normalized_match and not normalized_diagnostic and not exact_diagnostic:
        sentence_normalized = claim.normalized_sentence or ""
        sentence_candidates = [
            row for row in resolved_in_span
            if row[0].normalized_form in sentence_normalized
        ]
        # Widen the search to full paragraph if span search found nothing
        if not sentence_candidates:
            for mention in paragraph_mentions:
                resolution = resolutions_by_mention.get(mention.mention_id)
                if not resolution:
                    continue
                entity = entity_lookup.get(resolution.entity_id)
                if not entity:
                    continue
                if mention.normalized_form in sentence_normalized:
                    sentence_candidates.append((mention, entity))

        if sentence_candidates:
            sentence_match, sentence_diagnostic = _narrow_link_candidates(
                sentence_candidates, claim_link, claim_type
            )
            if sentence_match:
                return _apply_compatibility_penalty(
                    sentence_match[1], claim_type, claim_link.relation_type
                ), None

    diagnostic = normalized_diagnostic or exact_diagnostic
    if diagnostic:
        diagnostic.claim_id = claim.claim_id
        return None, diagnostic
    return None, _diagnostic_record(
        claim.claim_id,
        claim_link,
        "NO_RESOLVED_MENTION",
        detail="No resolved mention matched the claim link within the claim evidence span.",
    )


def _apply_compatibility_penalty(
    entity: EntityRecord,
    claim_type: str,
    relation_type: str,
) -> EntityRecord:
    """No-op for now: compatibility penalty is applied at the claim level downstream.

    The weak-tier signal is surfaced via diagnostics; callers that need to
    lower extraction_confidence on weak links should call
    ``get_relation_compatibility()`` on the resolved ClaimEntityLinkRecord.
    Returning the entity unchanged keeps this function pure and testable.
    """
    return entity


def extract_semantic(
    structure: StructureBundle,
    *,
    claim_extractor: ClaimExtractor | None = None,
    measurement_extractor: MeasurementExtractor | None = None,
    mention_extractor: MentionExtractor | None = None,
    resolver: EntityResolver | None = None,
    run_overrides: dict[str, str] | None = None,
    resources_dir: Path | None = None,
) -> SemanticBundle:
    claim_extractor = claim_extractor or HybridClaimExtractor(resources_dir=resources_dir)
    measurement_extractor = measurement_extractor or RuleBasedMeasurementExtractor(resources_dir=resources_dir)
    mention_extractor = mention_extractor or RuleBasedMentionExtractor(resources_dir=resources_dir)
    resolver = resolver or DictionaryFuzzyResolver(seed_entities=default_seed_entities(resources_dir))

    now = datetime.now(timezone.utc)
    overrides = run_overrides or {}
    run_id = overrides.get("run_id", make_run_id(overrides.get("ocr_engine", "unknown"), now))
    extraction_run = ExtractionRunRecord(
        run_id=run_id,
        ocr_engine=overrides.get("ocr_engine", "unknown"),
        ocr_version=overrides.get("ocr_version", "unknown"),
        normalizer_version=overrides.get("normalizer_version", "v1"),
        ner_model=overrides.get("ner_model", "hybrid-rules-llm"),
        relation_model=overrides.get("relation_model", "hybrid-rules-llm"),
        run_timestamp=overrides.get("run_timestamp", now.isoformat()),
        config_fingerprint=stable_hash(
            overrides.get("ocr_engine", "unknown"),
            overrides.get("ocr_version", "unknown"),
            overrides.get("normalizer_version", "v1"),
            overrides.get("ner_model", "hybrid-rules-llm"),
            overrides.get("relation_model", "hybrid-rules-llm"),
        ),
        claim_type_schema_version="v2",
    )

    claims: list[ClaimRecord] = []
    measurements: list[MeasurementRecord] = []
    mentions: list[MentionRecord] = []
    claim_links_by_claim: dict[str, list[ClaimLinkDraft]] = defaultdict(list)

    claim_counter = 0
    measurement_counter = 0

    paragraph_texts = {
        paragraph.paragraph_id: (paragraph.clean_text or paragraph.raw_ocr_text)
        for paragraph in structure.paragraphs
    }
    mentions_by_paragraph: dict[str, list[MentionRecord]] = defaultdict(list)
    claims_by_paragraph: dict[str, list[ClaimRecord]] = defaultdict(list)

    for paragraph in structure.paragraphs:
        paragraph_text = paragraph_texts[paragraph.paragraph_id]
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
                detection_confidence=draft.detection_confidence,
                ocr_suspect=bool(draft.ocr_flags),
            )
            mentions.append(mention)
            mentions_by_paragraph[paragraph.paragraph_id].append(mention)

        for idx, draft in enumerate(claim_drafts, start=1):
            valid, _reason = is_valid_claim_sentence(draft.source_sentence)
            if not valid:
                continue
            claim_counter += 1
            claim_id = make_claim_id(extraction_run.run_id, paragraph.paragraph_id, claim_counter + idx, draft.normalized_sentence)
            claim_date = draft.claim_date or _infer_claim_date(
                draft.source_sentence,
                fallback_year=structure.document.report_year,
            )
            claim = ClaimRecord(
                claim_id=claim_id,
                run_id=extraction_run.run_id,
                paragraph_id=paragraph.paragraph_id,
                claim_type=validate_claim_type(draft.claim_type),
                source_sentence=draft.source_sentence,
                normalized_sentence=draft.normalized_sentence,
                certainty=draft.epistemic_status,
                extraction_confidence=draft.extraction_confidence,
                evidence_start=draft.evidence_start,
                evidence_end=draft.evidence_end,
                claim_date=claim_date,
                notes=draft.notes,
            )
            claims.append(claim)
            claims_by_paragraph[paragraph.paragraph_id].append(claim)
            claim_links_by_claim[claim_id] = list(draft.claim_links)

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
                    measurement_date=claim_date,
                    methodology_note=measurement_draft.methodology_note,
                )
                measurements.append(measurement)

    entities, entity_resolutions = resolver.resolve(mentions)
    entity_lookup = {entity.entity_id: entity for entity in entities}
    resolutions_by_mention = {resolution.mention_id: resolution for resolution in entity_resolutions}

    claim_entity_links: list[ClaimEntityLinkRecord] = []
    claim_link_diagnostics: list[ClaimLinkDiagnosticRecord] = []
    claim_location_links: list[ClaimLocationLinkRecord] = []
    claim_period_links: list[ClaimPeriodLinkRecord] = []
    document_refuge_links: list[DocumentRefugeLinkRecord] = []
    document_period_links: list[DocumentPeriodLinkRecord] = []

    seen_claim_entity: set[tuple[str, str, str]] = set()
    seen_claim_location: set[tuple[str, str]] = set()

    # Create document-level anchor entity before link assembly so doc_anchor_id is in scope for doc-level links.
    doc_anchor_id: str | None = None
    anchor_cfg = load_domain_profile(resources_dir).get("document_anchor")
    if anchor_cfg:
        raw_kw = anchor_cfg.get("title_keywords") or anchor_cfg.get("title_keyword") or []
        keywords = [raw_kw] if isinstance(raw_kw, str) else list(raw_kw)
        title_lower = structure.document.title.lower()
        if any(kw.lower() in title_lower for kw in keywords):
            entity_type = anchor_cfg["entity_type"]
            name = anchor_cfg["name"]
            norm = anchor_cfg.get("normalized_form", name.lower())
            props = anchor_cfg.get("properties", {})
            doc_anchor_id = make_entity_id(entity_type, norm)
            if doc_anchor_id not in entity_lookup:
                anchor_entity = EntityRecord(
                    entity_id=doc_anchor_id,
                    entity_type=entity_type,
                    name=name,
                    normalized_form=norm,
                    properties=props,
                )
                entities.append(anchor_entity)
                entity_lookup[doc_anchor_id] = anchor_entity
            document_refuge_links.append(DocumentRefugeLinkRecord(doc_id=structure.document.doc_id, refuge_id=doc_anchor_id))

    for paragraph_id, paragraph_claims in claims_by_paragraph.items():
        paragraph_mentions = mentions_by_paragraph.get(paragraph_id, [])
        paragraph_text = paragraph_texts.get(paragraph_id, "")
        for claim in paragraph_claims:
            for claim_link in claim_links_by_claim.get(claim.claim_id, []):
                entity, diagnostic = _resolve_claim_link(
                    claim,
                    claim_link,
                    paragraph_mentions,
                    resolutions_by_mention,
                    entity_lookup,
                    paragraph_text,
                )
                if diagnostic:
                    claim_link_diagnostics.append(diagnostic)
                    continue
                if not entity:
                    continue
                if claim_link.relation_type == CLAIM_LOCATION_RELATION:
                    key = (claim.claim_id, entity.entity_id)
                    if key not in seen_claim_location:
                        claim_location_links.append(ClaimLocationLinkRecord(claim_id=claim.claim_id, entity_id=entity.entity_id))
                        seen_claim_location.add(key)
                    continue
                if claim_link.relation_type not in CLAIM_ENTITY_RELATIONS:
                    continue
                # Apply weak-tier confidence penalty before recording the link.
                if claim.claim_type and get_relation_compatibility(claim.claim_type, claim_link.relation_type) == "weak":
                    claim.extraction_confidence = max(
                        0.0, claim.extraction_confidence - _WEAK_CONFIDENCE_PENALTY
                    )
                    claim_link_diagnostics.append(
                        _diagnostic_record(
                            claim.claim_id,
                            claim_link,
                            "RELATION_COMPATIBILITY_WEAK",
                            detail=(
                                f"Relation {claim_link.relation_type!r} is weak for claim type "
                                f"{claim.claim_type!r}; confidence penalised by {_WEAK_CONFIDENCE_PENALTY}."
                            ),
                        )
                    )
                key = (claim.claim_id, entity.entity_id, claim_link.relation_type)
                if key not in seen_claim_entity:
                    claim_entity_links.append(
                        ClaimEntityLinkRecord(
                            claim_id=claim.claim_id,
                            entity_id=entity.entity_id,
                            relation_type=claim_link.relation_type,
                        )
                    )
                    seen_claim_entity.add(key)

    period_entity_id: str | None = None
    if structure.document.date_start or structure.document.date_end:
        period_entity_id = make_period_id(
            structure.document.date_start,
            structure.document.date_end,
            source_title=structure.document.title,
        )
        period_entity = EntityRecord(
            entity_id=period_entity_id,
            entity_type="Period",
            name=f"{structure.document.date_start or '?'} to {structure.document.date_end or '?'}",
            normalized_form=f"{structure.document.date_start or '?'} to {structure.document.date_end or '?'}",
            properties={
                "period_id": period_entity_id,
                "start_date": structure.document.date_start,
                "end_date": structure.document.date_end,
                "source_title": structure.document.title,
                "period_type": PERIOD_TYPE_PUBLICATION,
            },
        )
        if period_entity_id not in entity_lookup:
            entities.append(period_entity)
            entity_lookup[period_entity_id] = period_entity
        document_period_links.append(DocumentPeriodLinkRecord(doc_id=structure.document.doc_id, period_id=period_entity_id))
        for claim in claims:
            claim_period_links.append(ClaimPeriodLinkRecord(claim_id=claim.claim_id, period_id=period_entity_id))

    # Build Observations from eligible claims
    observations, years, obs_measurement_links, _ = build_observations(
        claims=claims,
        measurements=measurements,
        claim_entity_links=claim_entity_links,
        claim_location_links=claim_location_links,
        claim_period_links=claim_period_links,
        entity_lookup=entity_lookup,
        run_id=extraction_run.run_id,
        report_year=structure.document.report_year,
    )

    # Build Events from event-eligible claims
    events, event_obs_links, event_meas_links, event_years = build_events(
        claims=claims,
        measurements=measurements,
        claim_entity_links=claim_entity_links,
        claim_location_links=claim_location_links,
        claim_period_links=claim_period_links,
        entity_lookup=entity_lookup,
        observations=observations,
        run_id=extraction_run.run_id,
        report_year=structure.document.report_year,
    )
    # Merge event-derived year nodes without duplicates
    existing_year_ids = {y.year_id for y in years}
    for yr in event_years:
        if yr.year_id not in existing_year_ids:
            years.append(yr)
            existing_year_ids.add(yr.year_id)

    # Create Year node from document report_year and link document
    document_year_links: list[DocumentYearLinkRecord] = []
    if structure.document.report_year:
        doc_year_id = make_year_id(structure.document.report_year)
        if not any(y.year_id == doc_year_id for y in years):
            years.append(YearRecord(year_id=doc_year_id, year=structure.document.report_year, year_label=str(structure.document.report_year)))
        document_year_links.append(DocumentYearLinkRecord(doc_id=structure.document.doc_id, year_id=doc_year_id))

    # Place PART_OF Refuge links
    place_refuge_links: list[PlaceRefugeLinkRecord] = []
    if doc_anchor_id:
        for entity in entities:
            if entity.entity_type == "Place":
                place_refuge_links.append(PlaceRefugeLinkRecord(place_id=entity.entity_id, refuge_id=doc_anchor_id))

    # Concept assignment
    _concept_ids = {
        "concept_nesting_success", "concept_breeding_success",
        "concept_population_decline", "concept_drought_condition",
        "concept_flood_condition", "concept_temperature_extremes",
        "concept_precipitation_pattern", "concept_habitat_degradation",
        "concept_water_level_change", "concept_habitat_condition",
        "concept_ecological_restoration", "concept_infrastructure_rehabilitation",
        "concept_habitat_restoration", "concept_population_count",
        "concept_survey_result", "concept_breeding_activity",
        "concept_seasonal_condition",
    }
    claim_concept_links: list[ClaimConceptLinkRecord] = []
    for claim in claims:
        for assignment in assign_concepts(claim):
            if assignment.concept_id in _concept_ids:
                claim_concept_links.append(
                    ClaimConceptLinkRecord(
                        claim_id=claim.claim_id,
                        concept_id=assignment.concept_id,
                        confidence=assignment.confidence,
                        matched_rule=assignment.matched_rule,
                    )
                )

    return SemanticBundle(
        extraction_run=extraction_run,
        claims=claims,
        measurements=measurements,
        mentions=mentions,
        entities=entities,
        entity_resolutions=entity_resolutions,
        claim_entity_links=claim_entity_links,
        claim_link_diagnostics=claim_link_diagnostics,
        claim_location_links=claim_location_links,
        claim_period_links=claim_period_links,
        document_refuge_links=document_refuge_links,
        document_period_links=document_period_links,
        document_signed_by_links=[],
        person_affiliation_links=[],
        observations=observations,
        years=years,
        observation_measurement_links=obs_measurement_links,
        document_year_links=document_year_links,
        place_refuge_links=place_refuge_links,
        events=events,
        event_observation_links=event_obs_links,
        event_measurement_links=event_meas_links,
        claim_concept_links=claim_concept_links,
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
    paragraph_texts = {paragraph.paragraph_id: (paragraph.clean_text or paragraph.raw_ocr_text) for paragraph in structure.paragraphs}
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

    observation_count = len(semantic.observations)
    obs_with_species = sum(1 for obs in semantic.observations if obs.species_id)
    obs_with_year = sum(1 for obs in semantic.observations if obs.year_id)
    obs_with_evidence = sum(1 for obs in semantic.observations if obs.claim_id in claim_ids)
    safe_obs_count = max(1, observation_count)
    unclassified_claim_count = sum(1 for c in semantic.claims if c.claim_type == "unclassified_assertion")
    claim_entity_link_count = len(semantic.claim_entity_links)
    typed_claim_links = [link for link in semantic.claim_entity_links if link.relation_type in CLAIM_ENTITY_RELATIONS]
    safe_claim_entity_link_count = max(1, claim_entity_link_count)
    claim_entity_relation_counts: dict[str, int] = defaultdict(int)
    for link in semantic.claim_entity_links:
        claim_entity_relation_counts[link.relation_type] += 1
    typed_links_by_claim: dict[str, list[ClaimEntityLinkRecord]] = defaultdict(list)
    for link in typed_claim_links:
        typed_links_by_claim[link.claim_id].append(link)
    claims_with_many_typed_links = sum(1 for links in typed_links_by_claim.values() if len(links) > 5)
    claims_with_multiple_location_focuses = sum(
        1 for links in typed_links_by_claim.values() if sum(1 for link in links if link.relation_type == "LOCATION_FOCUS") > 1
    )
    claims_with_redundant_topic_focus = sum(
        1
        for links in typed_links_by_claim.values()
        if any(link.relation_type == "TOPIC_OF_CLAIM" for link in links)
        and len({link.relation_type for link in links if link.relation_type != "TOPIC_OF_CLAIM"}) >= 2
    )
    claims_by_id = {claim.claim_id: claim for claim in semantic.claims}
    claims_with_many_focus_roles_in_short_simple_sentence = sum(
        1
        for claim_id, links in typed_links_by_claim.items()
        if len({link.relation_type for link in links}) >= 3
        and claim_id in claims_by_id
        and _is_short_simple_sentence(claims_by_id[claim_id].source_sentence)
    )
    diagnostic_counts: dict[str, int] = defaultdict(int)
    for diagnostic in semantic.claim_link_diagnostics:
        diagnostic_counts[diagnostic.diagnostic_code] += 1

    metrics = {
        "claim_count": len(semantic.claims),
        "measurement_count": len(semantic.measurements),
        "mention_count": len(semantic.mentions),
        "observation_count": observation_count,
        "year_count": len(semantic.years),
        "claim_entity_link_count": claim_entity_link_count,
        "typed_claim_entity_link_count": len(typed_claim_links),
        "typed_claim_entity_link_share": len(typed_claim_links) / safe_claim_entity_link_count if claim_entity_link_count else 0.0,
        "claim_entity_relation_counts": dict(sorted(claim_entity_relation_counts.items())),
        "claims_with_evidence_pct": claims_with_evidence / claim_count,
        "measurements_linked_pct": measurement_with_claim / measurement_count,
        "mention_offset_valid_pct": mention_valid_offsets / mention_count,
        "observations_with_species_pct": obs_with_species / safe_obs_count if observation_count else 0.0,
        "observations_with_year_pct": obs_with_year / safe_obs_count if observation_count else 0.0,
        "unclassified_claim_count": unclassified_claim_count,
        "claims_with_many_typed_links": claims_with_many_typed_links,
        "claims_with_multiple_location_focuses": claims_with_multiple_location_focuses,
        "claims_with_redundant_topic_focus": claims_with_redundant_topic_focus,
        "claims_with_many_focus_roles_in_short_simple_sentence": claims_with_many_focus_roles_in_short_simple_sentence,
        "claim_link_diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "duplicate_id_violations": duplicate_counts,
        "manual_review_precision_target": 0.85,
    }
    metrics["quality_gates"] = {
        "claims_have_evidence": metrics["claims_with_evidence_pct"] >= 1.0,
        "measurements_linked": metrics["measurements_linked_pct"] >= 1.0,
        "no_duplicate_ids": sum(duplicate_counts.values()) == 0,
        "mention_offsets_valid": metrics["mention_offset_valid_pct"] >= 0.95,
        "observations_have_evidence": obs_with_evidence == observation_count if observation_count else True,
    }
    return metrics


def build_spelling_review_queue(structure: StructureBundle, semantic: SemanticBundle) -> list[dict[str, Any]]:
    return _build_spelling_review_queue(structure, semantic)


def _make_doc_run_id(input_path: str, now: datetime) -> str:
    """Generate a run_id that is unique per document even under heavy parallelism.

    Uses the input file path as entropy so two workers processing different files
    at the same instant always get distinct run_ids, eliminating claim_id collisions.
    """
    from graphrag_pipeline.core.ids import stable_hash
    ts = now.strftime("%Y%m%dT%H%M%S%fZ")
    return f"run_{ts}_{stable_hash(input_path, ts, size=8)}"


def _fetch_graph_entities(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    neo4j_trust_mode: str | None,
    neo4j_ca_cert: str | None,
) -> list[EntityRecord]:
    """Fetch all active Entity nodes from Neo4j as EntityRecord objects.

    Used by run_e2e(graph_resolve=True) to seed DictionaryFuzzyResolver with
    entities already in the graph so new documents resolve to existing IDs.
    Returns [] on any failure so the pipeline degrades to seed-only resolution.
    """
    from graphrag_pipeline.retrieval.executor import Neo4jQueryExecutor
    from graphrag_pipeline.core.graph.cypher import GRAPH_ENTITY_FETCH_QUERY
    executor = Neo4jQueryExecutor(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        trust_mode=neo4j_trust_mode or "system",
        ca_cert_path=neo4j_ca_cert,
    )
    try:
        rows = executor.run(GRAPH_ENTITY_FETCH_QUERY, {})
    except Exception as exc:
        _log.warning("graph_resolve: failed to fetch graph entities: %s", exc)
        return []
    finally:
        executor.close()
    result: list[EntityRecord] = []
    for row in rows:
        if not row.get("entity_id") or not row.get("entity_type") or not row.get("normalized_form"):
            continue
        result.append(EntityRecord(
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            name=row.get("name") or row["normalized_form"],
            normalized_form=row["normalized_form"],
            properties={},
        ))
    return result


def _process_single_document(
    input_item: str,
    out_dir: str,
    review_out_dir: str | None,
    domain_dir_str: str | None = None,
    run_id: str | None = None,
    supplementary_entity_rows: list[EntityRecord] | None = None,
) -> dict[str, Any]:
    """Worker function for parallel document processing (parse + extract + save + quality).

    Must be defined at module scope so ProcessPoolExecutor can pickle it on Windows.
    domain_dir_str is kept as str | None (not Path) so it is pickling-safe on Windows.
    """
    resources_dir = Path(domain_dir_str) if domain_dir_str else None
    structure = parse_source(input_item)
    resolver: EntityResolver | None = None
    if supplementary_entity_rows:
        resolver = DictionaryFuzzyResolver(
            seed_entities=default_seed_entities(resources_dir),
            supplementary_candidates=supplementary_entity_rows,
        )
    semantic = extract_semantic(
        structure,
        resolver=resolver,
        resources_dir=resources_dir,
        run_overrides={"run_id": run_id} if run_id else None,
    )

    # Sensitivity gate: quarantine high-confidence claims before graph write.
    # Runs after extraction but before saving bundles so quarantine_status
    # is persisted on disk and written to the graph atomically.
    try:
        from graphrag_pipeline.review.detectors import sensitivity_monitor as _sm
        _sm_proposals = _sm.detect(structure, semantic, snapshot_id="")
        _sm_cfg = _sm._load_config()
        _auto_quarantine_threshold = (
            _sm_cfg.get("pii_detection", {}).get("auto_quarantine_threshold", 0.85)
        )
        _now_ts = datetime.now(timezone.utc).isoformat()
        _quarantined_ids: set[str] = set()
        for _prop in _sm_proposals:
            if _prop.confidence >= _auto_quarantine_threshold:
                for _target in _prop.targets:
                    if _target.target_kind == "claim" and _target.target_id not in _quarantined_ids:
                        for _claim in semantic.claims:
                            if _claim.claim_id == _target.target_id:
                                _claim.quarantine_status = "quarantined"
                                _claim.quarantine_reason = _prop.issue_class
                                _claim.quarantine_timestamp = _now_ts
                                _quarantined_ids.add(_claim.claim_id)
    except Exception as _gate_exc:
        _log.error(
            "Sensitivity gate failed for %r: %s — quarantining all claims as precaution",
            input_item, _gate_exc, exc_info=True,
        )
        # Quarantine everything in the document rather than silently ingesting
        # un-screened content.  A reviewer can clear individual claims once the
        # detector issue is resolved.
        _now_ts = datetime.now(timezone.utc).isoformat()
        for _claim in semantic.claims:
            if _claim.quarantine_status == "active":
                _claim.quarantine_status = "quarantined"
                _claim.quarantine_reason = "sensitivity_gate_error"
                _claim.quarantine_timestamp = _now_ts

    stem = Path(input_item).stem
    output_dir = Path(out_dir)
    structure_path = output_dir / f"{stem}.structure.json"
    semantic_path = output_dir / f"{stem}.semantic.json"
    save_structure_bundle(structure_path, structure)
    save_semantic_bundle(semantic_path, semantic)

    metrics = quality_report(structure, semantic)
    doc_summary: dict[str, Any] = {
        "input": input_item,
        "doc_id": structure.document.doc_id,
        "run_id": semantic.extraction_run.run_id,
        "structure_output": str(structure_path),
        "semantic_output": str(semantic_path),
        "quality": metrics,
    }

    if review_out_dir is not None:
        review_rows = build_spelling_review_queue(structure, semantic)
        review_path = Path(review_out_dir) / f"{stem}.spelling_review.json"
        save_json(review_path, review_rows)
        doc_summary["spelling_review_output"] = str(review_path)
        doc_summary["spelling_review_issue_count"] = len(review_rows)

    return doc_summary


def run_e2e(
    inputs: list[str | Path],
    out_dir: str | Path,
    *,
    workers: int = 1,
    backend: str = "memory",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str = "neo4j",
    neo4j_trust_mode: str | None = None,
    neo4j_ca_cert: str | None = None,
    review_out_dir: str | Path | None = None,
    review_db_path: str | Path | None = None,
    resources_dir: Path | None = None,
    graph_resolve: bool = False,
) -> dict[str, Any]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_output_dir_str = str(Path(review_out_dir)) if review_out_dir is not None else None
    if review_output_dir_str is not None:
        Path(review_output_dir_str).mkdir(parents=True, exist_ok=True)
    domain_dir_str = str(resources_dir) if resources_dir is not None else None

    # --- Phase 0: pre-flight Neo4j checks (duplicate detection + graph entities) ---
    # Both run only when backend=="neo4j" and a URI is available, so the memory-
    # backend path is completely unaffected.
    all_str_inputs = [str(item) for item in inputs]
    doc_summaries: list[dict[str, Any]] = []
    supplementary_entities: list[EntityRecord] = []

    if backend == "neo4j" and neo4j_uri:
        import hashlib as _hashlib
        import json as _json

        # 0a. Compute file_hash for each input (same algorithm as source_parser.py).
        input_file_hashes: dict[str, str] = {}
        for item in all_str_inputs:
            try:
                payload = _json.loads(Path(item).read_text(encoding="utf-8"))
                h = _hashlib.sha1(
                    _json.dumps(payload, sort_keys=True).encode("utf-8")
                ).hexdigest()
                input_file_hashes[item] = h
            except Exception as _exc:
                _log.debug("Could not hash %s for duplicate check: %s", item, _exc)

        # 0b. Batch-check which hashes already exist in the graph.
        if input_file_hashes:
            from graphrag_pipeline.retrieval.executor import Neo4jQueryExecutor
            from graphrag_pipeline.core.graph.cypher import DUPLICATE_HASH_CHECK_QUERY
            _dup_executor = Neo4jQueryExecutor(
                uri=neo4j_uri,
                user=neo4j_user or "",
                password=neo4j_password or "",
                database=neo4j_database,
                trust_mode=neo4j_trust_mode or "system",
                ca_cert_path=neo4j_ca_cert,
            )
            try:
                dup_rows = _dup_executor.run(
                    DUPLICATE_HASH_CHECK_QUERY,
                    {"file_hashes": list(input_file_hashes.values())},
                )
                ingested_hashes: set[str] = {row["file_hash"] for row in dup_rows}
            except Exception as _exc:
                _log.warning("Duplicate hash check failed: %s — skipping pre-flight", _exc)
                ingested_hashes = set()
            finally:
                _dup_executor.close()

            for item in all_str_inputs:
                h = input_file_hashes.get(item)
                if h and h in ingested_hashes:
                    _log.info("Skipping already-ingested document: %s", Path(item).name)
                    doc_summaries.append({
                        "input": item,
                        "status": "already_ingested",
                        "skipped": True,
                        "file_hash": h,
                    })

        # 0c. Fetch graph entities for supplementary resolution if requested.
        if graph_resolve:
            supplementary_entities = _fetch_graph_entities(
                neo4j_uri,
                neo4j_user or "",
                neo4j_password or "",
                neo4j_database,
                neo4j_trust_mode,
                neo4j_ca_cert,
            )
            _log.info("graph_resolve: loaded %d graph entities as supplementary candidates", len(supplementary_entities))

    already_ingested_inputs = {d["input"] for d in doc_summaries if d.get("skipped")}
    str_inputs = [i for i in all_str_inputs if i not in already_ingested_inputs]

    # --- Phase 1: parallel extraction (parse + extract + save + quality) ---
    effective_workers = min(max(workers, 1), len(str_inputs)) if str_inputs else 1

    # Pre-generate a unique run_id per document in the parent process so that
    # parallel workers never collide even when they start within the same second.
    _batch_now = datetime.now(timezone.utc)
    doc_run_ids = {item: _make_doc_run_id(item, _batch_now) for item in str_inputs}

    _supp = supplementary_entities or None  # None when empty → workers use seed-only resolution

    if effective_workers <= 1:
        for input_item in str_inputs:
            doc_summaries.append(
                _process_single_document(input_item, str(output_dir), review_output_dir_str, domain_dir_str, doc_run_ids[input_item], _supp)
            )
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_input = {
                executor.submit(
                    _process_single_document,
                    input_item,
                    str(output_dir),
                    review_output_dir_str,
                    domain_dir_str,
                    doc_run_ids[input_item],
                    _supp,
                ): input_item
                for input_item in str_inputs
            }
            for i, future in enumerate(as_completed(future_to_input), 1):
                input_item = future_to_input[future]
                summary = future.result()  # re-raises on worker exception
                _log.info("[%d/%d] Processed: %s", i, len(str_inputs), Path(input_item).stem)
                doc_summaries.append(summary)

    # Restore deterministic output order (as_completed is unordered).
    # Skipped (already_ingested) entries are already in doc_summaries with an "input" key.
    doc_summaries.sort(key=lambda d: d["input"])

    # --- Phase 2: sequential graph loading and review detection ---
    neo4j_kwargs = dict(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        neo4j_trust_mode=neo4j_trust_mode,
        neo4j_ca_cert=neo4j_ca_cert,
    )

    from .checkpoint import append_checkpoint, load_checkpoint
    completed_doc_ids = load_checkpoint(output_dir)
    if completed_doc_ids:
        _log.info("Checkpoint: %d doc(s) already written — will skip them.", len(completed_doc_ids))

    review_store = None
    if review_db_path is not None:
        from graphrag_pipeline.review.store import ReviewStore
        review_store = ReviewStore(review_db_path)

    writer: GraphWriter | None = None
    try:
        for doc_summary in doc_summaries:
            if doc_summary.get("skipped"):
                continue  # already_ingested — no graph work needed
            doc_id = doc_summary["doc_id"]
            if doc_id in completed_doc_ids:
                _log.info("Skipping checkpointed doc: %s", doc_id)
                continue

            structure = load_structure_bundle(doc_summary["structure_output"])
            semantic = load_semantic_bundle(doc_summary["semantic_output"])

            if writer is None:
                writer = load_graph(structure, semantic, backend=backend, **neo4j_kwargs)
            else:
                writer.load_structure(structure)
                writer.load_semantic(structure, semantic)

            # Checkpoint after successful graph write — not before.
            # If the write throws, the doc stays uncheckpointed and will retry on re-run.
            append_checkpoint(output_dir, doc_id, doc_summary["input"])
            _log.info("Checkpointed doc %s (%s)", doc_id, Path(doc_summary["input"]).name)

            if review_store is not None:
                from graphrag_pipeline.review.detect import run_detection
                review_result = run_detection(
                    structure, semantic, review_store,
                    structure_bundle_path=str(Path(doc_summary["structure_output"]).resolve()),
                    semantic_bundle_path=str(Path(doc_summary["semantic_output"]).resolve()),
                )
                doc_summary["review_detect"] = review_result
    finally:
        if review_store is not None:
            review_store.close()

    already_ingested_count = sum(1 for d in doc_summaries if d.get("skipped"))
    result: dict[str, Any] = {
        "documents_processed": len(doc_summaries) - already_ingested_count,
        "already_ingested": already_ingested_count,
        "outputs": doc_summaries,
        "backend": backend,
    }
    if backend == "memory" and writer is not None:
        result["writer"] = writer
    return result


def load_saved_pair(structure_path: str | Path, semantic_path: str | Path) -> tuple[StructureBundle, SemanticBundle]:
    return load_structure_bundle(structure_path), load_semantic_bundle(semantic_path)


def _duplicate_counts(values: dict[str, list[str]]) -> dict[str, int]:
    duplicates: dict[str, int] = {}
    for key, items in values.items():
        duplicates[key] = len(items) - len(set(items))
    return duplicates


def resolve_mentions_targeted(
    semantic: SemanticBundle,
    *,
    resolver: EntityResolver | None = None,
    resources_dir: Path | None = None,
) -> tuple[SemanticBundle, dict[str, int]]:
    """Re-run entity resolution for unresolved mentions against the current seed_entities.csv.

    Finds mentions with no EntityResolutionRecord, runs the resolver against them,
    and merges results into the existing bundle. Mentions with existing
    POSSIBLY_REFERS_TO records are NOT retried — they already have a resolution.

    Claim-entity links are rebuilt for affected claims using a simplified heuristic
    (mention-in-span + entity-type compatibility via CLAIM_ENTITY_RELATION_PRECEDENCE)
    since ClaimLinkDraft objects are not stored in the bundle. OCCURRED_AT location
    links are not created in this pass; run full extract-semantic for those.

    Args:
        semantic: The semantic bundle to update (mutated in place).
        resolver: Optional resolver override. Defaults to DictionaryFuzzyResolver()
                  which reads the current seed_entities.csv.

    Returns:
        Tuple of (updated_bundle, stats_dict) with keys:
        unresolved_before, new_resolutions_count, new_entities_count,
        new_claim_links_count, unresolved_after.
    """
    if resolver is None:
        resolver = DictionaryFuzzyResolver(seed_entities=default_seed_entities(resources_dir))

    # Step 1: find unresolved mentions
    already_resolved: set[str] = {r.mention_id for r in semantic.entity_resolutions}
    unresolved = [m for m in semantic.mentions if m.mention_id not in already_resolved]
    unresolved_before = len(unresolved)

    if not unresolved:
        return semantic, {
            "unresolved_before": 0,
            "new_resolutions_count": 0,
            "new_entities_count": 0,
            "new_claim_links_count": 0,
            "unresolved_after": 0,
        }

    # Step 2: run resolver on unresolved mentions only
    new_entities_raw, new_resolutions = resolver.resolve(unresolved)

    # Step 3: merge new EntityResolutionRecords (guarded against double-invocation)
    for rec in new_resolutions:
        if rec.mention_id not in already_resolved:
            semantic.entity_resolutions.append(rec)
            already_resolved.add(rec.mention_id)

    # Step 4: merge new EntityRecords (dedup by entity_id)
    existing_entity_ids: set[str] = {e.entity_id for e in semantic.entities}
    new_entities_added: list[EntityRecord] = []
    for entity in new_entities_raw:
        if entity.entity_id not in existing_entity_ids:
            semantic.entities.append(entity)
            existing_entity_ids.add(entity.entity_id)
            new_entities_added.append(entity)

    if not new_resolutions:
        return semantic, {
            "unresolved_before": unresolved_before,
            "new_resolutions_count": 0,
            "new_entities_count": 0,
            "new_claim_links_count": 0,
            "unresolved_after": unresolved_before,
        }

    # Step 5: rebuild claim-entity links for claims in affected paragraphs
    newly_resolved_ids: set[str] = {r.mention_id for r in new_resolutions}
    affected_paragraphs: set[str] = {
        m.paragraph_id for m in semantic.mentions if m.mention_id in newly_resolved_ids
    }

    resolutions_by_mention: dict[str, Any] = {r.mention_id: r for r in semantic.entity_resolutions}
    entity_lookup: dict[str, EntityRecord] = {e.entity_id: e for e in semantic.entities}

    mentions_by_paragraph: dict[str, list[MentionRecord]] = defaultdict(list)
    for m in semantic.mentions:
        if m.paragraph_id in affected_paragraphs:
            mentions_by_paragraph[m.paragraph_id].append(m)

    existing_entity_keys: set[tuple[str, str, str]] = {
        (lnk.claim_id, lnk.entity_id, lnk.relation_type) for lnk in semantic.claim_entity_links
    }
    new_entity_links: list[ClaimEntityLinkRecord] = []

    for claim in semantic.claims:
        if claim.paragraph_id not in affected_paragraphs:
            continue
        # Use explicit evidence offsets; fall back to unbounded when unavailable
        # (paragraph text is not stored in the bundle without the structure bundle).
        span_start = claim.evidence_start if claim.evidence_start is not None else 0
        span_end = claim.evidence_end if claim.evidence_end is not None else 2**31

        for mention in mentions_by_paragraph.get(claim.paragraph_id, []):
            if mention.mention_id not in newly_resolved_ids:
                continue
            if mention.start_offset < span_start or mention.end_offset > span_end:
                continue
            resolution = resolutions_by_mention.get(mention.mention_id)
            if not resolution:
                continue
            entity = entity_lookup.get(resolution.entity_id)
            if not entity:
                continue

            # Find the first compatible relation type in precedence order
            best_relation: str | None = None
            for relation in CLAIM_ENTITY_RELATION_PRECEDENCE:
                if not entity_type_allowed_for_relation(relation, entity.entity_type):
                    continue
                if claim.claim_type and get_relation_compatibility(claim.claim_type, relation) == "forbidden":
                    continue
                best_relation = relation
                break

            if best_relation is None:
                continue

            ent_key = (claim.claim_id, entity.entity_id, best_relation)
            if ent_key not in existing_entity_keys:
                new_entity_links.append(
                    ClaimEntityLinkRecord(
                        claim_id=claim.claim_id,
                        entity_id=entity.entity_id,
                        relation_type=best_relation,
                    )
                )
                existing_entity_keys.add(ent_key)

    semantic.claim_entity_links.extend(new_entity_links)

    unresolved_after = sum(1 for m in semantic.mentions if m.mention_id not in already_resolved)

    return semantic, {
        "unresolved_before": unresolved_before,
        "new_resolutions_count": len(new_resolutions),
        "new_entities_count": len(new_entities_added),
        "new_claim_links_count": len(new_entity_links),
        "unresolved_after": unresolved_after,
    }


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
