"""Typed phase functions for the semantic extraction pipeline.

Each function accepts an :class:`ExtractionState` (and optionally injected
collaborators like extractors) and mutates the state in place.  The functions
are called sequentially by :func:`pipeline.extract_semantic`.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from gemynd.core.claim_contract import (
    CLAIM_ENTITY_RELATIONS,
    CLAIM_LOCATION_RELATION,
    entity_type_allowed_for_relation,
    get_preferred_entity_types,
    get_relation_compatibility,
    validate_claim_type,
)
from gemynd.core.claim_validator import is_valid_claim_sentence
from gemynd.core.ids import (
    make_claim_id,
    make_entity_id,
    make_measurement_id,
    make_mention_id,
    make_period_id,
    make_run_id,
    make_year_id,
    stable_hash,
)
from gemynd.core.models import (
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
    YearRecord,
)
from gemynd.ingest.concept_assigner import assign_concepts
from gemynd.ingest.derivation_context import build_derivation_contexts
from gemynd.ingest.event_builder import build_events
from gemynd.ingest.extraction_state import ExtractionState
from gemynd.ingest.extractors.claim_extractor import ClaimExtractor, ClaimLinkDraft
from gemynd.ingest.extractors.measurement_extractor import MeasurementExtractor
from gemynd.ingest.extractors.mention_extractor import MentionExtractor, ResolutionContext
from gemynd.ingest.observation_builder import build_observations
from gemynd.core.resolver import EntityResolver

PERIOD_TYPE_PUBLICATION = "publication_period"

MONTHS = {
    "jan": "01", "january": "01", "feb": "02", "february": "02",
    "mar": "03", "march": "03", "apr": "04", "april": "04",
    "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10", "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

CLAUSE_MARKERS = (",", ";", " and ", " or ")

_WEAK_CONFIDENCE_PENALTY: float = 0.10


# ── Helpers (moved from pipeline.py) ──────────────────────────────────────


def build_extraction_run(
    run_overrides: dict[str, str] | None = None,
    now: datetime | None = None,
) -> ExtractionRunRecord:
    """Create an :class:`ExtractionRunRecord` from optional overrides."""
    now = now or datetime.now(timezone.utc)
    overrides = run_overrides or {}
    run_id = overrides.get("run_id", make_run_id(overrides.get("ocr_engine", "unknown"), now))
    return ExtractionRunRecord(
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

    preferred = get_preferred_entity_types(claim_type) if claim_type else []
    if preferred and len(relation_filtered) > 1:
        def _preference_rank(row: tuple[MentionRecord, EntityRecord]) -> int:
            et = row[1].entity_type
            try:
                return preferred.index(et)
            except ValueError:
                return len(preferred)

        relation_filtered_sorted = sorted(relation_filtered, key=_preference_rank)
        if relation_filtered_sorted[0][1].entity_type != relation_filtered[0][1].entity_type:
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

    if not exact_match and not normalized_match and not normalized_diagnostic and not exact_diagnostic:
        sentence_normalized = claim.normalized_sentence or ""
        sentence_candidates = [
            row for row in resolved_in_span
            if row[0].normalized_form in sentence_normalized
        ]
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
    """No-op: compatibility penalty is applied at the claim level downstream."""
    return entity


# ── Phase functions ───────────────────────────────────────────────────────


def extract_paragraphs(
    state: ExtractionState,
    claim_extractor: ClaimExtractor,
    measurement_extractor: MeasurementExtractor,
    mention_extractor: MentionExtractor,
) -> None:
    """Phase 2: loop over paragraphs, extract claims, mentions, measurements."""
    state.paragraph_texts = {
        p.paragraph_id: (p.clean_text or p.raw_ocr_text)
        for p in state.structure.paragraphs
    }

    for paragraph in state.structure.paragraphs:
        paragraph_text = state.paragraph_texts[paragraph.paragraph_id]
        claim_drafts = claim_extractor.extract(paragraph_text)
        mention_drafts = mention_extractor.extract(paragraph_text)

        for draft in mention_drafts:
            mention_id = make_mention_id(
                state.extraction_run.run_id,
                paragraph.paragraph_id,
                draft.start_offset,
                draft.end_offset,
                draft.surface_form,
            )
            mention = MentionRecord(
                mention_id=mention_id,
                run_id=state.extraction_run.run_id,
                paragraph_id=paragraph.paragraph_id,
                surface_form=draft.surface_form,
                normalized_form=draft.normalized_form,
                start_offset=draft.start_offset,
                end_offset=draft.end_offset,
                detection_confidence=draft.detection_confidence,
                ocr_suspect=bool(draft.ocr_flags),
            )
            state.mentions.append(mention)
            state.mentions_by_paragraph[paragraph.paragraph_id].append(mention)

        for idx, draft in enumerate(claim_drafts, start=1):
            valid, _reason = is_valid_claim_sentence(draft.source_sentence)
            if not valid:
                continue
            state.claim_counter += 1
            claim_id = make_claim_id(
                state.extraction_run.run_id,
                paragraph.paragraph_id,
                state.claim_counter + idx,
                draft.normalized_sentence,
            )
            claim_date = draft.claim_date or _infer_claim_date(
                draft.source_sentence,
                fallback_year=state.structure.document.report_year,
            )
            claim = ClaimRecord(
                claim_id=claim_id,
                run_id=state.extraction_run.run_id,
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
            state.claims.append(claim)
            state.claims_by_paragraph[paragraph.paragraph_id].append(claim)
            state.claim_links_by_claim[claim_id] = list(draft.claim_links)

            measurement_drafts = measurement_extractor.extract(draft)
            for measurement_idx, measurement_draft in enumerate(measurement_drafts, start=1):
                state.measurement_counter += 1
                measurement = MeasurementRecord(
                    measurement_id=make_measurement_id(
                        state.extraction_run.run_id,
                        claim_id,
                        state.measurement_counter + measurement_idx,
                        measurement_draft.name,
                        measurement_draft.raw_value,
                    ),
                    claim_id=claim_id,
                    run_id=state.extraction_run.run_id,
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
                state.measurements.append(measurement)


def resolve_entities(state: ExtractionState, resolver: EntityResolver) -> None:
    """Phase 3: run entity resolution, build entity_lookup and resolutions_by_mention."""
    resolution_contexts: dict[str, ResolutionContext] = {
        p.paragraph_id: ResolutionContext(
            paragraph_id=p.paragraph_id,
            claim_types=sorted({c.claim_type for c in state.claims_by_paragraph.get(p.paragraph_id, [])}),
        )
        for p in state.structure.paragraphs
    }

    state.entities, state.entity_resolutions = resolver.resolve(
        state.mentions, contexts=resolution_contexts,
    )
    state.entity_lookup = {e.entity_id: e for e in state.entities}
    state.resolutions_by_mention = {r.mention_id: r for r in state.entity_resolutions}


def create_domain_anchor(state: ExtractionState) -> None:
    """Phase 4: create document-level anchor entity if domain config matches.

    No-ops when ``config.domain_anchor`` is ``None``.
    """
    anchor_cfg = state.config.domain_anchor
    if not anchor_cfg:
        return

    raw_kw = anchor_cfg.get("title_keywords") or anchor_cfg.get("title_keyword") or []
    keywords = [raw_kw] if isinstance(raw_kw, str) else list(raw_kw)
    title_lower = state.structure.document.title.lower()
    if not any(kw.lower() in title_lower for kw in keywords):
        return

    entity_type = anchor_cfg["entity_type"]
    name = anchor_cfg["name"]
    norm = anchor_cfg.get("normalized_form", name.lower())
    props = anchor_cfg.get("properties", {})
    doc_anchor_id = make_entity_id(entity_type, norm)

    if doc_anchor_id not in state.entity_lookup:
        anchor_entity = EntityRecord(
            entity_id=doc_anchor_id,
            entity_type=entity_type,
            name=name,
            normalized_form=norm,
            properties=props,
        )
        state.register_entity(anchor_entity)

    state.doc_anchor_id = doc_anchor_id
    state.document_refuge_links.append(
        DocumentRefugeLinkRecord(doc_id=state.structure.document.doc_id, refuge_id=doc_anchor_id)
    )


def resolve_claim_links(state: ExtractionState) -> None:
    """Phase 5: resolve claim links to entities with deduplication."""
    seen_claim_entity: set[tuple[str, str, str]] = set()
    seen_claim_location: set[tuple[str, str]] = set()

    for paragraph_id, paragraph_claims in state.claims_by_paragraph.items():
        paragraph_mentions = state.mentions_by_paragraph.get(paragraph_id, [])
        paragraph_text = state.paragraph_texts.get(paragraph_id, "")
        for claim in paragraph_claims:
            for claim_link in state.claim_links_by_claim.get(claim.claim_id, []):
                entity, diagnostic = _resolve_claim_link(
                    claim,
                    claim_link,
                    paragraph_mentions,
                    state.resolutions_by_mention,
                    state.entity_lookup,
                    paragraph_text,
                )
                if diagnostic:
                    state.claim_link_diagnostics.append(diagnostic)
                    continue
                if not entity:
                    continue
                if claim_link.relation_type == CLAIM_LOCATION_RELATION:
                    key = (claim.claim_id, entity.entity_id)
                    if key not in seen_claim_location:
                        state.claim_location_links.append(
                            ClaimLocationLinkRecord(claim_id=claim.claim_id, entity_id=entity.entity_id)
                        )
                        seen_claim_location.add(key)
                    continue
                if claim_link.relation_type not in CLAIM_ENTITY_RELATIONS:
                    continue
                if claim.claim_type and get_relation_compatibility(claim.claim_type, claim_link.relation_type) == "weak":
                    claim.extraction_confidence = max(
                        0.0, claim.extraction_confidence - _WEAK_CONFIDENCE_PENALTY
                    )
                    state.claim_link_diagnostics.append(
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
                key3 = (claim.claim_id, entity.entity_id, claim_link.relation_type)
                if key3 not in seen_claim_entity:
                    state.claim_entity_links.append(
                        ClaimEntityLinkRecord(
                            claim_id=claim.claim_id,
                            entity_id=entity.entity_id,
                            relation_type=claim_link.relation_type,
                        )
                    )
                    seen_claim_entity.add(key3)


def create_period_entity(state: ExtractionState) -> None:
    """Phase 6: create Period entity from document dates and link claims."""
    doc = state.structure.document
    if not doc.date_start and not doc.date_end:
        return

    period_entity_id = make_period_id(
        doc.date_start, doc.date_end, source_title=doc.title,
    )
    period_entity = EntityRecord(
        entity_id=period_entity_id,
        entity_type="Period",
        name=f"{doc.date_start or '?'} to {doc.date_end or '?'}",
        normalized_form=f"{doc.date_start or '?'} to {doc.date_end or '?'}",
        properties={
            "period_id": period_entity_id,
            "start_date": doc.date_start,
            "end_date": doc.date_end,
            "source_title": doc.title,
            "period_type": PERIOD_TYPE_PUBLICATION,
        },
    )
    if period_entity_id not in state.entity_lookup:
        state.register_entity(period_entity)

    state.document_period_links.append(
        DocumentPeriodLinkRecord(doc_id=doc.doc_id, period_id=period_entity_id)
    )
    for claim in state.claims:
        state.claim_period_links.append(
            ClaimPeriodLinkRecord(claim_id=claim.claim_id, period_id=period_entity_id)
        )


def build_derivation_phase(state: ExtractionState) -> None:
    """Phase 7: build shared per-claim derivation contexts."""
    state.derivation_contexts = build_derivation_contexts(
        claims=state.claims,
        measurements=state.measurements,
        claim_entity_links=state.claim_entity_links,
        claim_location_links=state.claim_location_links,
        claim_period_links=state.claim_period_links,
        entity_lookup=state.entity_lookup,
        run_id=state.extraction_run.run_id,
        report_year=state.structure.document.report_year,
        doc_date_start=state.structure.document.date_start,
        doc_date_end=state.structure.document.date_end,
        registry=state.config.derivation_registry or None,
        year_validation_cfg=state.config.year_validation,
    )


def build_observations_phase(state: ExtractionState) -> None:
    """Phase 8: build observations from eligible claims."""
    observations, years, obs_measurement_links, _ = build_observations(
        claims=state.claims,
        measurements=state.measurements,
        claim_entity_links=state.claim_entity_links,
        claim_location_links=state.claim_location_links,
        claim_period_links=state.claim_period_links,
        entity_lookup=state.entity_lookup,
        run_id=state.extraction_run.run_id,
        report_year=state.structure.document.report_year,
        _contexts=state.derivation_contexts,
    )
    state.observations = observations
    state.years = years
    state.obs_measurement_links = obs_measurement_links


def build_events_phase(state: ExtractionState) -> None:
    """Phase 9: build events from event-eligible claims and merge year nodes."""
    events, event_obs_links, event_meas_links, event_years = build_events(
        claims=state.claims,
        measurements=state.measurements,
        claim_entity_links=state.claim_entity_links,
        claim_location_links=state.claim_location_links,
        claim_period_links=state.claim_period_links,
        entity_lookup=state.entity_lookup,
        observations=state.observations,
        run_id=state.extraction_run.run_id,
        report_year=state.structure.document.report_year,
        _contexts=state.derivation_contexts,
    )
    state.events = events
    state.event_obs_links = event_obs_links
    state.event_meas_links = event_meas_links

    existing_year_ids = {y.year_id for y in state.years}
    for yr in event_years:
        if yr.year_id not in existing_year_ids:
            state.years.append(yr)
            existing_year_ids.add(yr.year_id)


def create_year_entities(state: ExtractionState) -> None:
    """Phase 10: create document-level year entity and link."""
    report_year = state.structure.document.report_year
    if not report_year:
        return

    doc_year_id = make_year_id(report_year)
    if not any(y.year_id == doc_year_id for y in state.years):
        state.years.append(
            YearRecord(year_id=doc_year_id, year=report_year, year_label=str(report_year))
        )
    state.document_year_links.append(
        DocumentYearLinkRecord(doc_id=state.structure.document.doc_id, year_id=doc_year_id)
    )


def create_place_refuge_links(state: ExtractionState) -> None:
    """Phase 11: link Place entities to the document anchor refuge."""
    if not state.doc_anchor_id:
        return
    for entity in state.entities:
        if entity.entity_type == "Place":
            state.place_refuge_links.append(
                PlaceRefugeLinkRecord(place_id=entity.entity_id, refuge_id=state.doc_anchor_id)
            )


def assign_concepts_phase(state: ExtractionState) -> None:
    """Phase 12: assign concepts to claims.

    Derives the allowed concept ID set from ``config.concept_rules`` so that
    new domains automatically use their own concept vocabulary.
    """
    if state.config.concept_rules:
        allowed_concept_ids = {rule[0] for rule in state.config.concept_rules}
    else:
        # Fallback for configs without concept_rules
        allowed_concept_ids = {
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

    for claim in state.claims:
        for assignment in assign_concepts(claim, config=state.config):
            if assignment.concept_id in allowed_concept_ids:
                state.claim_concept_links.append(
                    ClaimConceptLinkRecord(
                        claim_id=claim.claim_id,
                        concept_id=assignment.concept_id,
                        confidence=assignment.confidence,
                        matched_rule=assignment.matched_rule,
                    )
                )
