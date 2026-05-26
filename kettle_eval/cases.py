"""Case factory. Wraps every extracted pipeline object as an EvaluationCase.

Pure, post-pipeline pass — not a new phase function. Mirrors the
`(structure, semantic) -> structured output` shape of `quality_report()`.
Keeping case generation outside `extract_semantic` preserves the pure-extraction
contract and the pickle-safety of `extract_document` for ProcessPoolExecutor.
"""
from __future__ import annotations

from typing import Any

from gemynd.core.models import (
    ClaimEntityLinkRecord,
    ClaimRecord,
    EntityRecord,
    EntityResolutionRecord,
    MentionRecord,
    PageRecord,
    ParagraphRecord,
    SemanticBundle,
    StructureBundle,
)

from .ids import (
    make_case_family_id,
    make_case_id,
    make_decision_class_id,
    make_relation_object_id,
    make_resolution_object_id,
)
from .models import EvaluationCase


def cases_from_bundle(
    structure: StructureBundle,
    semantic: SemanticBundle,
    corpus_id: str,
) -> list[EvaluationCase]:
    """Emit one EvaluationCase per extracted object.

    Dispatch:
      - PageRecord            -> page_ocr
      - ClaimRecord           -> claim_extraction
      - MentionRecord         -> mention
      - EntityResolutionRecord -> entity_resolution (decision-level identity)
      - ClaimEntityLinkRecord -> relation
    """
    run = semantic.extraction_run
    run_id = run.run_id
    doc_id = structure.document.doc_id
    model_name = run.ner_model or run.relation_model or "unknown"
    config_hash = run.config_fingerprint or ""

    paragraphs_by_id = {p.paragraph_id: p for p in structure.paragraphs}
    pages_by_id = {p.page_id: p for p in structure.pages}
    claims_by_id = {c.claim_id: c for c in semantic.claims}

    cases: list[EvaluationCase] = []

    for page in structure.pages:
        cases.append(_page_case(
            page=page, doc_id=doc_id, corpus_id=corpus_id, run_id=run_id,
            model_name=model_name, config_hash=config_hash,
        ))

    for claim in semantic.claims:
        paragraph = paragraphs_by_id.get(claim.paragraph_id)
        page_id = paragraph.page_id if paragraph else None
        cases.append(_claim_case(
            claim=claim, page_id=page_id, doc_id=doc_id, corpus_id=corpus_id,
            run_id=run_id, model_name=model_name, config_hash=config_hash,
        ))

    for mention in semantic.mentions:
        paragraph = paragraphs_by_id.get(mention.paragraph_id)
        page_id = paragraph.page_id if paragraph else None
        cases.append(_mention_case(
            mention=mention, page_id=page_id, doc_id=doc_id, corpus_id=corpus_id,
            run_id=run_id, model_name=model_name, config_hash=config_hash,
        ))

    mentions_by_id = {m.mention_id: m for m in semantic.mentions}
    entities_by_id = {e.entity_id: e for e in semantic.entities}
    for resolution in semantic.entity_resolutions:
        mention = mentions_by_id.get(resolution.mention_id)
        entity = entities_by_id.get(resolution.entity_id)
        paragraph_id = mention.paragraph_id if mention else None
        paragraph = paragraphs_by_id.get(paragraph_id) if paragraph_id else None
        page_id = paragraph.page_id if paragraph else None
        cases.append(_resolution_case(
            resolution=resolution, mention=mention, entity=entity,
            paragraph=paragraph, paragraph_id=paragraph_id, page_id=page_id,
            doc_id=doc_id, corpus_id=corpus_id, run_id=run_id,
            model_name=model_name, config_hash=config_hash,
        ))

    for link in semantic.claim_entity_links:
        claim = claims_by_id.get(link.claim_id)
        paragraph_id = claim.paragraph_id if claim else None
        paragraph = paragraphs_by_id.get(paragraph_id) if paragraph_id else None
        page_id = paragraph.page_id if paragraph else None
        cases.append(_relation_case(
            link=link, claim=claim, paragraph_id=paragraph_id, page_id=page_id,
            doc_id=doc_id, corpus_id=corpus_id, run_id=run_id,
            model_name=model_name, config_hash=config_hash,
        ))

    # Useful for risk_features: prevents missing-page-anchor false positives when
    # pages are present but the paragraph index is stale.
    _ = pages_by_id  # keep reference; risk_features.compute_features uses pages

    return cases


def _page_case(
    *, page: PageRecord, doc_id: str, corpus_id: str, run_id: str,
    model_name: str, config_hash: str,
) -> EvaluationCase:
    object_id = page.page_id
    case_type = "page_ocr"
    return EvaluationCase(
        case_id=make_case_id(case_type, object_id, run_id),
        case_family_id=make_case_family_id(case_type, doc_id, page.page_id, object_id),
        # Pages are inherently unique — class id is one-per-page.
        decision_class_id=make_decision_class_id(case_type, page.page_id),
        case_type=case_type,
        corpus_id=corpus_id,
        document_id=doc_id,
        page_id=page.page_id,
        source_span_id=page.page_id,
        object_id=object_id,
        extraction_run_id=run_id,
        pipeline_stage="parse.page",
        model_name=model_name,
        config_hash=config_hash,
        input_text=(page.clean_text or page.raw_ocr_text or "")[:4000],
        model_output={"ocr_confidence": page.ocr_confidence, "page_number": page.page_number},
        payload={"page_number": page.page_number, "ocr_confidence": page.ocr_confidence},
    )


def _claim_case(
    *, claim: ClaimRecord, page_id: str | None, doc_id: str, corpus_id: str,
    run_id: str, model_name: str, config_hash: str,
) -> EvaluationCase:
    case_type = "claim_extraction"
    object_id = claim.claim_id
    return EvaluationCase(
        case_id=make_case_id(case_type, object_id, run_id),
        case_family_id=make_case_family_id(case_type, doc_id, claim.paragraph_id, object_id),
        # Boilerplate claims that repeat verbatim share a class.
        decision_class_id=make_decision_class_id(
            case_type, claim.normalized_sentence, claim.claim_type,
        ),
        case_type=case_type,
        corpus_id=corpus_id,
        document_id=doc_id,
        page_id=page_id,
        source_span_id=claim.paragraph_id,
        object_id=object_id,
        extraction_run_id=run_id,
        pipeline_stage="extract.claim",
        model_name=model_name,
        config_hash=config_hash,
        input_text=claim.source_sentence,
        model_output={
            "claim_type": claim.claim_type,
            "certainty": claim.certainty,
            "extraction_confidence": claim.extraction_confidence,
            "evidence_start": claim.evidence_start,
            "evidence_end": claim.evidence_end,
            "claim_date": claim.claim_date,
        },
        payload={
            "claim_type": claim.claim_type,
            "certainty": claim.certainty,
            "extraction_confidence": claim.extraction_confidence,
        },
    )


def _mention_case(
    *, mention: MentionRecord, page_id: str | None, doc_id: str, corpus_id: str,
    run_id: str, model_name: str, config_hash: str,
) -> EvaluationCase:
    case_type = "mention"
    object_id = mention.mention_id
    return EvaluationCase(
        case_id=make_case_id(case_type, object_id, run_id),
        case_family_id=make_case_family_id(case_type, doc_id, mention.paragraph_id, object_id),
        # Every extraction of the same normalized form belongs to one class.
        decision_class_id=make_decision_class_id(case_type, mention.normalized_form),
        case_type=case_type,
        corpus_id=corpus_id,
        document_id=doc_id,
        page_id=page_id,
        source_span_id=mention.paragraph_id,
        object_id=object_id,
        extraction_run_id=run_id,
        pipeline_stage="extract.mention",
        model_name=model_name,
        config_hash=config_hash,
        input_text=mention.surface_form,
        model_output={
            "surface_form": mention.surface_form,
            "normalized_form": mention.normalized_form,
            "start_offset": mention.start_offset,
            "end_offset": mention.end_offset,
            "detection_confidence": mention.detection_confidence,
            "ocr_suspect": mention.ocr_suspect,
        },
        payload={
            "detection_confidence": mention.detection_confidence,
            "ocr_suspect": mention.ocr_suspect,
        },
    )


_SNIPPET_RADIUS = 60


def _resolution_input_text(
    *,
    resolution: EntityResolutionRecord,
    mention: MentionRecord | None,
    entity: EntityRecord | None,
    paragraph: ParagraphRecord | None,
) -> str:
    """Render a reviewer-readable summary for an entity_resolution case.

    Format:
        <surface_form> -> <entity_name> [<entity_type>] (score=0.78, REFERS_TO)
        Context: ...<snippet around the mention offsets>...
    """
    surface = mention.surface_form if mention else "(no mention)"
    entity_name = entity.name if entity else resolution.entity_id
    entity_type = f" [{entity.entity_type}]" if entity else ""
    head = (
        f"{surface} -> {entity_name}{entity_type} "
        f"(score={resolution.match_score:.2f}, {resolution.relation_type})"
    )
    snippet = ""
    if mention is not None and paragraph is not None:
        text = paragraph.clean_text or paragraph.raw_ocr_text or ""
        if text:
            start = max(0, mention.start_offset - _SNIPPET_RADIUS)
            end = min(len(text), mention.end_offset + _SNIPPET_RADIUS)
            window = text[start:end].strip()
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            snippet = f"Context: {prefix}{window}{suffix}"
    return f"{head}\n{snippet}" if snippet else head


def _resolution_case(
    *, resolution: EntityResolutionRecord,
    mention: MentionRecord | None, entity: EntityRecord | None,
    paragraph: ParagraphRecord | None,
    paragraph_id: str | None,
    page_id: str | None, doc_id: str, corpus_id: str, run_id: str,
    model_name: str, config_hash: str,
) -> EvaluationCase:
    case_type = "entity_resolution"
    object_id = make_resolution_object_id(
        resolution.mention_id,
        resolution.entity_id,
        resolution.relation_type,
        run_id,
    )
    surface_norm = mention.normalized_form if mention else resolution.mention_id
    return EvaluationCase(
        case_id=make_case_id(case_type, object_id, run_id),
        case_family_id=make_case_family_id(case_type, doc_id, paragraph_id, object_id),
        # "Surface X resolves to entity Y as relation R" is the decision class.
        # Same triple across 14 newspaper articles -> one row in the dedup view.
        decision_class_id=make_decision_class_id(
            case_type, surface_norm, resolution.entity_id, resolution.relation_type,
        ),
        case_type=case_type,
        corpus_id=corpus_id,
        document_id=doc_id,
        page_id=page_id,
        source_span_id=paragraph_id,
        object_id=object_id,
        extraction_run_id=run_id,
        pipeline_stage="resolve.entity",
        model_name=model_name,
        config_hash=config_hash,
        input_text=_resolution_input_text(
            resolution=resolution, mention=mention, entity=entity, paragraph=paragraph,
        ),
        model_output={
            "mention_id": resolution.mention_id,
            "entity_id": resolution.entity_id,
            "relation_type": resolution.relation_type,
            "match_score": resolution.match_score,
            "surface_form": mention.surface_form if mention else None,
            "entity_name": entity.name if entity else None,
            "entity_type": entity.entity_type if entity else None,
        },
        payload={
            "mention_id": resolution.mention_id,
            "entity_id": resolution.entity_id,
            "relation_type": resolution.relation_type,
            "match_score": resolution.match_score,
            "surface_form": mention.surface_form if mention else None,
            "entity_name": entity.name if entity else None,
        },
    )


def _relation_case(
    *, link: ClaimEntityLinkRecord, claim: ClaimRecord | None,
    paragraph_id: str | None, page_id: str | None, doc_id: str, corpus_id: str,
    run_id: str, model_name: str, config_hash: str,
) -> EvaluationCase:
    case_type = "relation"
    object_id = make_relation_object_id(link.claim_id, link.entity_id, link.relation_type)
    claim_signature = claim.normalized_sentence if claim else link.claim_id
    return EvaluationCase(
        case_id=make_case_id(case_type, object_id, run_id),
        case_family_id=make_case_family_id(case_type, doc_id, paragraph_id, object_id),
        # Same claim-text + entity + relation across docs -> one class.
        decision_class_id=make_decision_class_id(
            case_type, claim_signature, link.entity_id, link.relation_type,
        ),
        case_type=case_type,
        corpus_id=corpus_id,
        document_id=doc_id,
        page_id=page_id,
        source_span_id=paragraph_id,
        object_id=object_id,
        extraction_run_id=run_id,
        pipeline_stage="link.claim_entity",
        model_name=model_name,
        config_hash=config_hash,
        input_text=claim.source_sentence if claim else "",
        model_output={
            "claim_id": link.claim_id,
            "entity_id": link.entity_id,
            "relation_type": link.relation_type,
        },
        payload={
            "claim_id": link.claim_id,
            "entity_id": link.entity_id,
            "relation_type": link.relation_type,
            "claim_type": claim.claim_type if claim else None,
        },
    )


def build_case_index(cases: list[EvaluationCase]) -> dict[tuple[str, str], str]:
    """Map (case_type, object_id) -> case_id for the detector mapping layer."""
    index: dict[tuple[str, str], str] = {}
    for case in cases:
        index[(case.case_type, case.object_id)] = case.case_id
    return index


def build_object_to_case_index(cases: list[EvaluationCase]) -> dict[str, list[EvaluationCase]]:
    """Reverse index: claim_id / mention_id / entity_id / page_id -> [case].

    Detector proposals reference target IDs without knowing the case_type
    discriminator. This index lets the mapping layer find every case that
    touches a given source object — e.g. an entity_id maps to every
    entity_resolution case referencing that entity.
    """
    out: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        keys = {case.object_id}
        # For entity_resolution, also index by the underlying mention_id and
        # entity_id so detectors targeting either can find this case.
        if case.case_type == "entity_resolution":
            mention_id = case.payload.get("mention_id")
            entity_id = case.payload.get("entity_id")
            if mention_id:
                keys.add(mention_id)
            if entity_id:
                keys.add(entity_id)
        elif case.case_type == "relation":
            claim_id = case.payload.get("claim_id")
            entity_id = case.payload.get("entity_id")
            if claim_id:
                keys.add(claim_id)
            if entity_id:
                keys.add(entity_id)
        for key in keys:
            out.setdefault(key, []).append(case)
    return out
