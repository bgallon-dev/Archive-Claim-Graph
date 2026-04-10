from __future__ import annotations

import logging
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extractors import HybridClaimExtractor, RuleBasedMeasurementExtractor, RuleBasedMentionExtractor
from .extractors.claim_extractor import ClaimExtractor
from .extractors.measurement_extractor import MeasurementExtractor
from .extractors.mention_extractor import MentionExtractor, ResolutionContext
from .graph.writer import GraphWriter, InMemoryGraphWriter, Neo4jGraphWriter
from gemynd.shared.io_utils import load_json, load_semantic_bundle, load_structure_bundle, save_json, save_semantic_bundle, save_structure_bundle
from gemynd.core.models import (
    ClaimEntityLinkRecord,
    ClaimRecord,
    EntityRecord,
    MentionRecord,
    SemanticBundle,
    StructureBundle,
)
from gemynd.core.claim_contract import (
    CLAIM_ENTITY_RELATIONS,
    CLAIM_ENTITY_RELATION_PRECEDENCE,
    entity_type_allowed_for_relation,
    get_relation_compatibility,
)
from gemynd.core.resolver import DictionaryFuzzyResolver, EntityResolver, default_seed_entities
from gemynd.core.domain_config import DomainConfig, load_domain_config
from .spelling_review import build_spelling_review_queue as _build_spelling_review_queue
from .source_parser import parse_source_file, parse_source_payload
from .extraction_result import (
    ExtractionResult,
    PersistResult,
    SensitivityGate,
)
from .sensitivity_gate import DefaultSensitivityGate

_log = logging.getLogger(__name__)


CLAUSE_MARKERS = (",", ";", " and ", " or ")


def parse_source(input_path: str | Path | dict[str, Any]) -> StructureBundle:
    if isinstance(input_path, dict):
        return parse_source_payload(input_path, source_file=None)
    path = Path(input_path)
    if path.suffix.lower() == ".json":
        return parse_source_file(path)
    payload = load_json(path)
    return parse_source_payload(payload, source_file=str(path))


def _is_short_simple_sentence(sentence: str) -> bool:
    normalized = f" {sentence.strip().lower()} "
    token_count = len(re.findall(r"\b\w+\b", sentence))
    return token_count < 25 and not any(marker in normalized for marker in CLAUSE_MARKERS)


def extract_semantic(
    structure: StructureBundle,
    *,
    claim_extractor: ClaimExtractor | None = None,
    measurement_extractor: MeasurementExtractor | None = None,
    mention_extractor: MentionExtractor | None = None,
    resolver: EntityResolver | None = None,
    run_overrides: dict[str, str] | None = None,
    resources_dir: Path | None = None,
    no_llm: bool = False,
    token_logger: Any | None = None,
) -> SemanticBundle:
    from .extraction_state import ExtractionState
    from .phases import (
        assign_concepts_phase,
        build_derivation_phase,
        build_events_phase,
        build_extraction_run,
        build_observations_phase,
        create_domain_anchor,
        create_period_entity,
        create_place_refuge_links,
        create_year_entities,
        extract_paragraphs,
        resolve_claim_links,
        resolve_entities,
    )

    _config = load_domain_config(resources_dir)

    # Set institution_id from domain config if not already set on the document.
    if not structure.document.institution_id and _config.institution_id:
        structure.document.institution_id = _config.institution_id

    # -- Extractor initialisation (kept here: couples to constructor args) --
    if claim_extractor is None:
        if no_llm:
            from gemynd.ingest.extractors.claim_extractor import (
                LLMClaimExtractor,
                NullLLMAdapter,
            )
            claim_extractor = HybridClaimExtractor(
                llm_extractor=LLMClaimExtractor(NullLLMAdapter()),
                resources_dir=resources_dir,
                config=_config,
            )
        else:
            claim_extractor = HybridClaimExtractor(resources_dir=resources_dir, config=_config, token_logger=token_logger)
    measurement_extractor = measurement_extractor or RuleBasedMeasurementExtractor(resources_dir=resources_dir, config=_config)
    mention_extractor = mention_extractor or RuleBasedMentionExtractor(resources_dir=resources_dir, config=_config)
    resolver = resolver or DictionaryFuzzyResolver(resources_dir=resources_dir, config=_config)

    # -- Build state and run phases --
    state = ExtractionState(
        structure=structure,
        config=_config,
        extraction_run=build_extraction_run(run_overrides),
    )

    extract_paragraphs(state, claim_extractor, measurement_extractor, mention_extractor)
    resolve_entities(state, resolver)
    create_domain_anchor(state)
    resolve_claim_links(state)
    create_period_entity(state)
    build_derivation_phase(state)
    build_observations_phase(state)
    build_events_phase(state)
    create_year_entities(state)
    create_place_refuge_links(state)
    assign_concepts_phase(state)

    return state.to_semantic_bundle()


def load_graph(
    structure: StructureBundle,
    semantic: SemanticBundle,
    *,
    config: DomainConfig,
    backend: str = "memory",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str = "neo4j",
    neo4j_trust_mode: str | None = None,
    neo4j_ca_cert: str | None = None,
) -> GraphWriter:
    entity_labels = config.entity_labels
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
            entity_labels=entity_labels,
        )
    else:
        writer = InMemoryGraphWriter(entity_labels=entity_labels)

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
    from gemynd.core.ids import stable_hash
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
    from gemynd.retrieval.executor import Neo4jQueryExecutor
    from gemynd.core.graph.cypher import GRAPH_ENTITY_FETCH_QUERY
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


def extract_document(
    input_item: str,
    domain_dir_str: str | None = None,
    run_id: str | None = None,
    supplementary_entity_rows: list[EntityRecord] | None = None,
    no_llm: bool = False,
) -> ExtractionResult:
    """Pure extraction: parse source and run semantic extraction.

    No side effects (no file I/O, no sensitivity gate).  Module-scope and
    pickle-safe so it can be submitted to :class:`ProcessPoolExecutor`.
    """
    resources_dir = Path(domain_dir_str) if domain_dir_str else None
    structure = parse_source(input_item)
    resolver: EntityResolver | None = None
    if supplementary_entity_rows:
        resolver = DictionaryFuzzyResolver(
            seed_entities=default_seed_entities(resources_dir),
            supplementary_candidates=supplementary_entity_rows,
            resources_dir=resources_dir,
        )
    semantic = extract_semantic(
        structure,
        resolver=resolver,
        resources_dir=resources_dir,
        run_overrides={"run_id": run_id} if run_id else None,
        no_llm=no_llm,
    )
    return ExtractionResult(
        structure=structure,
        semantic=semantic,
        doc_id=structure.document.doc_id,
        run_id=semantic.extraction_run.run_id,
        input_path=input_item,
    )


def persist_document(
    result: ExtractionResult,
    out_dir: str,
    review_out_dir: str | None = None,
    sensitivity_gate: SensitivityGate | None = None,
) -> PersistResult:
    """Effectful persistence: sensitivity gate, file I/O, quality report.

    The *sensitivity_gate* parameter allows institutions to inject custom
    screening logic.  When ``None``, :class:`DefaultSensitivityGate` is used.
    """
    gate = sensitivity_gate or DefaultSensitivityGate()
    gate_result = gate(result.structure, result.semantic)
    semantic = gate_result.semantic

    stem = Path(result.input_path).stem
    output_dir = Path(out_dir)
    structure_path = output_dir / f"{stem}.structure.json"
    semantic_path = output_dir / f"{stem}.semantic.json"
    save_structure_bundle(structure_path, result.structure)
    save_semantic_bundle(semantic_path, semantic)

    metrics = quality_report(result.structure, semantic)

    spelling_review_output: str | None = None
    spelling_review_issue_count = 0
    if review_out_dir is not None:
        review_rows = build_spelling_review_queue(result.structure, semantic)
        review_path = Path(review_out_dir) / f"{stem}.spelling_review.json"
        save_json(review_path, review_rows)
        spelling_review_output = str(review_path)
        spelling_review_issue_count = len(review_rows)

    return PersistResult(
        input_path=result.input_path,
        doc_id=result.doc_id,
        run_id=result.run_id,
        structure_output=str(structure_path),
        semantic_output=str(semantic_path),
        quality=metrics,
        quarantine_summary=gate_result.quarantine_summary,
        spelling_review_output=spelling_review_output,
        spelling_review_issue_count=spelling_review_issue_count,
    )


def _process_single_document(
    input_item: str,
    out_dir: str,
    review_out_dir: str | None,
    domain_dir_str: str | None = None,
    run_id: str | None = None,
    supplementary_entity_rows: list[EntityRecord] | None = None,
    no_llm: bool = False,
) -> dict[str, Any]:
    """Worker function for parallel document processing.

    Thin wrapper around :func:`extract_document` + :func:`persist_document`.
    Must be defined at module scope so ProcessPoolExecutor can pickle it on Windows.
    """
    result = extract_document(
        input_item, domain_dir_str, run_id, supplementary_entity_rows, no_llm,
    )
    persist = persist_document(result, out_dir, review_out_dir)
    return persist.to_summary_dict()


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
    no_llm: bool = False,
) -> dict[str, Any]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_output_dir_str = str(Path(review_out_dir)) if review_out_dir is not None else None
    if review_output_dir_str is not None:
        Path(review_output_dir_str).mkdir(parents=True, exist_ok=True)
    domain_dir_str = str(resources_dir) if resources_dir is not None else None
    _run_config = load_domain_config(resources_dir)

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
            from gemynd.retrieval.executor import Neo4jQueryExecutor
            from gemynd.core.graph.cypher import DUPLICATE_HASH_CHECK_QUERY
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

    # --- Phase 1 / Phase 2 setup ---
    effective_workers = min(max(workers, 1), len(str_inputs)) if str_inputs else 1

    # Pre-generate a unique run_id per document in the parent process so that
    # parallel workers never collide even when they start within the same second.
    _batch_now = datetime.now(timezone.utc)
    doc_run_ids = {item: _make_doc_run_id(item, _batch_now) for item in str_inputs}

    _supp = supplementary_entities or None  # None when empty → workers use seed-only resolution

    neo4j_kwargs = dict(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        neo4j_trust_mode=neo4j_trust_mode,
        neo4j_ca_cert=neo4j_ca_cert,
    )

    from .checkpoint import append_checkpoint, load_checkpoint, load_checkpoint_inputs
    completed_doc_ids = load_checkpoint(output_dir)
    completed_inputs = load_checkpoint_inputs(output_dir)
    if completed_doc_ids:
        _log.info("Checkpoint: %d doc(s) already written — will skip them.", len(completed_doc_ids))

    review_store = None
    if review_db_path is not None:
        from gemynd.review.store import ReviewStore
        review_store = ReviewStore(review_db_path)

    # Rolling counter of entity_id → cumulative REFERS_TO resolution count across
    # all documents processed so far in this run (document-level frequency prior).
    document_entity_counts: Counter[str] = Counter()

    writer: GraphWriter | None = None

    # -- Per-doc graph write helper (shared by single-worker and multi-worker paths) --
    def _write_doc_to_graph(doc_summary: dict[str, Any]) -> None:
        nonlocal writer
        structure = load_structure_bundle(doc_summary["structure_output"])
        semantic = load_semantic_bundle(doc_summary["semantic_output"])

        # A3: Re-resolve with frequency prior if we have data from prior docs.
        if document_entity_counts:
            domain_dir = Path(resources_dir) if resources_dir is not None else None
            _supp_for_prior = supplementary_entities if supplementary_entities else None
            freq_resolver = DictionaryFuzzyResolver(
                seed_entities=default_seed_entities(domain_dir),
                supplementary_candidates=_supp_for_prior,
                resources_dir=domain_dir,
            )
            _, new_resolutions = freq_resolver.resolve(
                semantic.mentions,
                contexts=_rebuild_resolution_contexts(semantic),
                document_entity_counts=document_entity_counts,
            )
            semantic.entity_resolutions = new_resolutions
            save_semantic_bundle(doc_summary["semantic_output"], semantic)

        # Update the rolling counter with this document's REFERS_TO resolutions.
        for resolution in semantic.entity_resolutions:
            if resolution.relation_type == "REFERS_TO":
                document_entity_counts[resolution.entity_id] += 1

        if writer is None:
            writer = load_graph(
                structure, semantic,
                config=_run_config,
                backend=backend,
                **neo4j_kwargs,
            )
        else:
            writer.load_structure(structure)
            writer.load_semantic(structure, semantic)

        # Checkpoint after successful graph write — not before.
        # If the write throws, the doc stays uncheckpointed and will retry on re-run.
        append_checkpoint(output_dir, doc_summary["doc_id"], doc_summary["input"])
        _log.info("Checkpointed doc %s (%s)", doc_summary["doc_id"], Path(doc_summary["input"]).name)

        if review_store is not None:
            from gemynd.review.detect import run_detection
            review_result = run_detection(
                structure, semantic, review_store,
                structure_bundle_path=str(Path(doc_summary["structure_output"]).resolve()),
                semantic_bundle_path=str(Path(doc_summary["semantic_output"]).resolve()),
            )
            doc_summary["review_detect"] = review_result

    try:
        if effective_workers <= 1:
            # --- Single-worker: interleaved extract + graph write per document ---
            # Sort for deterministic frequency-prior accumulation order.
            str_inputs_ordered = sorted(str_inputs)
            for i, input_item in enumerate(str_inputs_ordered, 1):
                if input_item in completed_inputs:
                    _log.info("Skipping checkpointed doc: %s", Path(input_item).stem)
                    # Update frequency prior from saved semantic bundle so
                    # subsequent docs still benefit from prior resolutions.
                    sem_path = output_dir / f"{Path(input_item).stem}.semantic.json"
                    if sem_path.exists():
                        _sem = load_semantic_bundle(sem_path)
                        for res in _sem.entity_resolutions:
                            if res.relation_type == "REFERS_TO":
                                document_entity_counts[res.entity_id] += 1
                    continue

                extraction = extract_document(
                    input_item, domain_dir_str, doc_run_ids[input_item],
                    _supp, no_llm=no_llm,
                )
                persist = persist_document(
                    extraction, str(output_dir), review_output_dir_str,
                )
                summary = persist.to_summary_dict()
                _log.info("[%d/%d] Processed: %s", i, len(str_inputs), Path(input_item).stem)

                _write_doc_to_graph(summary)
                doc_summaries.append(summary)

        else:
            # --- Multi-worker: parallel extraction (Phase 1) ---
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
                        no_llm,
                    ): input_item
                    for input_item in str_inputs
                }
                for i, future in enumerate(as_completed(future_to_input), 1):
                    input_item = future_to_input[future]
                    summary = future.result()  # re-raises on worker exception
                    _log.info("[%d/%d] Processed: %s", i, len(str_inputs), Path(input_item).stem)
                    doc_summaries.append(summary)

            # --- Multi-worker: sequential graph write (Phase 2) ---
            # Restore deterministic output order (as_completed is unordered).
            doc_summaries.sort(key=lambda d: d["input"])

            for doc_summary in doc_summaries:
                if doc_summary.get("skipped"):
                    continue  # already_ingested — no graph work needed
                doc_id = doc_summary["doc_id"]
                if doc_id in completed_doc_ids:
                    _log.info("Skipping checkpointed doc: %s", doc_id)
                    continue
                _write_doc_to_graph(doc_summary)
    finally:
        if review_store is not None:
            review_store.close()

    # Restore deterministic output order for the final result.
    doc_summaries.sort(key=lambda d: d["input"])

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


def _rebuild_resolution_contexts(semantic: SemanticBundle) -> dict[str, ResolutionContext]:
    """Reconstruct per-paragraph ResolutionContext objects from a loaded SemanticBundle.

    Used in the Phase 2 frequency-prior re-resolution loop of run_e2e() to avoid
    storing or re-extracting claim data when re-resolving with document_entity_counts.
    """
    from collections import defaultdict
    claim_types_by_para: dict[str, set[str]] = defaultdict(set)
    for claim in semantic.claims:
        claim_types_by_para[claim.paragraph_id].add(claim.claim_type)
    return {
        pid: ResolutionContext(paragraph_id=pid, claim_types=sorted(types))
        for pid, types in claim_types_by_para.items()
    }


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
        resolver = DictionaryFuzzyResolver(
            seed_entities=default_seed_entities(resources_dir),
            resources_dir=resources_dir,
        )

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


