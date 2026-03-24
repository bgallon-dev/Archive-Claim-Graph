from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .env import load_dotenv
from .io_utils import load_semantic_bundle, load_structure_bundle, save_json, save_rows_csv, save_semantic_bundle, save_structure_bundle

from .pipeline import build_spelling_review_queue, extract_semantic, load_graph, parse_source, quality_report, resolve_mentions_targeted, run_e2e


def _write_review_output(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    if target.suffix.lower() == ".csv":
        save_rows_csv(target, rows)
    else:
        save_json(target, rows)


def _add_neo4j_args(p: argparse.ArgumentParser) -> None:
    """Add Neo4j connection arguments, defaulting to environment variables."""
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    p.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    p.add_argument("--neo4j-trust", choices=["system", "all", "custom"], default=os.environ.get("NEO4J_TRUST"))
    p.add_argument("--neo4j-ca-cert", default=os.environ.get("NEO4J_CA_CERT") or None)


def _add_domain_args(p: argparse.ArgumentParser) -> None:
    """Add optional domain directory argument."""
    p.add_argument(
        "--domain-dir",
        default=None,
        help="Path to a domain resources directory containing domain_profile.yaml. "
             "Defaults to the built-in resources/ directory.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphrag", description="Claim-centric narrative report graph pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest-structure", help="Parse source OCR report into structure bundle JSON.")
    ingest_parser.add_argument("--input", required=True, help="Path to source report JSON.")
    ingest_parser.add_argument("--output", required=True, help="Path to structure bundle JSON output.")
    ingest_parser.add_argument(
        "--access-level",
        choices=["public", "staff_only", "restricted", "indigenous_restricted"],
        default="public",
        help="Sensitivity classification for this document (default: public).",
    )
    ingest_parser.add_argument(
        "--institution-id",
        default="turnbull",
        help="Institution identifier for multi-tenant isolation (default: turnbull).",
    )
    ingest_parser.add_argument(
        "--donor-restricted",
        action="store_true",
        default=False,
        help="Flag document as carrying donor reproduction restrictions.",
    )
    ingest_parser.add_argument(
        "--indigenous-consultation-confirmed",
        action="store_true",
        default=False,
        help="Required for indigenous_restricted documents — confirms tribal consultation has occurred.",
    )

    semantic_parser = subparsers.add_parser("extract-semantic", help="Extract claims/mentions/measurements from structure bundle.")
    semantic_parser.add_argument("--structure", required=True, help="Path to structure bundle JSON.")
    semantic_parser.add_argument("--output", required=True, help="Path to semantic bundle JSON output.")
    semantic_parser.add_argument("--ocr-engine", default="unknown", help="OCR engine name for ExtractionRun metadata.")
    semantic_parser.add_argument("--ocr-version", default="unknown", help="OCR version for ExtractionRun metadata.")
    _add_domain_args(semantic_parser)

    graph_parser = subparsers.add_parser("load-graph", help="Load structure and semantic bundles into graph backend.")
    graph_parser.add_argument("--structure", default=None, help="Path to structure bundle JSON.")
    graph_parser.add_argument("--semantic", default=None, help="Path to semantic bundle JSON.")
    graph_parser.add_argument("--input-dir", default=None, help="Directory containing *.structure.json / *.semantic.json pairs.")
    graph_parser.add_argument("--backend", choices=["memory", "neo4j"], default="memory")
    graph_parser.add_argument("--workers", type=int, default=1,
                              help="Parallel threads for loading bundles from disk (default: 1). Use 0 to auto-detect. Only applies with --input-dir.")
    _add_neo4j_args(graph_parser)

    e2e_parser = subparsers.add_parser("run-e2e", help="Run parse + extract + load over multiple reports.")
    e2e_parser.add_argument("--inputs", nargs="+", required=True, help="Paths to source report JSON files or a directory.")
    e2e_parser.add_argument("--out-dir", required=True, help="Output directory for structure/semantic bundles.")
    e2e_parser.add_argument("--review-out-dir", default=None, help="Optional directory to write spelling review queue JSON files.")
    e2e_parser.add_argument("--review-db", default=None, help="Optional path to review SQLite database for anti-pattern detection.")
    e2e_parser.add_argument("--backend", choices=["memory", "neo4j"], default="memory")
    e2e_parser.add_argument("--workers", type=int, default=1,
                            help="Number of parallel worker processes for extraction (default: 1). Use 0 to auto-detect CPU count.")
    _add_neo4j_args(e2e_parser)
    _add_domain_args(e2e_parser)

    report_parser = subparsers.add_parser("quality-report", help="Compute quality metrics for one structure/semantic pair.")
    report_parser.add_argument("--structure", required=True, help="Path to structure bundle JSON.")
    report_parser.add_argument("--semantic", required=True, help="Path to semantic bundle JSON.")
    report_parser.add_argument("--output", help="Optional path to write quality report JSON.")

    spelling_parser = subparsers.add_parser("spelling-review-report", help="Build a claim-linked spelling review queue.")
    spelling_parser.add_argument("--structure", required=True, help="Path to structure bundle JSON.")
    spelling_parser.add_argument("--semantic", required=True, help="Path to semantic bundle JSON.")
    spelling_parser.add_argument("--output", help="Optional path to write review queue JSON or CSV.")

    # -- Review subsystem commands ------------------------------------------
    detect_parser = subparsers.add_parser("review-detect", help="Run anti-pattern detectors and populate the review store.")
    detect_parser.add_argument("--structure", required=True, help="Path to structure bundle JSON.")
    detect_parser.add_argument("--semantic", required=True, help="Path to semantic bundle JSON.")
    detect_parser.add_argument("--review-db", required=True, help="Path to review SQLite database.")

    serve_parser = subparsers.add_parser("review-serve", help="Launch the local review web application.")
    serve_parser.add_argument("--review-db", required=True, help="Path to review SQLite database.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1).")
    serve_parser.add_argument("--port", type=int, default=8787, help="Port to bind (default 8787).")

    export_parser = subparsers.add_parser("review-export", help="Export review proposals, patches, or revision history.")
    export_parser.add_argument("--review-db", required=True, help="Path to review SQLite database.")
    export_parser.add_argument("--output", required=True, help="Output file path (JSON or CSV).")
    export_parser.add_argument("--mode", choices=["proposals", "patches", "revisions"], default="proposals",
                               help="Export mode: proposals (full), patches (accepted), or revisions (single proposal).")
    export_parser.add_argument("--status", default=None, help="Filter by proposal status.")
    export_parser.add_argument("--snapshot-id", default=None, help="Filter by snapshot ID.")
    export_parser.add_argument("--proposal-id", default=None, help="Proposal ID (required for revisions mode).")

    # -- Retrieval subsystem commands ---------------------------------------
    query_serve_parser = subparsers.add_parser(
        "query-serve",
        help="Launch the natural-language query API server.",
    )
    query_serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1).")
    query_serve_parser.add_argument("--port", type=int, default=8788, help="Port to bind (default 8788).")
    query_serve_parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for synthesis model response (default 4096).")
    _add_neo4j_args(query_serve_parser)
    _add_domain_args(query_serve_parser)

    resolve_parser = subparsers.add_parser(
        "resolve-mentions",
        help="Re-run entity resolution for unresolved mentions using the current seed_entities.csv.",
    )
    resolve_target = resolve_parser.add_mutually_exclusive_group(required=True)
    resolve_target.add_argument("--semantic", metavar="PATH", help="Path to a single semantic bundle JSON file.")
    resolve_target.add_argument("--semantic-dir", metavar="DIR", help="Directory; all *.semantic.json files are processed.")
    resolve_parser.add_argument("--output", metavar="PATH", default=None,
                                help="Output path for updated bundle. Only valid with --semantic; default: overwrite in place.")
    resolve_parser.add_argument("--dry-run", action="store_true", default=False,
                                help="Print stats without writing any files.")
    _add_domain_args(resolve_parser)

    verify_parser = subparsers.add_parser(
        "verify-integrity",
        help="Verify file_hash for all ingested documents against their source files.",
    )
    verify_parser.add_argument(
        "--institution-id",
        default=None,
        help="Limit check to one institution (default: all institutions).",
    )
    verify_parser.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this path. If omitted, prints summary to stdout.",
    )
    _add_neo4j_args(verify_parser)

    scan_parser = subparsers.add_parser(
        "sensitivity-scan",
        help="Scan all active claims in the graph for PII, Indigenous cultural sensitivity, and living person references.",
    )
    scan_parser.add_argument(
        "--institution-id",
        default=None,
        help="Limit scan to one institution (default: all institutions).",
    )
    scan_parser.add_argument(
        "--review-db",
        default=os.environ.get("REVIEW_DB", "data/review.db"),
        help="Path to review SQLite database. Quarantine proposals are stored here.",
    )
    scan_parser.add_argument(
        "--output",
        default=None,
        help="Write JSON scan report to this path. If omitted, prints summary to stdout.",
    )
    _add_neo4j_args(scan_parser)

    validate_parser = subparsers.add_parser(
        "validate-domain",
        help="Run sample documents through extraction and report domain quality metrics.",
    )
    validate_parser.add_argument("--samples", nargs="+", required=True, help="Paths to source report JSON files.")
    validate_parser.add_argument("--output", default=None, help="Optional path to write the quality report JSON.")
    validate_parser.add_argument("--threshold-unclassified", type=float, default=0.20,
                                 help="Fail if unclassified claim rate exceeds this fraction (default 0.20).")
    _add_domain_args(validate_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-structure":
        access_level = args.access_level
        if access_level == "indigenous_restricted" and not args.indigenous_consultation_confirmed:
            print(
                "ERROR: --access-level indigenous_restricted requires "
                "--indigenous-consultation-confirmed.\n"
                "You must confirm that the institution has completed documented tribal "
                "consultation with the relevant Indigenous nation(s) before ingesting "
                "materials in this category.",
                file=sys.stderr,
            )
            return 1
        structure = parse_source(args.input)
        structure.document.access_level = access_level
        structure.document.institution_id = args.institution_id
        structure.document.donor_restricted = args.donor_restricted
        save_structure_bundle(args.output, structure)
        audit_db = os.environ.get("WRITE_AUDIT_DB", "data/write_audit.db")
        try:
            from .retrieval.web.write_audit_log import WriteAuditLogger
            WriteAuditLogger(audit_db).log(
                event_type="ingestion",
                doc_id=structure.document.doc_id,
                doc_title=structure.document.title,
                institution_id=structure.document.institution_id,
                performed_by="cli",
                details={
                    "access_level": structure.document.access_level,
                    "source_file": structure.document.source_file,
                    "output": str(Path(args.output)),
                },
            )
        except Exception:
            pass  # audit log failure must never block ingestion
        print(json.dumps({"status": "ok", "output": str(Path(args.output))}, indent=2))
        return 0

    if args.command == "extract-semantic":
        resources_dir = Path(args.domain_dir) if args.domain_dir else None
        structure = load_structure_bundle(args.structure)
        semantic = extract_semantic(
            structure,
            run_overrides={"ocr_engine": args.ocr_engine, "ocr_version": args.ocr_version},
            resources_dir=resources_dir,
        )
        save_semantic_bundle(args.output, semantic)
        print(json.dumps({"status": "ok", "output": str(Path(args.output))}, indent=2))
        return 0

    if args.command == "load-graph":
        neo4j_kwargs = dict(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            neo4j_trust_mode=args.neo4j_trust,
            neo4j_ca_cert=args.neo4j_ca_cert,
        )
        if args.input_dir:
            pairs = sorted(Path(args.input_dir).glob("*.structure.json"))
            if not pairs:
                print(f"[warn] No *.structure.json files found in: {args.input_dir}")
                return 1

            workers = args.workers if args.workers != 0 else (os.cpu_count() or 1)
            effective_workers = min(max(workers, 1), len(pairs))

            def _load_pair(structure_path: Path) -> tuple[Path, object, object] | None:
                semantic_path = structure_path.with_suffix("").with_suffix(".semantic.json")
                if not semantic_path.exists():
                    print(f"[warn] Missing semantic bundle for {structure_path.name}, skipping.", file=sys.stderr)
                    return None
                return structure_path, load_structure_bundle(structure_path), load_semantic_bundle(semantic_path)

            if effective_workers <= 1:
                loaded_pairs = [_load_pair(p) for p in pairs]
            else:
                loaded_map: dict[Path, tuple[Path, object, object]] = {}
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    future_to_path = {executor.submit(_load_pair, p): p for p in pairs}
                    for i, future in enumerate(as_completed(future_to_path), 1):
                        result = future.result()
                        if result is not None:
                            loaded_map[result[0]] = result
                            print(f"[{i}/{len(pairs)}] Loaded: {result[0].stem}", file=sys.stderr)
                loaded_pairs = [loaded_map.get(p) for p in pairs]

            results = []
            writer = None
            for item in loaded_pairs:
                if item is None:
                    continue
                structure_path, structure, semantic = item
                semantic_path = structure_path.with_suffix("").with_suffix(".semantic.json")
                if writer is None:
                    writer = load_graph(structure, semantic, backend=args.backend, **neo4j_kwargs)
                else:
                    writer.load_structure(structure)
                    writer.load_semantic(structure, semantic)
                results.append({"doc_id": structure.document.doc_id, "structure": str(structure_path), "semantic": str(semantic_path)})
            print(json.dumps({"status": "ok", "backend": args.backend, "documents": results}, indent=2))
        elif args.structure and args.semantic:
            structure = load_structure_bundle(args.structure)
            semantic = load_semantic_bundle(args.semantic)
            writer = load_graph(structure, semantic, backend=args.backend, **neo4j_kwargs)
            result = {"status": "ok", "backend": args.backend, "doc_id": structure.document.doc_id}
            if args.backend == "memory":
                result["node_count"] = sum(len(v) for v in writer.node_store.values())  # type: ignore[attr-defined]
                result["relationship_count"] = len(writer.rel_store)  # type: ignore[attr-defined]
            print(json.dumps(result, indent=2))
        else:
            print("[error] Provide either --input-dir or both --structure and --semantic.", file=__import__("sys").stderr)
            return 2
        return 0

    if args.command == "run-e2e":
        resolved_inputs: list[str] = []
        for item in args.inputs:
            p = Path(item)
            if p.is_dir():
                found = sorted(p.rglob("*.json"))
                if not found:
                    print(f"[warn] No JSON files found in directory: {p}")
                resolved_inputs.extend(str(f) for f in found)
            else:
                resolved_inputs.append(item)
        workers = args.workers if args.workers != 0 else (os.cpu_count() or 1)
        summary = run_e2e(
            resolved_inputs,
            args.out_dir,
            workers=workers,
            backend=args.backend,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            neo4j_trust_mode=args.neo4j_trust,
            neo4j_ca_cert=args.neo4j_ca_cert,
            review_out_dir=args.review_out_dir,
            review_db_path=args.review_db,
            resources_dir=Path(args.domain_dir) if args.domain_dir else None,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "quality-report":
        structure = load_structure_bundle(args.structure)
        semantic = load_semantic_bundle(args.semantic)
        report = quality_report(structure, semantic)
        if args.output:
            save_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "spelling-review-report":
        structure = load_structure_bundle(args.structure)
        semantic = load_semantic_bundle(args.semantic)
        report = build_spelling_review_queue(structure, semantic)
        if args.output:
            _write_review_output(args.output, report)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "review-detect":
        from .review.detect import run_detection
        from .review.store import ReviewStore
        structure = load_structure_bundle(args.structure)
        semantic = load_semantic_bundle(args.semantic)
        store = ReviewStore(args.review_db)
        try:
            result = run_detection(
                structure, semantic, store,
                structure_bundle_path=str(Path(args.structure).resolve()),
                semantic_bundle_path=str(Path(args.semantic).resolve()),
            )
            print(json.dumps(result, indent=2))
        finally:
            store.close()
        return 0

    if args.command == "review-serve":
        from .review.web.app import create_app
        try:
            import uvicorn
        except ImportError:
            print("[error] uvicorn is required for review-serve. Install with: pip install uvicorn")
            return 1
        app = create_app(args.review_db)
        print(f"Starting review server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "review-export":
        from .review.store import ReviewStore
        from .review.export import (
            export_accepted_patches_json,
            export_proposals_csv,
            export_proposals_json,
            export_revision_history_json,
        )
        store = ReviewStore(args.review_db)
        try:
            output_path = args.output
            if args.mode == "proposals":
                if output_path.endswith(".csv"):
                    count = export_proposals_csv(store, output_path, status=args.status, snapshot_id=args.snapshot_id)
                else:
                    count = export_proposals_json(store, output_path, status=args.status, snapshot_id=args.snapshot_id)
            elif args.mode == "patches":
                count = export_accepted_patches_json(store, output_path, snapshot_id=args.snapshot_id)
            elif args.mode == "revisions":
                if not args.proposal_id:
                    print("[error] --proposal-id is required for revisions mode.")
                    return 2
                count = export_revision_history_json(store, output_path, args.proposal_id)
            else:
                print(f"[error] Unknown mode: {args.mode}")
                return 2
            print(json.dumps({"status": "ok", "output": output_path, "count": count}, indent=2))
        finally:
            store.close()
        return 0

    if args.command == "query-serve":
        from .retrieval.web.app import create_app
        try:
            import uvicorn
        except ImportError:
            print("[error] uvicorn is required for query-serve. Install with: pip install -e .[retrieval]")
            return 1
        app = create_app(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            neo4j_trust=args.neo4j_trust,
            neo4j_ca_cert=args.neo4j_ca_cert,
            max_tokens=args.max_tokens,
            domain_dir=args.domain_dir,
        )
        print(f"Starting query server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "resolve-mentions":
        if args.output and args.semantic_dir:
            print("[error] --output cannot be used with --semantic-dir.", file=sys.stderr)
            return 2
        paths = (
            sorted(Path(args.semantic_dir).glob("*.semantic.json"))
            if args.semantic_dir
            else [Path(args.semantic)]
        )
        if not paths:
            print(f"[warn] No *.semantic.json files found in: {args.semantic_dir}", file=sys.stderr)
            return 1
        results = []
        resources_dir = Path(args.domain_dir) if args.domain_dir else None
        for p in paths:
            bundle = load_semantic_bundle(p)
            updated, stats = resolve_mentions_targeted(bundle, resources_dir=resources_dir)
            row: dict = {"file": str(p), **stats}
            if not args.dry_run:
                out = Path(args.output) if args.output else p
                save_semantic_bundle(out, updated)
                row["output"] = str(out)
            results.append(row)
        print(json.dumps(results, indent=2))
        return 0

    if args.command == "validate-domain":
        resources_dir = Path(args.domain_dir) if args.domain_dir else None
        aggregate_claims = 0
        aggregate_unclassified = 0
        per_document = []
        for sample_path in args.samples:
            structure = parse_source(sample_path)
            semantic = extract_semantic(structure, resources_dir=resources_dir)
            metrics = quality_report(structure, semantic)
            aggregate_claims += metrics.get("claim_count", 0)
            aggregate_unclassified += metrics.get("unclassified_claim_count", 0)
            per_document.append({"file": sample_path, "doc_id": structure.document.doc_id, **metrics})

        safe_claims = max(1, aggregate_claims)
        unclassified_rate = aggregate_unclassified / safe_claims
        passes = unclassified_rate <= args.threshold_unclassified
        summary = {
            "pass": passes,
            "documents": len(per_document),
            "aggregate_claim_count": aggregate_claims,
            "unclassified_rate": round(unclassified_rate, 4),
            "threshold_unclassified": args.threshold_unclassified,
            "per_document": per_document,
        }
        if args.output:
            save_json(args.output, summary)
        print(json.dumps(summary, indent=2))
        return 0 if passes else 1

    if args.command == "verify-integrity":
        import hashlib
        from .retrieval.executor import Neo4jQueryExecutor
        from .graph.cypher import INTEGRITY_CHECK_QUERY
        executor = Neo4jQueryExecutor(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database,
            trust_mode=args.neo4j_trust,
            ca_cert_path=getattr(args, "neo4j_ca_cert", None),
        )
        rows = executor.run(INTEGRITY_CHECK_QUERY, {"institution_id": args.institution_id})
        executor.close()
        results = []
        ok_count = mismatch_count = missing_count = 0
        for row in rows:
            source_path = Path(row["source_file"]).resolve()
            try:
                source_path.relative_to(Path.cwd())
            except ValueError:
                results.append({
                    "doc_id": row["doc_id"], "title": row.get("title"),
                    "status": "error",
                    "error": f"source_file path outside working directory: {source_path}",
                })
                continue
            if not source_path.exists():
                results.append({
                    "doc_id": row["doc_id"], "title": row.get("title"),
                    "institution_id": row.get("institution_id"),
                    "status": "source_missing", "source_file": str(source_path),
                })
                missing_count += 1
                continue
            try:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                computed = hashlib.sha1(
                    json.dumps(payload, sort_keys=True).encode("utf-8")
                ).hexdigest()
            except Exception as exc:
                results.append({
                    "doc_id": row["doc_id"], "title": row.get("title"),
                    "status": "error", "error": str(exc),
                })
                continue
            status = "ok" if computed == row["file_hash"] else "mismatch"
            entry: dict = {
                "doc_id": row["doc_id"], "title": row.get("title"),
                "institution_id": row.get("institution_id"),
                "status": status,
            }
            if status == "mismatch":
                entry["stored_hash"] = row["file_hash"]
                entry["computed_hash"] = computed
                mismatch_count += 1
            else:
                ok_count += 1
            results.append(entry)
        summary = {
            "total": len(results),
            "ok": ok_count,
            "mismatch": mismatch_count,
            "source_missing": missing_count,
            "documents": results,
        }
        if args.output:
            save_json(args.output, summary)
        print(json.dumps(summary, indent=2))
        return 0 if mismatch_count == 0 and missing_count == 0 else 1

    if args.command == "sensitivity-scan":
        from .retrieval.executor import Neo4jQueryExecutor
        from .review.store import ReviewStore
        from .review.monitor import SensitivityMonitor
        executor = Neo4jQueryExecutor(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database,
            trust_mode=args.neo4j_trust,
            ca_cert_path=getattr(args, "neo4j_ca_cert", None),
        )
        store = ReviewStore(args.review_db)
        try:
            monitor = SensitivityMonitor(executor=executor, store=store)
            results = monitor.run_full_scan(institution_id=args.institution_id)
        finally:
            store.close()
            executor.close()
        if args.output:
            save_json(args.output, results)
        print(json.dumps(results, indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
