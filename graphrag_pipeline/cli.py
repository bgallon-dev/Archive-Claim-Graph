from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .env import load_dotenv
from .io_utils import load_semantic_bundle, load_structure_bundle, save_json, save_semantic_bundle, save_structure_bundle
from .pipeline import extract_semantic, load_graph, parse_source, quality_report, run_e2e


def _add_neo4j_args(p: argparse.ArgumentParser) -> None:
    """Add Neo4j connection arguments, defaulting to environment variables."""
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    p.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    p.add_argument("--neo4j-trust", choices=["system", "all", "custom"], default=os.environ.get("NEO4J_TRUST"))
    p.add_argument("--neo4j-ca-cert", default=os.environ.get("NEO4J_CA_CERT") or None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphrag", description="Claim-centric narrative report graph pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest-structure", help="Parse source OCR report into structure bundle JSON.")
    ingest_parser.add_argument("--input", required=True, help="Path to source report JSON.")
    ingest_parser.add_argument("--output", required=True, help="Path to structure bundle JSON output.")

    semantic_parser = subparsers.add_parser("extract-semantic", help="Extract claims/mentions/measurements from structure bundle.")
    semantic_parser.add_argument("--structure", required=True, help="Path to structure bundle JSON.")
    semantic_parser.add_argument("--output", required=True, help="Path to semantic bundle JSON output.")
    semantic_parser.add_argument("--ocr-engine", default="unknown", help="OCR engine name for ExtractionRun metadata.")
    semantic_parser.add_argument("--ocr-version", default="unknown", help="OCR version for ExtractionRun metadata.")

    graph_parser = subparsers.add_parser("load-graph", help="Load structure and semantic bundles into graph backend.")
    graph_parser.add_argument("--structure", default=None, help="Path to structure bundle JSON.")
    graph_parser.add_argument("--semantic", default=None, help="Path to semantic bundle JSON.")
    graph_parser.add_argument("--input-dir", default=None, help="Directory containing *.structure.json / *.semantic.json pairs.")
    graph_parser.add_argument("--backend", choices=["memory", "neo4j"], default="memory")
    _add_neo4j_args(graph_parser)

    e2e_parser = subparsers.add_parser("run-e2e", help="Run parse + extract + load over multiple reports.")
    e2e_parser.add_argument("--inputs", nargs="+", required=True, help="Paths to source report JSON files or a directory.")
    e2e_parser.add_argument("--out-dir", required=True, help="Output directory for structure/semantic bundles.")
    e2e_parser.add_argument("--backend", choices=["memory", "neo4j"], default="memory")
    _add_neo4j_args(e2e_parser)

    report_parser = subparsers.add_parser("quality-report", help="Compute quality metrics for one structure/semantic pair.")
    report_parser.add_argument("--structure", required=True, help="Path to structure bundle JSON.")
    report_parser.add_argument("--semantic", required=True, help="Path to semantic bundle JSON.")
    report_parser.add_argument("--output", help="Optional path to write quality report JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-structure":
        structure = parse_source(args.input)
        save_structure_bundle(args.output, structure)
        print(json.dumps({"status": "ok", "output": str(Path(args.output))}, indent=2))
        return 0

    if args.command == "extract-semantic":
        structure = load_structure_bundle(args.structure)
        semantic = extract_semantic(
            structure,
            run_overrides={"ocr_engine": args.ocr_engine, "ocr_version": args.ocr_version},
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
            results = []
            writer = None
            for structure_path in pairs:
                semantic_path = structure_path.with_suffix("").with_suffix(".semantic.json")
                if not semantic_path.exists():
                    print(f"[warn] Missing semantic bundle for {structure_path.name}, skipping.")
                    continue
                structure = load_structure_bundle(structure_path)
                semantic = load_semantic_bundle(semantic_path)
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
        summary = run_e2e(
            resolved_inputs,
            args.out_dir,
            backend=args.backend,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            neo4j_trust_mode=args.neo4j_trust,
            neo4j_ca_cert=args.neo4j_ca_cert,
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

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
