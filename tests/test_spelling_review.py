from __future__ import annotations

import json
from pathlib import Path

from graphrag_pipeline.cli import main as cli_main
from graphrag_pipeline.io_utils import save_semantic_bundle, save_structure_bundle
from graphrag_pipeline.pipeline import build_spelling_review_queue, extract_semantic, parse_source, run_e2e


def _build_queue(
    text: str,
    *,
    source_file: str | None = None,
    image_ref: str | None = None,
) -> tuple[object, object, list[dict]]:
    metadata: dict[str, object] = {"title": "Spelling Review Test"}
    if source_file is not None:
        metadata["source_file"] = source_file

    page: dict[str, object] = {"page_number": 1, "raw_text": text}
    if image_ref is not None:
        page["image_ref"] = image_ref

    structure = parse_source({"metadata": metadata, "pages": [page]})
    semantic = extract_semantic(structure)
    return structure, semantic, build_spelling_review_queue(structure, semantic)


def test_known_ocr_token_emits_high_severity_issue_with_image_ref() -> None:
    structure, semantic, queue = _build_queue(
        "Tumbull Refuge was observed by Spokane Bird Club.",
        source_file=r"C:\reports\turnbull.pdf",
        image_ref="images/page-1.png",
    )
    assert semantic.claims
    issue = next(item for item in queue if item["normalized_suspect_text"] == "tumbull")
    assert issue["severity"] == "high"
    assert issue["suggested_correction"] == "turnbull"
    assert issue["review_target_type"] == "image_ref"
    assert issue["image_ref"] == "images/page-1.png"
    assert issue["source_file"] == r"C:\reports\turnbull.pdf"
    assert issue["page_number"] == 1
    assert issue["page_id"] == structure.paragraphs[0].page_id


def test_domain_valid_claim_text_is_not_flagged() -> None:
    _, semantic, queue = _build_queue("Mallards were observed at Turnbull Refuge.")
    assert semantic.claims
    assert queue == []


def test_numeric_and_unit_tokens_are_not_flagged() -> None:
    _, semantic, queue = _build_queue("A fire burned 12 acres on August 2.")
    assert semantic.claims
    assert queue == []


def test_duplicate_claim_and_mention_signals_collapse_to_one_issue() -> None:
    _, semantic, queue = _build_queue("Tumbull Refuge and Tumbull marsh were observed.")
    assert semantic.claims
    tumbull_issues = [item for item in queue if item["normalized_suspect_text"] == "tumbull"]
    assert len(tumbull_issues) == 1
    assert "mention_ocr_suspect" in tumbull_issues[0]["flags"]


def test_review_queue_falls_back_to_pdf_page_target() -> None:
    _, semantic, queue = _build_queue(
        "Tumbull Refuge was observed.",
        source_file=r"C:\reports\turnbull.pdf",
    )
    assert semantic.claims
    issue = next(item for item in queue if item["normalized_suspect_text"] == "tumbull")
    assert issue["review_target_type"] == "pdf_page"
    assert issue["source_file"] == r"C:\reports\turnbull.pdf"
    assert issue["image_ref"] is None


def test_spelling_review_cli_writes_json_report(tmp_path: Path) -> None:
    structure, semantic, _ = _build_queue("Tumbull Refuge was observed.")
    structure_path = tmp_path / "sample.structure.json"
    semantic_path = tmp_path / "sample.semantic.json"
    review_path = tmp_path / "sample.review.json"
    save_structure_bundle(structure_path, structure)
    save_semantic_bundle(semantic_path, semantic)

    assert cli_main(
        [
            "spelling-review-report",
            "--structure",
            str(structure_path),
            "--semantic",
            str(semantic_path),
            "--output",
            str(review_path),
        ]
    ) == 0

    payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert any(item["normalized_suspect_text"] == "tumbull" for item in payload)


def test_run_e2e_can_write_review_queue_outputs(fixtures_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    review_dir = tmp_path / "review"
    summary = run_e2e(
        [str(fixtures_dir / "report3.json")],
        out_dir,
        backend="memory",
        review_out_dir=review_dir,
    )

    assert summary["documents_processed"] == 1
    output = summary["outputs"][0]
    assert "spelling_review_output" in output
    assert output["spelling_review_issue_count"] > 0

    review_path = Path(output["spelling_review_output"])
    assert review_path.exists()
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    suspect_tokens = {item["normalized_suspect_text"] for item in payload}
    assert suspect_tokens & {"tumbull", "emgman", "lightening"}
