"""JSONL bundle ingestion: stager unit tests + end-to-end parity with loose files.

Parity claim: given issue payloads that carry a stable ``metadata.doc_id`` (the
recommended shape — identity independent of bundling), ingesting a one-line-per-
issue JSONL produces the same documents, doc_ids and extracted content as
ingesting the same payloads as loose ``.json`` files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemynd.ingest.jsonl_source import stage_jsonl, iter_jsonl_payloads, JsonlLineError
from gemynd.ingest.pipeline import run_e2e, run_e2e_jsonl
from gemynd.shared.io_utils import load_structure_bundle, load_semantic_bundle

NEWS = Path("gemynd/resources_newspaper")


def _issue(doc_id: str, title: str, date: str, text: str) -> dict:
    return {
        "metadata": {"doc_id": doc_id, "title": title, "issue_date": date},
        "pages": [{"page_number": 1, "text": text}],
    }


_PAYLOADS = [
    _issue("sn111_1909-11-14_ed1", "The Spokane Press", "1909-11-14",
           "The Industrial Workers of the World organized a free speech fight in Spokane in 1909."),
    _issue("sn111_1909-11-15_ed1", "The Spokane Press", "1909-11-15",
           "Hundreds of members were arrested during the demonstration downtown."),
    _issue("sn111_1910-06-09_ed1", "The Spokane Press", "1910-06-09",
           "The Great Northern Railway extended its line through Cheney this spring."),
]


# --------------------------------------------------------------------------- #
# stager
# --------------------------------------------------------------------------- #
class TestStageJsonl:
    def test_expands_one_file_per_line_byte_preserving(self, tmp_path):
        jl = tmp_path / "wa_sn111.jsonl"
        raw_lines = [json.dumps(p) for p in _PAYLOADS]
        jl.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

        staged, errors = stage_jsonl(jl, tmp_path / "stage")
        assert errors == []
        assert [p.name for p in staged] == [
            "wa_sn111__000001.json", "wa_sn111__000002.json", "wa_sn111__000003.json",
        ]
        # exact bytes preserved → parsing is identical to a hand-authored file
        assert staged[0].read_text(encoding="utf-8") == raw_lines[0]

    def test_blank_lines_skipped_bad_line_reported(self, tmp_path):
        jl = tmp_path / "t.jsonl"
        jl.write_text(
            json.dumps(_PAYLOADS[0]) + "\n"
            + "\n"                                  # blank → skipped
            + "{not valid json}\n"                  # bad → error
            + json.dumps(_PAYLOADS[1]) + "\n",
            encoding="utf-8",
        )
        staged, errors = stage_jsonl(jl, tmp_path / "stage")
        assert len(staged) == 2
        assert len(errors) == 1 and errors[0].line_number == 3

    def test_non_object_line_raises_when_not_skipping(self, tmp_path):
        jl = tmp_path / "t.jsonl"
        jl.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(JsonlLineError):
            stage_jsonl(jl, tmp_path / "stage", skip_bad_lines=False)

    def test_iter_is_streaming_and_1_indexed(self, tmp_path):
        jl = tmp_path / "t.jsonl"
        jl.write_text(json.dumps(_PAYLOADS[0]) + "\n" + json.dumps(_PAYLOADS[1]) + "\n", encoding="utf-8")
        rows = list(iter_jsonl_payloads(jl))
        assert [r[0] for r in rows] == [1, 2]
        assert rows[0][2]["metadata"]["doc_id"] == "sn111_1909-11-14_ed1"


# --------------------------------------------------------------------------- #
# end-to-end parity: JSONL vs loose files
# --------------------------------------------------------------------------- #
def _digest(out_dir: Path) -> dict:
    """Map doc_id -> content fingerprint, ignoring run_id / source_file / names."""
    digest: dict = {}
    for sp in sorted(Path(out_dir).glob("*.structure.json")):
        struct = load_structure_bundle(sp)
        sem = load_semantic_bundle(sp.with_name(sp.name.replace(".structure.json", ".semantic.json")))
        digest[struct.document.doc_id] = {
            "title": struct.document.title,
            "pages": len(struct.pages),
            "paragraph_text": sorted(p.clean_text or p.raw_ocr_text for p in struct.paragraphs),
            "claims": sorted(c.source_sentence for c in sem.claims),
            "entities": sorted(e.name for e in sem.entities),
        }
    return digest


@pytest.mark.skipif(not NEWS.exists(), reason="newspaper domain resources not present")
class TestJsonlParity:
    def _run_loose(self, tmp_path) -> Path:
        loose_in = tmp_path / "loose_in"
        loose_in.mkdir()
        for i, p in enumerate(_PAYLOADS):
            (loose_in / f"issue_{i}.json").write_text(json.dumps(p), encoding="utf-8")
        out = tmp_path / "out_loose"
        run_e2e([str(f) for f in sorted(loose_in.glob("*.json"))], out,
                workers=1, backend="memory", resources_dir=NEWS, no_llm=True)
        return out

    def _run_jsonl(self, tmp_path) -> Path:
        jl = tmp_path / "wa_sn111.jsonl"
        jl.write_text("\n".join(json.dumps(p) for p in _PAYLOADS) + "\n", encoding="utf-8")
        out = tmp_path / "out_jsonl"
        summary = run_e2e_jsonl([jl], out, workers=1, backend="memory", resources_dir=NEWS, no_llm=True)
        assert summary["documents_staged"] == 3
        assert summary["lines_skipped"] == 0
        # staging cleaned up
        assert not (out / ".jsonl_staging").exists()
        return out

    def test_same_docids_and_content(self, tmp_path):
        loose = _digest(self._run_loose(tmp_path))
        jsonl = _digest(self._run_jsonl(tmp_path))
        assert set(loose) == set(jsonl), "doc_id sets differ between loose and JSONL ingest"
        assert loose == jsonl, "extracted content differs between loose and JSONL ingest"
        assert len(loose) == 3

    def test_docids_are_the_explicit_metadata_ids(self, tmp_path):
        # Identity is bundling-independent: doc_ids equal the payloads'
        # metadata.doc_id, never the staged filename.
        jsonl = _digest(self._run_jsonl(tmp_path))
        assert set(jsonl) == {
            "sn111_1909-11-14_ed1", "sn111_1909-11-15_ed1", "sn111_1910-06-09_ed1",
        }
