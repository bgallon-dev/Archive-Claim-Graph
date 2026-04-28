"""Unit tests for `_citation_label()` in web/app.py.

Covers structured-field citation rendering with series-aware provenance for
composite collections (e.g. BLM GLO Records ingested under the Turnbull
institution_id).
"""
from __future__ import annotations

from gemynd.retrieval.models import ProvenanceBlock
from gemynd.retrieval.web.app import _citation_label


def _make_block(**kwargs) -> ProvenanceBlock:
    defaults = dict(
        doc_title="Annual Report 1942",
        doc_date_start="1942-01-01",
        doc_date_end="1942-12-31",
        page_number=3,
        paragraph_id="para-001",
        claim_id="c-001",
        claim_type="POPULATION_TREND",
        extraction_confidence=0.85,
        epistemic_status="observed",
        source_sentence="Mallard numbers increased.",
        institution_id="turnbull",
    )
    defaults.update(kwargs)
    return ProvenanceBlock(**defaults)


class TestCitationLabel:
    def test_institution_date_page_default(self):
        block = _make_block(series=None)
        label = _citation_label(block)
        assert "Turnbull" in label
        assert "1942-01-01" in label
        assert "pp3" in label

    def test_series_prefixes_institution(self):
        block = _make_block(
            series="BLM GLO Records",
            institution_id="turnbull",
            doc_title="Land Patent 122388 — George Blackshaw — 1910-04-01",
            doc_date_start="1910-04-01",
            doc_date_end="1910-04-01",
            page_number=None,
        )
        label = _citation_label(block)
        assert "BLM GLO Records via Turnbull" in label
        assert "1910-04-01" in label

    def test_series_alone_when_no_institution(self):
        block = _make_block(series="BLM GLO Records", institution_id=None)
        label = _citation_label(block)
        assert label.startswith("BLM GLO Records")
        assert "via" not in label

    def test_no_series_falls_back_to_institution(self):
        block = _make_block(series=None, institution_id="spokane_newspaper")
        label = _citation_label(block)
        assert "Spokane Newspaper" in label
        assert "via" not in label

    def test_ocr_garbage_title_suppressed(self):
        block = _make_block(
            doc_title="G O N TIN U E D h o",
            institution_id=None,
            doc_date_start=None,
            page_number=None,
            doc_id="doc-fallback-id",
        )
        label = _citation_label(block)
        assert "G O N TIN" not in label
        assert label == "doc-fallback-id"

    def test_clean_title_used_when_no_structured_fields(self):
        block = _make_block(
            doc_title="The History of Spokane",
            institution_id=None,
            doc_date_start=None,
            page_number=None,
        )
        label = _citation_label(block)
        assert label == "The History of Spokane"
