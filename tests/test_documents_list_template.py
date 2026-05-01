from __future__ import annotations

from types import SimpleNamespace

from jinja2 import ChoiceLoader, Environment, FileSystemLoader

from gemynd.retrieval.web.app import _corpus_labels_for_composite


def _env() -> Environment:
    return Environment(
        loader=ChoiceLoader([
            FileSystemLoader("gemynd/retrieval/web/templates"),
            FileSystemLoader("gemynd/shared_templates"),
        ]),
        autoescape=True,
    )


def test_corpus_labels_for_composite_uses_human_labels() -> None:
    composite = SimpleNamespace(
        members=[
            SimpleNamespace(
                institution_id="turnbull",
                synthesis_context_label="Turnbull National Wildlife Refuge",
            ),
            SimpleNamespace(
                institution_id="spokane_newspaper",
                synthesis_context_label=None,
            ),
        ]
    )

    assert _corpus_labels_for_composite(composite) == {
        "turnbull": "Turnbull National Wildlife Refuge",
        "spokane_newspaper": "spokane_newspaper",
    }


def test_documents_list_fragment_renders_with_corpus_labels() -> None:
    html = _env().get_template("fragments/documents_list.html").render(
        request=None,
        documents=[
            {
                "doc_id": "doc-1",
                "title": "Annual Report",
                "institution_id": "turnbull",
                "source_links": [],
            }
        ],
        corpus_labels={"turnbull": "Turnbull National Wildlife Refuge"},
        total=1,
        page=1,
        total_pages=1,
    )

    assert "Turnbull National Wildlife Refuge" in html
    assert "Annual Report" in html


def test_documents_list_fragment_falls_back_without_corpus_labels() -> None:
    html = _env().get_template("fragments/documents_list.html").render(
        request=None,
        documents=[
            {
                "doc_id": "doc-1",
                "title": "",
                "title_raw": "bad ocr",
                "institution_id": "turnbull",
                "source_links": [],
            }
        ],
        total=1,
        page=1,
        total_pages=1,
    )

    assert "turnbull" in html
    assert "title OCR degraded" in html
