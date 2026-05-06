from __future__ import annotations

from types import SimpleNamespace

from jinja2 import ChoiceLoader, Environment, FileSystemLoader

from gemynd.retrieval.web.app import _build_source_links, _corpus_labels_for_composite

# Mirrors the spokane_newspaper.source_links block in data/corpus_registry.yaml.
# Kept inline so the test breaks loudly if the template form drifts from what
# we assert here (rather than silently following the YAML wherever it goes).
_SPOKANE_LINK_CONFIGS = [
    {
        "label": "View on Library of Congress",
        "template": "https://www.loc.gov/item/{lccn}/{date_start}/ed-{edition}/",
        "requires": ["lccn", "date_start", "edition"],
    },
]


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


def test_build_source_links_emits_canonical_loc_url() -> None:
    # edition arrives zero-stripped from source_parser so the template
    # renders the LoC canonical form (ed-1, not ed-01).
    doc = {
        "doc_id": "doc-1",
        "lccn": "sn88085947",
        "date_start": "1909-11-10",
        "edition": "1",
    }
    links = _build_source_links(doc, _SPOKANE_LINK_CONFIGS)
    assert len(links) == 1
    assert links[0]["url"] == "https://www.loc.gov/item/sn88085947/1909-11-10/ed-1/"
    assert links[0]["label"] == "View on Library of Congress"


def test_build_source_links_emits_nothing_when_required_field_missing() -> None:
    doc = {
        "doc_id": "doc-1",
        "lccn": None,
        "date_start": "1909-11-10",
        "edition": "1",
    }
    assert _build_source_links(doc, _SPOKANE_LINK_CONFIGS) == []


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
