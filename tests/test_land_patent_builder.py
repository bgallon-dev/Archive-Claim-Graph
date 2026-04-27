from __future__ import annotations

from gemynd.ingest.land_patent_builder import build_land_patent_semantic
from gemynd.ingest.source_parser import parse_source_payload


def _payload() -> dict:
    return {
        "metadata": {
            "title": "Land Patent 37515 — Avery A Smith — 1909-01-04",
            "doc_type": "land_patent",
            "date_start": "1909-01-04",
            "date_end": "1909-01-04",
            "report_year": 1909,
            "archive_ref": "37515",
            "meridian": "Willamette",
            "state": "WA",
            "grantees": [
                {
                    "entity_kind": "Person",
                    "person_entity_id": "patent_person_avery_a_smith",
                    "person_display": "Avery A Smith",
                    "person_norm": "avery a smith",
                    "name_raw": "SMITH, AVERY A",
                    "mention_id": "patent_mention_37515_avery_a_smith",
                }
            ],
            "parcels": [
                {
                    "entity_id": "parcel_t22n_r42e_s20_nene",
                    "name": "T22N R42E Section 20 NE/NE",
                    "township": "22N",
                    "range": "42E",
                    "section_number": 20,
                    "aliquot": "NE/NE",
                    "county": "Spokane",
                }
            ],
        },
        "pages": [
            {"page_number": 1, "raw_text": "Grantee(s): Avery A Smith\nParcels:\n- T22N R42E Section 20 NE/NE"}
        ],
    }


def test_land_patent_bundle_records() -> None:
    structure = parse_source_payload(_payload())
    semantic = build_land_patent_semantic(structure, run_overrides={"run_id": "run_test"})

    assert semantic.extraction_run.run_id == "run_test"
    assert len(semantic.claims) == 1
    claim = semantic.claims[0]
    assert claim.claim_type == "land_grant"
    assert claim.certainty == "certain"
    assert claim.extraction_confidence == 1.0
    assert claim.claim_date == "1909-01-04"

    entity_types = sorted(e.entity_type for e in semantic.entities)
    assert entity_types == ["Parcel", "Person"]

    grantee = next(e for e in semantic.entities if e.entity_type == "Person")
    parcel = next(e for e in semantic.entities if e.entity_type == "Parcel")
    assert grantee.entity_id == "patent_person_avery_a_smith"
    assert parcel.entity_id == "parcel_t22n_r42e_s20_nene"
    assert parcel.properties["township"] == "22N"
    assert parcel.properties["county"] == "Spokane"

    assert len(semantic.mentions) == 2
    assert len(semantic.entity_resolutions) == 2
    assert all(r.match_score == 1.0 and r.confirmed_by == "json_metadata" for r in semantic.entity_resolutions)

    assert [(l.relation_type, l.entity_id) for l in semantic.claim_entity_links] == [
        ("SUBJECT_OF_CLAIM", grantee.entity_id),
    ]
    assert [(l.relation_type, l.entity_id) for l in semantic.claim_location_links] == [
        ("OCCURRED_AT", parcel.entity_id),
    ]


def test_land_patent_bundle_is_deterministic() -> None:
    structure = parse_source_payload(_payload())
    first = build_land_patent_semantic(structure, run_overrides={"run_id": "run_test"})
    second = build_land_patent_semantic(structure, run_overrides={"run_id": "run_test"})

    assert [c.claim_id for c in first.claims] == [c.claim_id for c in second.claims]
    assert [e.entity_id for e in first.entities] == [e.entity_id for e in second.entities]
    assert [m.mention_id for m in first.mentions] == [m.mention_id for m in second.mentions]


def test_non_land_patent_unaffected() -> None:
    # Narrative reports should not be touched by land-patent path.
    payload = {
        "metadata": {"title": "Narrative", "doc_type": "narrative_report"},
        "pages": [{"page_number": 1, "raw_text": "Some content."}],
    }
    structure = parse_source_payload(payload)
    assert structure.document.doc_type == "narrative_report"
