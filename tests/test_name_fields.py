from pathlib import Path

from gemynd.core.models import EntityRecord, MentionRecord
from gemynd.ingest.pipeline import extract_semantic
from gemynd.ingest.source_parser import parse_source_file


def test_mention_and_entity_load_legacy_normalized_name_alias() -> None:
    mention = MentionRecord.from_dict(
        {
            "mention_id": "m1",
            "run_id": "run_1",
            "paragraph_id": "p1",
            "surface_form": "Turnbull Refuge",
            "normalized_name": "turnbull refuge",
            "start_offset": 0,
            "end_offset": 15,
            "detection_confidence": 0.9,
        }
    )
    entity = EntityRecord.from_dict(
        {
            "entity_id": "e1",
            "entity_type": "Refuge",
            "name": "Turnbull Refuge",
            "normalized_name": "turnbull refuge",
        }
    )

    assert mention.normalized_form == "turnbull refuge"
    assert "normalized_name" not in mention.to_dict()
    assert entity.normalized_form == "turnbull refuge"
    assert "normalized_name" not in entity.to_dict()


def test_period_entity_uses_source_title_not_label(fixtures_dir: Path) -> None:
    structure = parse_source_file(fixtures_dir / "report1.json")
    semantic = extract_semantic(
        structure,
        run_overrides={"run_id": "run_name_fields", "run_timestamp": "2026-03-10T00:00:00+00:00"},
    )

    period_entities = [entity for entity in semantic.entities if entity.entity_type == "Period"]
    assert period_entities, "Expected a Period entity for dated reports."
    for entity in period_entities:
        assert entity.properties["source_title"] == structure.document.title
        assert "label" not in entity.properties


def test_period_entity_from_dict_remaps_legacy_label_property() -> None:
    entity = EntityRecord.from_dict(
        {
            "entity_id": "period_1",
            "entity_type": "Period",
            "name": "1938-01-01 to 1938-12-31",
            "normalized_form": "1938-01-01 to 1938-12-31",
            "properties": {
                "label": "Turnbull Refuge Report 1938",
                "period_type": "publication_period",
            },
        }
    )

    assert entity.properties["source_title"] == "Turnbull Refuge Report 1938"
    assert "label" not in entity.properties
