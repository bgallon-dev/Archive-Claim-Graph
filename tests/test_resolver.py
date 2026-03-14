import pytest

from graphrag_pipeline.ids import make_entity_id
from graphrag_pipeline.models import MentionRecord
from graphrag_pipeline.resolver import (
    DictionaryFuzzyResolver,
    default_seed_entities,
    normalize_name,
    similarity_score,
)


def test_resolver_threshold_policy() -> None:
    resolver = DictionaryFuzzyResolver()
    mentions = [
        MentionRecord(
            mention_id="m1",
            run_id="run_x",
            paragraph_id="p1",
            surface_form="Turnbull Refuge",
            normalized_form="turnbull refuge",
            start_offset=0,
            end_offset=15,
            detection_confidence=0.9,
            ocr_suspect=False,
        ),
        MentionRecord(
            mention_id="m2",
            run_id="run_x",
            paragraph_id="p1",
            surface_form="Tumbull",
            normalized_form="tumbull",
            start_offset=16,
            end_offset=23,
            detection_confidence=0.7,
            ocr_suspect=True,
        ),
        MentionRecord(
            mention_id="m3",
            run_id="run_x",
            paragraph_id="p1",
            surface_form="Xyzabc",
            normalized_form="xyzabc",
            start_offset=24,
            end_offset=30,
            detection_confidence=0.5,
            ocr_suspect=False,
        ),
    ]

    entities, resolutions = resolver.resolve(mentions)

    assert entities
    by_mention = {item.mention_id: item for item in resolutions}
    assert by_mention["m1"].relation_type == "REFERS_TO"
    assert by_mention["m2"].relation_type == "POSSIBLY_REFERS_TO"
    assert "m3" not in by_mention


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mention(mention_id: str, normalized_form: str) -> MentionRecord:
    return MentionRecord(
        mention_id=mention_id,
        run_id="run_test",
        paragraph_id="p1",
        surface_form=normalized_form,
        normalized_form=normalized_form,
        start_offset=0,
        end_offset=len(normalized_form),
        detection_confidence=0.9,
        ocr_suspect=False,
    )


# ── SurveyMethod entity properties (via default_seed_entities) ────────────────

def test_survey_method_entity_id_is_stable() -> None:
    seeds = default_seed_entities()
    rec = next(e for e in seeds if e.entity_type == "SurveyMethod" and e.name == "aerial survey")
    expected_id = make_entity_id("SurveyMethod", normalize_name("aerial survey"))
    assert rec.entity_id == expected_id
    assert rec.entity_type == "SurveyMethod"
    assert rec.properties["method_class"] == "remote"


def test_survey_method_normalized_form() -> None:
    seeds = default_seed_entities()
    rec = next(e for e in seeds if e.entity_type == "SurveyMethod" and e.name == "water count")
    assert rec.normalized_form == "water count"


# ── default_seed_entities coverage ───────────────────────────────────────────

def test_default_seeds_contain_all_17_survey_methods() -> None:
    expected = {
        "aerial survey", "ground count", "nest survey", "trap survey", "banding",
        "aerial count", "roadside count", "call survey", "spotlight survey",
        "water count", "brood survey", "live trapping", "breeding survey",
        "population census", "harvest count", "pellet survey", "track count",
    }
    seeds = default_seed_entities()
    names = {e.normalized_form for e in seeds if e.entity_type == "SurveyMethod"}
    assert names == expected


def test_survey_method_seeds_have_no_duplicate_entity_ids() -> None:
    seeds = default_seed_entities()
    ids = [e.entity_id for e in seeds if e.entity_type == "SurveyMethod"]
    assert len(ids) == len(set(ids))


def test_survey_method_seeds_method_class_values() -> None:
    allowed = {"remote", "direct", "capture", "harvest", "indirect"}
    seeds = default_seed_entities()
    for e in seeds:
        if e.entity_type == "SurveyMethod":
            assert e.properties.get("method_class") in allowed, (
                f"{e.name!r} has unexpected method_class={e.properties.get('method_class')!r}"
            )


# ── Resolution integration tests ─────────────────────────────────────────────

def test_exact_survey_method_mentions_resolve_refers_to() -> None:
    resolver = DictionaryFuzzyResolver()
    new_seeds = [
        "aerial count", "roadside count", "call survey", "spotlight survey",
        "water count", "brood survey", "live trapping", "breeding survey",
        "population census", "harvest count", "pellet survey", "track count",
    ]
    mentions = [_mention(f"m_{i}", form) for i, form in enumerate(new_seeds)]
    _, resolutions = resolver.resolve(mentions)
    by_mention = {r.mention_id: r for r in resolutions}
    for i, form in enumerate(new_seeds):
        rec = by_mention.get(f"m_{i}")
        assert rec is not None, f"{form!r} produced no resolution"
        assert rec.relation_type == "REFERS_TO", (
            f"{form!r} resolved as {rec.relation_type} (match_score={rec.match_score})"
        )


def test_live_trap_resolves_to_live_trapping_via_substring() -> None:
    resolver = DictionaryFuzzyResolver()
    _, resolutions = resolver.resolve([_mention("m_lt", "live trap")])
    assert resolutions, "'live trap' produced no resolution"
    rec = resolutions[0]
    assert rec.relation_type == "REFERS_TO"
    assert rec.entity_id == make_entity_id("SurveyMethod", "live trapping")


def test_aerial_transect_does_not_resolve_refers_to() -> None:
    resolver = DictionaryFuzzyResolver()
    _, resolutions = resolver.resolve([_mention("m_at", "aerial transect")])
    for rec in resolutions:
        assert rec.relation_type != "REFERS_TO", (
            f"'aerial transect' unexpectedly resolved REFERS_TO {rec.entity_id} "
            f"(match_score={rec.match_score})"
        )


def test_new_seeds_do_not_cross_contaminate_existing_seeds() -> None:
    seeds = [e for e in default_seed_entities() if e.entity_type == "SurveyMethod"]
    for i, a in enumerate(seeds):
        for j, b in enumerate(seeds):
            if i >= j:
                continue
            match_score = similarity_score(a.normalized_form, b.normalized_form)
            assert match_score < 0.85, (
                f"Seeds {a.name!r} and {b.name!r} are too similar (match_score={match_score:.4f})"
            )
