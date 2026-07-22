"""Contextual entity discovery.

The suffix rules in entity_type_rules.yaml cannot see a bare proper noun, which
is what most newspaper entities are. These rules recognise an entity by the
frame it appears in instead. Cases below are drawn from real OR/WA OCR.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from gemynd.ingest.extractors.entity_discovery import (
    load_discovery_rules,
    mine_entities,
    resolve_type_conflicts,
    write_seed_csv,
)

_RULES_PATH = (
    Path(__file__).resolve().parents[1]
    / "gemynd"
    / "resources_newspaper"
    / "entity_discovery_rules.yaml"
)


@pytest.fixture(scope="module")
def rules_and_stops():
    """Production patterns and stopwords, with evidence thresholds relaxed.

    The shipped thresholds are calibrated for an 81,000-document mine, so a
    three-document fixture can never clear them. These tests are about what the
    *frames and filters* do; the thresholds themselves are asserted separately
    in ``test_shipped_thresholds_are_calibrated_for_corpus_scale``.
    """
    rules, stopwords = load_discovery_rules(_RULES_PATH)
    relaxed = [
        dataclasses.replace(rule, min_occurrences=1, min_documents=1)
        for rule in rules
    ]
    return relaxed, stopwords


def test_shipped_thresholds_are_calibrated_for_corpus_scale() -> None:
    """Guards the tuning the relaxed fixture deliberately bypasses."""
    rules, _ = load_discovery_rules(_RULES_PATH)
    by_name = {r.name: r for r in rules}
    # The noisiest frame carries the highest bar.
    prepositional = by_name["prepositional_place"]
    assert prepositional.min_occurrences >= 10
    assert prepositional.min_documents >= 5
    # Every rule must require recurrence — OCR damage appears exactly once.
    for rule in rules:
        assert rule.min_documents >= 2, rule.name


def _names(mined) -> set[str]:
    return {e.name for e in mined}


def test_bare_proper_nouns_are_discovered(rules_and_stops) -> None:
    """The whole point: forms no suffix rule can ever type."""
    rules, stops = rules_and_stops
    docs = [
        "PORTLAND, Or., July 3. The train arrived at Portland this morning.",
        "A shipment was sent from Portland to Corvallis last week.",
        "Corvallis, Or. The college at Corvallis opened; freight to Portland.",
        "Visitors travelled to Corvallis and went on to Portland.",
    ]
    found = _names(mine_entities(docs, rules, stops))
    assert "Portland" in found
    assert "Corvallis" in found
    # All-caps dateline must not mint a second, differently-cased entity.
    assert "PORTLAND" not in found


def test_common_nouns_in_entity_frames_are_rejected(rules_and_stops) -> None:
    """"in July", "at Cost" sit in the same frame as "at Portland"."""
    rules, stops = rules_and_stops
    docs = [
        "Sold in July at Cost. Goods arriving in Saturday shipments.",
        "Again in July, and at Cost, the American trade moved North.",
        "Still in July at Cost, going North to the West.",
    ]
    found = _names(mine_entities(docs, rules, stops))
    for junk in ("July", "Cost", "Saturday", "American", "North", "West"):
        assert junk not in found, junk


def test_advertisement_phrases_are_rejected(rules_and_stops) -> None:
    """Page OCR is not article-segmented, so ad copy shares the news frames.

    Each phrase below was mined as a real entity and became a resolution magnet
    ("Half Price" alone pulled 29 edges).
    """
    rules, stops = rules_and_stops
    docs = [
        "Goods sold at Half Price. Dealer in Live Stock and Horse Furnishing.",
        "Again at Half Price, a line of Live Stock, with Horse Furnishing.",
        "Still at Half Price; Live Stock offered, Horse Furnishing too.",
    ]
    found = _names(mine_entities(docs, rules, stops))
    for junk in ("Half Price", "Live Stock", "Horse Furnishing"):
        assert junk not in found, junk


def test_date_fragments_are_rejected(rules_and_stops) -> None:
    """"July 1st" OCRs to "July Ist" and looks like a two-word name (147 edges)."""
    rules, stops = rules_and_stops
    docs = [
        "Payable on July Ist at the office, and due in July Ist terms.",
        "Notice: on July Ist the rate changes; from July Ist onward.",
        "Set for July Ist, effective to July Ist next.",
    ]
    assert "July Ist" not in _names(mine_entities(docs, rules, stops))


def test_real_names_built_from_ordinary_words_survive(rules_and_stops) -> None:
    """The ad filter must not take "Hood River" or "Baker City" with it."""
    rules, stops = rules_and_stops
    docs = [
        "Shipped from Hood River to Baker City on Monday.",
        "Hood River, Or. Fruit moved from Hood River to Baker City by rail.",
        "Travellers went to Hood River, then on to Baker City.",
    ]
    found = _names(mine_entities(docs, rules, stops))
    assert "Hood River" in found
    assert "Baker City" in found


def test_common_words_are_rejected_by_capitalisation(rules_and_stops) -> None:
    """The signal no stopword list can supply.

    "of Them", "to Sell" and "of Ladies" mined Them/Sell/Ladies as places (697,
    108 and 135 edges). None are enumerable in advance, but the corpus shows
    them overwhelmingly in lowercase, whereas a real name is capitalised.
    """
    rules, stops = rules_and_stops
    docs = [
        "Word of Them came; we saw them and told them we must sell to Sell.",
        "More of Them arrived; them again, them once more, and we sell to Sell.",
        "Still of Them, them nearby, them again; they sell and sell, to Sell it.",
    ]
    found = _names(mine_entities(docs, rules, stops))
    assert "Them" not in found
    assert "Sell" not in found


def test_multiword_names_with_short_leading_words_are_captured(rules_and_stops) -> None:
    """Requiring a 4+ char first word restarted the match at the second word.

    "San Francisco" was mined as "Francisco" (1,922 sightings) and the real name
    never entered the gazetteer at all.
    """
    rules, stops = rules_and_stops
    docs = [
        "Sailed to San Francisco and on to Los Angeles this week.",
        "From San Francisco the party went to Los Angeles by rail.",
        "San Francisco, Cal. Freight bound to Los Angeles.",
    ]
    found = _names(mine_entities(docs, rules, stops))
    assert "San Francisco" in found
    assert "Los Angeles" in found
    assert "Francisco" not in found


def test_low_capitalisation_terms_are_reported_for_the_resolver() -> None:
    """Words the corpus shows are not names, so mentions of them cannot resolve."""
    from gemynd.ingest.extractors.entity_discovery import low_capitalization_terms

    profile = {"portland": 0.98, "white": 0.10, "season": 0.05, "rarely": 0.0}
    sightings = {"portland": 500, "white": 400, "season": 300, "rarely": 3}
    terms = low_capitalization_terms(profile, sightings, min_seen=25)
    assert "white" in terms and "season" in terms
    assert "portland" not in terms, "a real name must not be condemned"
    assert "rarely" not in terms, "too little evidence to judge"


def test_capitalisation_profile_measures_name_likeness() -> None:
    from gemynd.ingest.extractors.entity_discovery import capitalization_profile

    profile = capitalization_profile(
        ["Portland is busy. them and them again.", "Portland grew; them, them."]
    )
    assert profile["portland"] == 1.0
    assert profile["them"] == 0.0


def test_holidays_are_not_places(rules_and_stops) -> None:
    """Always capitalised, so the capitalisation profile reads them as names."""
    rules, stops = rules_and_stops
    docs = [
        "Closed at Christmas and again at Christmas next year.",
        "Shops shut at Christmas; travel at Christmas was heavy.",
        "At Christmas the mail was late, and at Christmas again.",
    ]
    assert "Christmas" not in _names(mine_entities(docs, rules, stops))


def test_bare_forenames_are_suppressed_by_corpus_evidence(rules_and_stops) -> None:
    """"Mr. John," mints the forename, which then magnets unrelated people.

    A forename leads other names; a surname rarely does. No name list needed.
    """
    rules, stops = rules_and_stops
    docs = [
        "Mr. John Smith arrived. Mr. John Gavin followed. Dr. King spoke.",
        "Later Mr. John Smith left; Mr. John Gavin too. Dr. King returned.",
        "Mr. John, of the firm, called on Dr. King again today.",
    ]
    found = _names(mine_entities(docs, rules, stops))
    assert "John" not in found, "bare forename must not become an entity"
    assert "John Smith" in found
    assert "King" in found, "a surname that never leads must survive"


def test_county_rule_mints_the_full_name(rules_and_stops) -> None:
    rules, stops = rules_and_stops
    docs = ["Wasco County court met.", "The Wasco County assessor reported."]
    found = _names(mine_entities(docs, rules, stops))
    assert "Wasco County" in found
    assert "Wasco" not in found  # bare token reaches it via the resolver's gate


def test_single_sighting_is_not_enough() -> None:
    """OCR damage appears once; real names recur. This is the noise filter.

    Uses the SHIPPED thresholds deliberately — this test is *about* them, so the
    relaxed fixture would defeat its purpose.
    """
    rules, stops = load_discovery_rules(_RULES_PATH)
    docs = ["Bakkb City, Or., May 2.", "Nothing further to report."]
    assert "Bakkb City" not in _names(mine_entities(docs, rules, stops))


def test_honorific_captures_the_whole_name(rules_and_stops) -> None:
    """Capturing one word mines the given name, which is a useless entity."""
    rules, stops = rules_and_stops
    docs = ["Mr. John Smith spoke.", "Later Mr. John Smith departed."]
    found = _names(mine_entities(docs, rules, stops))
    assert "John Smith" in found
    assert "John" not in found


def test_names_never_contain_newlines(rules_and_stops) -> None:
    """`\\s` in a capture spans line breaks; "Oregon\\nCity" was mined as a name."""
    rules, stops = rules_and_stops
    docs = [
        "Freight moved to Oregon\nCity today, and on to Oregon\nCity again.",
        "Shipments from Oregon\nCity; more from Oregon\nCity by rail.",
        "Bound to Oregon\nCity, then to Oregon\nCity once more.",
    ]
    for entity in mine_entities(docs, rules, stops):
        assert "\n" not in entity.name and "\r" not in entity.name, repr(entity.name)


def test_ocr_variants_fold_into_the_well_evidenced_form() -> None:
    from gemynd.ingest.extractors.entity_discovery import MinedEntity, merge_ocr_variants

    canonical = MinedEntity(
        name="Prineville", entity_type="Place", occurrences=300, document_count=120
    )
    variant = MinedEntity(
        name="Prineviile", entity_type="Place", occurrences=21, document_count=13
    )
    merged = merge_ocr_variants([canonical, variant])
    assert [e.name for e in merged] == ["Prineville"]
    assert merged[0].occurrences == 321, "variant evidence should carry over"


def test_a_more_specific_name_is_not_swallowed_by_a_shorter_one() -> None:
    """Containment is not a spelling variant.

    "William County" merging into "William" destroyed the county and credited a
    forename with the county rule's evidence (3,086 sightings, typed Place).
    "The Dalles" disappeared into "Dalle" the same way.
    """
    from gemynd.ingest.extractors.entity_discovery import MinedEntity, merge_ocr_variants

    short = MinedEntity(name="William", entity_type="Place", occurrences=3000, document_count=900)
    specific = MinedEntity(name="William County", entity_type="Place", occurrences=40, document_count=30)
    names = {e.name for e in merge_ocr_variants([short, specific])}
    assert names == {"William", "William County"}

    the_dalles = MinedEntity(name="The Dalles", entity_type="Place", occurrences=50, document_count=40)
    dalle = MinedEntity(name="Dalle", entity_type="Place", occurrences=900, document_count=700)
    assert "The Dalles" in {e.name for e in merge_ocr_variants([dalle, the_dalles])}


def test_similar_names_of_similar_weight_do_not_merge() -> None:
    """Two real towns must not swallow one another just for looking alike."""
    from gemynd.ingest.extractors.entity_discovery import MinedEntity, merge_ocr_variants

    a = MinedEntity(name="Benton", entity_type="Place", occurrences=100, document_count=40)
    b = MinedEntity(name="Renton", entity_type="Place", occurrences=90, document_count=38)
    assert len(merge_ocr_variants([a, b])) == 2


def test_type_conflicts_collapse_to_best_evidenced() -> None:
    from gemynd.ingest.extractors.entity_discovery import MinedEntity

    place = MinedEntity(name="Lincoln", entity_type="Place", occurrences=9, document_count=2)
    person = MinedEntity(name="Lincoln", entity_type="Person", occurrences=2, document_count=1)
    resolved = resolve_type_conflicts([place, person])
    assert len(resolved) == 1
    assert resolved[0].entity_type == "Place"


def test_written_csv_loads_as_seed_entities(tmp_path: Path, rules_and_stops) -> None:
    """Mined output must round-trip through the seed-entity loader."""
    from gemynd.core.resolver import default_seed_entities

    rules, stops = rules_and_stops
    docs = [
        "PORTLAND, Or., July 3. Freight left from Portland today.",
        "The party set out from Corvallis, Or., bound to Portland.",
        "At Corvallis the fair opened; travellers came to Portland.",
    ]
    mined = resolve_type_conflicts(mine_entities(docs, rules, stops))
    # A domain dir needs the curated file too; discovery only ever adds to it.
    # Filenames resolve through domain_profile.yaml, so declare them there.
    (tmp_path / "domain_profile.yaml").write_text(
        'version: "1"\n'
        "domain: test_domain\n"
        "resources:\n"
        "  seed_entities: seed_entities.csv\n"
        "  discovered_entities: discovered_entities.csv\n",
        encoding="utf-8",
    )
    (tmp_path / "seed_entities.csv").write_text(
        "entity_type,name,prop_key,prop_value\nPlace,Spokane,,\n", encoding="utf-8"
    )
    write_seed_csv(mined, tmp_path / "discovered_entities.csv")

    entities = default_seed_entities(tmp_path)
    by_name = {e.name for e in entities}
    assert "Spokane" in by_name          # curated survives
    assert "Portland" in by_name         # discovered is unioned in
    forms = [e.normalized_form for e in entities]
    assert len(forms) == len(set(forms)), "duplicate surface forms would split mentions"
