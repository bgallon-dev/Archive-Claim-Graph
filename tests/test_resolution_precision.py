"""Guards against the two mistyping/misresolution defects found on the
newspaper corpus. Each case below is a real failure observed in the graph.
"""
from __future__ import annotations

from gemynd.core.entity_type_inference import infer_entity_type
from gemynd.core.resolver import similarity_score, token_anchor_ok

_FUZZY_THRESHOLD = 0.88  # mirrors entity_type_inference / resolver


# ── containment boost must require a substantial length ratio ────────────────
# These four produced 45.3% spurious REFERS_TO (242,464 of 535,624): a short
# fragment substring-matched a long seed name and hit exactly 0.88.
def test_short_fragment_no_longer_resolves_to_long_entity() -> None:
    for frag, entity in [
        ("fall", "spokane falls"),
        ("west", "western federation of miners"),
        ("chicago", "chicago milwaukee and st. paul railway"),
        ("world", "industrial workers of the world"),
    ]:
        assert similarity_score(frag, entity) < _FUZZY_THRESHOLD, (frag, entity)


def test_genuine_partial_matches_still_resolve() -> None:
    # shorter side is a substantial part of the longer one → still a match
    assert similarity_score("new york", "new york city") >= _FUZZY_THRESHOLD
    assert similarity_score("wasco county", "wasco county, oregon") >= _FUZZY_THRESHOLD


def test_exact_match_unaffected() -> None:
    assert similarity_score("spokane falls", "spokane falls") == 1.0


# ── suffix-anchored keyword matching ─────────────────────────────────────────
_PLACE = {
    "name": "place_admin", "target_type": "Place",
    "match_mode": "suffix_token_uppercase", "keywords": frozenset({"COUNTY"}),
}
_ORG = {
    "name": "corp", "target_type": "Organization",
    "match_mode": "suffix_token_uppercase",
    "keywords": frozenset({"COMPANY", "DEPARTMENT"}),
}


def test_suffix_rule_matches_qualified_names() -> None:
    assert infer_entity_type("Wasco County", rules=[_PLACE]) == "Place"
    assert infer_entity_type("Standard Wagon Company", rules=[_ORG]) == "Organization"


def test_suffix_rule_rejects_bare_common_nouns() -> None:
    # bare nouns are not entities — these were typed Place/Organization before
    for bare in ("COUNTY", "Department", "STATE"):
        assert infer_entity_type(bare, rules=[_PLACE, _ORG]) is None


def test_suffix_rule_rejects_keyword_in_non_final_position() -> None:
    # "County Judge Geo" typed as Place under any_token matching
    assert infer_entity_type("County Judge Geo", rules=[_PLACE, _ORG]) is None


# ── token-anchored resolution ────────────────────────────────────────────────
# Every pair below is a real CP1 resolution edge, with its observed count.
# Character similarity alone accepted all of them: 44.3% of resolutions were
# spurious and 95% of POSSIBLY_REFERS_TO edges were wrong.
def test_character_coincidence_is_rejected() -> None:
    for surface, entity in [
        ("range", "sprague"),          # 67 edges
        ("spain", "spokane"),          # 37
        ("phone", "spokane"),          # 24
        ("newport", "new york"),       # 23
        ("col", "colfax"),             # 20
        ("hill", "hillyard"),          # 19
        ("work", "new york"),          # 18
        ("house", "palouse"),          # 18
        ("com", "tacoma"),             # 14
        ("alsea", "palouse"),          # 12
        ("report", "davenport"),       # 9
        ("plain", "pullman"),          # 9
        ("thomas", "tacoma"),          # 8
        ("lord", "colorado"),          # 8
        ("cholera", "colorado"),       # 7
        ("oregonian", "oregon"),       # 25 — was a REFERS_TO
    ]:
        assert not token_anchor_ok(surface, entity), (surface, entity)


def test_shared_generic_token_is_not_evidence() -> None:
    # Same generic tail, different entity — these were POSSIBLY_REFERS_TO edges.
    assert not token_anchor_ok("wasco county", "adams county")
    assert not token_anchor_ok("benton county", "lincoln county")
    assert not token_anchor_ok("prosser falls", "spokane falls")
    # A bare generic noun resolved to a named place (65 edges, a REFERS_TO).
    assert not token_anchor_ok("county", "adams county")


def test_leading_token_must_match() -> None:
    # Shares only the trailing "pacific" — a different railroad.
    assert not token_anchor_ok("southern pacific", "northern pacific railway")
    assert token_anchor_ok("northern pacific", "northern pacific railway")


def test_genuine_references_still_anchor() -> None:
    for surface, entity in [
        ("washington", "washington state"),
        ("chronicle", "spokane chronicle"),
        ("great northern", "great northern railway"),
        ("new york city", "new york"),
        ("roosevelt", "theodore roosevelt"),
        ("lincoln", "lincoln county"),
        ("turnbull", "turnbull national wildlife refuge"),
    ]:
        assert token_anchor_ok(surface, entity), (surface, entity)


def test_ocr_damaged_tokens_still_anchor() -> None:
    """Fuzzy matching exists to absorb OCR damage; the gate must not undo that."""
    assert token_anchor_ok("turnbul", "turnbull national wildlife refuge")
    assert token_anchor_ok("tumbull", "turnbull")  # the classic rn -> m confusion
    assert token_anchor_ok("spokane chronicie", "spokane chronicle")


def test_shared_given_name_is_not_the_same_person() -> None:
    for surface, entity in [
        ("william watson", "william howard taft"),
        ("william grant", "william howard taft"),
        ("william hays", "william d. haywood"),
        ("frank wilson", "frank little"),
        ("james wilson", "james h. walsh"),
        ("robert fulton", "robert b. strahorn"),
        ("elizabeth bonn", "elizabeth gurley flynn"),
        ("washington street", "washington state"),
    ]:
        assert not token_anchor_ok(surface, entity), (surface, entity)


def test_inflected_forms_still_anchor() -> None:
    """Plurals are not damage: 'coots' -> seed 'coot' must survive the gate."""
    assert token_anchor_ok("coots", "coot")
    assert token_anchor_ok("sportsmens assn", "sportsmen assn")


def test_prefix_append_is_a_different_name() -> None:
    """"harris"/"harrison" fits the same 2-edit budget as tumbull/turnbull."""
    assert not token_anchor_ok("harris", "harrison")
    assert not token_anchor_ok("stevens", "stevenson")
    # ...but truncation damage, which loses one character, must still resolve.
    assert token_anchor_ok("turnbul", "turnbull")


def test_bare_compass_direction_is_not_a_place() -> None:
    """A bare "north" resolved to the mined "North Yak" 60 times."""
    assert not token_anchor_ok("north", "north yak")
    assert not token_anchor_ok("eastern", "eastern oregon")
    # The qualifier is still usable as part of a fuller name.
    assert token_anchor_ok("north yakima", "north yakima city")


def test_plural_rule_needs_a_substantial_stem() -> None:
    """Stripping an "s" off a 3-letter token matches anything.

    The month "May" collapsed onto the mined surname "Mays" 89 times.
    """
    assert not token_anchor_ok("may", "mays")
    assert not token_anchor_ok("ore", "ores")


def test_one_edit_on_a_short_token_is_not_a_match() -> None:
    """"work" -> "york" is a single edit and resolved to New York 18 times."""
    assert not token_anchor_ok("work", "york")
    assert not token_anchor_ok("lord", "ford")


# ── person name-form rule must not swallow abbreviations ─────────────────────
_PERSON_RULES_PATH = "gemynd/resources_newspaper/entity_type_rules.yaml"


def _newspaper_rules():
    from pathlib import Path

    from gemynd.core.domain_config import load_domain_config

    root = Path(__file__).resolve().parents[1]
    return load_domain_config(root / "gemynd" / "resources_newspaper").entity_type_rules


def test_blocking_index_is_exact_not_approximate() -> None:
    """Blocking must not silently drop resolutions to buy speed.

    The mined gazetteer is ~72k entities, so scanning all of them per mention is
    3.2 billion comparisons per 200 documents. Blocking is only legitimate if it
    returns exactly what the exhaustive scan would.
    """
    from gemynd.core.models import EntityRecord
    from gemynd.core.resolver import DictionaryFuzzyResolver

    names = [
        ("Place", "portland"), ("Place", "spokane"), ("Place", "spokane falls"),
        ("Place", "hood river"), ("Place", "wasco county"), ("Person", "turnbull"),
        ("Organization", "northern pacific railway"), ("Place", "new york"),
        ("Person", "harrison"), ("Place", "corvallis"), ("Place", "walla walla"),
    ]
    entities = [
        EntityRecord(
            entity_id=f"e{i}", entity_type=t, name=n.title(), normalized_form=n,
            properties={},
        )
        for i, (t, n) in enumerate(names)
    ]
    resolver = DictionaryFuzzyResolver(seed_entities=entities)

    probes = [
        "portland", "spokane", "spokane fall", "hood river", "wasco county",
        "turnbul", "northern pacific", "new york city", "harris", "corvalis",
        "walla walla", "nonsense", "them", "york",
    ]
    for probe in probes:
        blocked = {e.entity_id for e in resolver._blocked_candidates(probe)}
        exhaustive = {
            e.entity_id
            for e in entities
            if similarity_score(probe, e.normalized_form) >= 0.65
            and token_anchor_ok(probe, e.normalized_form)
        }
        assert exhaustive <= blocked, (probe, exhaustive - blocked)


def test_loader_carries_keywords_for_both_keyword_modes() -> None:
    """Regression: the loader only populated keywords for `any_token_uppercase`.

    Adding `suffix_token_uppercase` without updating the loader left every
    newspaper Place/Organization suffix rule inert — it matched nothing, and the
    failure was silent because inference simply fell through to the next rule.
    """
    rules = _newspaper_rules()
    by_mode: dict[str, list[dict]] = {}
    for rule in rules:
        by_mode.setdefault(rule["match_mode"], []).append(rule)
    suffix_rules = by_mode.get("suffix_token_uppercase") or []
    assert suffix_rules, "newspaper domain should define suffix rules"
    for rule in suffix_rules:
        assert rule.get("keywords"), f"{rule['name']} lost its keywords in loading"


def test_suffix_rules_actually_fire_end_to_end() -> None:
    rules = _newspaper_rules()
    assert infer_entity_type("Wasco County", rules=rules) == "Place"
    assert infer_entity_type("Condon Globe", rules=rules) == "Organization"
    assert infer_entity_type("Standard Wagon Company", rules=rules) == "Organization"


def test_initials_rule_rejects_abbreviations() -> None:
    rules = _newspaper_rules()
    # All were typed Person at CP1: degrees, states, and an initialised college.
    for abbrev in ("M. D.", "N. Y.", "U. S.", "A. D.", "O. R.", "O. A. C.", "W. H."):
        assert infer_entity_type(abbrev, rules=rules) != "Person", abbrev


def test_initials_rule_still_matches_real_names() -> None:
    rules = _newspaper_rules()
    for name in ("C. Piatt", "F. J. Cheney", "W. H. Leeds", "Dr. King"):
        assert infer_entity_type(name, rules=rules) == "Person", name


def test_any_token_mode_still_works_for_existing_domains() -> None:
    """Turnbull's rules use any_token_uppercase (e.g. 'BOARD of Education')."""
    any_rule = {
        "name": "gov", "target_type": "Organization",
        "match_mode": "any_token_uppercase", "keywords": frozenset({"BOARD"}),
    }
    assert infer_entity_type("Board of Education", rules=[any_rule]) == "Organization"
