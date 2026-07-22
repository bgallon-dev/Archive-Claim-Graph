"""The rules file is authoritative: no invented type when rules don't match.

Regression: ``infer_entity_type`` unconditionally fell back to "Person". On the
newspaper corpus that typed 94.9% of mentions as Person — including ad
fragments ("IN", "OUR", "COST") — and a wrong hint filters resolution
candidates, blocking the correct match.
"""
from __future__ import annotations

import re

from gemynd.core.entity_type_inference import infer_entity_type

_ORG_RULE = {
    "name": "corporate_suffixes",
    "target_type": "Organization",
    "match_mode": "any_token_uppercase",
    "keywords": frozenset({"COMPANY", "RAILWAY"}),
}
_FALLBACK_PERSON = {
    "name": "default_person",
    "target_type": "Person",
    "match_mode": "fallback",
}


def test_rules_supplied_but_none_match_returns_none() -> None:
    # ad fragments must NOT become Person
    for junk in ("IN", "OUR", "COST", "SHIRTS", "Summer Trade"):
        assert infer_entity_type(junk, rules=[_ORG_RULE]) is None


def test_matching_rule_still_wins() -> None:
    assert infer_entity_type("Standard Wagon Company", rules=[_ORG_RULE]) == "Organization"


def test_explicit_fallback_rule_is_honored() -> None:
    """A domain that wants a catch-all declares one (Turnbull does)."""
    rules = [_ORG_RULE, _FALLBACK_PERSON]
    assert infer_entity_type("Arthur Smith", rules=rules) == "Person"
    assert infer_entity_type("Standard Wagon Company", rules=rules) == "Organization"


def test_no_rules_supplied_keeps_default_for_backward_compat() -> None:
    assert infer_entity_type("Arthur Smith") == "Person"
    assert infer_entity_type("Arthur Smith", default="Place") == "Place"


def test_seed_vocabulary_still_wins_over_rules() -> None:
    class _Seed:
        def __init__(self, name, etype):
            self.name = name
            self.normalized_form = name.lower()
            self.entity_type = etype

    seeds = [_Seed("John Church", "Person")]
    # "CHURCH" would hit an Organization keyword rule, but the seed wins.
    org_church = {
        "name": "org", "target_type": "Organization",
        "match_mode": "any_token_uppercase", "keywords": frozenset({"CHURCH"}),
    }
    assert infer_entity_type("John Church", seed_entities=seeds, rules=[org_church]) == "Person"


def test_regex_rule_matches_uppercased_surface() -> None:
    rule = {
        "name": "person_initials", "target_type": "Person", "match_mode": "regex_any",
        "patterns": [re.compile(r"^[A-Z]\. [A-Z]")],
    }
    assert infer_entity_type("C. Piatt", rules=[rule]) == "Person"
    assert infer_entity_type("Portland", rules=[rule]) is None
