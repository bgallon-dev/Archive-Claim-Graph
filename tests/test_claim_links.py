from __future__ import annotations

import pytest

from gemynd.ingest.extractors.claim_extractor import ClaimDraft, ClaimLinkDraft
from gemynd.ingest.extractors.mention_extractor import MentionDraft
from gemynd.core.models import EntityRecord, EntityResolutionRecord
from gemynd.ingest.pipeline import extract_semantic, parse_source, quality_report


class StaticClaimExtractor:
    def __init__(self, drafts: list[ClaimDraft]) -> None:
        self._drafts = drafts

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        return list(self._drafts)


class StaticMentionExtractor:
    def __init__(self, drafts: list[MentionDraft]) -> None:
        self._drafts = drafts

    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        return list(self._drafts)


class MappingResolver:
    def __init__(self, mapping: dict[tuple[int, int], tuple[EntityRecord, str]]) -> None:
        self._mapping = mapping

    def resolve(self, mentions: list, contexts=None, document_entity_counts=None) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        entities: dict[str, EntityRecord] = {}
        resolutions: list[EntityResolutionRecord] = []
        for mention in mentions:
            resolved = self._mapping.get((mention.start_offset, mention.end_offset))
            if not resolved:
                continue
            entity, relation_type = resolved
            entities[entity.entity_id] = entity
            resolutions.append(
                EntityResolutionRecord(
                    mention_id=mention.mention_id,
                    entity_id=entity.entity_id,
                    relation_type=relation_type,
                    match_score=0.99 if relation_type == "REFERS_TO" else 0.72,
                )
            )
        return list(entities.values()), resolutions


def _single_paragraph_structure(text: str):
    return parse_source(
        {
            "metadata": {"title": "Claim Link Test", "report_year": 1938},
            "pages": [{"page_number": 1, "raw_ocr_text": text}],
        }
    )


# ---------------------------------------------------------------------------
# Parametrised edge-case table
#
# Paragraph offsets are verified inline:
#   "Mallards were observed."          Mallards=(0,8)  len=23
#   "The deer was managed."            deer=(4,8)      len=21
#   "Mallards were found in the wetland." Mallards=(0,8) len=35
#   "Marsh near marsh dried."          Marsh=(0,5) marsh=(11,16) len=23
#   "Turnbull and Turnbull were discussed." T1=(0,8) T2=(13,21) len=37
#   "Wetland habitat was flooded."     Wetland=(0,7)   len=28
#   "A fire at Turnbull NWR burned."   TurnbullNWR=(10,22) len=30
#   "Fire burned all grass. Turnbull NWR was affected." TurnbullNWR=(23,35) len=49
#   "Surveys using line transects found mallards." linetransects=(14,28) len=44
#   "At Turnbull Turnbull fire burned." T1=(3,11) T2=(12,20) len=33
#   "Meeting held at Turnbull."        Turnbull=(16,24) len=25
#   "The wetland dried up."            wetland=(4,11)  len=21
#   "Mallards at Turnbull Refuge."     Mallards=(0,8) TurnbullRefuge=(12,27) len=28
# ---------------------------------------------------------------------------
CLAIM_LINK_EDGE_CASES: list[dict] = [
    # ------------------------------------------------------------------
    # 1. Exact span match, strong compatibility → direct link, no diag
    # ------------------------------------------------------------------
    {
        "id": "exact_span_strong_compat",
        "paragraph": "Mallards were observed.",
        "claim_type": "species_presence",
        "claim_links": [
            {"surface": "Mallards", "normalized": "mallards",
             "relation": "SPECIES_FOCUS", "span": (0, 8)},
        ],
        "mentions": [
            {"surface": "Mallards", "normalized": "mallards", "span": (0, 8)},
        ],
        "resolutions": {
            (0, 8): ("species_mallard", "Species", "REFERS_TO"),
        },
        "expected_entity_links": [("species_mallard", "SPECIES_FOCUS")],
        "expected_diagnostics": [],
    },
    # ------------------------------------------------------------------
    # 2. Normalized-form fallback (no span on link) → unique match
    # ------------------------------------------------------------------
    {
        "id": "normalized_form_fallback_unique",
        "paragraph": "Mallards were observed.",
        "claim_type": "species_presence",
        "claim_links": [
            {"surface": "Mallards", "normalized": "mallards", "relation": "SPECIES_FOCUS"},
        ],
        "mentions": [
            {"surface": "Mallards", "normalized": "mallards", "span": (0, 8)},
        ],
        "resolutions": {
            (0, 8): ("species_mallard", "Species", "REFERS_TO"),
        },
        "expected_entity_links": [("species_mallard", "SPECIES_FOCUS")],
        "expected_diagnostics": [],
    },
    # ------------------------------------------------------------------
    # 3. Forbidden compatibility → link dropped, RELATION_COMPATIBILITY_FORBIDDEN
    #    weather_observation + MANAGEMENT_TARGET = forbidden
    # ------------------------------------------------------------------
    {
        "id": "forbidden_compat_drops_link",
        "paragraph": "The deer was managed.",
        "claim_type": "weather_observation",
        "claim_links": [
            {"surface": "deer", "normalized": "deer",
             "relation": "MANAGEMENT_TARGET", "span": (4, 8)},
        ],
        "mentions": [
            {"surface": "deer", "normalized": "deer", "span": (4, 8)},
        ],
        "resolutions": {
            (4, 8): ("species_deer", "Species", "REFERS_TO"),
        },
        "expected_entity_links": [],
        "expected_diagnostics": ["RELATION_COMPATIBILITY_FORBIDDEN"],
    },
    # ------------------------------------------------------------------
    # 4. Weak compatibility → link kept, confidence penalised −0.10
    #    habitat_condition + SPECIES_FOCUS = weak
    # ------------------------------------------------------------------
    {
        "id": "weak_compat_penalizes_confidence",
        "paragraph": "Mallards were found in the wetland.",
        "claim_type": "habitat_condition",
        "claim_links": [
            {"surface": "Mallards", "normalized": "mallards",
             "relation": "SPECIES_FOCUS", "span": (0, 8)},
        ],
        "mentions": [
            {"surface": "Mallards", "normalized": "mallards", "span": (0, 8)},
        ],
        "resolutions": {
            (0, 8): ("species_mallard", "Species", "REFERS_TO"),
        },
        "expected_entity_links": [("species_mallard", "SPECIES_FOCUS")],
        "expected_diagnostics": ["RELATION_COMPATIBILITY_WEAK"],
        "expected_confidence_after": 0.80,
    },
    # ------------------------------------------------------------------
    # 5. Two candidates, same normalized form, no entity_type_hint
    #    → AMBIGUOUS_FALLBACK
    # ------------------------------------------------------------------
    {
        "id": "ambiguous_same_normalized_form_no_hint",
        "paragraph": "Marsh near marsh dried.",
        "claim_type": "habitat_condition",
        "claim_links": [
            {"surface": "marsh", "normalized": "marsh", "relation": "HABITAT_FOCUS"},
        ],
        "mentions": [
            {"surface": "Marsh", "normalized": "marsh", "span": (0, 5)},
            {"surface": "marsh", "normalized": "marsh", "span": (11, 16)},
        ],
        "resolutions": {
            (0, 5):  ("habitat_marsh_a", "Habitat", "REFERS_TO"),
            (11, 16): ("habitat_marsh_b", "Habitat", "REFERS_TO"),
        },
        "expected_entity_links": [],
        "expected_diagnostics": ["AMBIGUOUS_FALLBACK"],
    },
    # ------------------------------------------------------------------
    # 6. entity_type_hint disambiguates two candidates of different types
    #    public_contact + LOCATION_FOCUS = weak → penalty too
    # ------------------------------------------------------------------
    {
        "id": "entity_type_hint_breaks_ambiguity",
        "paragraph": "Turnbull and Turnbull were discussed.",
        "claim_type": "public_contact",
        "claim_links": [
            {"surface": "Turnbull", "normalized": "turnbull",
             "relation": "LOCATION_FOCUS", "hint": "Refuge"},
        ],
        "mentions": [
            {"surface": "Turnbull", "normalized": "turnbull", "span": (0, 8)},
            {"surface": "Turnbull", "normalized": "turnbull", "span": (13, 21)},
        ],
        "resolutions": {
            (0, 8):  ("place_turnbull",  "Place",  "REFERS_TO"),
            (13, 21): ("refuge_turnbull", "Refuge", "REFERS_TO"),
        },
        "expected_entity_links": [("refuge_turnbull", "LOCATION_FOCUS")],
        "expected_diagnostics": ["RELATION_COMPATIBILITY_WEAK"],
        "expected_confidence_after": 0.80,
    },
    # ------------------------------------------------------------------
    # 7. Resolved entity type conflicts with relation's allowed types
    #    SPECIES_FOCUS allows {Species}; entity is Habitat → RELATION_TYPE_CONFLICT
    # ------------------------------------------------------------------
    {
        "id": "relation_type_conflict",
        "paragraph": "Wetland habitat was flooded.",
        "claim_type": "species_presence",
        "claim_links": [
            {"surface": "Wetland", "normalized": "wetland",
             "relation": "SPECIES_FOCUS", "span": (0, 7)},
        ],
        "mentions": [
            {"surface": "Wetland", "normalized": "wetland", "span": (0, 7)},
        ],
        "resolutions": {
            (0, 7): ("habitat_wetland", "Habitat", "REFERS_TO"),
        },
        "expected_entity_links": [],
        "expected_diagnostics": ["RELATION_TYPE_CONFLICT"],
    },
    # ------------------------------------------------------------------
    # 8. entity_type_hint names a type absent from resolved candidates
    #    → TYPE_HINT_CONFLICT
    # ------------------------------------------------------------------
    {
        "id": "type_hint_conflict_no_matching_entity",
        "paragraph": "Mallards were observed.",
        "claim_type": "species_presence",
        "claim_links": [
            {"surface": "Mallards", "normalized": "mallards",
             "relation": "SPECIES_FOCUS", "hint": "Habitat"},
        ],
        "mentions": [
            {"surface": "Mallards", "normalized": "mallards", "span": (0, 8)},
        ],
        "resolutions": {
            (0, 8): ("species_mallard", "Species", "REFERS_TO"),
        },
        "expected_entity_links": [],
        "expected_diagnostics": ["TYPE_HINT_CONFLICT"],
    },
    # ------------------------------------------------------------------
    # 9. Sentence-level fallback: mention's normalized_form differs from
    #    the claim link's normalized_form but appears in normalized_sentence
    #    (in-span path — resolved_in_span is non-empty)
    # ------------------------------------------------------------------
    {
        "id": "sentence_fallback_normalized_form_mismatch_in_span",
        "paragraph": "A fire at Turnbull NWR burned.",
        # claim link says "turnbull"; mention normalises to "turnbull nwr"
        # → normalized match fails; sentence fallback finds it via sentence text
        "normalized_sentence": "a fire at turnbull nwr burned.",
        "claim_type": "fire_incident",
        "claim_links": [
            {"surface": "Turnbull", "normalized": "turnbull", "relation": "LOCATION_FOCUS"},
        ],
        "mentions": [
            {"surface": "Turnbull NWR", "normalized": "turnbull nwr", "span": (10, 22)},
        ],
        "resolutions": {
            (10, 22): ("refuge_turnbull", "Refuge", "REFERS_TO"),
        },
        "expected_entity_links": [("refuge_turnbull", "LOCATION_FOCUS")],
        "expected_diagnostics": [],
    },
    # ------------------------------------------------------------------
    # 10. Sentence-level fallback: mention is OUTSIDE the claim evidence
    #     span → paragraph-widening sub-path
    # ------------------------------------------------------------------
    {
        "id": "sentence_fallback_mention_outside_evidence_span",
        "paragraph": "Fire burned all grass. Turnbull NWR was affected.",
        # evidence span covers only the first sentence
        "evidence_end": 22,
        "normalized_sentence": "fire burned all grass at turnbull nwr",
        "claim_type": "fire_incident",
        "claim_links": [
            {"surface": "Turnbull", "normalized": "turnbull", "relation": "LOCATION_FOCUS"},
        ],
        "mentions": [
            {"surface": "Turnbull NWR", "normalized": "turnbull nwr", "span": (23, 35)},
        ],
        "resolutions": {
            (23, 35): ("refuge_turnbull", "Refuge", "REFERS_TO"),
        },
        "expected_entity_links": [("refuge_turnbull", "LOCATION_FOCUS")],
        "expected_diagnostics": [],
    },
    # ------------------------------------------------------------------
    # 11. SurveyMethod entity with REFERS_TO resolution passes the guard
    #     "Birds were counted using line transects."  line transects=(25,39)
    # ------------------------------------------------------------------
    {
        "id": "survey_method_refers_to_matched",
        "paragraph": "Birds were counted using line transects.",
        "claim_type": "species_presence",
        "claim_links": [
            {"surface": "line transects", "normalized": "line transects",
             "relation": "METHOD_FOCUS", "span": (25, 39)},
        ],
        "mentions": [
            {"surface": "line transects", "normalized": "line transects", "span": (25, 39)},
        ],
        "resolutions": {
            (25, 39): ("method_line_transect", "SurveyMethod", "REFERS_TO"),
        },
        "expected_entity_links": [("method_line_transect", "METHOD_FOCUS")],
        "expected_diagnostics": [],
    },
    # ------------------------------------------------------------------
    # 12. SurveyMethod entity with POSSIBLY_REFERS_TO is blocked by the
    #     guard; mention's normalized_form absent from normalized_sentence
    #     so widening also fails → NO_RESOLVED_MENTION
    # ------------------------------------------------------------------
    {
        "id": "survey_method_possibly_refers_to_blocked",
        "paragraph": "Birds were counted using line transects.",
        "normalized_sentence": "mallards were found this year",
        "claim_type": "species_presence",
        "claim_links": [
            {"surface": "line transects", "normalized": "line transects",
             "relation": "METHOD_FOCUS", "span": (25, 39)},
        ],
        "mentions": [
            {"surface": "line transects", "normalized": "line transects", "span": (25, 39)},
        ],
        "resolutions": {
            (25, 39): ("method_line_transect", "SurveyMethod", "POSSIBLY_REFERS_TO"),
        },
        "expected_entity_links": [],
        "expected_diagnostics": ["NO_RESOLVED_MENTION"],
    },
    # ------------------------------------------------------------------
    # 13. preferred_entity_types reranking: two candidates (Place, Refuge),
    #     preferred order puts Place first, but no hint → still AMBIGUOUS
    # ------------------------------------------------------------------
    {
        "id": "preferred_reranking_two_types_no_hint_ambiguous",
        "paragraph": "At Turnbull Turnbull fire burned.",
        "claim_type": "fire_incident",   # preferred: [Place, Refuge, Habitat, Activity]
        "claim_links": [
            {"surface": "Turnbull", "normalized": "turnbull", "relation": "LOCATION_FOCUS"},
        ],
        "mentions": [
            {"surface": "Turnbull", "normalized": "turnbull", "span": (3, 11)},
            {"surface": "Turnbull", "normalized": "turnbull", "span": (12, 20)},
        ],
        "resolutions": {
            (3, 11):  ("place_turnbull",  "Place",  "REFERS_TO"),
            (12, 20): ("refuge_turnbull", "Refuge", "REFERS_TO"),
        },
        "expected_entity_links": [],
        "expected_diagnostics": ["AMBIGUOUS_FALLBACK"],
    },
    # ------------------------------------------------------------------
    # 14. Same two-candidate setup as #13; entity_type_hint picks Refuge
    #     fire_incident + LOCATION_FOCUS = strong → no penalty
    # ------------------------------------------------------------------
    {
        "id": "preferred_reranking_hint_picks_refuge",
        "paragraph": "At Turnbull Turnbull fire burned.",
        "claim_type": "fire_incident",
        "claim_links": [
            {"surface": "Turnbull", "normalized": "turnbull",
             "relation": "LOCATION_FOCUS", "hint": "Refuge"},
        ],
        "mentions": [
            {"surface": "Turnbull", "normalized": "turnbull", "span": (3, 11)},
            {"surface": "Turnbull", "normalized": "turnbull", "span": (12, 20)},
        ],
        "resolutions": {
            (3, 11):  ("place_turnbull",  "Place",  "REFERS_TO"),
            (12, 20): ("refuge_turnbull", "Refuge", "REFERS_TO"),
        },
        "expected_entity_links": [("refuge_turnbull", "LOCATION_FOCUS")],
        "expected_diagnostics": [],
    },
    # ------------------------------------------------------------------
    # 15. Compound: sentence-level fallback fires AND weak-compat penalty
    #     applied in the same resolution.
    #     public_contact + LOCATION_FOCUS = weak.
    #     Claim link normalised to "turnbull nwr"; mention normalises to
    #     "turnbull" (no exact/normalized match) → sentence fallback;
    #     "turnbull" ∈ "a meeting was held at turnbull nwr" → found; weak penalty.
    #     "A meeting was held at Turnbull."  Turnbull=(22,30) len=31
    # ------------------------------------------------------------------
    {
        "id": "compound_sentence_fallback_plus_weak_compat",
        "paragraph": "A meeting was held at Turnbull.",
        "normalized_sentence": "a meeting was held at turnbull nwr",
        "claim_type": "public_contact",
        "claim_links": [
            {"surface": "Turnbull NWR", "normalized": "turnbull nwr",
             "relation": "LOCATION_FOCUS"},
        ],
        "mentions": [
            {"surface": "Turnbull", "normalized": "turnbull", "span": (22, 30)},
        ],
        "resolutions": {
            (22, 30): ("refuge_turnbull", "Refuge", "REFERS_TO"),
        },
        "expected_entity_links": [("refuge_turnbull", "LOCATION_FOCUS")],
        "expected_diagnostics": ["RELATION_COMPATIBILITY_WEAK"],
        "expected_confidence_after": 0.80,
    },
    # ------------------------------------------------------------------
    # 16. No span, normalized_form mismatch, mention absent from
    #     normalized_sentence → complete fallthrough → NO_RESOLVED_MENTION
    # ------------------------------------------------------------------
    {
        "id": "no_span_no_normalized_match_fallthrough",
        "paragraph": "The wetland dried up.",
        # claim link says "marsh"; only mention is "wetland";
        # normalized_sentence also uses "marsh" → widening misses "wetland"
        "normalized_sentence": "the marsh dried up",
        "claim_type": "habitat_condition",
        "claim_links": [
            {"surface": "marsh", "normalized": "marsh", "relation": "HABITAT_FOCUS"},
        ],
        "mentions": [
            {"surface": "wetland", "normalized": "wetland", "span": (4, 11)},
        ],
        "resolutions": {
            (4, 11): ("habitat_wetland", "Habitat", "REFERS_TO"),
        },
        "expected_entity_links": [],
        "expected_diagnostics": ["NO_RESOLVED_MENTION"],
    },
    # ------------------------------------------------------------------
    # 17. Two claim links on one claim; both resolve successfully
    #     "Mallards were observed at Turnbull Refuge."
    #     Mallards=(0,8)  Turnbull Refuge=(26,41)  len=42
    # ------------------------------------------------------------------
    {
        "id": "multiple_claim_links_both_resolved",
        "paragraph": "Mallards were observed at Turnbull Refuge.",
        "claim_type": "species_presence",
        "claim_links": [
            {"surface": "Mallards",        "normalized": "mallards",
             "relation": "SPECIES_FOCUS",  "span": (0, 8)},
            {"surface": "Turnbull Refuge", "normalized": "turnbull refuge",
             "relation": "LOCATION_FOCUS", "span": (26, 41)},
        ],
        "mentions": [
            {"surface": "Mallards",        "normalized": "mallards",        "span": (0, 8)},
            {"surface": "Turnbull Refuge", "normalized": "turnbull refuge", "span": (26, 41)},
        ],
        "resolutions": {
            (0, 8):   ("species_mallard", "Species", "REFERS_TO"),
            (26, 41): ("refuge_turnbull", "Refuge",  "REFERS_TO"),
        },
        "expected_entity_links": [
            ("species_mallard", "SPECIES_FOCUS"),
            ("refuge_turnbull", "LOCATION_FOCUS"),
        ],
        "expected_diagnostics": [],
    },
]


@pytest.mark.parametrize("case", CLAIM_LINK_EDGE_CASES, ids=lambda c: c["id"])
def test_claim_link_resolution(case) -> None:
    paragraph = case["paragraph"]
    source_sentence = case.get("source_sentence", paragraph)
    normalized_sentence = case.get("normalized_sentence", paragraph.lower())
    evidence_start = case.get("evidence_start", 0)
    evidence_end = case.get("evidence_end", len(paragraph))

    claim_links = [
        ClaimLinkDraft(
            surface_form=cl["surface"],
            normalized_form=cl["normalized"],
            relation_type=cl["relation"],
            start_offset=cl["span"][0] if cl.get("span") else None,
            end_offset=cl["span"][1] if cl.get("span") else None,
            entity_type_hint=cl.get("hint"),
        )
        for cl in case["claim_links"]
    ]

    entity_records: dict[str, EntityRecord] = {}
    for entity_id, entity_type, _ in case["resolutions"].values():
        if entity_id not in entity_records:
            entity_records[entity_id] = EntityRecord(entity_id, entity_type, entity_id, entity_id)

    resolver = MappingResolver({
        (start, end): (entity_records[entity_id], rel_type)
        for (start, end), (entity_id, entity_type, rel_type) in case["resolutions"].items()
    })

    structure = _single_paragraph_structure(paragraph)

    semantic = extract_semantic(
        structure,
        claim_extractor=StaticClaimExtractor([
            ClaimDraft(
                claim_type=case["claim_type"],
                source_sentence=source_sentence,
                normalized_sentence=normalized_sentence,
                epistemic_status="certain",
                extraction_confidence=0.90,
                evidence_start=evidence_start,
                evidence_end=evidence_end,
                claim_links=claim_links,
            )
        ]),
        mention_extractor=StaticMentionExtractor([
            MentionDraft(m["surface"], m["normalized"], m["span"][0], m["span"][1], 0.90, [])
            for m in case["mentions"]
        ]),
        resolver=resolver,
        run_overrides={
            "run_id": f"run_{case['id']}",
            "run_timestamp": "2026-01-01T00:00:00+00:00",
        },
    )

    actual_links = [(lnk.entity_id, lnk.relation_type) for lnk in semantic.claim_entity_links]
    actual_diags = sorted(d.diagnostic_code for d in semantic.claim_link_diagnostics)

    assert actual_links == case["expected_entity_links"]
    assert actual_diags == sorted(case["expected_diagnostics"])

    if "expected_confidence_after" in case:
        assert semantic.claims[0].extraction_confidence == pytest.approx(
            case["expected_confidence_after"]
        )
