"""Unit tests for Layer 0: query intent classifier."""
from __future__ import annotations

import pytest

from gemynd.retrieval.classifier import classify_query


class TestBucketAssignment:
    def test_analytical_signals(self):
        intent = classify_query("how many mallard observations between 1940 and 1950?")
        assert intent.bucket == "analytical"
        assert intent.classifier_confidence > 0.5

    def test_conversational_signals(self):
        intent = classify_query("what did the 1942 report say about habitat conditions?")
        assert intent.bucket == "conversational"
        assert intent.classifier_confidence > 0.5

    def test_hybrid_signals(self):
        intent = classify_query(
            "what species reached their highest count between 1938 and 1945 "
            "and what evidence supports that?"
        )
        assert intent.bucket == "hybrid"

    def test_no_signals_defaults_conversational(self):
        intent = classify_query("turnbull")
        assert intent.bucket == "conversational"
        assert intent.classifier_confidence == 0.5

    def test_count_keyword_analytical(self):
        intent = classify_query("count the total nesting records per year")
        assert intent.bucket == "analytical"

    def test_describe_keyword_conversational(self):
        intent = classify_query("describe the wetland conditions in the 1950s")
        assert intent.bucket == "conversational"


class TestYearExtraction:
    def test_between_pattern(self):
        intent = classify_query("observations between 1940 and 1950")
        assert intent.year_min == 1940
        assert intent.year_max == 1950

    def test_single_year(self):
        intent = classify_query("the 1945 report mentions mallards")
        assert intent.year_min == 1945
        assert intent.year_max == 1945

    def test_explicit_year_range_overrides_text(self):
        intent = classify_query("between 1940 and 1950", year_range=(1960, 1970))
        assert intent.year_min == 1960
        assert intent.year_max == 1970

    def test_no_years(self):
        intent = classify_query("describe habitat conditions")
        assert intent.year_min is None
        assert intent.year_max is None

    def test_multiple_years_min_max(self):
        intent = classify_query("from 1935 to 1942 to 1950 records")
        assert intent.year_min == 1935
        assert intent.year_max == 1950


class TestEntityExtraction:
    def test_capitalised_entity_extracted(self):
        intent = classify_query("what did the Mallard population show in 1945?")
        assert any("Mallard" in e for e in intent.entities)

    def test_entity_hints_included(self):
        intent = classify_query("tell me about ducks", entity_hints=["mallard"])
        assert "mallard" in intent.entities

    def test_no_duplicate_hints(self):
        intent = classify_query("tell me about Mallard", entity_hints=["Mallard"])
        assert intent.entities.count("Mallard") == 1

    def test_common_words_not_extracted_as_entities(self):
        intent = classify_query(
            "Describe what was happening at Turnbull Refuge during the 1940s. "
            "What patterns stand out across that decade?"
        )
        extracted_lower = [e.lower() for e in intent.entities]
        for word in ("during", "describe", "across", "decade", "that"):
            assert word not in extracted_lower, (
                f"Common word '{word}' should not be extracted as an entity candidate"
            )

    def test_turnbull_not_extracted_when_in_extra_stopwords(self):
        intent = classify_query(
            "What happened at Turnbull Refuge in 1942?",
            extra_stopwords=frozenset({"turnbull"}),
        )
        extracted_lower = [e.lower() for e in intent.entities]
        # "turnbull" alone should not appear — only "Turnbull Refuge" as a phrase
        assert "turnbull" not in extracted_lower
        assert any("Turnbull" in e for e in intent.entities)

    def test_extra_stopwords_merged_with_base(self):
        """Domain-specific stopwords are merged with the base set."""
        intent = classify_query(
            "What myinstitution data shows about mallard",
            extra_stopwords=frozenset({"myinstitution"}),
        )
        extracted_lower = [e.lower() for e in intent.entities]
        assert "myinstitution" not in extracted_lower


# ---------------------------------------------------------------------------
# Tier 1 — structural-lookup intent detection
# ---------------------------------------------------------------------------

from gemynd.retrieval.classifier import detect_tier1_intent
from gemynd.retrieval.models import EntityContext, ResolvedEntity


def _resolved(entity_id: str = "person_abc", confidence: float = 1.0,
              relation: str = "REFERS_TO", entity_type: str = "Person",
              surface_form: str = "Edward Copeland") -> ResolvedEntity:
    return ResolvedEntity(
        surface_form=surface_form,
        entity_id=entity_id,
        entity_type=entity_type,
        resolution_confidence=confidence,
        resolution_relation=relation,
    )


class TestCrossCorpusOverlapDetection:
    """Layer 0 must flag queries asking for entities present in two corpora.

    The flag drives the CROSS_CORPUS_OVERLAP cascade entry in
    `_select_retrieval_plan`.  False positives are costly (they spend the
    cascade's first attempt on an empty result set), so the patterns require
    BOTH a join verb ("appear in both", "also appear", etc.) AND two distinct
    corpus references via doc_type/series aliases."""

    def test_canonical_failing_query_fires(self):
        intent = classify_query(
            "Which individuals named in Spokane newspaper coverage of labor "
            "disputes also appear as patentees in GLO records for the "
            "surrounding region?"
        )
        assert intent.cross_corpus_overlap is True

    def test_overlap_between_phrasing_fires(self):
        intent = classify_query(
            "Show me the overlap between Spokane Press articles and BLM GLO records."
        )
        assert intent.cross_corpus_overlap is True

    def test_who_appears_in_both_phrasing_fires(self):
        intent = classify_query(
            "Who appears in both the newspaper articles and the land patents?"
        )
        assert intent.cross_corpus_overlap is True

    def test_single_corpus_with_both_does_not_fire(self):
        """A query mentioning 'both' but only one corpus reference must NOT
        fire — this is the in-both-years false-positive case."""
        intent = classify_query(
            "What did the Spokane Press say about labor in both 1908 and 1909?"
        )
        assert intent.cross_corpus_overlap is False

    def test_two_corpora_without_join_verb_does_not_fire(self):
        """Just naming two corpora is not enough; the query must explicitly
        ask for entities appearing across them."""
        intent = classify_query(
            "Tell me about Spokane Press articles and BLM GLO records."
        )
        assert intent.cross_corpus_overlap is False

    def test_unrelated_query_does_not_fire(self):
        intent = classify_query("describe wetland conditions in 1950")
        assert intent.cross_corpus_overlap is False


class TestDetectTier1Intent:
    def test_how_many_with_single_high_confidence_entity_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("how many claims mention Edward Copeland?", ctx)
        assert intent is not None
        assert intent.operation == "count_claims_for_entity"
        assert intent.entity_id == "person_abc"
        assert intent.entity_name == "Edward Copeland"
        assert intent.entity_type == "Person"

    def test_no_resolved_entity_does_not_fire(self):
        ctx = EntityContext(resolved=[])
        assert detect_tier1_intent("how many claims about Spokane?", ctx) is None

    def test_multiple_resolved_entities_does_not_fire(self):
        # Ambiguity: refuses to short-circuit when more than one entity matched.
        ctx = EntityContext(resolved=[_resolved("a"), _resolved("b")])
        assert detect_tier1_intent("how many claims mention X?", ctx) is None

    def test_possibly_refers_to_does_not_fire(self):
        ctx = EntityContext(resolved=[_resolved(relation="POSSIBLY_REFERS_TO")])
        assert detect_tier1_intent("how many claims mention X?", ctx) is None

    def test_low_confidence_does_not_fire(self):
        # 0.84 is below the resolver's REFERS_TO floor (0.85) — the
        # resolver should not have classified this as REFERS_TO in the
        # first place, but if a stale fixture passes one through, Tier 1
        # still refuses to short-circuit.
        ctx = EntityContext(resolved=[_resolved(confidence=0.84)])
        assert detect_tier1_intent("how many claims mention X?", ctx) is None

    def test_substring_match_confidence_fires(self):
        # The resolver awards 0.88 for substring matches like "Turnbull
        # Refuge" → canonical "Turnbull" entity.  This is above the
        # REFERS_TO floor and should route to Tier 1 — gating it out via
        # a stricter Tier 1 threshold blocks legitimate surface-form
        # variants without buying additional safety.
        ctx = EntityContext(resolved=[_resolved(confidence=0.88)])
        assert detect_tier1_intent("how many claims mention X?", ctx) is not None

    def test_query_not_starting_with_how_many_does_not_fire(self):
        # "how many" must lead the query — buried mid-sentence falls through.
        ctx = EntityContext(resolved=[_resolved()])
        assert detect_tier1_intent("Tell me how many claims mention X.", ctx) is None
        assert detect_tier1_intent("Describe Edward Copeland's role.", ctx) is None

    def test_leading_whitespace_is_tolerated(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("   how many claims mention X?", ctx)
        assert intent is not None

    def test_case_insensitive(self):
        ctx = EntityContext(resolved=[_resolved()])
        assert detect_tier1_intent("HOW MANY claims mention X?", ctx) is not None
        assert detect_tier1_intent("How Many claims mention X?", ctx) is not None


class TestDetectTier1YearRange:
    def test_what_years_with_entity_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("what years was Edward Copeland documented?", ctx)
        assert intent is not None
        assert intent.operation == "year_range_for_entity"
        assert intent.entity_id == "person_abc"

    def test_when_was_with_entity_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("when was Edward Copeland active?", ctx)
        assert intent is not None
        assert intent.operation == "year_range_for_entity"

    def test_date_range_for_with_entity_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("date range for Edward Copeland", ctx)
        assert intent is not None
        assert intent.operation == "year_range_for_entity"

    def test_year_range_without_entity_does_not_fire(self):
        ctx = EntityContext(resolved=[])
        assert detect_tier1_intent("what years are in the corpus?", ctx) is None

    def test_year_range_buried_in_sentence_does_not_fire(self):
        ctx = EntityContext(resolved=[_resolved()])
        # Must be leading-anchored.
        assert detect_tier1_intent("Tell me what years X covers.", ctx) is None


class TestDetectTier1ListDocumentsForEntity:
    def test_list_documents_about_entity_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("list documents about Edward Copeland", ctx)
        assert intent is not None
        assert intent.operation == "list_documents_for_entity"

    def test_show_me_documents_mentioning_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("show me documents mentioning Edward Copeland", ctx)
        assert intent is not None
        assert intent.operation == "list_documents_for_entity"

    def test_which_documents_discussing_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("which documents discussing Edward Copeland", ctx)
        assert intent is not None
        assert intent.operation == "list_documents_for_entity"

    def test_what_documents_on_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("what documents on Edward Copeland", ctx)
        assert intent is not None
        assert intent.operation == "list_documents_for_entity"

    def test_list_documents_without_connective_does_not_fire(self):
        # "list documents" alone is too generic — needs about/mentioning/etc.
        ctx = EntityContext(resolved=[_resolved()])
        assert detect_tier1_intent("list documents", ctx) is None

    def test_list_documents_without_entity_does_not_fire(self):
        ctx = EntityContext(resolved=[])
        assert detect_tier1_intent("list documents about something", ctx) is None


class TestDetectTier1ListDocumentsForYear:
    def test_list_documents_from_year_fires(self):
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent(
            "list documents from 1909", ctx, year_min=1909, year_max=1909
        )
        assert intent is not None
        assert intent.operation == "list_documents_for_year"
        assert intent.target_year == 1909
        assert intent.entity_type == "Year"

    def test_show_me_documents_in_year_fires(self):
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent(
            "show me documents in 1909", ctx, year_min=1909, year_max=1909
        )
        assert intent is not None
        assert intent.operation == "list_documents_for_year"

    def test_what_documents_published_in_year_fires(self):
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent(
            "what documents published in 1909", ctx, year_min=1909, year_max=1909
        )
        assert intent is not None

    def test_year_range_does_not_fire(self):
        # year_min != year_max means a range was specified, not a single year
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent(
            "list documents from 1900 to 1910", ctx, year_min=1900, year_max=1910
        )
        assert intent is None

    def test_no_year_does_not_fire(self):
        ctx = EntityContext(resolved=[])
        assert detect_tier1_intent("list documents from somewhere", ctx) is None


class TestDetectTier1LocationsForEntity:
    def test_where_does_appear_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("where does Edward Copeland appear", ctx)
        assert intent is not None
        assert intent.operation == "locations_for_entity"

    def test_where_is_mentioned_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("where is Edward Copeland mentioned", ctx)
        assert intent is not None
        assert intent.operation == "locations_for_entity"

    def test_what_places_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("what places are associated with Edward Copeland", ctx)
        assert intent is not None
        assert intent.operation == "locations_for_entity"

    def test_which_locations_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("which locations are linked to him", ctx)
        assert intent is not None
        assert intent.operation == "locations_for_entity"

    def test_locations_without_entity_does_not_fire(self):
        ctx = EntityContext(resolved=[])
        assert detect_tier1_intent("where does X appear", ctx) is None


class TestDetectTier1CoOccurrence:
    def test_who_appears_with_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("who appears with Edward Copeland", ctx)
        assert intent is not None
        assert intent.operation == "co_occurring_entities_for_entity"

    def test_who_is_mentioned_with_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("who is mentioned with Edward Copeland", ctx)
        assert intent is not None

    def test_what_entities_associated_with_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("what entities are associated with him", ctx)
        assert intent is not None

    def test_co_occurs_with_hyphen_fires(self):
        ctx = EntityContext(resolved=[_resolved()])
        intent = detect_tier1_intent("who co-occurs with Edward", ctx)
        assert intent is not None

    def test_who_is_x_does_not_fire(self):
        # Vague biographical question — should fall through to synthesis.
        ctx = EntityContext(resolved=[_resolved()])
        assert detect_tier1_intent("who is Edward Copeland", ctx) is None


class TestDetectTier1FrequencyRanking:
    def test_top_n_people_fires(self):
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent("top 10 people by claim count", ctx)
        assert intent is not None
        assert intent.operation == "frequency_ranking_for_type"
        assert intent.target_label == "Person"
        assert intent.limit == 10

    def test_top_n_places_fires(self):
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent("top 25 places", ctx)
        assert intent is not None
        assert intent.target_label == "Place"
        assert intent.limit == 25

    def test_most_common_species_fires(self):
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent("most common species", ctx)
        assert intent is not None
        assert intent.target_label == "Species"
        assert intent.limit == 10  # default when N not specified

    def test_which_people_appear_most_fires(self):
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent("which people appear most", ctx)
        assert intent is not None
        assert intent.target_label == "Person"

    def test_unknown_type_word_does_not_fire(self):
        ctx = EntityContext(resolved=[])
        # "claims" is not in the type-keyword vocabulary.
        assert detect_tier1_intent("top 10 claims", ctx) is None

    def test_limit_capped_at_100(self):
        ctx = EntityContext(resolved=[])
        intent = detect_tier1_intent("top 99999 people", ctx)
        assert intent is not None
        assert intent.limit == 100  # sanity-capped

    def test_no_pattern_does_not_fire(self):
        ctx = EntityContext(resolved=[])
        assert detect_tier1_intent("show me people", ctx) is None
