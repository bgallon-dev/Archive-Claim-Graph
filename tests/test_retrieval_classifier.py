"""Unit tests for Layer 0: query intent classifier."""
from __future__ import annotations

import pytest

from graphrag_pipeline.retrieval.classifier import classify_query


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
