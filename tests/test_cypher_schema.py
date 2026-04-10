from gemynd.core.graph.cypher import INDEX_STATEMENTS, build_constraint_statements


def test_schema_contains_required_constraints_and_indexes() -> None:
    entity_labels = frozenset({
        "Refuge", "Place", "Person", "Organization", "Species",
        "Activity", "Period", "Habitat", "SurveyMethod",
    })
    schema = build_constraint_statements(entity_labels) + INDEX_STATEMENTS
    statements = "\n".join(schema)
    for label in [
        "Document",
        "Page",
        "Section",
        "Paragraph",
        "Annotation",
        "ExtractionRun",
        "Claim",
        "Measurement",
        "Mention",
        "Observation",
        "Year",
        "Refuge",
        "Place",
        "Person",
        "Organization",
        "Species",
        "Activity",
        "Period",
        "Habitat",
        "SurveyMethod",
    ]:
        assert f"FOR (n:{label}) REQUIRE" in statements

    assert "document_report_year" in statements
    assert "claim_type" in statements
    assert "measurement_name" in statements
    assert "period_dates" in statements
    assert "mention_ocr_suspect" in statements
    assert "observation_type" in statements
    assert "year_value" in statements
    assert "observation_year_int" in statements
