from gemynd.ingest.extractors.mention_extractor import RuleBasedMentionExtractor


def test_mention_offsets_and_ocr_flags() -> None:
    extractor = RuleBasedMentionExtractor()
    text = "Tumbull refuge received mallard counts from Spokane Bird Club."
    mentions = extractor.extract(text)
    assert mentions

    for mention in mentions:
        assert text[mention.start_offset : mention.end_offset] == mention.surface_form

    assert any(mention.ocr_flags for mention in mentions if mention.normalized_form == "tumbull")
