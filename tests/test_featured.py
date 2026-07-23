"""FeaturedStore: normalization, suggestions, and the curated answer cache."""
from __future__ import annotations

import json

import pytest

from gemynd.retrieval.featured import FeaturedStore, normalize_question


@pytest.fixture
def store(tmp_path):
    return FeaturedStore(tmp_path / "featured.db")


def test_normalize_folds_case_whitespace_and_trailing_punctuation():
    n = normalize_question
    assert n("What happened at Turnbull?") == "what happened at turnbull"
    assert n("  what   happened  at Turnbull ") == "what happened at turnbull"
    assert n("What happened at Turnbull!!!") == "what happened at turnbull"
    # a tapped suggestion and its stored form must collide
    assert n("Which animals are most prevalent in Turnbull?") \
        == n("which animals are most prevalent in turnbull")


def test_empty_store_is_a_no_op(store):
    assert store.list_questions() == []
    assert store.get_cached("anything") is None


def test_add_shows_as_suggestion_but_is_not_cached(store):
    store.add("What animals live at Turnbull?")
    qs = store.list_questions()
    assert len(qs) == 1
    assert qs[0]["display"] == "What animals live at Turnbull?"
    assert qs[0]["cached"] is False
    # not captured yet => a query still runs live (cache miss)
    assert store.get_cached("What animals live at Turnbull?") is None


def test_capture_makes_it_answer_instantly_on_normalised_match(store):
    payload = {"answer": "Mostly waterfowl.", "answer_sentences": []}
    store.capture("What animals live at Turnbull?", json.dumps(payload))
    # exact
    assert store.get_cached("What animals live at Turnbull?") == payload
    # case / whitespace / trailing punctuation all still hit
    assert store.get_cached("  what animals live at turnbull  ") == payload
    assert store.get_cached("WHAT ANIMALS LIVE AT TURNBULL!!") == payload
    # a genuine paraphrase misses (never serve a wrong answer)
    assert store.get_cached("what birds are at the refuge") is None
    assert store.list_questions()[0]["cached"] is True


def test_capture_upserts_over_add_and_preserves_display(store):
    store.add("What animals live at Turnbull?")
    store.capture("What animals live at Turnbull?", json.dumps({"answer": "x"}))
    qs = store.list_questions()
    assert len(qs) == 1  # same normalised key => one row, not two
    assert qs[0]["cached"] is True


def test_add_after_capture_keeps_the_captured_answer(store):
    store.capture("Q?", json.dumps({"answer": "kept"}))
    store.add("Q?")  # re-adding as a suggestion must not wipe the answer
    assert store.get_cached("Q?") == {"answer": "kept"}


def test_malformed_cached_json_is_treated_as_a_miss_not_an_error(store):
    # capture() validates, so inject a bad row directly to simulate corruption
    import sqlite3
    conn = sqlite3.connect(store._path)
    conn.execute(
        "INSERT INTO featured (question_norm, question_display, response_json, updated_at) "
        "VALUES (?,?,?,?)",
        ("broken", "broken", "{not json", "2026-01-01"),
    )
    conn.commit(); conn.close()
    assert store.get_cached("broken") is None  # miss, no exception


def test_capture_rejects_malformed_json_loudly(store):
    with pytest.raises(json.JSONDecodeError):
        store.capture("Q?", "{not json")


def test_remove(store):
    store.add("Q?")
    assert store.remove("q?") is True   # normalised match
    assert store.remove("q?") is False
    assert store.list_questions() == []


def test_sort_order_controls_suggestion_order(store):
    store.add("second", sort_order=2)
    store.add("first", sort_order=1)
    assert [q["display"] for q in store.list_questions()] == ["first", "second"]


def test_cached_dict_round_trips_through_queryresponse():
    """A captured answer must reconstruct as a QueryResponse, since the request
    path returns QueryResponse.model_validate(cached). Real captures store a full
    model_dump_json(), so round-trip a full dump (not a hand-written subset)."""
    from gemynd.retrieval.web.app import QueryResponse  # noqa: PLC0415
    original = QueryResponse(
        answer="Mostly waterfowl.",
        confidence_assessment="high",
        supporting_claim_ids=["claim_1", "claim_2"],
        caveats=["OCR quality varies."],
    )
    dumped = json.loads(original.model_dump_json())   # what capture() persists
    restored = QueryResponse.model_validate(dumped)   # what the request path does
    assert restored.answer == "Mostly waterfowl."
    assert restored.supporting_claim_ids == ["claim_1", "claim_2"]


def test_store_get_cached_feeds_queryresponse_validate():
    """End-to-end of the request-path contract: capture a full QueryResponse
    dump, read it back via get_cached, and it validates."""
    from gemynd.retrieval.web.app import QueryResponse  # noqa: PLC0415
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    try:
        st = FeaturedStore(path)
        qr = QueryResponse(
            answer="A.", confidence_assessment="medium",
            supporting_claim_ids=[], caveats=[],
        )
        st.capture("Featured Q?", qr.model_dump_json())
        cached = st.get_cached("featured q")
        assert cached is not None
        assert QueryResponse.model_validate(cached).answer == "A."
    finally:
        os.unlink(path)
