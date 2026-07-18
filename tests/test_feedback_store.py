"""Tests for tester feedback capture.

A pilot's value is learning what breaks, so losing a report is worse than
most failures this codebase guards against — hence the emphasis below on
never dropping a submission (unknown category, odd whitespace, long text).
"""
from __future__ import annotations

import pytest

from gemynd.retrieval.feedback_store import FEEDBACK_CATEGORIES, FeedbackStore


@pytest.fixture()
def store(tmp_path):
    s = FeedbackStore(tmp_path / "feedback.db")
    yield s
    s.close()


class TestAdd:
    def test_roundtrip(self, store):
        fid = store.add(
            "The answer cited a document that doesn't mention the species.",
            category="wrong_answer",
            user_id="u-1",
            user_email="archivist@example.org",
            user_role="researcher",
            page_path="/",
            query_text="mallard counts 1938",
        )
        assert fid
        rows = store.list_recent()
        assert len(rows) == 1
        row = rows[0]
        assert row["feedback_id"] == fid
        assert row["category"] == "wrong_answer"
        assert row["user_email"] == "archivist@example.org"
        assert row["user_role"] == "researcher"
        assert row["page_path"] == "/"
        assert row["query_text"] == "mallard counts 1938"
        assert row["resolved"] == 0
        assert "doesn't mention" in row["message"]

    def test_unknown_category_falls_back_rather_than_raising(self, store):
        """Never lose a report to validation."""
        store.add("something odd", category="not-a-real-category")
        assert store.list_recent()[0]["category"] == "other"

    @pytest.mark.parametrize("category", FEEDBACK_CATEGORIES)
    def test_every_declared_category_is_accepted(self, store, category):
        store.add("msg", category=category)
        assert store.list_recent()[0]["category"] == category

    def test_message_is_trimmed_and_capped(self, store):
        store.add("   padded   ")
        assert store.list_recent()[0]["message"] == "padded"
        store.add("x" * 9000)
        assert len(store.list_recent()[0]["message"]) == 4000

    def test_anonymous_submission_is_allowed(self, store):
        """User context is best-effort; a report with none is still a report."""
        store.add("no user attached")
        row = store.list_recent()[0]
        assert row["user_id"] is None
        assert row["user_email"] is None

    def test_empty_query_text_stored_as_null_not_empty_string(self, store):
        store.add("msg", query_text="")
        assert store.list_recent()[0]["query_text"] is None


class TestListing:
    def test_newest_first(self, store):
        for i in range(3):
            store.add(f"report {i}")
        messages = [r["message"] for r in store.list_recent()]
        assert messages == ["report 2", "report 1", "report 0"]

    def test_limit_respected(self, store):
        for i in range(5):
            store.add(f"r{i}")
        assert len(store.list_recent(limit=2)) == 2

    def test_empty_store_returns_empty_list(self, store):
        assert store.list_recent() == []
        assert store.count_unresolved() == 0


class TestResolution:
    def test_mark_resolved_and_count(self, store):
        a = store.add("first")
        store.add("second")
        assert store.count_unresolved() == 2
        store.mark_resolved(a)
        assert store.count_unresolved() == 1
        assert not any(
            r["resolved"] and r["feedback_id"] != a for r in store.list_recent()
        )

    def test_exclude_resolved(self, store):
        a = store.add("first")
        store.add("second")
        store.mark_resolved(a)
        rows = store.list_recent(include_resolved=False)
        assert [r["message"] for r in rows] == ["second"]

    def test_unmark(self, store):
        a = store.add("first")
        store.mark_resolved(a)
        store.mark_resolved(a, resolved=False)
        assert store.count_unresolved() == 1


class TestSchemaRegistration:
    def test_registered_with_database_manager(self, tmp_path):
        """The store must migrate through the shared manager like every other."""
        import dataclasses

        from gemynd.shared.database_manager import DatabaseManager
        from gemynd.shared.settings import Settings

        settings = dataclasses.replace(
            Settings(), feedback_db=str(tmp_path / "feedback.db")
        )
        mgr = DatabaseManager(settings)
        mgr.run_migrations()
        conn = mgr.get_connection("feedback")
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "feedback" in tables
        # skip_init: the manager already ran the migration.
        s = FeedbackStore(conn=conn, skip_init=True)
        s.add("via manager")
        assert s.list_recent()[0]["message"] == "via manager"
        mgr.close_all()
