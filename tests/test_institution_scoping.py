"""A non-admin account with no institution must fail loudly, not silently.

Retrieval scopes documents with `d.institution_id IN $institution_ids`.
`UserContext` derives that list from the user's `institution_id`, and an empty
one yields an EMPTY list — and `x IN []` matches nothing in Cypher.

So a tester created without an institution logs in fine and then gets zero
results for every query, with no error. The archive looks empty rather than
the account looking misconfigured, which is the worst possible failure mode
for a pilot: it burns tester goodwill on a five-minute settings fix.

Verified against a real graph on 2026-07-18: `institution_ids=[]` returned 0
of 112 Turnbull documents.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

from gemynd.auth.models import UserContext  # noqa: E402
from gemynd.retrieval.web.app import _apply_composite_scope  # noqa: E402


def _user(role: str, institution_id: str = "") -> UserContext:
    return UserContext(
        role=role,
        institution_id=institution_id,
        permitted_levels=["public"],
        user_id="u-1",
        email="tester@example.org",
    )


def _composite(*institution_ids: str):
    return SimpleNamespace(institution_ids=list(institution_ids))


class TestEmptyInstitutionIsRejected:
    def test_context_derives_empty_permitted_list(self):
        """The root cause: no institution -> no permitted institutions."""
        assert _user("researcher").permitted_institution_ids == []

    @pytest.mark.parametrize("role", ["readonly", "researcher", "archivist"])
    def test_unscoped_non_admin_gets_explanatory_403(self, role):
        with pytest.raises(HTTPException) as exc:
            _apply_composite_scope(_user(role), _composite("turnbull"), admin_only=True)
        assert exc.value.status_code == 403
        detail = str(exc.value.detail)
        # The message must point at the account, not at the archive.
        assert "not assigned to a collection" in detail
        assert "not an empty archive" in detail

    def test_scoped_non_admin_passes_through_unchanged(self):
        user = _apply_composite_scope(
            _user("researcher", "turnbull"), _composite("turnbull", "spokane"), admin_only=True
        )
        # Tenant isolation: NOT widened to every registered corpus.
        assert user.permitted_institution_ids == ["turnbull"]


class TestAdminScoping:
    def test_admin_is_widened_to_every_registered_corpus(self):
        user = _apply_composite_scope(
            _user("admin"), _composite("turnbull", "spokane"), admin_only=True
        )
        assert user.permitted_institution_ids == ["turnbull", "spokane"]

    def test_admin_without_institution_is_not_rejected(self):
        """Admins legitimately carry no home institution."""
        user = _apply_composite_scope(_user("admin"), _composite("turnbull"), admin_only=True)
        assert user.permitted_institution_ids == ["turnbull"]

    def test_admin_only_false_widens_any_role(self):
        user = _apply_composite_scope(
            _user("archivist", "turnbull"), _composite("turnbull", "spokane"), admin_only=False
        )
        assert user.permitted_institution_ids == ["turnbull", "spokane"]
