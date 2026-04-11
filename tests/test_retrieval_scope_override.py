"""Tests for the composite-scope override applied to UserContext at
request time in the retrieval web app.

The retrieval layer's Cypher templates accept ``$institution_ids`` as a
list, but every per-request handler passes ``user.permitted_institution_ids``
— which ``UserContext.__post_init__`` defaults to a single-element list
containing the user's home institution. ``_apply_composite_scope``
re-expands this to the full registered-corpus list for admin callers so
cross-corpus retrieval works, while leaving non-admin roles pinned for
tenant isolation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from gemynd.auth.models import UserContext
from gemynd.retrieval.web.app import _apply_composite_scope


@dataclass
class _FakeComposite:
    institution_ids: list[str]


def _make_user(role: str, institution_id: str = "turnbull") -> UserContext:
    return UserContext(
        role=role,
        institution_id=institution_id,
        permitted_levels=["public"],
    )


class TestApplyCompositeScope:
    def test_admin_gets_full_composite_list(self):
        user = _make_user("admin")
        assert user.permitted_institution_ids == ["turnbull"]

        composite = _FakeComposite(institution_ids=["turnbull", "spokane_newspaper"])
        result = _apply_composite_scope(user, composite, admin_only=True)

        assert result is user
        assert result.permitted_institution_ids == ["turnbull", "spokane_newspaper"]

    def test_readonly_user_stays_pinned_when_admin_only(self):
        user = _make_user("readonly", institution_id="turnbull")
        composite = _FakeComposite(institution_ids=["turnbull", "spokane_newspaper"])

        result = _apply_composite_scope(user, composite, admin_only=True)

        assert result.permitted_institution_ids == ["turnbull"]

    def test_archivist_stays_pinned_when_admin_only(self):
        user = _make_user("archivist", institution_id="spokane_newspaper")
        composite = _FakeComposite(institution_ids=["turnbull", "spokane_newspaper"])

        result = _apply_composite_scope(user, composite, admin_only=True)

        assert result.permitted_institution_ids == ["spokane_newspaper"]

    def test_admin_only_false_expands_any_role(self):
        """admin_only=False is used by require_admin_scoped, where the admin
        role is already guaranteed by require_admin. The helper itself should
        still honour the flag for any role.
        """
        user = _make_user("readonly", institution_id="turnbull")
        composite = _FakeComposite(institution_ids=["turnbull", "spokane_newspaper"])

        result = _apply_composite_scope(user, composite, admin_only=False)

        assert result.permitted_institution_ids == ["turnbull", "spokane_newspaper"]

    def test_none_composite_is_safe(self):
        """If the web app hasn't finished its lifespan startup yet, state
        may not have composite_config populated. The helper must not crash.
        """
        user = _make_user("admin")
        result = _apply_composite_scope(user, None, admin_only=True)
        assert result.permitted_institution_ids == ["turnbull"]

    def test_empty_composite_zeros_out_list(self):
        """An empty composite (no corpora registered) translates to an empty
        list. Downstream Cypher treats this as 'any institution' via the
        ``size($institution_ids) = 0`` branch, which is the correct
        degenerate behaviour.
        """
        user = _make_user("admin")
        composite = _FakeComposite(institution_ids=[])

        result = _apply_composite_scope(user, composite, admin_only=True)

        assert result.permitted_institution_ids == []

    def test_scope_copy_is_independent_of_composite(self):
        """The helper must copy the composite list so later mutation of the
        composite (unlikely but possible) does not leak into the user's
        per-request scope.
        """
        composite = _FakeComposite(institution_ids=["turnbull", "spokane_newspaper"])
        user = _make_user("admin")

        _apply_composite_scope(user, composite, admin_only=True)

        composite.institution_ids.append("intruder")
        assert "intruder" not in user.permitted_institution_ids


class TestDefaultBehaviorOfUserContext:
    """Guardrails against regressions in the UserContext default that the
    scope override depends on.
    """

    def test_default_permitted_institution_ids_is_single_home(self):
        user = _make_user("readonly", institution_id="spokane_newspaper")
        assert user.permitted_institution_ids == ["spokane_newspaper"]

    def test_empty_institution_yields_empty_list(self):
        user = UserContext(role="readonly", institution_id="", permitted_levels=["public"])
        assert user.permitted_institution_ids == []
