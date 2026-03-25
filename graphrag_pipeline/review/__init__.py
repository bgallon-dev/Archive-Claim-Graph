"""Anti-pattern review layer v1 – detect, propose, validate, review, export."""
from __future__ import annotations

from graphrag_pipeline.review.models import Proposal
from graphrag_pipeline.shared.settings import Settings


class ReviewService:
    """Programmatic interface to the review system.

    Usage::

        from graphrag_pipeline.shared.settings import Settings
        from graphrag_pipeline.review import ReviewService

        settings = Settings.from_env()
        svc = ReviewService(settings)
        proposals = svc.get_proposals()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _store(self):
        from graphrag_pipeline.review.store import ReviewStore
        return ReviewStore(self._settings.review_db)

    def detect(self, structure, semantic) -> list[Proposal]:
        """Run all anti-pattern detectors and upsert proposals into the review store."""
        from graphrag_pipeline.review.detect import run_detection
        store = self._store()
        try:
            run_detection(structure, semantic, store)
        finally:
            store.close()
        return self.get_proposals()

    def get_proposals(
        self, status: str | None = None
    ) -> list[Proposal]:
        """Return proposals, optionally filtered by status."""
        store = self._store()
        try:
            return store.list_proposals(status=status)
        finally:
            store.close()

    def accept(self, proposal_id: str) -> None:
        """Accept a proposal and apply the associated patch."""
        from graphrag_pipeline.review.actions import accept_proposal
        store = self._store()
        try:
            accept_proposal(store, proposal_id)
        finally:
            store.close()

    def reject(self, proposal_id: str, reason: str = "") -> None:
        """Reject a proposal."""
        from graphrag_pipeline.review.actions import reject_proposal
        store = self._store()
        try:
            reject_proposal(store, proposal_id, reason=reason)
        finally:
            store.close()

    def export(self, path: str, fmt: str = "json") -> int:
        """Export proposals to a file. Returns the count of records written."""
        from graphrag_pipeline.review.export import (
            export_proposals_json,
            export_proposals_csv,
        )
        store = self._store()
        try:
            if fmt == "csv":
                return export_proposals_csv(store, path)
            return export_proposals_json(store, path)
        finally:
            store.close()


__all__ = ["ReviewService"]
