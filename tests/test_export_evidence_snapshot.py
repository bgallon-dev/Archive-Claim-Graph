from __future__ import annotations

from scripts.export_evidence_snapshot import EXPORT_QUERY, _rows


class FakeExecutor:
    def __init__(self) -> None:
        self.calls = []

    def run(self, query, params):
        self.calls.append((query, params))
        if len(self.calls) > 1:
            return []
        return [
            {
                "claim_id": "claim-1",
                "source_sentence": "Ada served on the Spokane board.",
                "confidence": 0.9,
                "claim_type": "role",
                "doc_id": "doc-1",
                "doc_title": "Example Gazette",
                "institution_id": "spokane",
                "series": "gazette",
                "date_start": "1909-01-02",
                "lccn": "sn-example",
                "edition": "1",
                "page_number": 4,
                "paragraph_id": "p-1",
                "subjects": [{"entity_id": "person-1", "name": "Ada"}],
            }
        ]


def test_export_is_scoped_latest_public_and_provenance_complete() -> None:
    assert "$institution_ids" in EXPORT_QUERY
    assert "newer.run_timestamp > lr.run_timestamp" in EXPORT_QUERY
    assert "access_level" in EXPORT_QUERY
    assert "donor_restricted" in EXPORT_QUERY

    executor = FakeExecutor()
    rows = list(_rows(executor, 50, None, ["spokane"]))

    assert executor.calls[0][1]["institution_ids"] == ["spokane"]
    assert rows[0]["source"]["page_number"] == 4
    assert rows[0]["source"]["date_start"] == "1909-01-02"
    assert rows[0]["source"]["source_url"] == "https://www.loc.gov/item/sn-example/1909-01-02/ed-1/"
    assert rows[0]["source"]["evidence_fingerprint"]
    assert rows[0]["provenance_id"] == rows[0]["source"]["evidence_fingerprint"]
    assert rows[0]["subjects"][0]["name"] == "Ada"
