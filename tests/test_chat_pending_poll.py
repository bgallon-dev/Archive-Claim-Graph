from pathlib import Path

from gemynd.core.graph.cypher import GET_DOCUMENT_HEADER_QUERY


ROOT = Path(__file__).resolve().parents[1]


def test_refresh_recovery_uses_quiet_pending_poll() -> None:
    template = (
        ROOT / "gemynd" / "retrieval" / "web" / "templates" / "chat.html"
    ).read_text(encoding="utf-8")

    assert "'?pending=1'" in template
    assert "Explicit pending mode returns a quiet 200" in template


def test_missing_optional_loc_sequence_does_not_emit_property_warning() -> None:
    assert "properties(d).loc_sequence AS loc_sequence" in GET_DOCUMENT_HEADER_QUERY
    assert "d.loc_sequence AS loc_sequence" not in GET_DOCUMENT_HEADER_QUERY
