"""Step 6 — auth DDL default + EAD tag map domain-agnosticism.

* auth: the ``user.institution_id`` DDL default was the Turnbull literal
  ``'turnbull'``; it is now ``''`` (matching the Python-layer default).
* EAD: the entity_type → EAD element map is now overlaid from the domain's
  ``ead_tag_map`` so a corpus's own entity types map to the right access point.
"""
from __future__ import annotations

import sqlite3

from gemynd.shared.io_utils import save_semantic_bundle
from gemynd.ingest.export.ead_xml import render_ead_xml
from gemynd.core.models import (
    EntityRecord,
    ExtractionRunRecord,
    SemanticBundle,
)


# --------------------------------------------------------------------------- #
# auth DDL default
# --------------------------------------------------------------------------- #
class TestAuthInstitutionDefault:
    def test_ddl_default_is_empty_not_turnbull(self):
        from gemynd.auth.store import _SCHEMA
        con = sqlite3.connect(":memory:")
        try:
            con.executescript(_SCHEMA)
            # INSERT omitting institution_id → exercises the column DEFAULT
            con.execute(
                "INSERT INTO user (user_id, email, hashed_password, role, created_at) "
                "VALUES (?,?,?,?,?)",
                ("u1", "a@example.com", "hash", "admin", "2026-01-01T00:00:00Z"),
            )
            got = con.execute("SELECT institution_id FROM user WHERE user_id='u1'").fetchone()[0]
        finally:
            con.close()
        assert got == ""

    def test_schema_string_has_no_turnbull_literal(self):
        from gemynd.auth.store import _SCHEMA
        assert "'turnbull'" not in _SCHEMA


# --------------------------------------------------------------------------- #
# EAD tag map
# --------------------------------------------------------------------------- #
def _bundle(entities):
    return SemanticBundle(
        extraction_run=ExtractionRunRecord(run_id="r1"),
        claims=[], measurements=[], mentions=[], entities=entities,
        entity_resolutions=[], claim_entity_links=[], claim_link_diagnostics=[],
        claim_location_links=[], claim_period_links=[], document_anchor_links=[],
        document_period_links=[], document_signed_by_links=[], person_affiliation_links=[],
    )


def _render(tmp_path, entities, ead_tag_map):
    save_semantic_bundle(tmp_path / "doc1.semantic.json", _bundle(entities))
    out = tmp_path / "fa.xml"
    render_ead_xml(tmp_path, out, institution_id="x", ead_tag_map=ead_tag_map)
    return out.read_text(encoding="utf-8")


_ENTITIES = [
    EntityRecord(entity_id="e_p", entity_type="Person", name="John Doe", normalized_form="john doe"),
    EntityRecord(entity_id="e_c", entity_type="City", name="Metropolis", normalized_form="metropolis"),
    EntityRecord(entity_id="e_r", entity_type="Refuge", name="Turnbull Refuge", normalized_form="turnbull refuge"),
]


class TestEadTagMapParity:
    def test_default_map_preserved(self, tmp_path):
        xml = _render(tmp_path, _ENTITIES, ead_tag_map=None)
        assert "<persname>John Doe</persname>" in xml          # universal default
        assert "<geogname>Turnbull Refuge</geogname>" in xml   # Refuge → geogname default
        # an unmapped custom type falls back to the generic subject element
        assert "<subject>Metropolis</subject>" in xml
        assert "<geogname>Metropolis</geogname>" not in xml


class TestEadTagMapAgnostic:
    def test_domain_map_overlays_default(self, tmp_path):
        xml = _render(tmp_path, _ENTITIES, ead_tag_map={"City": "geogname"})
        assert "<geogname>Metropolis</geogname>" in xml   # domain mapping applied
        assert "<subject>Metropolis</subject>" not in xml
        # default mappings still present (overlay, not replace)
        assert "<persname>John Doe</persname>" in xml
        assert "<geogname>Turnbull Refuge</geogname>" in xml

    def test_domain_map_can_override_a_default(self, tmp_path):
        # A domain may reclassify a default type (e.g. Person → subject).
        xml = _render(tmp_path, _ENTITIES, ead_tag_map={"Person": "subject"})
        assert "<subject>John Doe</subject>" in xml
        assert "<persname>John Doe</persname>" not in xml


class TestEadConfigField:
    def test_shipped_domains_declare_no_ead_map(self):
        from pathlib import Path
        from gemynd.core.domain_config import load_domain_config
        from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR
        for rd in (TURNBULL_RESOURCES_DIR, Path("gemynd/resources_newspaper")):
            assert load_domain_config(rd).ead_tag_map == {}
