"""EAD 2002 XML finding-aid export from pipeline output bundles.

Produces a single EAD 2002-compliant XML file from all *.semantic.json
files in a directory.  One <c level="item"> component is written per
document.  Access-restricted documents (restricted, indigenous_restricted)
are silently skipped — EAD exports are for public-facing finding aids.

Returns the count of document components written.
"""
from __future__ import annotations

import io
import logging
import xml.etree.ElementTree as ET
from datetime import date, timezone, datetime
from pathlib import Path

_log = logging.getLogger(__name__)

# Documents with these access levels are excluded from EAD exports.
_SKIP_LEVELS = {"restricted", "indigenous_restricted"}

# EAD access-point element name by entity type.
_EAD_TAG: dict[str, str] = {
    "Organization": "corpname",
    "Person": "persname",
    "Place": "geogname",
    "Refuge": "geogname",
    "Species": "subject",
    "Habitat": "subject",
    "Activity": "subject",
    "SurveyMethod": "subject",
    "Event": "subject",
}

_DOCTYPE = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE ead PUBLIC "+//ISBN 1-931666-00-8//DTD ead.dtd'
    ' (EAD Version 2002)//EN"\n'
    '    "http://lcweb2.loc.gov/xmlcommon/dtds/ead2002/ead.dtd">\n'
)

_EAD_NS = "urn:isbn:1-931666-00-8"


def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrib: str) -> ET.Element:
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = text
    return el


def render_ead_xml(
    bundles_dir: Path,
    output_path: Path,
    *,
    institution_id: str = "",
    collection_title: str | None = None,
    glob_pattern: str = "*.semantic.json",
) -> int:
    """Write an EAD 2002 XML finding aid from bundle files.

    Returns the count of document ``<c>`` components written.
    Documents with *restricted* or *indigenous_restricted* access level are
    skipped (a warning is logged per skipped document).
    """
    from gemynd.shared.io_utils import load_semantic_bundle, load_structure_bundle

    semantic_paths = sorted(bundles_dir.glob(glob_pattern))
    if not semantic_paths:
        _log.warning("render_ead_xml: no files matching %r in %s", glob_pattern, bundles_dir)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    title = collection_title or f"{institution_id} Collection"

    # Collect per-document data so we can compute collection-level date span.
    docs: list[dict] = []
    for sem_path in semantic_paths:
        try:
            sem = load_semantic_bundle(sem_path)
        except Exception as exc:
            _log.warning("render_ead_xml: skipping %s: %s", sem_path.name, exc)
            continue

        struct_path = sem_path.with_suffix("").with_suffix(".structure.json")
        doc = sem_path.stem  # fallback doc_id
        doc_title = doc
        date_start: str | None = None
        date_end: str | None = None
        archive_ref: str | None = None
        page_count = 0
        access_level = "public"

        if struct_path.exists():
            try:
                struct = load_structure_bundle(struct_path)
                d = struct.document
                doc = d.doc_id
                doc_title = d.title
                date_start = d.date_start
                date_end = d.date_end or d.date_start
                archive_ref = d.archive_ref
                page_count = d.page_count
                access_level = d.access_level
            except Exception as exc:
                _log.warning("render_ead_xml: cannot load structure for %s: %s", sem_path.name, exc)

        if access_level in _SKIP_LEVELS:
            _log.warning(
                "render_ead_xml: skipping %s (access_level=%s)", doc, access_level
            )
            continue

        docs.append({
            "doc_id": doc,
            "doc_title": doc_title,
            "date_start": date_start,
            "date_end": date_end,
            "archive_ref": archive_ref,
            "page_count": page_count,
            "entities": list(sem.entities),
            "claims": sorted(
                sem.claims,
                key=lambda c: c.extraction_confidence,
                reverse=True,
            )[:5],
        })

    # Compute collection date span.
    all_starts = [d["date_start"] for d in docs if d["date_start"]]
    all_ends = [d["date_end"] for d in docs if d["date_end"]]
    earliest = min(all_starts) if all_starts else ""
    latest = max(all_ends) if all_ends else ""
    total_pages = sum(d["page_count"] for d in docs)

    # ------------------------------------------------------------------ #
    # Build EAD tree                                                       #
    # ------------------------------------------------------------------ #
    root = ET.Element("ead", {"xmlns": _EAD_NS})

    # <eadheader>
    eadheader = _sub(root, "eadheader", langencoding="iso639-2b")
    eadid_text = f"{institution_id}_{generated_at.replace('-', '')}"
    _sub(eadheader, "eadid", eadid_text)
    filedesc = _sub(eadheader, "filedesc")
    titlestmt = _sub(filedesc, "titlestmt")
    _sub(titlestmt, "titleproper", title)
    profiledesc = _sub(eadheader, "profiledesc")
    creation = _sub(profiledesc, "creation", "Created by Gemynd on ")
    _sub(creation, "date", generated_at)

    # <archdesc>
    archdesc = _sub(root, "archdesc", level="collection")
    did = _sub(archdesc, "did")
    _sub(did, "unittitle", title)
    if earliest or latest:
        span = f"{earliest}–{latest}" if earliest != latest else earliest
        normal_attr = {}
        if earliest and latest:
            normal_attr["normal"] = f"{earliest}/{latest}"
        elif earliest:
            normal_attr["normal"] = earliest
        unitdate = _sub(did, "unitdate", span, type="inclusive", **normal_attr)  # noqa: F841
    extent_text = f"{len(docs)} document{'s' if len(docs) != 1 else ''}; {total_pages} page{'s' if total_pages != 1 else ''}"
    physdesc = _sub(did, "physdesc")
    _sub(physdesc, "extent", extent_text)

    # <dsc>
    dsc = _sub(archdesc, "dsc", type="combined")

    for d in docs:
        c = _sub(dsc, "c", level="item", id=d["doc_id"])
        c_did = _sub(c, "did")
        _sub(c_did, "unittitle", d["doc_title"])
        _sub(c_did, "unitid", d["archive_ref"] or d["doc_id"])
        if d["date_start"]:
            ds = d["date_start"]
            de = d["date_end"] or ds
            date_text = f"{ds}–{de}" if ds != de else ds
            normal = f"{ds}/{de}" if ds != de else ds
            _sub(c_did, "unitdate", date_text, normal=normal)
        if d["page_count"]:
            c_phys = _sub(c_did, "physdesc")
            _sub(c_phys, "extent", f"{d['page_count']} page{'s' if d['page_count'] != 1 else ''}")

        # <controlaccess> — one access-point per unique entity name+tag combo
        seen_access: set[tuple[str, str]] = set()
        access_points: list[tuple[str, str]] = []  # (tag, name)
        for entity in d["entities"]:
            tag = _EAD_TAG.get(entity.entity_type, "subject")
            key = (tag, entity.name)
            if key not in seen_access:
                seen_access.add(key)
                access_points.append(key)

        if access_points:
            ca = _sub(c, "controlaccess")
            for tag, name in access_points:
                _sub(ca, tag, name)

        # <scopecontent> — top-5 claims by confidence
        if d["claims"]:
            sc = _sub(c, "scopecontent")
            lst = _sub(sc, "list")
            for claim in d["claims"]:
                item_text = f"{claim.source_sentence} ({claim.claim_type})"
                _sub(lst, "item", item_text)

    # ------------------------------------------------------------------ #
    # Serialise and write                                                  #
    # ------------------------------------------------------------------ #
    buf = io.StringIO()
    ET.ElementTree(root).write(buf, encoding="unicode", xml_declaration=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_DOCTYPE + buf.getvalue(), encoding="utf-8")
    _log.info("render_ead_xml: wrote %d components to %s", len(docs), output_path)
    return len(docs)
