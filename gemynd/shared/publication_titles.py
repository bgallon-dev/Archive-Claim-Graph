"""Human-readable publication names for newspaper catalog identifiers.

The newspaper corpus stores the Library of Congress Control Number (LCCN) as
``Document.series``.  LCCNs are stable routing/provenance keys, but they are
not useful as the primary source label in the research UI.  This deliberately
small local catalog keeps rendering deterministic and offline.  Unknown IDs
are returned unchanged so newly ingested titles remain visible until the
catalog is extended.
"""
from __future__ import annotations

import re

PUBLICATION_TITLES: dict[str, str] = {
    "2003260222": "The Dalles Weekly Chronicle",
    "2004260241": "The Daily Gazette-Times",
    "2004260419": "St. Helens Mist",
    "2004260421": "The Oregon Mist",
    "sn00063520": "The Ontario Argus",
    "sn85042399": "Ashland Tidings",
    "sn85042400": "The Morning Astorian",
    "sn85042448": "The Dalles Daily Chronicle",
    "sn83045604": "The Seattle Post-Intelligencer",
    "sn86072021": "Spokane Falls Review",
    "sn87008085": "The Newport Miner",
    "sn87093055": "Washington State Journal",
    "sn87093056": "Adams County News",
    "sn87093057": "Washington State Journal and Adams County News",
    "sn88085318": "The Colville Examiner",
    "sn88085460": "The Colfax Gazette",
    "sn88085488": "Pullman Herald",
    "sn88085947": "The Spokane Press",
    "sn88086023": "East Oregonian",
    "sn88087129": "The Spokane Woman",
    "sn93051669": "The Dalles Times-Mountaineer",
    "sn94049698": "Heppner Gazette",
    "sn96088180": "Rogue River Courier",
    "sn97071090": "Medford Mail Tribune",
    "sn98047755": "Washington Farmer",
    "sn99063957": "Daily Capital Journal",
}

_LCCN_RE = re.compile(r"^(?:sn\d{8}|\d{10})$", re.IGNORECASE)


def publication_title(catalog_id: str | None) -> str | None:
    """Return a human-readable title, preserving unknown catalog IDs."""
    value = (catalog_id or "").strip()
    if not value:
        return None
    return PUBLICATION_TITLES.get(value.lower(), value)


def is_lccn(value: str | None) -> bool:
    """Return whether *value* has the LCCN forms used by this corpus."""
    return bool(_LCCN_RE.fullmatch((value or "").strip()))
