"""Per-claim derivation state shared between observation_builder and event_builder.

``DerivationContext`` captures all entity bindings, year provenance, and routing
metadata that both builders previously computed independently for each claim.
``build_derivation_contexts()`` performs the lookup work once and returns a list
of contexts that the builders can consume directly.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import functools

from gemynd.core.ids import make_year_id

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig
    from gemynd.core.models import (
        ClaimEntityLinkRecord,
        ClaimLocationLinkRecord,
        ClaimPeriodLinkRecord,
        ClaimRecord,
        EntityRecord,
        MeasurementRecord,
    )

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")


@functools.cache
def _default_registry() -> dict:
    """Lazily load the derivation registry for the no-config fallback path."""
    from gemynd.shared.resource_loader import load_derivation_registry
    return load_derivation_registry()


def _extract_year(claim_date: str | None, report_year: int | None) -> tuple[int | None, str]:
    """Return (year_int, year_source).

    year_source is one of 'claim_date', 'document_primary_year', or 'unknown'.
    """
    if claim_date:
        m = _YEAR_RE.search(claim_date)
        if m:
            return int(m.group(1)), "claim_date"
    if report_year is not None:
        return report_year, "document_primary_year"
    return None, "unknown"


def _parse_year(date_str: str) -> int | None:
    """Extract the first 4-digit year from *date_str*, or None if not found."""
    m = _YEAR_RE.search(date_str)
    return int(m.group(1)) if m else None


def _check_year_plausibility(
    year: int,
    year_source: str,
    doc_date_start: str | None,
    doc_date_end: str | None,
    report_year: int | None,
    cfg: dict,
) -> tuple[int | None, str]:
    """Return (year, year_source) after applying plausibility bounds.

    When the year falls outside [window_lo, window_hi]:
    - action='flag'    → year preserved, year_source set to 'suspect'
    - action='exclude' → year set to None, year_source set to 'unknown'
    When no window can be derived, the year is returned unchanged.
    """
    tol = int(cfg.get("tolerance", 5))
    action = cfg.get("action", "flag")

    start_year = _parse_year(doc_date_start) if doc_date_start else report_year
    end_year = _parse_year(doc_date_end) if doc_date_end else report_year

    if start_year is None and end_year is None:
        return year, year_source

    lo = (start_year if start_year is not None else end_year) - tol  # type: ignore[operator]
    hi = (end_year if end_year is not None else start_year) + tol    # type: ignore[operator]

    if lo <= year <= hi:
        return year, year_source

    if action == "exclude":
        return None, "unknown"
    return year, "suspect"


@dataclass(slots=True)
class DerivationContext:
    """All per-claim state needed by observation_builder and event_builder.

    Constructed by :func:`build_derivation_contexts` and consumed by both
    builders when passed as ``_contexts``.  ``observation_id`` starts as
    ``None`` and is filled in by ``build_observations()`` after each
    ``ObservationRecord`` is created.

    ``role_entities`` holds the resolved role → entity_id bindings (e.g.
    ``{"species": "ent_123", "refuge": "ent_456"}``) driven by
    ``DomainConfig.role_resolution``. ``place_id`` / ``period_id`` /
    ``year_id`` remain first-class because they participate in distinct
    edge-emission paths that are domain-neutral.
    """

    claim: ClaimRecord
    role_entities: dict[str, str]
    place_id: str | None
    period_id: str | None
    observation_type: str | None     # None = not observation-eligible
    event_type: str | None           # None = not event-eligible
    year: int | None
    year_source: str                 # "claim_date"|"document_primary_year"|"unknown"|"suspect"
    year_id: str | None
    measurement_owner: str           # "observation"|"event"|"none"
    measurement_ids: list[str]
    missing_required_entities: list[str]
    observation_id: str | None       # set by build_observations() post-construction


def _get_spec(
    registry: dict | None,
    claim_type: str,
) -> tuple[str | None, str | None, tuple, tuple]:
    """Return ``(observation_type, event_type, required, optional)`` from a registry entry.

    Accepts both :class:`~gemynd.core.domain_config.ClaimDerivationSpec`
    objects (from :func:`~gemynd.core.domain_config.load_domain_config`) and
    raw dicts (from :func:`~gemynd.shared.resource_loader.load_derivation_registry`).
    Returns four-``None``/empty-tuple defaults when the registry is absent or the
    claim_type is not found.
    """
    if registry is None:
        return None, None, (), ()
    entry = registry.get(claim_type)
    if entry is None:
        return None, None, (), ()
    if hasattr(entry, "observation_type"):  # ClaimDerivationSpec
        return (
            entry.observation_type,
            entry.event_type,
            entry.required_entities,
            entry.optional_entities,
        )
    # raw dict (from load_derivation_registry)
    return (
        entry.get("observation_type") or None,
        entry.get("event_type") or None,
        tuple(entry.get("required_entities") or []),
        tuple(entry.get("optional_entities") or []),
    )


def build_derivation_contexts(
    claims: list[ClaimRecord],
    measurements: list[MeasurementRecord],
    claim_entity_links: list[ClaimEntityLinkRecord],
    claim_location_links: list[ClaimLocationLinkRecord],
    claim_period_links: list[ClaimPeriodLinkRecord],
    entity_lookup: dict[str, EntityRecord],
    run_id: str,
    report_year: int | None,
    *,
    config: DomainConfig | None = None,
    doc_date_start: str | None = None,
    doc_date_end: str | None = None,
    registry: dict[str, dict] | None = None,
    year_validation_cfg: dict | None = None,
) -> list[DerivationContext]:
    """Build one :class:`DerivationContext` per claim.

    Consolidates the entity-binding and year-extraction logic that was
    previously duplicated inside ``build_observations`` and ``build_events``.

    Parameters
    ----------
    registry:
        When provided (loaded by ``load_derivation_registry()``), used for
        ``observation_type``, ``event_type``, and ``required_entities``.
        Falls back to the Python constants in ``claim_contract`` when absent.
    year_validation_cfg:
        When provided (from ``domain_profile.yaml``'s ``year_validation``
        block), years outside the document date window are either flagged or
        excluded per the ``action`` key.
    """
    # ── Index lookups ────────────────────────────────────────────────────────
    measurements_by_claim: dict[str, list[MeasurementRecord]] = defaultdict(list)
    for m in measurements:
        measurements_by_claim[m.claim_id].append(m)

    entity_links_by_claim: dict[str, list[tuple[str, EntityRecord]]] = defaultdict(list)
    for link in claim_entity_links:
        entity = entity_lookup.get(link.entity_id)
        if entity:
            entity_links_by_claim[link.claim_id].append((link.relation_type, entity))

    locations_by_claim: dict[str, list[EntityRecord]] = defaultdict(list)
    for link in claim_location_links:
        entity = entity_lookup.get(link.entity_id)
        if entity:
            locations_by_claim[link.claim_id].append(entity)

    periods_by_claim: dict[str, str] = {}
    for link in claim_period_links:
        periods_by_claim[link.claim_id] = link.period_id

    # Resolve role → resolver spec map from config, with a no-config fallback.
    role_resolution = config.role_resolution if config is not None else {}
    role_location_entity_types: set[str] = {
        resolver.entity_type
        for resolver in role_resolution.values()
        if resolver.source == "location_links" and resolver.entity_type
    }

    # Build a reverse index: entity_type → role_name for the missing-required
    # check, which receives entity-type strings from the derivation registry.
    role_for_entity_type: dict[str, str] = {}
    for role_name, resolver in role_resolution.items():
        if resolver.source == "location_links" and resolver.entity_type:
            role_for_entity_type[resolver.entity_type] = role_name

    # ── Per-claim construction ───────────────────────────────────────────────
    contexts: list[DerivationContext] = []

    for claim in claims:
        ct = claim.claim_type

        # Routing: registry wins when present, otherwise load from YAML.
        effective_registry = registry if registry else _default_registry()
        obs_type, evt_type, _req, _opt = _get_spec(effective_registry, ct)
        required_entities: list[str] = list(_req)

        # measurement_owner
        if obs_type is not None:
            measurement_owner = "observation"
        elif evt_type is not None:
            measurement_owner = "event"
        else:
            measurement_owner = "none"

        # Config-driven role resolution.
        role_entities: dict[str, str] = {}
        place_id: str | None = None

        for role_name, resolver in role_resolution.items():
            if resolver.source == "entity_links":
                for relation_type, entity in entity_links_by_claim.get(claim.claim_id, []):
                    if relation_type in resolver.relations:
                        role_entities[role_name] = entity.entity_id
                        break
            elif resolver.source == "location_links":
                for entity in locations_by_claim.get(claim.claim_id, []):
                    if resolver.entity_type and entity.entity_type == resolver.entity_type:
                        role_entities[role_name] = entity.entity_id
                        break

        # place_id stays first-class: pick the first non-role-claimed location.
        for entity in locations_by_claim.get(claim.claim_id, []):
            if entity.entity_type == "Place":
                place_id = entity.entity_id
                break
            if entity.entity_type not in role_location_entity_types:
                place_id = entity.entity_id
                break

        # Year extraction + optional plausibility check.
        year_value, year_source = _extract_year(claim.claim_date, report_year)
        if year_validation_cfg is not None and year_value is not None:
            year_value, year_source = _check_year_plausibility(
                year_value, year_source,
                doc_date_start, doc_date_end, report_year,
                year_validation_cfg,
            )
        year_id: str | None = make_year_id(year_value) if year_value is not None else None

        # Missing required entities: map required entity_type → role, fall back
        # to checking place_id for entity types that are not role-resolved.
        missing: list[str] = []
        for required_type in required_entities:
            role_name = role_for_entity_type.get(required_type)
            if role_name is not None:
                if role_name not in role_entities:
                    missing.append(required_type)
                continue
            # Also check entity_links sources (species/habitat/method) by
            # matching entity_type on claim links.
            resolved_via_entity_links = any(
                entity.entity_type == required_type
                for _, entity in entity_links_by_claim.get(claim.claim_id, [])
            )
            if resolved_via_entity_links:
                continue
            # Place is first-class.
            if required_type == "Place" and place_id:
                continue
            missing.append(required_type)

        measurement_ids = [m.measurement_id for m in measurements_by_claim.get(claim.claim_id, [])]
        period_id = periods_by_claim.get(claim.claim_id)

        contexts.append(DerivationContext(
            claim=claim,
            role_entities=role_entities,
            place_id=place_id,
            period_id=period_id,
            observation_type=obs_type,
            event_type=evt_type,
            year=year_value,
            year_source=year_source,
            year_id=year_id,
            measurement_owner=measurement_owner,
            measurement_ids=measurement_ids,
            missing_required_entities=missing,
            observation_id=None,
        ))

    return contexts
