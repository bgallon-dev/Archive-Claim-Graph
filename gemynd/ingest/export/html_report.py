"""Standalone HTML collection statistics report.

Queries Neo4j using the same STATS_* Cypher constants used by the /stats
endpoint, then writes a self-contained HTML file with inline CSS.  The file
can be opened offline in any browser and printed to PDF.
"""
from __future__ import annotations

import html as _html
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_ALL_LEVELS = ["public", "staff_only", "restricted", "indigenous_restricted"]


def _esc(v: object) -> str:
    return _html.escape(str(v) if v is not None else "")


_CSS = """
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: Georgia, "Times New Roman", serif;
    background: #fff; color: #212529;
    max-width: 900px; margin: 2rem auto; padding: 0 1.5rem;
    line-height: 1.6;
  }
  h1 { font-size: 1.6rem; border-bottom: 2px solid #2c3e50; padding-bottom: 0.5rem; }
  h2 { font-size: 1.15rem; color: #2c3e50; margin: 1.5rem 0 0.5rem; }
  p.meta { color: #6c757d; font-size: 0.85rem; margin: 0 0 1.5rem; }
  .stat-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1rem; margin: 1rem 0 1.5rem;
  }
  .stat-card {
    border: 1px solid #dee2e6; border-radius: 6px; padding: 0.75rem 1rem; text-align: center;
  }
  .stat-card .value { font-size: 1.8rem; font-weight: 700; color: #0d6efd; }
  .stat-card .label { font-size: 0.75rem; color: #6c757d; margin-top: 0.2rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-bottom: 1rem; }
  th { text-align: left; border-bottom: 2px solid #dee2e6; padding: 0.4rem 0.6rem;
       font-size: 0.8rem; color: #6c757d; white-space: nowrap; }
  td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #f1f3f5; }
  tr:last-child td { border-bottom: none; }
  .bar-wrap { background: #e9ecef; border-radius: 3px; height: 8px; min-width: 60px; }
  .bar { background: #0d6efd; height: 8px; border-radius: 3px; }
  footer { border-top: 1px solid #dee2e6; padding-top: 0.75rem;
           font-size: 0.75rem; color: #6c757d; margin-top: 2rem; }
  @media print {
    body { max-width: 100%; margin: 0; padding: 1cm; }
    .stat-card .value { color: #000; }
  }
</style>
"""


def render_html_report(
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    neo4j_trust_mode: str | None = None,
    neo4j_ca_cert: str | None = None,
    institution_id: str,
    permitted_levels: list[str] | None = None,
    output_path: Path,
    generated_by: str = "gemynd export-corpus",
) -> None:
    """Query Neo4j and write a self-contained HTML statistics report.

    *permitted_levels* defaults to all levels (admin-grade export).
    """
    from gemynd.retrieval.executor import Neo4jQueryExecutor
    from gemynd.core.graph.cypher import (
        STATS_DOC_OVERVIEW_QUERY,
        STATS_DOC_TYPE_QUERY,
        STATS_CLAIM_TYPE_QUERY,
        STATS_ENTITY_TYPE_QUERY,
        STATS_TEMPORAL_COVERAGE_QUERY,
        STATS_CONFIDENCE_DISTRIBUTION_QUERY,
    )

    levels = permitted_levels or _ALL_LEVELS
    params: dict[str, Any] = {"institution_id": institution_id, "permitted_levels": levels}

    executor = Neo4jQueryExecutor(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        trust_mode=neo4j_trust_mode or "system",
        ca_cert_path=neo4j_ca_cert,
    )
    try:
        overview = executor.run(STATS_DOC_OVERVIEW_QUERY, params)
        doc_types = executor.run(STATS_DOC_TYPE_QUERY, params)
        claim_types = executor.run(STATS_CLAIM_TYPE_QUERY, params)
        entity_types = executor.run(STATS_ENTITY_TYPE_QUERY, params)
        temporal = executor.run(STATS_TEMPORAL_COVERAGE_QUERY, params)
        confidence = executor.run(STATS_CONFIDENCE_DISTRIBUTION_QUERY, params)
    finally:
        executor.close()

    ov = overview[0] if overview else {}
    cf = confidence[0] if confidence else {}
    total_docs = ov.get("total_docs") or 0
    total_pages = ov.get("total_pages") or 0
    earliest = ov.get("earliest_year") or "—"
    latest = ov.get("latest_year") or "—"
    total_claims = cf.get("total_claims") or 0
    avg_conf = cf.get("avg_confidence")
    avg_conf_str = f"{avg_conf:.2f}" if avg_conf is not None else "—"
    donor_count = ov.get("donor_restricted_count") or 0

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Summary stat cards ---
    stat_cards = "".join(
        f"<div class='stat-card'><div class='value'>{_esc(val)}</div>"
        f"<div class='label'>{_esc(label)}</div></div>"
        for val, label in [
            (total_docs, "Documents"),
            (total_claims, "Claims extracted"),
            (f"{earliest}–{latest}", "Date span"),
            (total_pages, "Pages"),
            (avg_conf_str, "Avg. confidence"),
        ]
    )

    # --- Document types table ---
    doc_type_rows = ""
    for row in doc_types:
        doc_type_rows += (
            f"<tr><td>{_esc(row.get('doc_type','—'))}</td>"
            f"<td>{_esc(row.get('count',0))}</td></tr>"
        )

    # --- Claim types table ---
    max_claim_count = max((r.get("count") or 0 for r in claim_types), default=1)
    claim_type_rows = ""
    for row in claim_types:
        ct = row.get("claim_type") or "—"
        cnt = row.get("count") or 0
        avg = row.get("avg_confidence")
        avg_s = f"{avg:.2f}" if avg is not None else "—"
        pct = int(cnt / max(max_claim_count, 1) * 100)
        bar = f"<div class='bar-wrap'><div class='bar' style='width:{pct}%'></div></div>"
        claim_type_rows += (
            f"<tr><td>{_esc(ct)}</td><td>{_esc(cnt)}</td>"
            f"<td>{avg_s}</td><td>{bar}</td></tr>"
        )

    # --- Entity types table ---
    entity_type_rows = ""
    for row in entity_types:
        entity_type_rows += (
            f"<tr><td>{_esc(row.get('entity_type','—'))}</td>"
            f"<td>{_esc(row.get('count',0))}</td></tr>"
        )

    # --- Temporal coverage table ---
    temporal_rows = ""
    for row in temporal:
        temporal_rows += (
            f"<tr><td>{_esc(row.get('year','—'))}</td>"
            f"<td>{_esc(row.get('doc_count',0))}</td></tr>"
        )

    # --- Confidence distribution ---
    high = cf.get("high_count") or 0
    med = cf.get("medium_count") or 0
    low = cf.get("low_count") or 0
    unc = cf.get("uncertain_epistemic_count") or 0
    conf_dist = (
        "<table><thead><tr><th>Tier</th><th>Claims</th></tr></thead><tbody>"
        f"<tr><td>High (&ge;0.85)</td><td>{_esc(high)}</td></tr>"
        f"<tr><td>Medium (0.70–0.84)</td><td>{_esc(med)}</td></tr>"
        f"<tr><td>Low (&lt;0.70)</td><td>{_esc(low)}</td></tr>"
        f"<tr><td>Uncertain (epistemic flag)</td><td>{_esc(unc)}</td></tr>"
        "</tbody></table>"
    )

    donor_note = (
        f"<p style='color:#6c757d;font-size:0.85rem'>"
        f"Note: {donor_count} document(s) carry donor reproduction restrictions.</p>"
        if donor_count
        else ""
    )

    html = (
        "<!doctype html><html lang='en'><head>"
        "<meta charset='UTF-8'>"
        f"<title>{_esc(institution_id)} — Collection Statistics Report</title>"
        + _CSS +
        "</head><body>"
        f"<h1>{_esc(institution_id)} Collection Statistics Report</h1>"
        f"<p class='meta'>Generated {_esc(generated_at)} by {_esc(generated_by)}</p>"
        "<div class='stat-grid'>" + stat_cards + "</div>"
        + donor_note +
        "<h2>Document types</h2>"
        "<table><thead><tr><th>Type</th><th>Count</th></tr></thead>"
        f"<tbody>{doc_type_rows}</tbody></table>"
        "<h2>Claim types</h2>"
        "<table><thead><tr><th>Type</th><th>Count</th><th>Avg conf.</th><th></th></tr></thead>"
        f"<tbody>{claim_type_rows}</tbody></table>"
        "<h2>Entity types</h2>"
        "<table><thead><tr><th>Type</th><th>Unique entities</th></tr></thead>"
        f"<tbody>{entity_type_rows}</tbody></table>"
        "<h2>Temporal coverage</h2>"
        "<table><thead><tr><th>Year</th><th>Documents</th></tr></thead>"
        f"<tbody>{temporal_rows}</tbody></table>"
        "<h2>Extraction confidence</h2>"
        + conf_dist +
        f"<footer>Gemynd &middot; {_esc(institution_id)} &middot; {_esc(generated_at)}</footer>"
        "</body></html>"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    _log.info("render_html_report: written to %s", output_path)
