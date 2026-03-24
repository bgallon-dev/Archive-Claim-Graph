"""FastAPI + HTMX review application."""
from __future__ import annotations

import functools
import html as _html
import json
from pathlib import Path
from typing import Any

from ..actions import (
    ReviewActionError,
    accept_proposal,
    defer_proposal,
    edit_proposal,
    reject_proposal,
)
from ..store import ReviewStore

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _render(template_name: str, **ctx: Any) -> str:
    """Minimal template rendering – just string format with context."""
    path = _TEMPLATES_DIR / template_name
    return path.read_text(encoding="utf-8").format(**ctx)


_JUNK_ISSUE_CLASSES = frozenset(
    {"header_contamination", "boilerplate_contamination", "short_generic_token", "ocr_garbage_mention"}
)

_CLAIM_SENTENCES: dict[str, str] = {
    "header_contamination": (
        "This surface form appears primarily in report headers rather than body text, "
        "so mentions are likely spurious and should be suppressed."
    ),
    "boilerplate_contamination": (
        "This surface form appears in boilerplate elements (form fields, letterhead, "
        "institutional headers) rather than substantive content, so mentions are likely spurious."
    ),
    "short_generic_token": (
        "This surface form is a short or generic word commonly used in text and "
        "unlikely to represent a meaningful named entity."
    ),
    "ocr_garbage_mention": (
        "This mention appears to be OCR noise — the text contains patterns consistent "
        "with misread characters rather than real entity names."
    ),
}

_SUPPRESSION_REASON_LABELS: dict[str, str] = {
    "header_contamination": "Appears in page headers / footers",
    "boilerplate_contamination": "Appears in boilerplate / institutional text",
    "short_generic_token": "Short or generic word unlikely to be a named entity",
    "ocr_garbage": "OCR noise — not a real word or entity name",
}

_SCOPE_LABELS: dict[str, str] = {
    "semantic_only": "Suppress from graph extraction only (body text preserved)",
    "full": "Full suppression from all graph outputs",
}


@functools.lru_cache(maxsize=8)
def _load_structure_json(struct_path: str) -> dict:  # type: ignore[type-arg]
    """Read and cache a structure JSON file by path string.

    Cached so repeated Load More button presses don't re-read the file.
    """
    try:
        return json.loads(Path(struct_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_paragraph_context(
    evidence: dict,  # type: ignore[type-arg]
    out_dir: str = "out",
    sample_size: int = 20,
    offset: int = 0,
) -> list[dict]:  # type: ignore[type-arg]
    """Return up to sample_size mention dicts enriched with 'paragraph_text'.

    Paginates the flat mention list (all mentions in document order, no per-page
    deduplication).  Structure JSON is cached after the first read so subsequent
    Load More requests are fast.
    Returns [] if the structure JSON cannot be found or parsed.
    """
    mentions = evidence.get("affected_mentions", [])
    if not mentions:
        return []
    source_file = mentions[0].get("source_file", "")
    if not source_file:
        return []
    doc_name = Path(source_file.replace("\\", "/")).parent.name
    struct_path = Path(out_dir) / f"{doc_name}.structure.json"
    if not struct_path.exists():
        return []

    struct = _load_structure_json(str(struct_path))
    if not struct:
        return []

    all_paragraphs = struct.get("paragraphs", [])
    para_lookup = {
        p["paragraph_id"]: p.get("clean_text") or p.get("raw_ocr_text", "")
        for p in all_paragraphs
    }

    # Paragraph IDs that are flagged (excluded from contrast search)
    flagged_para_ids = {m.get("paragraph_id", "") for m in mentions}

    # Flat pagination — all mentions in document order
    batch = [dict(m) for m in mentions[offset : offset + sample_size]]

    # Find contrast examples for surface forms in this batch only
    needed_forms = {m.get("surface_form", "").lower() for m in batch if m.get("surface_form")}
    contrast_by_surface: dict[str, str] = {}  # type: ignore[type-arg]
    for para in all_paragraphs:
        if para["paragraph_id"] in flagged_para_ids:
            continue
        text = para.get("clean_text") or para.get("raw_ocr_text", "")
        if len(text) < 100:
            continue
        text_lower = text.lower()
        for sf_lower in needed_forms:
            if sf_lower and sf_lower in text_lower:
                if len(text) > len(contrast_by_surface.get(sf_lower, "")):
                    contrast_by_surface[sf_lower] = text

    for m in batch:
        m["paragraph_text"] = para_lookup.get(m.get("paragraph_id", ""), "")
        sf_lower = m.get("surface_form", "").lower()
        m["contrast_example"] = contrast_by_surface.get(sf_lower, "")

    return batch


def _count_unique_mention_pages(evidence: dict) -> int:  # type: ignore[type-arg]
    """Return the total number of affected mentions."""
    return len(evidence.get("affected_mentions", []))


def _load_more_row_html(proposal_id: str, next_offset: int, total_pages: int) -> str:
    """Return a full-width <tr> with a Load next 20 button, or empty string if done."""
    remaining = total_pages - next_offset
    if remaining <= 0:
        return ""
    batch = min(20, remaining)
    pid = _html.escape(proposal_id)
    row_id = f"load-more-{pid.replace(':', '-')}"
    return (
        f'<tr id="{row_id}">'
        f'<td colspan="4" style="text-align:center;padding:0.75rem;background:#f8f9fa">'
        f'<button hx-get="/proposals/{pid}/mentions?offset={next_offset}&amp;limit=20" '
        f'hx-target="#{row_id}" hx-swap="outerHTML" class="btn-defer">'
        f"Load next {batch}</button>"
        f'<span style="color:#888;font-size:0.85rem;margin-left:0.75rem">'
        f"{remaining} more mention{'s' if remaining != 1 else ''} not yet shown"
        f"</span>"
        f"</td></tr>"
    )


def _render_risk_panel(
    issue_class: str,
    confidence: float,
    impact_size: int,
    sampled_mentions: list,  # type: ignore[type-arg]
) -> str:
    """Return a row of risk badges: Confidence / Impact / False-positive risk."""
    def _badge(label: str, level: str) -> str:
        return f'<span class="risk-badge risk-{level}">{label}: {level}</span>'

    conf_level = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.70 else "low")
    impact_level = "high" if impact_size >= 100 else ("medium" if impact_size >= 10 else "low")

    if issue_class == "ocr_garbage_mention":
        fp_level = "low"
    elif issue_class == "short_generic_token":
        fp_level = "high"
    elif issue_class in ("header_contamination", "boilerplate_contamination"):
        has_contrast = any(m.get("contrast_example") for m in sampled_mentions)
        fp_level = "medium" if has_contrast else "low"
    else:
        return ""

    return (
        '<div style="margin:0.75rem 0">'
        + _badge("Confidence", conf_level)
        + _badge("Impact", impact_level)
        + _badge("False-positive risk", fp_level)
        + "</div>"
    )


def _render_proposed_action_panel(
    patch_spec: dict,  # type: ignore[type-arg]
    impact_size: int,
    excluded_count: int,
    evidence: dict,  # type: ignore[type-arg]
) -> str:
    """Return a compact action summary panel for suppress_mention proposals."""
    if not patch_spec or patch_spec.get("proposal_type") != "suppress_mention":
        return ""

    normalized_form = ""
    mentions = evidence.get("affected_mentions", [])
    if mentions:
        normalized_form = (
            mentions[0].get("normalized_form", "")
            or mentions[0].get("surface_form", "")
        )

    nf_display = _html.escape(normalized_form) if normalized_form else "(see evidence below)"
    proposed_action = f"Suppress mentions where normalized form = <strong>{nf_display}</strong>"

    raw_reason = patch_spec.get("suppression_reason", "")
    reason_label = _SUPPRESSION_REASON_LABELS.get(raw_reason, _html.escape(raw_reason))

    scope_raw = patch_spec.get("scope", "")
    scope_label = _SCOPE_LABELS.get(scope_raw, _html.escape(scope_raw))

    exceptions_text = (
        f"{excluded_count} mention(s) will be preserved"
        if excluded_count > 0
        else "None"
    )

    return (
        '<section class="action-summary">'
        '<h2 style="margin-top:0;font-size:1.1rem">Proposed Action</h2>'
        "<table>"
        f"<tr><th>Action</th><td>{proposed_action}</td></tr>"
        f"<tr><th>Reason</th><td>{reason_label}</td></tr>"
        f"<tr><th>Estimated affected mentions</th><td><strong>{impact_size}</strong></td></tr>"
        f"<tr><th>Scope</th><td>{scope_label}</td></tr>"
        f"<tr><th>Exceptions currently preserved</th><td>{exceptions_text}</td></tr>"
        "</table>"
        "</section>"
    )


def _render_provenance_section(
    snapshot_id: str,
    detector_name: str,
    detector_version: str,
    created_at: str,
    validation_state: str,
) -> str:
    """Return a provenance metadata block."""
    def esc(v: object) -> str:
        return _html.escape(str(v))

    return (
        '<section class="provenance-section" style="margin:1.5rem 0">'
        '<h2 style="margin-top:0;font-size:1rem;color:#6c757d">Provenance</h2>'
        "<table>"
        f"<tr><th>Snapshot ID</th><td><code>{esc(snapshot_id)}</code></td></tr>"
        f"<tr><th>Detector</th><td>{esc(detector_name)}</td></tr>"
        f"<tr><th>Detector Version</th><td>{esc(detector_version)}</td></tr>"
        f"<tr><th>Proposal Generated</th><td>{esc(created_at)}</td></tr>"
        f"<tr><th>Validation Run</th><td>{esc(validation_state)}</td></tr>"
        "<tr><th>Sampling Method</th><td>All mentions, 20 per page (structure JSON cached)</td></tr>"
        "</table>"
        "</section>"
    )


def _mention_row(
    m: dict,  # type: ignore[type-arg]
    proposal_id: str,
    is_excluded: bool,
) -> str:
    """Return a single <tr> HTML fragment for a mention with its override button."""
    def esc(v: object) -> str:
        return _html.escape(str(v))

    mid = m.get("mention_id", "")
    sf = m.get("surface_form", "")
    pg = m.get("page_number", "")
    para_text = m.get("paragraph_text", "")
    contrast = m.get("contrast_example", "")
    row_id = "tgt-" + mid.replace(":", "-")

    def _highlight(text: str, term: str) -> str:
        if not text:
            return "<em style='color:#999'>No paragraph context</em>"
        escaped = esc(text[:300]) + ("…" if len(text) > 300 else "")
        if term:
            escaped = escaped.replace(esc(term), f"<mark>{esc(term)}</mark>", 1)
        return escaped

    flagged_html = (
        f'<div style="background:#fff3cd;padding:0.4rem;border-radius:3px;margin-bottom:0.3rem">'
        f'<span style="color:#856404;font-size:0.8rem;font-weight:700;display:block;margin-bottom:0.2rem">Bad Context</span>'
        f'<span style="font-size:0.85rem">{_highlight(para_text, sf)}</span>'
        f"</div>"
    )
    contrast_html = ""
    if contrast:
        contrast_html = (
            f'<div style="background:#d1e7dd;padding:0.4rem;border-radius:3px;margin-top:0.3rem">'
            f'<span style="color:#0f5132;font-size:0.8rem;font-weight:700;display:block;margin-bottom:0.2rem">Valid Context</span>'
            f'<span style="font-size:0.85rem">{_highlight(contrast, sf)}</span>'
            f"</div>"
        )
    else:
        contrast_html = (
            '<div style="font-size:0.75rem;color:#888;margin-top:0.3rem;padding:0.3rem;'
            'background:#f8f9fa;border-radius:3px">'
            "No legitimate use found in document — likely always junk.</div>"
        )

    from urllib.parse import quote as _q
    if is_excluded:
        btn = (
            f'<button hx-post="/proposals/{esc(proposal_id)}/targets/include'
            f'?target_id={_q(mid)}" '
            f'hx-target="#{row_id}" hx-swap="outerHTML" '
            f'class="btn-accept" style="white-space:nowrap">Exclude This Mention</button>'
        )
        row_style = ' style="opacity:0.5"'
    else:
        btn = (
            f'<button hx-post="/proposals/{esc(proposal_id)}/targets/exclude'
            f'?target_id={_q(mid)}" '
            f'hx-target="#{row_id}" hx-swap="outerHTML" '
            f'class="btn-reject" style="white-space:nowrap">Keep This Mention</button>'
        )
        row_style = ""

    return (
        f'<tr id="{row_id}"{row_style}>'
        f"<td><strong>{esc(sf)}</strong></td>"
        f'<td style="max-width:40rem">{flagged_html}{contrast_html}</td>'
        f"<td>{esc(pg)}</td>"
        f"<td>{btn}</td></tr>"
    )


def _render_evidence_summary(
    issue_class: str,
    evidence: dict,  # type: ignore[type-arg]
    proposal_id: str = "",
    sampled_mentions: list | None = None,  # type: ignore[type-arg]
    excluded_target_ids: set | None = None,  # type: ignore[type-arg]
    patch_spec: dict | None = None,  # type: ignore[type-arg]
    confidence: float = 0.0,
    impact_size: int = 0,
) -> str:
    """Return a human-readable HTML block extracted from the evidence snapshot."""
    if not evidence:
        return ""

    def esc(v: object) -> str:
        return _html.escape(str(v))

    parts: list[str] = [
        '<section style="margin:1rem 0;padding:1rem;background:#fff3cd;border-radius:6px;border:1px solid #ffc107">',
        '<h2 style="margin-top:0">What to Review</h2>',
    ]

    if issue_class in _JUNK_ISSUE_CLASSES:
        claim = _CLAIM_SENTENCES.get(issue_class, "")
        if claim:
            parts.append(f'<div class="claim-box">{_html.escape(claim)}</div>')

        reason = evidence.get("suppression_reason", issue_class)
        parts.append(f"<p><strong>Suppression reason:</strong> {esc(reason)}</p>")
        all_mentions = evidence.get("affected_mentions", [])
        total = len(all_mentions)
        excluded_ids = excluded_target_ids or set()
        excluded_count = len(excluded_ids)
        display = sampled_mentions if sampled_mentions else all_mentions[:20]

        if display:
            parts.append(
                f"<p>Showing <strong>{len(display)}</strong> of <strong>{total}</strong> "
                f"mentions. "
                f"<strong>{excluded_count}</strong> excluded from suppression.</p>"
            )
            parts.append(_render_risk_panel(issue_class, confidence, impact_size, display))
            parts.append(
                "<table><thead><tr>"
                "<th>Surface Form</th><th>Paragraph Context</th><th>Page</th><th>Action</th>"
                "</tr></thead><tbody>"
            )
            for m in display:
                mid = m.get("mention_id", "")
                parts.append(_mention_row(m, proposal_id, mid in excluded_ids))
            total_pages = _count_unique_mention_pages(evidence)
            parts.append(_load_more_row_html(proposal_id, 20, total_pages))
            parts.append("</tbody></table>")

    elif issue_class == "ocr_spelling_variant":
        canon = evidence.get("canonical_entity", {})
        variants = evidence.get("merge_entities", [])
        avg_sim = evidence.get("average_similarity", "")
        canon_count = evidence.get("canonical_mention_count", "")
        merge_counts = evidence.get("merge_mention_counts", {})
        parts.append(
            f"<p><strong>Canonical entity:</strong> {esc(canon.get('name',''))} "
            f"({esc(canon.get('entity_type',''))}) — {esc(canon_count)} mentions</p>"
        )
        parts.append(f"<p><strong>Average similarity:</strong> {esc(avg_sim)}</p>")
        if variants:
            parts.append(
                "<table><thead><tr><th>Variant Name</th><th>Type</th><th>Mentions</th></tr></thead><tbody>"
            )
            for v in variants:
                count = merge_counts.get(v.get("entity_id", ""), "")
                parts.append(
                    f"<tr><td>{esc(v.get('name',''))}</td>"
                    f"<td>{esc(v.get('entity_type',''))}</td>"
                    f"<td>{esc(count)}</td></tr>"
                )
            parts.append("</tbody></table>")

    elif issue_class == "duplicate_entity_alias":
        canon = evidence.get("canonical_entity", {})
        alias = evidence.get("alias_entity", {})
        sim = evidence.get("similarity_score", "")
        canon_count = evidence.get("canonical_mention_count", "")
        alias_count = evidence.get("alias_mention_count", "")
        parts.append(
            "<table><thead><tr><th>Role</th><th>Name</th><th>Type</th><th>Mentions</th></tr></thead><tbody>"
        )
        parts.append(
            f"<tr><td>Canonical</td><td>{esc(canon.get('name',''))}</td>"
            f"<td>{esc(canon.get('entity_type',''))}</td><td>{esc(canon_count)}</td></tr>"
        )
        parts.append(
            f"<tr><td>Alias</td><td>{esc(alias.get('name',''))}</td>"
            f"<td>{esc(alias.get('entity_type',''))}</td><td>{esc(alias_count)}</td></tr>"
        )
        parts.append("</tbody></table>")
        parts.append(f"<p><strong>Similarity score:</strong> {esc(sim)}</p>")

    elif issue_class in ("missing_species_focus", "missing_event_location"):
        sentence = evidence.get("source_sentence", "")
        claim_type = evidence.get("claim_type", "")
        key = "candidate_species" if issue_class == "missing_species_focus" else "candidate_location"
        candidate = evidence.get(key, {})
        label = "Species" if issue_class == "missing_species_focus" else "Location"
        parts.append(f"<p><strong>Claim type:</strong> {esc(claim_type)}</p>")
        if sentence:
            parts.append(
                f"<p><strong>Source sentence:</strong></p>"
                f'<blockquote style="border-left:4px solid #ccc;margin:0;padding:0 1rem">'
                f"<p>{esc(sentence)}</p></blockquote>"
            )
        if candidate:
            parts.append(
                f"<p><strong>Candidate {label}:</strong> {esc(candidate.get('name',''))} "
                f"({esc(candidate.get('entity_type',''))})</p>"
            )

    elif issue_class == "method_overtrigger":
        sentence = evidence.get("source_sentence", "")
        claim_type = evidence.get("claim_type", "")
        method = evidence.get("method_entity", {})
        compat = evidence.get("compatibility", "")
        parts.append(f"<p><strong>Claim type:</strong> {esc(claim_type)}</p>")
        if sentence:
            parts.append(
                f"<p><strong>Source sentence:</strong></p>"
                f'<blockquote style="border-left:4px solid #ccc;margin:0;padding:0 1rem">'
                f"<p>{esc(sentence)}</p></blockquote>"
            )
        parts.append(
            f"<p><strong>Method entity:</strong> {esc(method.get('name',''))} "
            f"&mdash; compatibility: <strong>{esc(compat)}</strong></p>"
        )

    elif issue_class == "pii_exposure":
        matched = evidence.get("matched_pattern", "")
        all_patterns = evidence.get("all_patterns", [])
        claim_id = evidence.get("claim_id", "")
        redacted = evidence.get("redacted_sentence", "")
        parts.append(
            f"<p><strong>Detection type:</strong> {esc(matched)}</p>"
        )
        if all_patterns:
            parts.append(
                f"<p><strong>All matched patterns:</strong> {esc(', '.join(all_patterns))}</p>"
            )
        if claim_id:
            parts.append(f"<p><strong>Claim ID:</strong> {esc(claim_id)}</p>")
        if redacted:
            parts.append(
                "<p><strong>Source sentence (PII redacted):</strong></p>"
                '<blockquote style="border-left:4px solid #dc3545;margin:0;padding:0 1rem">'
                f"<p>{esc(redacted)}</p></blockquote>"
            )
        parts.append(
            '<p style="color:#dc3545"><strong>Action required:</strong> '
            "Review whether this claim contains personal information that should remain quarantined "
            "or be permanently restricted. Do not un-quarantine without data governance approval.</p>"
        )

    elif issue_class == "indigenous_sensitivity":
        matched_term = evidence.get("matched_term", "")
        category = evidence.get("category", "")
        sensitivity = evidence.get("sensitivity", "")
        nations = evidence.get("nations", [])
        claim_id = evidence.get("claim_id", "")
        require_consultation = evidence.get("require_tribal_consultation_before_clear", True)
        parts.append(
            f"<p><strong>Matched term:</strong> <em>{esc(matched_term)}</em></p>"
        )
        parts.append(
            f"<p><strong>Vocabulary category:</strong> {esc(category)} "
            f"&mdash; sensitivity level: <strong>{esc(sensitivity)}</strong></p>"
        )
        if nations:
            parts.append(f"<p><strong>Associated nations:</strong> {esc(', '.join(nations))}</p>")
        if claim_id:
            parts.append(f"<p><strong>Claim ID:</strong> {esc(claim_id)}</p>")
        if require_consultation:
            parts.append(
                '<p style="color:#dc3545;font-weight:bold">&#9888; Tribal consultation required: '
                "Clearing this quarantine requires documented consultation with the relevant "
                "Indigenous nation(s). Archivist judgment alone is not sufficient.</p>"
            )

    elif issue_class == "living_person_reference":
        person_names = evidence.get("person_names", [])
        most_recent_year = evidence.get("most_recent_year", "")
        source_sentence = evidence.get("source_sentence", "")
        year_threshold = evidence.get("year_threshold", "")
        claim_id = evidence.get("claim_id", "")
        parts.append(
            f"<p><strong>Person entity/entities:</strong> {esc(', '.join(person_names) if person_names else '(unknown)')}</p>"
        )
        parts.append(
            f"<p><strong>Most recent associated year:</strong> {esc(most_recent_year)} "
            f"(within the last {esc(year_threshold)} years)</p>"
        )
        if claim_id:
            parts.append(f"<p><strong>Claim ID:</strong> {esc(claim_id)}</p>")
        if source_sentence:
            parts.append(
                "<p><strong>Source sentence:</strong></p>"
                '<blockquote style="border-left:4px solid #fd7e14;margin:0;padding:0 1rem">'
                f"<p>{esc(source_sentence)}</p></blockquote>"
            )
        parts.append(
            '<p style="color:#856404"><strong>Note:</strong> '
            "Review whether this individual is a living person and whether publication of this "
            "claim would violate their privacy rights.</p>"
        )

    else:
        return ""

    parts.append("</section>")
    return "".join(parts)


def create_app(db_path: str, users_db_path: str = "data/users.db") -> Any:
    """Create and return a FastAPI app for the review UI.

    Requires `fastapi` and `uvicorn` to be installed.
    """
    try:
        from fastapi import Depends, FastAPI, Form, Query, Request
        from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse as _RedirectResponse
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for review-serve. Install with: pip install fastapi uvicorn"
        ) from e

    from graphrag_pipeline.auth.dependencies import (
        NeedsLoginException,
        require_archivist_or_admin,
        require_login,
    )
    from graphrag_pipeline.auth.models import UserContext
    from graphrag_pipeline.auth.router import create_auth_router
    from graphrag_pipeline.auth.setup import is_setup_needed

    app = FastAPI(title="GraphRAG Review UI")
    app.include_router(create_auth_router(users_db_path), prefix="/auth")

    @app.exception_handler(NeedsLoginException)
    async def _needs_login_handler(request: Request, exc: NeedsLoginException):
        return _RedirectResponse(url=exc.redirect_url, status_code=303)

    @app.middleware("http")
    async def _setup_guard(request: Request, call_next):
        if not request.url.path.startswith("/auth/setup"):
            if is_setup_needed(users_db_path):
                return _RedirectResponse(url="/auth/setup", status_code=303)
        return await call_next(request)

    store = ReviewStore(db_path)

    @app.get("/", response_class=HTMLResponse)
    async def index(_user: UserContext = Depends(require_login)) -> str:
        counts_by_status = store.proposal_counts_by_status()
        counts_by_queue = store.proposal_counts_by_queue()
        total = sum(counts_by_status.values())

        sensitivity_count = counts_by_queue.get("sensitivity", 0)
        other_queues = {q: c for q, c in counts_by_queue.items() if q != "sensitivity"}

        status_rows = "".join(
            f"<tr><td>{s}</td><td>{c}</td></tr>" for s, c in sorted(counts_by_status.items())
        )
        # Sensitivity queue first with red styling when non-zero
        sensitivity_style = ' style="color:#dc3545;font-weight:bold"' if sensitivity_count > 0 else ""
        queue_rows = (
            f'<tr{sensitivity_style}>'
            f'<td><a href="/proposals?queue_name=sensitivity">sensitivity</a></td>'
            f"<td>{sensitivity_count}</td></tr>"
        )
        queue_rows += "".join(
            f"<tr><td>{q}</td><td>{c}</td></tr>" for q, c in sorted(other_queues.items())
        )

        return _INDEX_HTML.format(
            total=total,
            status_rows=status_rows,
            queue_rows=queue_rows,
        )

    @app.get("/proposals", response_class=HTMLResponse)
    async def list_proposals(
        status: str | None = Query(None),
        issue_class: str | None = Query(None),
        queue_name: str | None = Query(None),
        doc_id: str | None = Query(None),
        limit: int = Query(50),
        offset: int = Query(0),
        _user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        proposals = store.list_proposals(
            status=status,
            issue_class=issue_class,
            queue_name=queue_name,
            doc_id=doc_id,
            limit=limit,
            offset=offset,
        )
        rows = ""
        for p in proposals:
            rows += f"""<tr id="proposal-{p.proposal_id}">
                <td><a href="/proposals/{p.proposal_id}">{p.proposal_id[:12]}...</a></td>
                <td>{p.issue_class}</td>
                <td>{p.proposal_type}</td>
                <td>{p.status}</td>
                <td>{p.confidence:.2f}</td>
                <td>{p.priority_score:.2f}</td>
                <td>{p.impact_size}</td>
                <td>
                    <button hx-post="/proposals/{p.proposal_id}/accept" hx-target="#proposal-{p.proposal_id}" hx-swap="outerHTML"
                            class="btn-accept" {"disabled" if p.status not in ("queued", "deferred") else ""}>Accept</button>
                    <button hx-post="/proposals/{p.proposal_id}/reject" hx-target="#proposal-{p.proposal_id}" hx-swap="outerHTML"
                            class="btn-reject" {"disabled" if p.status not in ("queued", "deferred") else ""}>Reject</button>
                    <button hx-post="/proposals/{p.proposal_id}/defer" hx-target="#proposal-{p.proposal_id}" hx-swap="outerHTML"
                            class="btn-defer" {"disabled" if p.status != "queued" else ""}>Defer</button>
                </td>
            </tr>"""

        filter_params = "&".join(
            f"{k}={v}" for k, v in [("status", status), ("issue_class", issue_class), ("queue_name", queue_name), ("doc_id", doc_id)]
            if v
        )
        next_offset = offset + limit
        prev_offset = max(0, offset - limit)

        return _LIST_HTML.format(
            rows=rows,
            filter_params=filter_params,
            next_offset=next_offset,
            prev_offset=prev_offset,
            limit=limit,
            current_status=status or "",
            current_issue_class=issue_class or "",
            current_queue=queue_name or "",
        )

    @app.get("/proposals/{proposal_id}", response_class=HTMLResponse)
    async def proposal_detail(
        proposal_id: str,
        _user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        proposal = store.get_proposal(proposal_id)
        if not proposal:
            return "<h2>Proposal not found</h2>"
        targets = store.get_proposal_targets(proposal_id)
        revisions = store.get_revisions(proposal_id)
        events = store.get_correction_events(proposal_id)

        target_rows = "".join(
            f"<tr><td>{t.target_kind}</td><td>{t.target_id}</td><td>{t.target_role}</td><td>{'yes' if t.exists_in_snapshot else 'no'}</td></tr>"
            for t in targets
        )
        revision_rows = "".join(
            f"<tr><td>{r.revision_number}</td><td>{r.revision_kind}</td><td>{r.detector_name}</td><td>{r.validation_state}</td><td>{r.created_by}</td><td>{r.created_at}</td></tr>"
            for r in revisions
        )
        event_rows = "".join(
            f"<tr><td>{e.action}</td><td>{e.reviewer}</td><td>{e.reviewer_note}</td><td>{e.created_at}</td></tr>"
            for e in events
        )

        latest_rev = revisions[-1] if revisions else None
        patch_json = latest_rev.patch_spec_json if latest_rev else "{}"
        evidence_json = latest_rev.evidence_snapshot_json if latest_rev else "{}"

        try:
            patch_formatted = json.dumps(json.loads(patch_json), indent=2)
        except Exception:
            patch_formatted = patch_json
        evidence_dict: dict = {}  # type: ignore[type-arg]
        try:
            evidence_dict = json.loads(evidence_json)
            evidence_formatted = json.dumps(evidence_dict, indent=2)
        except Exception:
            evidence_formatted = evidence_json

        patch_dict: dict = {}  # type: ignore[type-arg]
        try:
            patch_dict = json.loads(patch_json)
        except Exception:
            pass

        detector_name = latest_rev.detector_name if latest_rev else ""
        detector_version = latest_rev.detector_version if latest_rev else ""
        validation_state = latest_rev.validation_state if latest_rev else ""

        sampled_mentions = _load_paragraph_context(evidence_dict)
        excluded_ids = {t.target_id for t in targets if t.reviewer_override == "excluded"}
        excluded_count = len(excluded_ids)

        evidence_summary = _render_evidence_summary(
            proposal.issue_class,
            evidence_dict,
            proposal_id=proposal.proposal_id,
            sampled_mentions=sampled_mentions,
            excluded_target_ids=excluded_ids,
            patch_spec=patch_dict,
            confidence=proposal.confidence,
            impact_size=proposal.impact_size,
        )

        proposed_action_panel = _render_proposed_action_panel(
            patch_dict, proposal.impact_size, excluded_count, evidence_dict
        )
        provenance_html = _render_provenance_section(
            proposal.snapshot_id,
            detector_name,
            detector_version,
            proposal.created_at,
            validation_state,
        )
        accept_exceptions_btn = ""
        if excluded_count > 0 and proposal.status in ("queued", "deferred"):
            n = excluded_count
            accept_exceptions_btn = (
                f'<button hx-post="/proposals/{proposal.proposal_id}/accept" '
                f'hx-include="#reviewer-note" '
                f'class="btn-accept-exceptions" style="margin-left:0.5rem">'
                f'Accept with {n} Exception{"s" if n != 1 else ""}'
                f"</button>"
            )

        return _DETAIL_HTML.format(
            proposal_id=proposal.proposal_id,
            issue_class=proposal.issue_class,
            proposal_type=proposal.proposal_type,
            status=proposal.status,
            confidence=f"{proposal.confidence:.2f}",
            priority_score=f"{proposal.priority_score:.2f}",
            impact_size=proposal.impact_size,
            anti_pattern_id=proposal.anti_pattern_id,
            proposed_action_panel=proposed_action_panel,
            evidence_summary=evidence_summary,
            provenance_html=provenance_html,
            target_rows=target_rows,
            revision_rows=revision_rows,
            event_rows=event_rows,
            patch_json=patch_formatted,
            evidence_json=evidence_formatted,
            disabled="" if proposal.status in ("queued", "deferred") else "disabled",
            accept_exceptions_btn=accept_exceptions_btn,
        )

    @app.post("/proposals/{proposal_id}/accept", response_class=HTMLResponse)
    async def do_accept(
        proposal_id: str,
        reviewer_note: str = Form(""),
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        reviewer = user.identity
        try:
            accept_proposal(store, proposal_id, reviewer, reviewer_note)
        except ReviewActionError as e:
            return f"<tr><td colspan='8'>Error: {e}</td></tr>"
        proposal = store.get_proposal(proposal_id)
        if not proposal:
            return "<tr><td colspan='8'>Error</td></tr>"
        return f"""<tr id="proposal-{proposal.proposal_id}">
            <td><a href="/proposals/{proposal.proposal_id}">{proposal.proposal_id[:12]}...</a></td>
            <td>{proposal.issue_class}</td><td>{proposal.proposal_type}</td>
            <td>{proposal.status}</td><td>{proposal.confidence:.2f}</td>
            <td>{proposal.priority_score:.2f}</td><td>{proposal.impact_size}</td>
            <td>Accepted</td></tr>"""

    @app.post("/proposals/{proposal_id}/reject", response_class=HTMLResponse)
    async def do_reject(
        proposal_id: str,
        reviewer_note: str = Form(""),
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        reviewer = user.identity
        try:
            reject_proposal(store, proposal_id, reviewer, reviewer_note)
        except ReviewActionError as e:
            return f"<tr><td colspan='8'>Error: {e}</td></tr>"
        proposal = store.get_proposal(proposal_id)
        if not proposal:
            return "<tr><td colspan='8'>Error</td></tr>"
        return f"""<tr id="proposal-{proposal.proposal_id}">
            <td><a href="/proposals/{proposal.proposal_id}">{proposal.proposal_id[:12]}...</a></td>
            <td>{proposal.issue_class}</td><td>{proposal.proposal_type}</td>
            <td>{proposal.status}</td><td>{proposal.confidence:.2f}</td>
            <td>{proposal.priority_score:.2f}</td><td>{proposal.impact_size}</td>
            <td>Rejected</td></tr>"""

    @app.post("/proposals/{proposal_id}/defer", response_class=HTMLResponse)
    async def do_defer(
        proposal_id: str,
        reviewer_note: str = Form(""),
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        reviewer = user.identity
        try:
            defer_proposal(store, proposal_id, reviewer, reviewer_note)
        except ReviewActionError as e:
            return f"<tr><td colspan='8'>Error: {e}</td></tr>"
        proposal = store.get_proposal(proposal_id)
        if not proposal:
            return "<tr><td colspan='8'>Error</td></tr>"
        return f"""<tr id="proposal-{proposal.proposal_id}">
            <td><a href="/proposals/{proposal.proposal_id}">{proposal.proposal_id[:12]}...</a></td>
            <td>{proposal.issue_class}</td><td>{proposal.proposal_type}</td>
            <td>{proposal.status}</td><td>{proposal.confidence:.2f}</td>
            <td>{proposal.priority_score:.2f}</td><td>{proposal.impact_size}</td>
            <td>Deferred</td></tr>"""

    @app.get("/api/proposals", response_class=JSONResponse)
    async def api_proposals(
        status: str | None = Query(None),
        snapshot_id: str | None = Query(None),
        _user: UserContext = Depends(require_archivist_or_admin),
    ) -> Any:
        return store.export_proposals_json(status=status, snapshot_id=snapshot_id)

    @app.get("/api/patches", response_class=JSONResponse)
    async def api_patches(
        snapshot_id: str | None = Query(None),
        _user: UserContext = Depends(require_archivist_or_admin),
    ) -> Any:
        return store.export_accepted_patches(snapshot_id=snapshot_id)

    @app.post("/proposals/{proposal_id}/targets/exclude", response_class=HTMLResponse)
    async def exclude_target(
        proposal_id: str,
        target_id: str = Query(...),
        _user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        store.set_target_override(proposal_id, "mention", target_id, "excluded")
        # Look up the mention from the evidence snapshot to re-render its row
        revisions = store.get_revisions(proposal_id)
        latest = revisions[-1] if revisions else None
        evidence: dict = {}  # type: ignore[type-arg]
        if latest:
            try:
                evidence = json.loads(latest.evidence_snapshot_json)
            except Exception:
                pass
        mention = next(
            (m for m in evidence.get("affected_mentions", []) if m.get("mention_id") == target_id),
            {"mention_id": target_id, "surface_form": target_id, "page_number": ""},
        )
        return _mention_row(mention, proposal_id, is_excluded=True)

    @app.post("/proposals/{proposal_id}/targets/include", response_class=HTMLResponse)
    async def include_target(
        proposal_id: str,
        target_id: str = Query(...),
        _user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        store.set_target_override(proposal_id, "mention", target_id, None)
        revisions = store.get_revisions(proposal_id)
        latest = revisions[-1] if revisions else None
        evidence: dict = {}  # type: ignore[type-arg]
        if latest:
            try:
                evidence = json.loads(latest.evidence_snapshot_json)
            except Exception:
                pass
        mention = next(
            (m for m in evidence.get("affected_mentions", []) if m.get("mention_id") == target_id),
            {"mention_id": target_id, "surface_form": target_id, "page_number": ""},
        )
        # Re-enrich with paragraph text
        sampled = _load_paragraph_context(evidence)
        para = next((m.get("paragraph_text", "") for m in sampled if m.get("mention_id") == target_id), "")
        mention = dict(mention)
        mention["paragraph_text"] = para
        return _mention_row(mention, proposal_id, is_excluded=False)

    @app.get("/proposals/{proposal_id}/mentions", response_class=HTMLResponse)
    async def load_more_mentions(
        proposal_id: str,
        offset: int = Query(20),
        limit: int = Query(20),
        _user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        revisions = store.get_revisions(proposal_id)
        latest = revisions[-1] if revisions else None
        evidence: dict = {}  # type: ignore[type-arg]
        if latest:
            try:
                evidence = json.loads(latest.evidence_snapshot_json)
            except Exception:
                pass

        targets = store.get_proposal_targets(proposal_id)
        excluded_ids = {t.target_id for t in targets if t.reviewer_override == "excluded"}

        sampled = _load_paragraph_context(evidence, offset=offset, sample_size=limit)
        total_pages = _count_unique_mention_pages(evidence)

        rows = "".join(
            _mention_row(m, proposal_id, m.get("mention_id", "") in excluded_ids)
            for m in sampled
        )
        next_offset = offset + len(sampled)
        return rows + _load_more_row_html(proposal_id, next_offset, total_pages)

    return app


# ---------------------------------------------------------------------------
# Inline HTML templates (avoids external file dependency for single-file deploy)
# ---------------------------------------------------------------------------

_BASE_CSS = """
<style>
    body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; }}
    h1, h2, h3 {{ color: #343a40; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #dee2e6; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #e9ecef; font-weight: 600; }}
    tr:hover {{ background: #f1f3f5; }}
    a {{ color: #0d6efd; }}
    .btn-accept {{ background: #198754; color: white; border: none; padding: 0.25rem 0.5rem; cursor: pointer; border-radius: 4px; }}
    .btn-reject {{ background: #dc3545; color: white; border: none; padding: 0.25rem 0.5rem; cursor: pointer; border-radius: 4px; }}
    .btn-defer {{ background: #ffc107; color: #212529; border: none; padding: 0.25rem 0.5rem; cursor: pointer; border-radius: 4px; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    pre {{ background: #e9ecef; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }}
    .nav {{ margin-bottom: 1rem; }}
    .nav a {{ margin-right: 1rem; }}
    .filters {{ margin: 1rem 0; }}
    .filters select, .filters input {{ padding: 0.25rem; margin-right: 0.5rem; }}
    .btn-accept-exceptions {{ background: #0d9488; color: white; border: none; padding: 0.25rem 0.75rem; cursor: pointer; border-radius: 4px; }}
    .risk-badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px; font-size: 0.78rem; font-weight: 600; margin-left: 0.35rem; }}
    .risk-high {{ background: #f8d7da; color: #842029; }}
    .risk-medium {{ background: #fff3cd; color: #664d03; }}
    .risk-low {{ background: #d1e7dd; color: #0f5132; }}
    .claim-box {{ background: #cfe2ff; border-left: 4px solid #0d6efd; padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.95rem; }}
    .action-summary {{ border: 1px solid #dee2e6; border-radius: 6px; padding: 1rem; margin-bottom: 1rem; background: #fff; }}
    .action-summary table {{ margin: 0.5rem 0; }}
    .action-summary th {{ background: #f8f9fa; width: 12rem; }}
    .provenance-section th {{ background: #f8f9fa; color: #6c757d; width: 12rem; }}
    details summary {{ cursor: pointer; font-weight: 600; margin: 1rem 0; color: #495057; }}
</style>
"""

_HTMX_SCRIPT = '<script src="https://unpkg.com/htmx.org@1.9.10"></script>'

_NAV = """<div class="nav">
    <a href="/">Dashboard</a>
    <a href="/proposals">All Proposals</a>
    <a href="/proposals?status=queued">Queued</a>
    <a href="/proposals?status=accepted_pending_apply">Accepted</a>
    <a href="/proposals?status=rejected">Rejected</a>
    <a href="/proposals?status=deferred">Deferred</a>
</div>"""

_INDEX_HTML = f"""<!DOCTYPE html><html><head><title>Review Dashboard</title>{_BASE_CSS}{_HTMX_SCRIPT}</head><body>
{_NAV}
<h1>Anti-Pattern Review Dashboard</h1>
<h2>By Status</h2>
<table><tr><th>Status</th><th>Count</th></tr>{{status_rows}}</table>
<h2>By Queue</h2>
<table><tr><th>Queue</th><th>Count</th></tr>{{queue_rows}}</table>
<p>Total proposals: {{total}}</p>
</body></html>"""

_LIST_HTML = f"""<!DOCTYPE html><html><head><title>Proposals</title>{_BASE_CSS}{_HTMX_SCRIPT}</head><body>
{_NAV}
<h1>Proposals</h1>
<div class="filters">
    <a href="/proposals?queue_name=sensitivity" style="color:#dc3545;font-weight:bold">&#9888; Sensitivity</a>
    <a href="/proposals?status=queued">Queued</a>
    <a href="/proposals?status=accepted_pending_apply">Accepted</a>
    <a href="/proposals?status=rejected">Rejected</a>
    <a href="/proposals?status=deferred">Deferred</a>
    <a href="/proposals?queue_name=ocr_entity">OCR/Entity</a>
    <a href="/proposals?queue_name=junk_mention">Junk Mention</a>
    <a href="/proposals?queue_name=builder_repair">Builder Repair</a>
    <a href="/proposals">All</a>
</div>
<table>
<thead><tr>
    <th>ID</th><th>Issue Class</th><th>Type</th><th>Status</th>
    <th>Confidence</th><th>Priority</th><th>Impact</th><th>Actions</th>
</tr></thead>
<tbody>{{rows}}</tbody>
</table>
<div>
    <a href="/proposals?{{filter_params}}&offset={{prev_offset}}&limit={{limit}}">Previous</a>
    <a href="/proposals?{{filter_params}}&offset={{next_offset}}&limit={{limit}}">Next</a>
</div>
</body></html>"""

_DETAIL_HTML = f"""<!DOCTYPE html><html><head><title>Proposal Detail</title>{_BASE_CSS}{_HTMX_SCRIPT}</head><body>
{_NAV}
<h1>Proposal {{proposal_id}}</h1>
<table>
<tr><th>Issue Class</th><td>{{issue_class}}</td></tr>
<tr><th>Proposal Type</th><td>{{proposal_type}}</td></tr>
<tr><th>Status</th><td>{{status}}</td></tr>
<tr><th>Confidence</th><td>{{confidence}}</td></tr>
<tr><th>Priority Score</th><td>{{priority_score}}</td></tr>
<tr><th>Impact Size</th><td>{{impact_size}}</td></tr>
<tr><th>Anti-Pattern</th><td>{{anti_pattern_id}}</td></tr>
</table>

{{proposed_action_panel}}

{{evidence_summary}}

<h2>Actions</h2>
<form style="margin:1rem 0">
    <label>Note: <input type="text" id="reviewer-note" value="" size="40"></label>
</form>
<div>
    <button hx-post="/proposals/{{proposal_id}}/accept" hx-include="#reviewer-note" {{disabled}} class="btn-accept">Accept</button>
    {{accept_exceptions_btn}}
    <button hx-post="/proposals/{{proposal_id}}/reject" hx-include="#reviewer-note" {{disabled}} class="btn-reject" style="margin-left:0.5rem">Reject</button>
    <button hx-post="/proposals/{{proposal_id}}/defer" hx-include="#reviewer-note" {{disabled}} class="btn-defer" style="margin-left:0.5rem">Defer</button>
</div>

<h2>Targets</h2>
<table><thead><tr><th>Kind</th><th>ID</th><th>Role</th><th>In Snapshot</th></tr></thead>
<tbody>{{target_rows}}</tbody></table>

{{provenance_html}}

<details>
<summary>Patch Spec (raw JSON)</summary>
<pre>{{patch_json}}</pre>
</details>

<details>
<summary>Evidence Snapshot (raw JSON)</summary>
<pre>{{evidence_json}}</pre>
</details>

<h2>Revision History</h2>
<table><thead><tr><th>#</th><th>Kind</th><th>Detector</th><th>Validation</th><th>By</th><th>At</th></tr></thead>
<tbody>{{revision_rows}}</tbody></table>

<h2>Correction Events</h2>
<table><thead><tr><th>Action</th><th>Reviewer</th><th>Note</th><th>At</th></tr></thead>
<tbody>{{event_rows}}</tbody></table>

</body></html>"""
