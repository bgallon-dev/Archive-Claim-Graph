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
    batch_accept_proposals,
    batch_reject_proposals,
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

_ISSUE_CLASS_LABELS: dict[str, str] = {
    "header_contamination": "Header contamination",
    "boilerplate_contamination": "Boilerplate contamination",
    "short_generic_token": "Short generic token",
    "ocr_garbage_mention": "OCR garbage mention",
    "ocr_spelling_variant": "OCR spelling variant",
    "duplicate_entity_alias": "Duplicate entity alias",
    "missing_species_focus": "Missing species link",
    "missing_event_location": "Missing location link",
    "method_overtrigger": "Method overtrigger",
    "pii_exposure": "PII exposure",
    "indigenous_sensitivity": "Indigenous cultural sensitivity",
    "living_person_reference": "Living person reference",
}

# Issue classes where batch-accepting with minimal inspection is appropriate.
# These map to suppress_mention proposals for low-consequence junk.
_BATCH_ELIGIBLE_CLASSES: frozenset[str] = frozenset({
    "header_contamination",
    "boilerplate_contamination",
    "short_generic_token",
    "ocr_garbage_mention",
})


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
    base = Path(out_dir).resolve()
    struct_path = (base / f"{doc_name}.structure.json").resolve()
    if not struct_path.is_relative_to(base):
        return []
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


def _collapsible_technical(inner_html: str) -> str:
    """Wrap technical detail HTML in a native collapsible <details> block."""
    return (
        '<details style="margin-top:1rem">'
        '<summary style="cursor:pointer;color:#6c757d;font-size:0.9rem;'
        'user-select:none">Technical details</summary>'
        '<div style="margin-top:0.5rem;padding:0.75rem;background:#f8f9fa;'
        'border-radius:4px;font-size:0.85rem">'
        + inner_html
        + "</div></details>"
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
    """Return a human-readable HTML block extracted from the evidence snapshot.

    Default view: plain-language summary + source context.
    Technical fields (IDs, raw scores, claim types) are behind a collapsible.
    """
    if not evidence:
        return ""

    def esc(v: object) -> str:
        return _html.escape(str(v))

    parts: list[str] = [
        '<section style="margin:1rem 0;padding:1rem;background:#fff3cd;border-radius:6px;border:1px solid #ffc107">',
        '<h2 style="margin-top:0">What to Review</h2>',
    ]

    if issue_class in _JUNK_ISSUE_CLASSES:
        all_mentions = evidence.get("affected_mentions", [])
        total = len(all_mentions)
        excluded_ids = excluded_target_ids or set()
        excluded_count = len(excluded_ids)
        display = sampled_mentions if sampled_mentions else all_mentions[:20]

        # Plain-language summary
        surface_form = ""
        if all_mentions:
            surface_form = (
                all_mentions[0].get("normalized_form", "")
                or all_mentions[0].get("surface_form", "")
            )
        sf_display = f"\u201c{esc(surface_form)}\u201d" if surface_form else "This surface form"
        _junk_plain: dict[str, str] = {
            "header_contamination": (
                "it appears primarily in page headers or footers rather than the body of the document"
            ),
            "boilerplate_contamination": (
                "it appears in boilerplate text such as form fields or institutional letterhead, "
                "not in substantive content"
            ),
            "short_generic_token": (
                "it is a short or generic word that is unlikely to be a meaningful named entity"
            ),
            "ocr_garbage_mention": (
                "the text appears to be OCR noise \u2014 a misread character sequence rather than "
                "a real entity name"
            ),
        }
        reason_desc = _junk_plain.get(issue_class, "it does not appear to be a real named entity")
        parts.append(
            f'<div class="claim-box">'
            f"The word {sf_display} was detected as a named entity, but {reason_desc}. "
            f"The system proposes to ignore it. "
            f"Review the mentions below to confirm.</div>"
        )

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

        raw_reason = evidence.get("suppression_reason", issue_class)
        parts.append(_collapsible_technical(
            f"<p><strong>Issue class:</strong> {esc(issue_class)}</p>"
            f"<p><strong>Suppression reason (raw):</strong> {esc(raw_reason)}</p>"
            f"<p><strong>Total affected mentions:</strong> {esc(total)}</p>"
        ))

    elif issue_class == "ocr_spelling_variant":
        canon = evidence.get("canonical_entity", {})
        variants = evidence.get("merge_entities", [])
        avg_sim = evidence.get("average_similarity", "")
        canon_count = evidence.get("canonical_mention_count", "")
        merge_counts = evidence.get("merge_mention_counts", {})

        # Plain-language summary
        canon_name = esc(canon.get("name", ""))
        if variants:
            variant_name_list = " and ".join(
                f"\u201c{esc(v.get('name', ''))}\u201d" for v in variants[:3]
            )
            suffix = f" (and {len(variants) - 3} more)" if len(variants) > 3 else ""
            parts.append(
                f'<div class="claim-box">'
                f"The system found name(s) that appear to refer to the same thing as "
                f"<strong>{canon_name}</strong>: {variant_name_list}{esc(suffix)}. "
                f"It is proposing to treat them as the same entity going forward. "
                f"Does this look correct?</div>"
            )
        else:
            parts.append(
                f'<div class="claim-box">'
                f"The system found a name that may be a spelling variant of "
                f"<strong>{canon_name}</strong> and is proposing to merge them. "
                f"Does this look correct?</div>"
            )

        # Technical details
        tech: list[str] = [
            f"<p><strong>Canonical entity:</strong> {esc(canon.get('name', ''))} "
            f"({esc(canon.get('entity_type', ''))}) \u2014 {esc(canon_count)} mentions</p>",
            f"<p><strong>Average similarity score:</strong> {esc(avg_sim)}</p>",
        ]
        if variants:
            tech.append(
                "<table><thead><tr>"
                "<th>Variant Name</th><th>Type</th><th>Mentions</th>"
                "</tr></thead><tbody>"
            )
            for v in variants:
                count = merge_counts.get(v.get("entity_id", ""), "")
                tech.append(
                    f"<tr><td>{esc(v.get('name', ''))}</td>"
                    f"<td>{esc(v.get('entity_type', ''))}</td>"
                    f"<td>{esc(count)}</td></tr>"
                )
            tech.append("</tbody></table>")
        parts.append(_collapsible_technical("".join(tech)))

    elif issue_class == "duplicate_entity_alias":
        canon = evidence.get("canonical_entity", {})
        alias = evidence.get("alias_entity", {})
        sim = evidence.get("similarity_score", "")
        canon_count = evidence.get("canonical_mention_count", "")
        alias_count = evidence.get("alias_mention_count", "")

        # Plain-language summary
        canon_name = esc(canon.get("name", ""))
        alias_name = esc(alias.get("name", ""))
        parts.append(
            f'<div class="claim-box">'
            f"The system found two names that appear to refer to the same thing: "
            f"<strong>{alias_name}</strong> and <strong>{canon_name}</strong>. "
            f"It is proposing to link them as aliases of the same entity going forward. "
            f"Does this look correct?</div>"
        )

        # Technical details
        tech_html = (
            "<table><thead><tr>"
            "<th>Role</th><th>Name</th><th>Type</th><th>Mentions</th>"
            "</tr></thead><tbody>"
            f"<tr><td>Canonical</td><td>{esc(canon.get('name', ''))}</td>"
            f"<td>{esc(canon.get('entity_type', ''))}</td><td>{esc(canon_count)}</td></tr>"
            f"<tr><td>Alias</td><td>{esc(alias.get('name', ''))}</td>"
            f"<td>{esc(alias.get('entity_type', ''))}</td><td>{esc(alias_count)}</td></tr>"
            "</tbody></table>"
            f"<p><strong>Similarity score:</strong> {esc(sim)}</p>"
        )
        parts.append(_collapsible_technical(tech_html))

    elif issue_class in ("missing_species_focus", "missing_event_location"):
        sentence = evidence.get("source_sentence", "")
        claim_type = evidence.get("claim_type", "")
        key = "candidate_species" if issue_class == "missing_species_focus" else "candidate_location"
        candidate = evidence.get(key, {})
        label = "species" if issue_class == "missing_species_focus" else "location"
        candidate_name = candidate.get("name", "")

        # Plain-language summary
        if candidate_name:
            parts.append(
                f'<div class="claim-box">'
                f"This sentence mentions a {label} but the system wasn\u2019t able to connect "
                f"them automatically. The likely match is "
                f"<strong>{esc(candidate_name)}</strong>. Is that correct?</div>"
            )
        else:
            parts.append(
                f'<div class="claim-box">'
                f"This sentence appears to reference a {label}, but the system could not "
                f"automatically identify a matching entity. Please review and link the "
                f"appropriate {label} if one exists.</div>"
            )

        # Source sentence (primary context)
        if sentence:
            parts.append(
                "<p><strong>Source sentence:</strong></p>"
                '<blockquote style="border-left:4px solid #ccc;margin:0;padding:0 1rem">'
                f"<p>{esc(sentence)}</p></blockquote>"
            )

        # Technical details
        tech: list[str] = [f"<p><strong>Claim type:</strong> {esc(claim_type)}</p>"]
        if candidate:
            tech.append(
                f"<p><strong>Candidate entity type:</strong> "
                f"{esc(candidate.get('entity_type', ''))}</p>"
            )
        parts.append(_collapsible_technical("".join(tech)))

    elif issue_class == "method_overtrigger":
        sentence = evidence.get("source_sentence", "")
        claim_type = evidence.get("claim_type", "")
        method = evidence.get("method_entity", {})
        compat = evidence.get("compatibility", "")
        method_name = method.get("name", "")

        # Plain-language summary
        method_display = f" (<strong>{esc(method_name)}</strong>)" if method_name else ""
        parts.append(
            f'<div class="claim-box">'
            f"The system linked a survey method{method_display} to this claim, but it may not "
            f"belong there. Approving this will remove that connection.</div>"
        )

        # Source sentence (primary context)
        if sentence:
            parts.append(
                "<p><strong>Source sentence:</strong></p>"
                '<blockquote style="border-left:4px solid #ccc;margin:0;padding:0 1rem">'
                f"<p>{esc(sentence)}</p></blockquote>"
            )

        # Technical details
        tech_html = (
            f"<p><strong>Claim type:</strong> {esc(claim_type)}</p>"
            f"<p><strong>Method entity:</strong> {esc(method_name)} "
            f"({esc(method.get('entity_type', ''))})</p>"
            f"<p><strong>Compatibility score:</strong> {esc(compat)}</p>"
        )
        parts.append(_collapsible_technical(tech_html))

    elif issue_class == "pii_exposure":
        matched = evidence.get("matched_pattern", "")
        all_patterns = evidence.get("all_patterns", [])
        claim_id = evidence.get("claim_id", "")
        redacted = evidence.get("redacted_sentence", "")

        # Severity warning (stays visible — do not collapse)
        parts.append(
            '<p style="color:#dc3545"><strong>Action required:</strong> '
            "Review whether this claim contains personal information that should remain "
            "quarantined or be permanently restricted. Do not un-quarantine without data "
            "governance approval.</p>"
        )

        # Plain-language summary
        pattern_label = esc(matched) if matched else "personal information"
        parts.append(
            f'<div class="claim-box">'
            f"This claim appears to contain {pattern_label}. "
            f"The sentence below has been shown with the sensitive content redacted.</div>"
        )

        if redacted:
            parts.append(
                "<p><strong>Source sentence (PII redacted):</strong></p>"
                '<blockquote style="border-left:4px solid #dc3545;margin:0;padding:0 1rem">'
                f"<p>{esc(redacted)}</p></blockquote>"
            )

        # Technical details
        tech: list[str] = [f"<p><strong>Detection type:</strong> {esc(matched)}</p>"]
        if all_patterns:
            tech.append(
                f"<p><strong>All matched patterns:</strong> "
                f"{esc(', '.join(all_patterns))}</p>"
            )
        if claim_id:
            tech.append(f"<p><strong>Claim ID:</strong> {esc(claim_id)}</p>")
        parts.append(_collapsible_technical("".join(tech)))

    elif issue_class == "indigenous_sensitivity":
        matched_term = evidence.get("matched_term", "")
        category = evidence.get("category", "")
        sensitivity = evidence.get("sensitivity", "")
        nations = evidence.get("nations", [])
        claim_id = evidence.get("claim_id", "")
        require_consultation = evidence.get("require_tribal_consultation_before_clear", True)

        # Consultation warning (stays visible — do not collapse)
        if require_consultation:
            parts.append(
                '<p style="color:#dc3545;font-weight:bold">&#9888; Tribal consultation required: '
                "Clearing this quarantine requires documented consultation with the relevant "
                "Indigenous nation(s). Archivist judgment alone is not sufficient.</p>"
            )

        # Plain-language summary
        nations_text = (
            f" associated with {esc(', '.join(nations))}" if nations else ""
        )
        parts.append(
            f'<div class="claim-box">'
            f"This claim contains the term <em>{esc(matched_term)}</em>{nations_text}, "
            f"which is flagged as Indigenous cultural material. Community consultation may be "
            f"required before this information is published.</div>"
        )

        # Technical details
        tech: list[str] = [
            f"<p><strong>Vocabulary category:</strong> {esc(category)}</p>",
            f"<p><strong>Sensitivity level:</strong> {esc(sensitivity)}</p>",
        ]
        if nations:
            tech.append(
                f"<p><strong>Associated nations:</strong> {esc(', '.join(nations))}</p>"
            )
        if claim_id:
            tech.append(f"<p><strong>Claim ID:</strong> {esc(claim_id)}</p>")
        parts.append(_collapsible_technical("".join(tech)))

    elif issue_class == "living_person_reference":
        person_names = evidence.get("person_names", [])
        most_recent_year = evidence.get("most_recent_year", "")
        source_sentence = evidence.get("source_sentence", "")
        year_threshold = evidence.get("year_threshold", "")
        claim_id = evidence.get("claim_id", "")

        # Plain-language summary
        names_display = esc(", ".join(person_names)) if person_names else "an individual"
        year_text = (
            f" (most recent associated year: {esc(most_recent_year)})"
            if most_recent_year else ""
        )
        parts.append(
            f'<div class="claim-box">'
            f"This claim may refer to a living person: <strong>{names_display}</strong>"
            f"{year_text}. Approving publication requires a privacy review.</div>"
        )

        if source_sentence:
            parts.append(
                "<p><strong>Source sentence:</strong></p>"
                '<blockquote style="border-left:4px solid #fd7e14;margin:0;padding:0 1rem">'
                f"<p>{esc(source_sentence)}</p></blockquote>"
            )

        # Privacy note (stays visible)
        parts.append(
            '<p style="color:#856404"><strong>Note:</strong> '
            "Review whether this individual is a living person and whether publication of this "
            "claim would violate their privacy rights.</p>"
        )

        # Technical details
        tech: list[str] = []
        if year_threshold:
            tech.append(
                f"<p><strong>Year threshold (within last N years):</strong> "
                f"{esc(year_threshold)}</p>"
            )
        if claim_id:
            tech.append(f"<p><strong>Claim ID:</strong> {esc(claim_id)}</p>")
        if tech:
            parts.append(_collapsible_technical("".join(tech)))

    else:
        return ""

    parts.append("</section>")
    return "".join(parts)


def _key_info_html(issue_class: str, evidence: dict) -> str:  # type: ignore[type-arg]
    """Return the single most informative HTML fragment for a proposal's evidence.

    Used by both _batch_sample_card (batch summary page) and _priority_card
    (landing page). No mention tables — just the key identification field.
    """
    def esc(v: object) -> str:
        return _html.escape(str(v))

    ic = issue_class

    if ic in _JUNK_ISSUE_CLASSES:
        mentions = evidence.get("affected_mentions", [])
        sf = (
            (mentions[0].get("normalized_form") or mentions[0].get("surface_form", ""))
            if mentions else ""
        )
        return f'Surface form: <strong>{esc(sf)}</strong>' if sf else "(no surface form)"

    if ic == "ocr_spelling_variant":
        canon = evidence.get("canonical_entity", {})
        variants = evidence.get("merge_entities", [])
        names = " \u2022 ".join(esc(v.get("name", "")) for v in variants[:3])
        suffix = f" (+{len(variants) - 3} more)" if len(variants) > 3 else ""
        return (
            f'<strong>{names}{esc(suffix)}</strong> \u2192 '
            f'<strong>{esc(canon.get("name", ""))}</strong>'
        )

    if ic == "duplicate_entity_alias":
        canon = evidence.get("canonical_entity", {})
        alias = evidence.get("alias_entity", {})
        return (
            f'<strong>{esc(alias.get("name", ""))}</strong> \u2192 '
            f'<strong>{esc(canon.get("name", ""))}</strong>'
        )

    if ic in ("missing_species_focus", "missing_event_location"):
        sentence = evidence.get("source_sentence", "")
        return (
            f'<em>{esc(sentence[:150])}{"&hellip;" if len(sentence) > 150 else ""}</em>'
            if sentence else "(no source sentence)"
        )

    if ic == "method_overtrigger":
        method = evidence.get("method_entity", {})
        sentence = evidence.get("source_sentence", "")
        return (
            f'Method: <strong>{esc(method.get("name", ""))}</strong>'
            + (
                f'<br><em style="font-size:0.9rem">'
                f'{esc(sentence[:120])}{"&hellip;" if len(sentence) > 120 else ""}</em>'
                if sentence else ""
            )
        )

    if ic == "pii_exposure":
        matched = evidence.get("matched_pattern", "")
        redacted = evidence.get("redacted_sentence", "")
        return (
            f'Pattern: <strong>{esc(matched)}</strong>'
            + (
                f'<br><em style="font-size:0.9rem">{esc(redacted[:120])}&hellip;</em>'
                if redacted else ""
            )
        )

    if ic == "indigenous_sensitivity":
        term = evidence.get("matched_term", "")
        nations = evidence.get("nations", [])
        nations_text = f" ({esc(', '.join(nations))})" if nations else ""
        return f'Term: <strong>{esc(term)}</strong>{nations_text}'

    if ic == "living_person_reference":
        names = evidence.get("person_names", [])
        year = evidence.get("most_recent_year", "")
        year_text = f" \u2014 most recent year: {esc(year)}" if year else ""
        return f'Person: <strong>{esc(", ".join(names))}</strong>{year_text}'

    return f'<em style="color:#6c757d">({esc(ic)})</em>'


def _batch_sample_card(proposal: Any, evidence: dict) -> str:  # type: ignore[type-arg]
    """Compact evidence card for the batch review summary page (5-sample spot-check)."""
    def esc(v: object) -> str:
        return _html.escape(str(v))

    return (
        f'<div style="border:1px solid #dee2e6;border-radius:6px;'
        f'padding:0.75rem;margin:0.5rem 0;background:#fff">'
        f'<p style="margin:0 0 0.4rem;font-size:0.82rem;color:#6c757d">'
        f'<code>{esc(proposal.proposal_id[:16])}&hellip;</code>'
        f' &nbsp;&middot;&nbsp; confidence {proposal.confidence:.2f}'
        f' &nbsp;&middot;&nbsp; '
        f'<a href="/proposals/{esc(proposal.proposal_id)}">view detail &rarr;</a></p>'
        f'<p style="margin:0">{_key_info_html(proposal.issue_class, evidence)}</p>'
        f'</div>'
    )


def _priority_card(proposal: Any, evidence: dict) -> str:  # type: ignore[type-arg]
    """Priority card for the landing page 'What needs your attention' section.

    Shows the plain-language key info plus inline Accept and Defer HTMX buttons.
    Buttons use hx-swap='delete' so accepted/deferred cards disappear without
    a page reload.
    """
    def esc(v: object) -> str:
        return _html.escape(str(v))

    pid = proposal.proposal_id
    pid_esc = esc(pid)
    card_id = f"priority-card-{pid_esc.replace(':', '-')}"
    ic_label = _html.escape(_ISSUE_CLASS_LABELS.get(proposal.issue_class, proposal.issue_class))

    is_sensitivity = proposal.issue_class in (
        "pii_exposure", "indigenous_sensitivity", "living_person_reference"
    )
    border_color = "#dc3545" if is_sensitivity else "#0d6efd"
    badge_style = (
        'background:#f8d7da;color:#842029'
        if is_sensitivity else
        'background:#cfe2ff;color:#084298'
    )

    actionable = proposal.status in ("queued", "deferred")
    accept_btn = (
        f'<button hx-post="/proposals/{pid_esc}/accept" '
        f'hx-target="#{card_id}" hx-swap="delete" '
        f'class="btn-accept" style="font-size:0.8rem;padding:0.2rem 0.5rem"'
        f'{"" if actionable else " disabled"}>Accept</button>'
    )
    defer_btn = (
        f'<button hx-post="/proposals/{pid_esc}/defer" '
        f'hx-target="#{card_id}" hx-swap="delete" '
        f'class="btn-defer" style="font-size:0.8rem;padding:0.2rem 0.5rem;margin-left:0.3rem"'
        f'{"" if actionable and proposal.status == \"queued\" else \" disabled"}>Defer</button>'
    )

    return (
        f'<div id="{card_id}" style="border:1px solid {border_color};border-radius:6px;'
        f'padding:0.85rem;margin:0.5rem 0;background:#fff">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        f'margin-bottom:0.4rem">'
        f'<span style="font-size:0.8rem;padding:0.15rem 0.5rem;border-radius:3px;'
        f'font-weight:600;{badge_style}">{ic_label}</span>'
        f'<span style="font-size:0.8rem;color:#6c757d">'
        f'priority {proposal.priority_score:.2f} &nbsp;&middot;&nbsp; '
        f'confidence {proposal.confidence:.2f} &nbsp;&middot;&nbsp; '
        f'{proposal.impact_size} target{"s" if proposal.impact_size != 1 else ""}'
        f'</span>'
        f'</div>'
        f'<p style="margin:0.3rem 0 0.5rem">'
        f'{_key_info_html(proposal.issue_class, evidence)}</p>'
        f'<div style="font-size:0.82rem">'
        f'{accept_btn}{defer_btn}'
        f'&nbsp;&nbsp;<a href="/proposals/{pid_esc}" style="color:#6c757d">'
        f'Full detail &rarr;</a>'
        f'</div>'
        f'</div>'
    )


def create_app(db_path: str, users_db_path: str = "data/users.db") -> Any:
    """Create and return a FastAPI app for the review UI.

    Requires `fastapi` and `uvicorn` to be installed.
    """
    from graphrag_pipeline.shared.logging_config import setup_logging
    setup_logging()

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

    import logging as _logging
    from datetime import datetime as _datetime, timezone as _timezone
    from fastapi import HTTPException as _HTTPException
    from fastapi.responses import JSONResponse as _JSONResponse

    _log = _logging.getLogger(__name__)

    @app.exception_handler(Exception)
    async def _internal_error_handler(request: Request, exc: Exception) -> _JSONResponse:
        if isinstance(exc, _HTTPException):
            raise exc
        _log.error(
            "Unhandled exception on %s %s: %s",
            request.method, request.url, exc, exc_info=True,
        )
        return _JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred."},
        )

    @app.middleware("http")
    async def _setup_guard(request: Request, call_next):
        if not request.url.path.startswith("/auth/setup") and request.url.path != "/health":
            if is_setup_needed(users_db_path):
                return _RedirectResponse(url="/auth/setup", status_code=303)
        return await call_next(request)

    store = ReviewStore(db_path)

    # ------------------------------------------------------------------
    # GET /health — liveness + ReviewStore connectivity check
    # NOTE: This endpoint MUST remain unauthenticated. It is used by uptime
    # monitors and load balancers that have no session credentials.
    # ------------------------------------------------------------------
    @app.get("/health", include_in_schema=True)
    def health():
        ts = _datetime.now(_timezone.utc).isoformat()
        try:
            store.proposal_counts_by_status()
            db_status = "connected"
        except Exception:
            db_status = "unavailable"
        db_ok = db_status == "connected"
        return _JSONResponse(
            status_code=200 if db_ok else 503,
            content={"status": "ok" if db_ok else "degraded", "review_db": db_status, "timestamp": ts},
        )

    @app.get("/", response_class=HTMLResponse)
    async def index(_user: UserContext = Depends(require_login)) -> str:
        from datetime import date as _date

        # --- Section 1: top 5 highest-priority queued proposals ---
        top5 = store.list_proposals(status="queued", limit=5)
        priority_cards = ""
        for p in top5:
            rev = store.get_latest_revision(p.proposal_id)
            ev: dict = {}  # type: ignore[type-arg]
            if rev:
                try:
                    ev = json.loads(rev.evidence_snapshot_json)
                except Exception:
                    pass
            priority_cards += _priority_card(p, ev)
        if not priority_cards:
            priority_cards = (
                '<p style="color:#6c757d;font-style:italic">No queued proposals — queue is clear.</p>'
            )

        # Sensitivity alert (shown above everything if non-zero)
        sensitivity_queued = store.list_proposals(
            queue_name="sensitivity", status="queued", limit=1
        )
        if sensitivity_queued:
            sc = store.proposal_counts_by_issue_class(status="queued")
            sensitivity_total = sum(
                v["count"] for k, v in sc.items()
                if k in ("pii_exposure", "indigenous_sensitivity", "living_person_reference")
            )
            sensitivity_alert = (
                f'<div style="background:#f8d7da;border:1px solid #f5c2c7;border-radius:6px;'
                f'padding:0.75rem 1rem;margin-bottom:1rem">'
                f'<strong style="color:#842029">&#9888; Sensitivity queue:</strong> '
                f'{sensitivity_total} proposal{"s" if sensitivity_total != 1 else ""} '
                f'require urgent review. '
                f'<a href="/proposals?queue_name=sensitivity" style="color:#842029;font-weight:600">'
                f'Review now &rarr;</a>'
                f'</div>'
            )
        else:
            sensitivity_alert = ""

        # --- Section 2: batch-eligible categories ---
        ic_counts = store.proposal_counts_by_issue_class(status="queued")
        batch_rows = ""
        for ic, stats in ic_counts.items():
            if ic not in _BATCH_ELIGIBLE_CLASSES:
                continue
            label = _html.escape(_ISSUE_CLASS_LABELS.get(ic, ic))
            count = stats["count"]
            avg_conf = stats["avg_confidence"]
            batch_href = f"/proposals/batch?issue_class={_html.escape(ic)}"
            batch_rows += (
                f"<tr>"
                f"<td>{label}</td>"
                f"<td>{count}</td>"
                f"<td>{avg_conf:.2f}</td>"
                f'<td><a href="{batch_href}" class="btn-batch-review">Review as batch</a></td>'
                f"</tr>"
            )
        if not batch_rows:
            batch_rows = (
                '<tr><td colspan="4" style="color:#6c757d;font-style:italic">'
                'No batch-eligible proposals queued.</td></tr>'
            )

        # --- Section 3: done today ---
        today_iso = _date.today().isoformat()  # "YYYY-MM-DD" prefix match
        event_counts = store.correction_event_counts(since_iso=today_iso)
        accepted_today = event_counts.get("accept", 0)
        rejected_today = event_counts.get("reject", 0)
        deferred_today = event_counts.get("defer", 0)

        return _INDEX_HTML.format(
            sensitivity_alert=sensitivity_alert,
            priority_cards=priority_cards,
            batch_rows=batch_rows,
            accepted_today=accepted_today,
            rejected_today=rejected_today,
            deferred_today=deferred_today,
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
            f"{k}={v}" for k, v in [
                ("status", status), ("issue_class", issue_class),
                ("queue_name", queue_name), ("doc_id", doc_id),
            ]
            if v
        )
        next_offset = offset + limit
        prev_offset = max(0, offset - limit)

        # Issue-class summary table (shown when viewing the queued list unfiltered,
        # or filtered by queue_name only — gives a quick count-per-class overview)
        show_summary = not issue_class and not status or status == "queued"
        if show_summary:
            ic_counts = store.proposal_counts_by_issue_class(status="queued")
            if ic_counts:
                ic_rows = ""
                for ic, stats in ic_counts.items():
                    batch_href = f"/proposals/batch?issue_class={_html.escape(ic)}"
                    ic_rows += (
                        f"<tr>"
                        f"<td>{_html.escape(ic)}</td>"
                        f"<td>{stats['count']}</td>"
                        f"<td>{stats['avg_confidence']:.2f}</td>"
                        f'<td><a href="{batch_href}" class="btn-batch-review">Review as batch</a></td>'
                        f"</tr>"
                    )
                queue_summary = (
                    '<section style="margin:1rem 0;padding:1rem;background:#e7f1ff;'
                    'border-radius:6px;border:1px solid #b6d4fe">'
                    '<h2 style="margin-top:0;font-size:1rem">Queued proposals by issue class</h2>'
                    '<table><thead><tr>'
                    '<th>Issue Class</th><th>Count</th><th>Avg Confidence</th><th></th>'
                    '</tr></thead><tbody>'
                    + ic_rows
                    + '</tbody></table></section>'
                )
            else:
                queue_summary = ""
        else:
            # When filtered to a specific issue class, show the batch review link inline
            if issue_class:
                batch_href = f"/proposals/batch?issue_class={_html.escape(issue_class)}"
                queue_summary = (
                    f'<p style="margin:0.5rem 0">'
                    f'<a href="{batch_href}" class="btn-batch-review">'
                    f'Review all &ldquo;{_html.escape(issue_class)}&rdquo; as batch &rarr;</a></p>'
                )
            else:
                queue_summary = ""

        return _LIST_HTML.format(
            rows=rows,
            filter_params=filter_params,
            next_offset=next_offset,
            prev_offset=prev_offset,
            limit=limit,
            current_status=status or "",
            current_issue_class=issue_class or "",
            current_queue=queue_name or "",
            queue_summary=queue_summary,
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

    @app.get("/proposals/batch", response_class=HTMLResponse)
    async def batch_review_summary(
        issue_class: str | None = Query(None),
        queue_name: str | None = Query(None),
        _user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        """Batch review page: stats + 5-sample evidence cards + accept/reject form."""
        proposals = store.list_proposals(
            issue_class=issue_class,
            queue_name=queue_name,
            status="queued",
            limit=10000,
        )
        if not proposals:
            filter_label = _html.escape(issue_class or queue_name or "selected filter")
            return (
                f'<!DOCTYPE html><html><head><title>Batch Review</title>'
                f'{_BASE_CSS}</head><body>{_NAV}'
                f'<h1>Batch Review</h1>'
                f'<p>No queued proposals found for <strong>{filter_label}</strong>.</p>'
                f'<p><a href="/proposals">Back to queue</a></p>'
                f'</body></html>'
            )

        count = len(proposals)
        confidences = [p.confidence for p in proposals]
        conf_min = min(confidences)
        conf_max = max(confidences)
        conf_avg = sum(confidences) / count

        # Build 5-card sample
        sample_cards = ""
        for p in proposals[:5]:
            rev = store.get_latest_revision(p.proposal_id)
            ev: dict = {}  # type: ignore[type-arg]
            if rev:
                try:
                    ev = json.loads(rev.evidence_snapshot_json)
                except Exception:
                    pass
            sample_cards += _batch_sample_card(p, ev)

        filter_label = _html.escape(issue_class or queue_name or "all")
        back_qs = "&".join(
            f"{k}={_html.escape(v)}"
            for k, v in [("issue_class", issue_class), ("queue_name", queue_name)]
            if v
        )
        back_href = f"/proposals?{back_qs}" if back_qs else "/proposals"

        return _BATCH_HTML.format(
            filter_label=filter_label,
            count=count,
            conf_min=f"{conf_min:.2f}",
            conf_max=f"{conf_max:.2f}",
            conf_avg=f"{conf_avg:.2f}",
            sample_cards=sample_cards,
            issue_class=_html.escape(issue_class or ""),
            queue_name=_html.escape(queue_name or ""),
            back_href=back_href,
        )

    @app.post("/proposals/batch-accept", response_class=HTMLResponse)
    async def do_batch_accept(
        issue_class: str = Form(""),
        queue_name: str = Form(""),
        reviewer_note: str = Form(""),
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        if not reviewer_note.strip():
            return (
                f'<!DOCTYPE html><html><head><title>Batch Accept</title>'
                f'{_BASE_CSS}</head><body>{_NAV}'
                f'<p style="color:#dc3545;font-weight:bold">'
                f'A reviewer note is required for batch actions. '
                f'<a href="javascript:history.back()">Go back</a></p>'
                f'</body></html>'
            )
        proposals = store.list_proposals(
            issue_class=issue_class or None,
            queue_name=queue_name or None,
            status="queued",
            limit=10000,
        )
        proposal_ids = [p.proposal_id for p in proposals]
        result = batch_accept_proposals(store, proposal_ids, user.identity, reviewer_note)
        accepted = result["accepted"]
        skipped = result["skipped"]
        label = _html.escape(issue_class or queue_name or "all")
        back_qs = "&".join(
            f"{k}={_html.escape(v)}"
            for k, v in [("issue_class", issue_class), ("queue_name", queue_name)]
            if v
        )
        back_href = f"/proposals?{back_qs}" if back_qs else "/proposals"
        skip_note = f" ({skipped} already resolved)" if skipped else ""
        return (
            f'<!DOCTYPE html><html><head><title>Batch Accept</title>'
            f'{_BASE_CSS}</head><body>{_NAV}'
            f'<h1>Batch Accept Complete</h1>'
            f'<div style="padding:1rem;background:#d1e7dd;border-radius:6px;margin:1rem 0">'
            f'<strong>{accepted} proposals accepted</strong>{_html.escape(skip_note)} '
            f'for <em>{label}</em>.'
            f'</div>'
            f'<p>Note: <em>{_html.escape(reviewer_note)}</em></p>'
            f'<p><a href="{back_href}">Back to queue &rarr;</a></p>'
            f'</body></html>'
        )

    @app.post("/proposals/batch-reject", response_class=HTMLResponse)
    async def do_batch_reject(
        issue_class: str = Form(""),
        queue_name: str = Form(""),
        reviewer_note: str = Form(""),
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> str:
        if not reviewer_note.strip():
            return (
                f'<!DOCTYPE html><html><head><title>Batch Reject</title>'
                f'{_BASE_CSS}</head><body>{_NAV}'
                f'<p style="color:#dc3545;font-weight:bold">'
                f'A reviewer note is required for batch actions. '
                f'<a href="javascript:history.back()">Go back</a></p>'
                f'</body></html>'
            )
        proposals = store.list_proposals(
            issue_class=issue_class or None,
            queue_name=queue_name or None,
            status="queued",
            limit=10000,
        )
        proposal_ids = [p.proposal_id for p in proposals]
        result = batch_reject_proposals(store, proposal_ids, user.identity, reviewer_note)
        rejected = result["rejected"]
        skipped = result["skipped"]
        label = _html.escape(issue_class or queue_name or "all")
        back_qs = "&".join(
            f"{k}={_html.escape(v)}"
            for k, v in [("issue_class", issue_class), ("queue_name", queue_name)]
            if v
        )
        back_href = f"/proposals?{back_qs}" if back_qs else "/proposals"
        skip_note = f" ({skipped} already resolved)" if skipped else ""
        return (
            f'<!DOCTYPE html><html><head><title>Batch Reject</title>'
            f'{_BASE_CSS}</head><body>{_NAV}'
            f'<h1>Batch Reject Complete</h1>'
            f'<div style="padding:1rem;background:#f8d7da;border-radius:6px;margin:1rem 0">'
            f'<strong>{rejected} proposals rejected</strong>{_html.escape(skip_note)} '
            f'for <em>{label}</em>.'
            f'</div>'
            f'<p>Note: <em>{_html.escape(reviewer_note)}</em></p>'
            f'<p><a href="{back_href}">Back to queue &rarr;</a></p>'
            f'</body></html>'
        )

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
    .btn-batch-review {{ background: #0d6efd; color: white; padding: 0.25rem 0.75rem; border-radius: 4px; text-decoration: none; font-size: 0.875rem; display: inline-block; }}
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

_INDEX_HTML = f"""<!DOCTYPE html><html><head><title>Review Queue</title>{_BASE_CSS}{_HTMX_SCRIPT}</head><body>
{_NAV}
<h1>Review Queue</h1>
{{sensitivity_alert}}
<section style="margin:1.5rem 0">
<h2 style="font-size:1.1rem;border-bottom:2px solid #dee2e6;padding-bottom:0.4rem">
  What needs your attention today?</h2>
<p style="color:#6c757d;font-size:0.9rem;margin-top:0">
  Top 5 proposals by priority score. Accept or defer directly, or open for full detail.
</p>
{{priority_cards}}
<p style="margin-top:0.5rem;font-size:0.9rem">
  <a href="/proposals?status=queued">See all queued proposals &rarr;</a>
</p>
</section>

<section style="margin:1.5rem 0">
<h2 style="font-size:1.1rem;border-bottom:2px solid #dee2e6;padding-bottom:0.4rem">
  What can be handled quickly?</h2>
<p style="color:#6c757d;font-size:0.9rem;margin-top:0">
  Low-consequence proposals suitable for batch review. Spot-check a sample, then accept or reject the group with one action.
</p>
<table>
<thead><tr><th>Issue class</th><th>Queued</th><th>Avg confidence</th><th></th></tr></thead>
<tbody>{{batch_rows}}</tbody>
</table>
</section>

<section style="margin:1.5rem 0">
<h2 style="font-size:1.1rem;border-bottom:2px solid #dee2e6;padding-bottom:0.4rem">
  What has been done today?</h2>
<table style="width:auto">
<tr><th>Accepted</th><td><strong>{{accepted_today}}</strong></td></tr>
<tr><th>Rejected</th><td><strong>{{rejected_today}}</strong></td></tr>
<tr><th>Deferred</th><td><strong>{{deferred_today}}</strong></td></tr>
</table>
<p style="font-size:0.85rem;color:#6c757d">Counts since midnight UTC today.</p>
</section>
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
{{queue_summary}}
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

_BATCH_HTML = f"""<!DOCTYPE html><html><head><title>Batch Review</title>{_BASE_CSS}{_HTMX_SCRIPT}</head><body>
{_NAV}
<h1>Batch Review &mdash; {{filter_label}}</h1>

<section style="margin:1rem 0;padding:1rem;background:#e7f1ff;border-radius:6px;border:1px solid #b6d4fe">
<h2 style="margin-top:0;font-size:1rem">Summary</h2>
<table style="width:auto">
<tr><th>Queued proposals</th><td><strong>{{count}}</strong></td></tr>
<tr><th>Confidence range</th><td>{{conf_min}} &ndash; {{conf_max}}</td></tr>
<tr><th>Average confidence</th><td>{{conf_avg}}</td></tr>
</table>
</section>

<h2 style="font-size:1rem">Sample (up to 5 proposals)</h2>
{{sample_cards}}

<section style="margin:1.5rem 0;padding:1rem;background:#fff;border:1px solid #dee2e6;border-radius:6px">
<h2 style="margin-top:0;font-size:1rem">Apply batch decision</h2>
<p style="color:#6c757d;font-size:0.9rem">
  This will apply your decision to all <strong>{{count}}</strong> currently queued proposals
  matching this filter. A note is required and will be recorded on every proposal.
</p>
<form method="post" style="margin:0">
  <input type="hidden" name="issue_class" value="{{issue_class}}">
  <input type="hidden" name="queue_name" value="{{queue_name}}">
  <div style="margin:0.75rem 0">
    <label for="batch-note" style="font-weight:600">Reviewer note (required):</label><br>
    <textarea id="batch-note" name="reviewer_note" rows="3"
      style="width:100%;max-width:40rem;margin-top:0.25rem;padding:0.4rem;border:1px solid #ced4da;border-radius:4px"
      placeholder="Briefly describe why you are accepting or rejecting this batch&hellip;"
      required></textarea>
  </div>
  <button type="submit" formaction="/proposals/batch-accept" class="btn-accept"
    onclick="return confirm('Accept all {{count}} queued proposals in this batch?')">
    Accept all {{count}}
  </button>
  <button type="submit" formaction="/proposals/batch-reject" class="btn-reject"
    style="margin-left:0.5rem"
    onclick="return confirm('Reject all {{count}} queued proposals in this batch?')">
    Reject all {{count}}
  </button>
  <a href="{{back_href}}" style="margin-left:1rem;color:#6c757d">Cancel</a>
</form>
</section>
</body></html>"""
