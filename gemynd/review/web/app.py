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
        '<h2 style="margin-top:0;font-size:1rem;color:var(--text-secondary,#6B6054)">Provenance</h2>'
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
        '<summary style="cursor:pointer;color:var(--text-secondary,#6B6054);font-size:0.9rem;'
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
            '<p style="color:var(--accent-red-text,#8B2114)"><strong>Action required:</strong> '
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
                '<blockquote style="border-left:4px solid var(--accent-red-border,#F0C4BC);margin:0;padding:0 1rem">'
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
                '<p style="color:var(--accent-red-text,#8B2114);font-weight:bold">&#9888; Tribal consultation required: '
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

    return f'<em style="color:var(--text-tertiary,#9C9488)">({esc(ic)})</em>'


def _batch_sample_card(proposal: Any, evidence: dict) -> str:  # type: ignore[type-arg]
    """Compact evidence card for the batch review summary page (5-sample spot-check)."""
    def esc(v: object) -> str:
        return _html.escape(str(v))

    return (
        f'<div style="border:1px solid var(--border-medium,rgba(61,43,31,.18));border-radius:8px;'
        f'padding:0.75rem;margin:0.5rem 0;background:var(--white,#fff)">'
        f'<p style="margin:0 0 0.4rem;font-size:12px;color:var(--text-secondary,#6B6054)">'
        f'<code style="font-size:11px">{esc(proposal.proposal_id[:16])}&hellip;</code>'
        f' &nbsp;&middot;&nbsp; confidence {proposal.confidence:.2f}'
        f' &nbsp;&middot;&nbsp; '
        f'<a href="/proposals/{esc(proposal.proposal_id)}" class="action-link" style="font-size:12px">view detail &rarr;</a></p>'
        f'<p style="margin:0;font-size:13px">{_key_info_html(proposal.issue_class, evidence)}</p>'
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
    border_color = "var(--accent-red-border,#F0C4BC)" if is_sensitivity else "var(--border-medium,rgba(61,43,31,.18))"
    badge_cls = "badge-red" if is_sensitivity else "badge-blue"

    actionable = proposal.status in ("queued", "deferred")
    accept_btn = (
        f'<button hx-post="/proposals/{pid_esc}/accept" '
        f'hx-target="#{card_id}" hx-swap="delete" '
        f'class="btn-sm btn-accept"'
        f'{"" if actionable else " disabled"}>Accept</button>'
    )
    defer_btn = (
        f'<button hx-post="/proposals/{pid_esc}/defer" '
        f'hx-target="#{card_id}" hx-swap="delete" '
        f'class="btn-sm btn-defer" style="margin-left:4px"'
        f'{"" if actionable and proposal.status == \"queued\" else \" disabled"}>Defer</button>'
    )

    return (
        f'<div id="{card_id}" style="border:1px solid {border_color};border-radius:var(--radius-md,8px);'
        f'padding:0.85rem;margin:0.5rem 0;background:var(--white,#fff)">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        f'margin-bottom:0.4rem">'
        f'<span class="badge {badge_cls}">{ic_label}</span>'
        f'<span style="font-size:12px;color:var(--text-secondary,#6B6054)">'
        f'priority {proposal.priority_score:.2f} &nbsp;&middot;&nbsp; '
        f'confidence {proposal.confidence:.2f} &nbsp;&middot;&nbsp; '
        f'{proposal.impact_size} target{"s" if proposal.impact_size != 1 else ""}'
        f'</span>'
        f'</div>'
        f'<p style="margin:0.3rem 0 0.5rem;font-size:13px">'
        f'{_key_info_html(proposal.issue_class, evidence)}</p>'
        f'<div style="display:flex;align-items:center;gap:4px">'
        f'{accept_btn}{defer_btn}'
        f'<a href="/proposals/{pid_esc}" class="action-link" style="margin-left:8px;font-size:12px">'
        f'Full detail &rarr;</a>'
        f'</div>'
        f'</div>'
    )


def create_app(db_path: str, users_db_path: str = "data/users.db") -> Any:
    """Create and return a FastAPI app for the review UI.

    Requires `fastapi` and `uvicorn` to be installed.
    """
    from gemynd.shared.logging_config import setup_logging
    setup_logging()

    try:
        from fastapi import Depends, FastAPI, Form, Query, Request
        from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse as _RedirectResponse
        from fastapi.templating import Jinja2Templates
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for review-serve. Install with: pip install fastapi uvicorn"
        ) from e

    _shared_dir = str(Path(__file__).parent.parent.parent / "shared_templates")
    _templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    _templates.env.loader = __import__("jinja2").FileSystemLoader(
        [str(_TEMPLATES_DIR), _shared_dir]
    )

    from gemynd.auth.dependencies import (
        NeedsLoginException,
        require_archivist_or_admin,
        require_login,
    )
    from gemynd.auth.models import UserContext
    from gemynd.auth.router import create_auth_router
    from gemynd.auth.setup import is_setup_needed

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
    async def index(request: Request, _user: UserContext = Depends(require_login)):
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
                '<p style="color:var(--text-secondary,#6B6054);font-style:italic;font-size:13px">'
                'No queued proposals — queue is clear.</p>'
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
                f'<div class="alert-bar" style="margin-bottom:1rem">'
                f'<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" style="width:16px;height:16px;flex-shrink:0">'
                f'<path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>'
                f'<span><strong>Sensitivity queue:</strong> '
                f'{sensitivity_total} proposal{"s" if sensitivity_total != 1 else ""} '
                f'require urgent review. '
                f'<a href="/proposals?queue_name=sensitivity" class="action-link" style="font-weight:600">'
                f'Review now &rarr;</a></span>'
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
                '<tr><td colspan="4" style="color:var(--text-secondary,#6B6054);font-style:italic;font-size:13px">'
                'No batch-eligible proposals queued.</td></tr>'
            )

        # --- Section 3: done today ---
        today_iso = _date.today().isoformat()  # "YYYY-MM-DD" prefix match
        event_counts = store.correction_event_counts(since_iso=today_iso)
        accepted_today = event_counts.get("accept", 0)
        rejected_today = event_counts.get("reject", 0)
        deferred_today = event_counts.get("defer", 0)

        return _templates.TemplateResponse("index.html", {
            "request": request,
            "active_page": "dashboard",
            "sensitivity_alert": sensitivity_alert,
            "priority_cards": priority_cards,
            "batch_rows": batch_rows,
            "accepted_today": accepted_today,
            "rejected_today": rejected_today,
            "deferred_today": deferred_today,
        })

    @app.get("/proposals", response_class=HTMLResponse)
    async def list_proposals(
        request: Request,
        status: str | None = Query(None),
        issue_class: str | None = Query(None),
        queue_name: str | None = Query(None),
        doc_id: str | None = Query(None),
        limit: int = Query(50),
        offset: int = Query(0),
        _user: UserContext = Depends(require_archivist_or_admin),
    ):
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
                    '<section style="margin:1rem 0;padding:1rem;background:var(--accent-blue-bg,#EFF4FB);'
                    'border-radius:var(--radius-md,8px);border:1px solid var(--accent-blue-border,#B5CDE8)">'
                    '<p class="section-heading" style="margin-top:0">Queued proposals by issue class</p>'
                    '<table class="mini-table"><thead><tr>'
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

        return _templates.TemplateResponse("list.html", {
            "request": request,
            "active_page": "proposals",
            "rows": rows,
            "filter_params": filter_params,
            "next_offset": next_offset,
            "prev_offset": prev_offset,
            "limit": limit,
            "queue_summary": queue_summary,
        })

    @app.get("/proposals/{proposal_id}", response_class=HTMLResponse)
    async def proposal_detail(
        proposal_id: str,
        request: Request,
        _user: UserContext = Depends(require_archivist_or_admin),
    ):
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

        return _templates.TemplateResponse("detail.html", {
            "request": request,
            "active_page": "proposals",
            "proposal_id": proposal.proposal_id,
            "issue_class": proposal.issue_class,
            "proposal_type": proposal.proposal_type,
            "status": proposal.status,
            "confidence": f"{proposal.confidence:.2f}",
            "priority_score": f"{proposal.priority_score:.2f}",
            "impact_size": proposal.impact_size,
            "anti_pattern_id": proposal.anti_pattern_id,
            "proposed_action_panel": proposed_action_panel,
            "evidence_summary": evidence_summary,
            "provenance_html": provenance_html,
            "target_rows": target_rows,
            "revision_rows": revision_rows,
            "event_rows": event_rows,
            "patch_json": patch_formatted,
            "evidence_json": evidence_formatted,
            "is_active": proposal.status in ("queued", "deferred"),
            "accept_exceptions_btn": accept_exceptions_btn,
        })

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
        request: Request,
        issue_class: str | None = Query(None),
        queue_name: str | None = Query(None),
        _user: UserContext = Depends(require_archivist_or_admin),
    ):
        """Batch review page: stats + 5-sample evidence cards + accept/reject form."""
        proposals = store.list_proposals(
            issue_class=issue_class,
            queue_name=queue_name,
            status="queued",
            limit=10000,
        )
        filter_label = _html.escape(issue_class or queue_name or "selected filter")
        if not proposals:
            return _templates.TemplateResponse("batch.html", {
                "request": request,
                "is_empty": True,
                "filter_label": filter_label,
            })

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

        return _templates.TemplateResponse("batch.html", {
            "request": request,
            "is_empty": False,
            "filter_label": filter_label,
            "count": count,
            "conf_min": f"{conf_min:.2f}",
            "conf_max": f"{conf_max:.2f}",
            "conf_avg": f"{conf_avg:.2f}",
            "sample_cards": sample_cards,
            "issue_class": _html.escape(issue_class or ""),
            "queue_name": _html.escape(queue_name or ""),
            "back_href": back_href,
        })

    @app.post("/proposals/batch-accept", response_class=HTMLResponse)
    async def do_batch_accept(
        request: Request,
        issue_class: str = Form(""),
        queue_name: str = Form(""),
        reviewer_note: str = Form(""),
        user: UserContext = Depends(require_archivist_or_admin),
    ):
        if not reviewer_note.strip():
            return _templates.TemplateResponse("batch_result.html", {
                "request": request,
                "action_label": "Accept",
                "action_past": "",
                "error": "A reviewer note is required for batch actions.",
                "count": 0, "label": "", "skip_note": "", "reviewer_note": "", "back_href": "/proposals",
                "result_bg": "#f8d7da",
            })
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
        return _templates.TemplateResponse("batch_result.html", {
            "request": request,
            "action_label": "Accept",
            "action_past": "accepted",
            "error": "",
            "count": accepted,
            "label": label,
            "skip_note": _html.escape(skip_note),
            "reviewer_note": _html.escape(reviewer_note),
            "back_href": back_href,
            "result_bg": "#d1e7dd",
        })

    @app.post("/proposals/batch-reject", response_class=HTMLResponse)
    async def do_batch_reject(
        request: Request,
        issue_class: str = Form(""),
        queue_name: str = Form(""),
        reviewer_note: str = Form(""),
        user: UserContext = Depends(require_archivist_or_admin),
    ):
        if not reviewer_note.strip():
            return _templates.TemplateResponse("batch_result.html", {
                "request": request,
                "action_label": "Reject",
                "action_past": "",
                "error": "A reviewer note is required for batch actions.",
                "count": 0, "label": "", "skip_note": "", "reviewer_note": "", "back_href": "/proposals",
                "result_bg": "#f8d7da",
            })
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
        return _templates.TemplateResponse("batch_result.html", {
            "request": request,
            "action_label": "Reject",
            "action_past": "rejected",
            "error": "",
            "count": rejected,
            "label": label,
            "skip_note": _html.escape(skip_note),
            "reviewer_note": _html.escape(reviewer_note),
            "back_href": back_href,
            "result_bg": "#f8d7da",
        })

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
