"""Layer 4 — FastAPI Retrieval API.

Exposes the retrieval pipeline as HTTP endpoints.

Endpoints
---------
POST /query
    Accept a natural-language query, route through all four layers, and return
    a synthesised answer with provenance metadata.

POST /query/provenance
    Accept a claim_id and return the full raw provenance chain without LLM
    synthesis — useful for citation verification and the review web app.

Usage
-----
Start via the CLI:
    graphrag query-serve --port 8788
"""
from __future__ import annotations

import dataclasses
import hashlib as _hashlib
import logging
import os
import re
import sqlite3 as _sqlite3
import statistics as _statistics
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import Depends, FastAPI, Form, HTTPException, Query
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field, field_validator
except Exception:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "fastapi/pydantic not installed. Install with: pip install -e .[retrieval]"
    )

from .auth import UserContext, require_user
from graphrag_pipeline.auth.dependencies import NeedsLoginException, require_admin, require_archivist_or_admin
from graphrag_pipeline.auth.router import create_auth_router
from ..classifier import classify_query
from ..context_assembler import ProvenanceContextAssembler, _serialise_block
from ..conversation_log import ClaimInteraction, ConversationLogger, LogRecord, make_conversation_id
from graphrag_pipeline.core.graph.cypher import (
    CORPUS_STATS_QUERY,
    COUNT_QUARANTINED_CLAIMS_QUERY,
    SOFT_DELETE_DOCUMENT_QUERY,
    RESTORE_DOCUMENT_QUERY,
    LIST_DOCUMENTS_QUERY,
    STATS_DOC_OVERVIEW_QUERY,
    STATS_DOC_TYPE_QUERY,
    STATS_CLAIM_TYPE_QUERY,
    STATS_ENTITY_TYPE_QUERY,
    STATS_TEMPORAL_COVERAGE_QUERY,
    STATS_CONFIDENCE_DISTRIBUTION_QUERY,
    GAP_TEMPORAL_DENSITY_QUERY,
    GAP_ENTITY_DEPTH_QUERY,
    GAP_GEOGRAPHIC_COVERAGE_QUERY,
    ENTITY_SEARCH_QUERY,
    ENTITY_DETAIL_QUERY,
    ENTITY_NEIGHBORHOOD_QUERY,
)
from ..entity_gateway import EntityResolutionGateway
from ..executor import Neo4jQueryExecutor
from ..models import RetrievalStats, SynthesisResult
from ..query_builder import CypherQueryBuilder
from ..synthesis import SynthesisEngine


# Allowlist for entity_id URL path parameters.
# Format: lowercase letters/underscores (2-30 chars) + underscore + 8-32 hex chars
# e.g. entity_id_a3f1b2c4d5e6f7a8
_ENTITY_ID_RE = re.compile(r'^[a-z_]{2,30}_[0-9a-f]{8,32}$')

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    text: str = Field(..., description="Natural-language query")
    year_range: tuple[int, int] | None = Field(
        None, description="Optional year bounds [min, max]"
    )
    entity_hints: list[str] = Field(
        default_factory=list,
        description="Additional entity surface forms to help resolution",
    )
    mode: Literal["analytical", "conversational", "auto"] = Field(
        "auto", description="Force a specific retrieval mode or use auto-classification"
    )
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Prior Q&A turns (client-managed). Empty list = stateless behaviour.",
    )
    session_id: str | None = Field(
        None,
        max_length=128,
        pattern=r'^[a-zA-Z0-9_\-]{1,128}$',
        description="Client-generated session ID grouping multi-turn exchanges.",
    )

    @field_validator("conversation_history")
    @classmethod
    def cap_history(cls, v: list[ConversationTurn]) -> list[ConversationTurn]:
        if len(v) > 10:
            raise ValueError("conversation_history may not exceed 10 turns")
        return v


class ProvenanceRequest(BaseModel):
    claim_id: str = Field(..., description="Canonical claim_id to retrieve provenance for")


class QueryResponse(BaseModel):
    answer: str
    confidence_assessment: str
    supporting_claim_ids: list[str]
    caveats: list[str]
    min_extraction_confidence: float | None = None
    analytical_result: dict[str, Any] | None = None
    ambiguous_entities: list[str] = Field(default_factory=list)
    retrieval_stats: dict[str, int] | None = None
    synthesis_available: bool = True  # False when API call failed; provenance blocks still returned
    quarantined_claims_count: int = 0  # claims excluded from this answer due to sensitivity review
    supporting_claims_detail: list[dict[str, Any]] = Field(default_factory=list)
    # Each entry: {claim_id, confidence, epistemic_status, claim_type, doc_id, doc_title}
    user_can_annotate: bool = False  # True when the requesting user has archivist/admin role


# ---------------------------------------------------------------------------
# Chat UI (self-contained HTML page served at GET /)
# ---------------------------------------------------------------------------

_CHAT_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Turnbull GraphRAG</title>
<style>
:root {
  --bg: #f0f2f5;
  --header-bg: #1a1a2e;
  --header-fg: #e5e7eb;
  --panel-bg: #ffffff;
  --bubble-user-bg: #2563eb;
  --bubble-user-fg: #ffffff;
  --bubble-ai-bg: #e8eaed;
  --bubble-ai-fg: #1a1a1a;
  --input-bg: #ffffff;
  --border: #d1d5db;
  --warning-bg: #fef3c7;
  --warning-fg: #92400e;
  --error-bg: #fee2e2;
  --error-fg: #991b1b;
  --badge-green: #16a34a;
  --badge-amber: #d97706;
  --badge-red: #dc2626;
  --badge-fg: #ffffff;
  --table-header: #f3f4f6;
  --table-border: #e5e7eb;
  --label: #6b7280;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111827;
    --header-bg: #0f172a;
    --header-fg: #e5e7eb;
    --panel-bg: #1e293b;
    --bubble-user-bg: #1d4ed8;
    --bubble-user-fg: #ffffff;
    --bubble-ai-bg: #1e293b;
    --bubble-ai-fg: #e5e7eb;
    --input-bg: #1e293b;
    --border: #374151;
    --warning-bg: #451a03;
    --warning-fg: #fcd34d;
    --error-bg: #450a0a;
    --error-fg: #fca5a5;
    --table-header: #0f172a;
    --table-border: #374151;
    --label: #9ca3af;
  }
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 14px;
  background: var(--bg);
  color: var(--bubble-ai-fg);
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

/* ---- Header ---- */
header {
  background: var(--header-bg);
  color: var(--header-fg);
  padding: 0 1rem;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}
header h1 { font-size: 1rem; font-weight: 600; letter-spacing: 0.02em; }
#settings-toggle,
#new-conv-btn {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.25);
  color: var(--header-fg);
  border-radius: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 13px;
  transition: background 0.15s;
}
#settings-toggle:hover,
#new-conv-btn:hover { background: rgba(255,255,255,0.1); }

/* ---- Settings panel ---- */
#settings-panel {
  display: none;
  background: var(--panel-bg);
  border-bottom: 1px solid var(--border);
  padding: 0.75rem 1rem;
  flex-shrink: 0;
}
#settings-panel.open { display: block; }
.settings-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem;
  align-items: flex-end;
  max-width: 760px;
  margin: 0 auto;
}
.settings-group { display: flex; flex-direction: column; gap: 4px; }
.settings-group label { font-size: 12px; color: var(--label); font-weight: 500; }
.settings-group input[type="text"],
.settings-group input[type="number"] {
  background: var(--input-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 13px;
  color: var(--bubble-ai-fg);
  width: 100px;
}
.settings-group input[type="text"] { width: 200px; }
.radio-row { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
.radio-row label { font-size: 13px; color: var(--bubble-ai-fg); display: flex; align-items: center; gap: 4px; cursor: pointer; }

/* ---- Chat area ---- */
#chat-area {
  flex: 1;
  overflow-y: auto;
  padding: 1rem 1rem 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.bubble-wrap {
  display: flex;
  max-width: 760px;
  width: 100%;
  margin: 0 auto;
}
.bubble-wrap.user { justify-content: flex-end; }
.bubble {
  padding: 0.65rem 0.9rem;
  border-radius: 1rem;
  max-width: 82%;
  line-height: 1.55;
  word-break: break-word;
}
.bubble.user {
  background: var(--bubble-user-bg);
  color: var(--bubble-user-fg);
  border-bottom-right-radius: 4px;
  white-space: pre-wrap;
}
.bubble.assistant {
  background: var(--bubble-ai-bg);
  color: var(--bubble-ai-fg);
  border-bottom-left-radius: 4px;
}
.bubble.loading { opacity: 0.6; font-style: italic; }
.bubble.error { background: var(--error-bg); color: var(--error-fg); }

/* ---- Response internals ---- */
.answer { margin-bottom: 0.5rem; white-space: pre-wrap; }
.meta-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin: 0.35rem 0; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  color: var(--badge-fg);
}
.badge-green { background: var(--badge-green); }
.badge-amber { background: var(--badge-amber); }
.badge-red { background: var(--badge-red); }
.mode-tag {
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(0,0,0,0.12);
  color: inherit;
  opacity: 0.8;
}
.warning {
  background: var(--warning-bg);
  color: var(--warning-fg);
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 13px;
  margin: 0.35rem 0;
}
.caveats { margin: 0.35rem 0; font-size: 13px; }
.caveats ul { margin: 0.25rem 0 0 1.25rem; }
.caveats li { margin-bottom: 2px; }
details { margin: 0.35rem 0; font-size: 13px; }
summary { cursor: pointer; user-select: none; color: var(--label); font-size: 12px; }
summary:hover { opacity: 0.8; }
.table-wrap { overflow-x: auto; margin-top: 0.4rem; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th { background: var(--table-header); padding: 4px 8px; text-align: left; border: 1px solid var(--table-border); font-weight: 600; }
td { padding: 4px 8px; border: 1px solid var(--table-border); }
.claim-ids { font-family: monospace; font-size: 12px; margin-top: 0.25rem; opacity: 0.85; }
.coverage { font-size: 11px; color: var(--label); margin-top: 0.4rem; border-top: 1px solid var(--border); padding-top: 0.3rem; }
.sources-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 0.4rem; }
.sources-table th, .sources-table td { padding: 3px 8px; border: 1px solid var(--table-border); text-align: left; }
.sources-table th { background: var(--table-header); font-weight: 600; }
.src-id { font-family: monospace; font-size: 11px; }
.src-label { font-size: 11px; background: var(--table-header); border-radius: 3px; padding: 1px 5px; white-space: nowrap; }

/* ---- Input bar ---- */
#input-bar {
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  border-top: 1px solid var(--border);
  background: var(--panel-bg);
  flex-shrink: 0;
}
#input-bar > div { max-width: 760px; width: 100%; margin: 0 auto; display: flex; gap: 0.5rem; align-items: flex-end; }
#user-input {
  flex: 1;
  resize: none;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 12px;
  font-size: 14px;
  font-family: inherit;
  line-height: 1.4;
  background: var(--input-bg);
  color: var(--bubble-ai-fg);
  max-height: 140px;
  overflow-y: auto;
}
#user-input:focus { outline: none; border-color: var(--bubble-user-bg); }
#send-btn {
  flex-shrink: 0;
  background: var(--bubble-user-bg);
  color: #fff;
  border: none;
  border-radius: 10px;
  padding: 8px 16px;
  font-size: 14px;
  cursor: pointer;
  height: 38px;
  transition: opacity 0.15s;
}
#send-btn:disabled { opacity: 0.45; cursor: not-allowed; }
#send-btn:not(:disabled):hover { opacity: 0.85; }
.note-btn { font-size:11px; padding:2px 7px; border-radius:4px; background:transparent;
            border:1px solid var(--border); color:var(--label); cursor:pointer; }
.note-btn:hover { background:var(--table-header); }
.notes-panel { margin-top:6px; padding:8px 10px; background:var(--warning-bg);
               border-radius:6px; font-size:12px; }
.notes-panel form { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
.notes-panel textarea { flex:1; min-width:160px; font-size:12px; font-family:inherit;
  border:1px solid var(--border); border-radius:4px; padding:4px 6px;
  background:var(--input-bg); color:var(--bubble-ai-fg); resize:vertical; min-height:48px; }
.notes-panel button[type=submit] { align-self:flex-end; padding:4px 12px; font-size:12px;
  border:none; border-radius:4px; background:var(--bubble-user-bg); color:#fff; cursor:pointer; }
</style>
<script src="https://unpkg.com/htmx.org@1.9.12" crossorigin="anonymous"></script>
</head>
<body>

<header>
  <h1>Turnbull GraphRAG</h1>
  <div style="display:flex;gap:0.5rem;align-items:center;">
    <a href="/stats" style="color:var(--header-fg);font-size:13px;text-decoration:none;border:1px solid rgba(255,255,255,0.25);border-radius:6px;padding:4px 10px;white-space:nowrap;">&#128202; Collection Stats</a>
    <a href="/gaps" style="color:var(--header-fg);font-size:13px;text-decoration:none;border:1px solid rgba(255,255,255,0.25);border-radius:6px;padding:4px 10px;white-space:nowrap;">&#128270; Gap Analysis</a>
    <a href="/map" style="color:var(--header-fg);font-size:13px;text-decoration:none;border:1px solid rgba(255,255,255,0.25);border-radius:6px;padding:4px 10px;white-space:nowrap;">&#128101; Relationship Map</a>
    <a href="/history" style="color:var(--header-fg);font-size:13px;text-decoration:none;border:1px solid rgba(255,255,255,0.25);border-radius:6px;padding:4px 10px;white-space:nowrap;">&#128203; History</a>
    <button id="new-conv-btn">&#10011; New Conversation</button>
    <button id="settings-toggle">&#9881; Settings</button>
  </div>
</header>

<div id="settings-panel">
  <div class="settings-row">
    <div class="settings-group">
      <label>Mode</label>
      <div class="radio-row">
        <label><input type="radio" name="mode" value="auto" checked> Auto</label>
        <label><input type="radio" name="mode" value="conversational"> Conversational</label>
        <label><input type="radio" name="mode" value="analytical"> Analytical</label>
      </div>
    </div>
    <div class="settings-group">
      <label>Year min</label>
      <input type="number" id="year-min" placeholder="e.g. 1940" min="1900" max="2100">
    </div>
    <div class="settings-group">
      <label>Year max</label>
      <input type="number" id="year-max" placeholder="e.g. 1960" min="1900" max="2100">
    </div>
    <div class="settings-group">
      <label>Entity hints (comma-separated)</label>
      <input type="text" id="entity-hints" placeholder="mallard, pintail">
    </div>
  </div>
</div>

<div id="chat-area"></div>

<div id="input-bar">
  <div>
    <textarea id="user-input" rows="1" placeholder="Ask about the Turnbull refuge archive\u2026"></textarea>
    <button id="send-btn">Send &#9654;</button>
  </div>
</div>

<script>
var conversationHistory = [];
var HISTORY_LIMIT = 10;

function generateSessionId() {
  return 'sess_' + Date.now().toString(36) + Math.random().toString(36).slice(2);
}
var currentSessionId = generateSessionId();

function esc(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function cellValue(v) {
  if (v === null || v === undefined) return '\u2014';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function confidenceBadge(conf) {
  if (conf === null || conf === undefined) return '';
  const pct = Math.round(conf * 100);
  const cls = conf >= 0.8 ? 'green' : conf >= 0.6 ? 'amber' : 'red';
  return '<span class="badge badge-' + cls + '">' + pct + '% confidence</span>';
}

function formatTable(ar) {
  if (!ar || !ar.rows || ar.rows.length === 0) return '';
  const cols = ar.columns;
  const headers = cols.map(function(c) { return '<th>' + esc(c) + '</th>'; }).join('');
  const rows = ar.rows.map(function(r) {
    var cells = cols.map(function(c) { return '<td>' + esc(cellValue(r[c])) + '</td>'; }).join('');
    return '<tr>' + cells + '</tr>';
  }).join('');
  return '<details class="analytical-details"><summary>Analytical result \u2014 ' + esc(ar.query_name) + ' (' + ar.rows.length + ' rows)</summary>'
    + '<div class="table-wrap"><table><thead><tr>' + headers + '</tr></thead><tbody>' + rows + '</tbody></table></div></details>';
}

function formatResponse(data) {
  var html = '<p class="answer">' + esc(data.answer) + '</p>';

  if (data.ambiguous_entities && data.ambiguous_entities.length > 0) {
    html += '<div class="warning">\u26a0 Ambiguous: ' + data.ambiguous_entities.map(esc).join(', ') + '</div>';
  }

  var badge = confidenceBadge(data.min_extraction_confidence);
  if (badge || data.analytical_result) {
    html += '<div class="meta-row">' + badge + '</div>';
  }

  if (data.analytical_result) {
    html += formatTable(data.analytical_result);
  }

  if (data.caveats && data.caveats.length > 0) {
    var items = data.caveats.map(function(c) { return '<li>' + esc(c) + '</li>'; }).join('');
    html += '<div class="caveats"><strong>Caveats:</strong><ul>' + items + '</ul></div>';
  }

  if (data.supporting_claims_detail && data.supporting_claims_detail.length > 0) {
    var rows = data.supporting_claims_detail.map(function(c) {
      var badge = (c.confidence != null) ? confidenceBadge(c.confidence) : '';
      var epis = c.epistemic_status ? '<span class="src-label">' + esc(c.epistemic_status) + '</span>' : '';
      var ct = c.claim_type ? '<span class="src-label">' + esc(c.claim_type) + '</span>' : '';
      return '<tr><td class="src-id">' + esc(c.claim_id) + '</td><td>' + badge + '</td><td>' + epis + '</td><td>' + ct + '</td></tr>';
    }).join('');
    html += '<details><summary>Sources (' + data.supporting_claims_detail.length + ' claims)</summary>'
      + '<table class="sources-table"><thead><tr>'
      + '<th>Claim ID</th><th>Confidence</th><th>Epistemic Status</th><th>Type</th>'
      + '</tr></thead><tbody>' + rows + '</tbody></table></details>';
  } else if (data.supporting_claim_ids && data.supporting_claim_ids.length > 0) {
    html += '<details><summary>Sources (' + data.supporting_claim_ids.length + ' claims)</summary>'
      + '<p class="claim-ids">' + data.supporting_claim_ids.map(esc).join(', ') + '</p></details>';
  }

  // Archivist note buttons — one per unique cited document (archivist/admin role only).
  if (data.user_can_annotate && data.supporting_claims_detail && data.supporting_claims_detail.length > 0) {
    var seenDocs = {};
    data.supporting_claims_detail.forEach(function(c) {
      if (c.doc_id && !seenDocs[c.doc_id]) seenDocs[c.doc_id] = c.doc_title || c.doc_id;
    });
    var docIds = Object.keys(seenDocs);
    if (docIds.length > 0) {
      html += '<div style="margin-top:8px">';
      docIds.forEach(function(docId) {
        var eid = esc(docId);
        var etitle = esc(seenDocs[docId]);
        html += '<div id="notes-panel-' + eid + '">'
          + '<button class="note-btn"'
          + ' hx-get="/document/' + eid + '/notes-panel"'
          + ' hx-target="#notes-panel-' + eid + '"'
          + ' hx-swap="outerHTML">'
          + 'Add/view note \u2014 ' + etitle
          + '</button></div>';
      });
      html += '</div>';
    }
  }

  if (data.retrieval_stats) {
    var s = data.retrieval_stats;
    var coverageText = s.claims_in_context + ' claims ('
      + s.paragraphs_in_context + ' paragraphs, '
      + s.documents_in_context + ' docs) of \u2248'
      + s.corpus_total_paragraphs + ' paragraphs / '
      + s.corpus_total_documents + ' docs in corpus';
    if (s.ocr_dropped > 0) {
      coverageText += ' \u2014 ' + s.ocr_dropped + ' OCR-corrupted claims excluded';
    }
    html += '<div class="coverage">' + coverageText + '</div>';
  }

  return html;
}

function appendBubble(role, htmlContent) {
  var chat = document.getElementById('chat-area');
  var wrap = document.createElement('div');
  wrap.className = 'bubble-wrap ' + role;
  var bubble = document.createElement('div');
  bubble.className = 'bubble ' + role;
  bubble.innerHTML = htmlContent;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function autoResize(ta) {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
}

async function sendMessage() {
  var ta = document.getElementById('user-input');
  var text = ta.value.trim();
  if (!text) return;

  var mode = document.querySelector('input[name="mode"]:checked').value;
  var yearMinVal = document.getElementById('year-min').value;
  var yearMaxVal = document.getElementById('year-max').value;
  var hintsRaw = document.getElementById('entity-hints').value;
  var entityHints = hintsRaw ? hintsRaw.split(',').map(function(s){ return s.trim(); }).filter(Boolean) : [];
  var year_range = (yearMinVal && yearMaxVal) ? [parseInt(yearMinVal, 10), parseInt(yearMaxVal, 10)] : null;

  appendBubble('user', esc(text));
  ta.value = '';
  ta.style.height = 'auto';
  ta.disabled = true;
  document.getElementById('send-btn').disabled = true;

  var loadingWrap = appendBubble('assistant loading', '<em>Thinking\u2026</em>');
  loadingWrap.querySelector('.bubble').className = 'bubble assistant loading';

  try {
    var resp = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text, mode: mode, year_range: year_range, entity_hints: entityHints, conversation_history: conversationHistory, session_id: currentSessionId}),
    });
    var bubble = loadingWrap.querySelector('.bubble');
    if (!resp.ok) {
      var detail = resp.statusText;
      try { detail = (await resp.json()).detail || detail; } catch(e) {}
      bubble.className = 'bubble assistant error';
      bubble.innerHTML = esc('Error ' + resp.status + ': ' + detail);
    } else {
      var data = await resp.json();
      bubble.className = 'bubble assistant';
      bubble.innerHTML = formatResponse(data);
      conversationHistory.push({role: 'user', content: text});
      conversationHistory.push({role: 'assistant', content: data.answer});
      if (conversationHistory.length > HISTORY_LIMIT) {
        conversationHistory = conversationHistory.slice(conversationHistory.length - HISTORY_LIMIT);
      }
    }
  } catch(err) {
    var bubble2 = loadingWrap.querySelector('.bubble');
    bubble2.className = 'bubble assistant error';
    bubble2.innerHTML = esc('Network error: ' + err.message);
  }

  document.getElementById('chat-area').scrollTop = document.getElementById('chat-area').scrollHeight;
  ta.disabled = false;
  document.getElementById('send-btn').disabled = false;
  ta.focus();
}

document.addEventListener('DOMContentLoaded', function() {
  var ta = document.getElementById('user-input');
  ta.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  ta.addEventListener('input', function() { autoResize(ta); });

  document.getElementById('settings-toggle').addEventListener('click', function() {
    document.getElementById('settings-panel').classList.toggle('open');
  });

  document.getElementById('new-conv-btn').addEventListener('click', function() {
    conversationHistory = [];
    currentSessionId = generateSessionId();
    console.log('[GraphRAG] New session:', currentSessionId);
    document.getElementById('chat-area').innerHTML = '';
    appendBubble('assistant',
      '<p>New conversation started. Ask me anything about the Turnbull National Wildlife Refuge archive.</p>'
    );
  });

  document.getElementById('send-btn').addEventListener('click', sendMessage);

  appendBubble('assistant',
    '<p>Hello! Ask me anything about the Turnbull National Wildlife Refuge archive. '
    + 'Use <strong>\u2699 Settings</strong> to adjust mode, year range, or entity hints. '
    + '<em>Shift+Enter</em> for a new line.</p>'
  );

  // Auto-submit if ?q= URL param is present (e.g., "Re-run" from the history page).
  var _urlQ = new URLSearchParams(window.location.search).get('q');
  if (_urlQ) {
    ta.value = _urlQ;
    autoResize(ta);
    history.replaceState(null, '', '/');
    sendMessage();
  }
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Collection statistics HTML renderer
# ---------------------------------------------------------------------------

# Expected claim-type share for a wildlife refuge annual-report corpus.
# Used by gap detection to identify underrepresented topical areas.
_EXPECTED_CLAIM_SHARES: dict[str, float] = {
    "population_estimate": 0.22,
    "species_presence":    0.18,
    "management_action":   0.14,
    "habitat_condition":   0.10,
    "breeding_activity":   0.07,
    "migration_timing":    0.05,
    "weather_observation": 0.04,
}

_STATS_CSS = """
<style>
.stat-cards{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}
.stat-card{background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:1rem 1.5rem;min-width:150px;flex:1}
.stat-card .value{font-size:2rem;font-weight:700;color:#0d6efd}
.stat-card .label{font-size:0.8rem;color:#6c757d;text-transform:uppercase;letter-spacing:.05em}
.stats-section{margin:2rem 0}
.stats-section h2{font-size:1.1rem;border-bottom:2px solid #dee2e6;padding-bottom:0.3rem;margin-bottom:0.75rem}
table.stats{width:100%;border-collapse:collapse;font-size:0.9rem}
table.stats th{background:#f8f9fa;text-align:left;padding:6px 10px;border:1px solid #dee2e6}
table.stats td{padding:5px 10px;border:1px solid #dee2e6}
.tier-high{color:#198754;font-weight:600}
.tier-medium{color:#fd7e14;font-weight:600}
.tier-low{color:#dc3545;font-weight:600}
.gap-row td{color:#adb5bd;font-style:italic}
a.nav-back{display:inline-block;margin-bottom:1rem;color:#0d6efd;text-decoration:none;font-size:0.9rem}
</style>"""


def _conf_tier_class(avg: float | None) -> str:
    if avg is None:
        return ""
    if avg >= 0.85:
        return "tier-high"
    if avg >= 0.70:
        return "tier-medium"
    return "tier-low"


_DOC_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,80}$')
# UUID v4 as produced by str(uuid.uuid4())
_SAVED_ID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def _render_notes_panel(doc_id: str, current: "dict | None", history: "list[dict]") -> str:
    """Render the HTMX notes panel partial for *doc_id*."""
    import html as _html

    eid = _html.escape(doc_id)
    prefill = _html.escape(current["note_text"]) if current else ""

    current_html = ""
    if current:
        current_html = (
            f"<p style='margin:0 0 4px'><strong>Current note</strong> — "
            f"<em>{_html.escape(current.get('created_by', ''))}, "
            f"{_html.escape((current.get('created_at') or '')[:16])} UTC</em>:</p>"
            f"<p style='margin:0 0 8px;white-space:pre-wrap'>{_html.escape(current['note_text'])}</p>"
        )

    older = [h for h in history if not h.get("is_current")]
    history_html = ""
    if older:
        items = "".join(
            f"<p style='margin:2px 0'>"
            f"<em>{_html.escape(h.get('created_by', ''))}, "
            f"{_html.escape((h.get('created_at') or '')[:16])}:</em> "
            f"{_html.escape(h.get('note_text', ''))}</p>"
            for h in older
        )
        history_html = (
            f"<details style='margin-top:4px;font-size:11px;color:#6c757d'>"
            f"<summary>History ({len(older)} previous)</summary>{items}</details>"
        )

    return (
        f"<div id='notes-panel-{eid}' class='notes-panel'>"
        + current_html
        + history_html
        + f"<form hx-post='/document/{eid}/notes'"
        f"      hx-target='#notes-panel-{eid}' hx-swap='outerHTML'"
        f"      hx-encoding='application/x-www-form-urlencoded'>"
        f"  <textarea name='note_text' rows='3' maxlength='2000'"
        f"    placeholder='Add an archivist note for this document...'>{prefill}</textarea>"
        f"  <button type='submit'>Save note</button>"
        f"</form></div>"
    )


def _render_notes_panel_unavailable(doc_id: str) -> str:
    import html as _html
    eid = _html.escape(doc_id)
    return (
        f"<div id='notes-panel-{eid}' class='notes-panel' style='color:#6c757d'>"
        "Annotation store not configured. Start the server with <code>--annotation-db</code>."
        "</div>"
    )


def _render_notes_panel_error(doc_id: str, msg: str) -> str:
    import html as _html
    eid = _html.escape(doc_id)
    return (
        f"<div id='notes-panel-{eid}' class='notes-panel' style='color:#dc3545'>"
        f"{_html.escape(msg)}</div>"
    )


def _render_history_html(
    rows: "list[dict]",
    total: int,
    saved: "list[dict]",
    q: str,
    bucket: str,
    limit: int,
    offset: int,
) -> str:
    """Render the query history & saved searches HTML page."""
    import html as _html
    import urllib.parse as _urlparse

    def esc(v: object) -> str:
        return _html.escape(str(v) if v is not None else "")

    # -- Pagination maths --------------------------------------------------
    prev_offset = max(0, offset - limit)
    next_offset = offset + limit
    has_prev = offset > 0
    has_next = next_offset < total

    base_qs = ""
    if q:
        base_qs += f"&q={_urlparse.quote(q)}"
    if bucket:
        base_qs += f"&bucket={_urlparse.quote(bucket)}"

    # -- Query history rows ------------------------------------------------
    rows_html = ""
    for row in rows:
        qtext = row.get("query_text") or ""
        qtrun = qtext[:120] + ("\u2026" if len(qtext) > 120 else "")
        bkt = row.get("bucket") or ""
        yr_min = row.get("year_min")
        yr_max = row.get("year_max")
        if yr_min and yr_max:
            yr_str = f"{yr_min}\u2013{yr_max}"
        elif yr_min:
            yr_str = str(yr_min)
        elif yr_max:
            yr_str = str(yr_max)
        else:
            yr_str = "\u2014"
        ts = (row.get("created_at") or "")[:16]
        conv_id = row.get("conversation_id") or ""
        rerun_url = esc("/?q=" + _urlparse.quote(qtext))
        yr_min_str = esc(yr_min) if yr_min is not None else ""
        yr_max_str = esc(yr_max) if yr_max is not None else ""
        save_form = (
            "<details class=\"save-details\">"
            "<summary>&#11088; Save</summary>"
            "<form class=\"save-form\" method=\"POST\" action=\"/history/saved\">"
            f"<input type=\"hidden\" name=\"query_text\" value=\"{esc(qtext)}\">"
            f"<input type=\"hidden\" name=\"bucket\" value=\"{esc(bkt)}\">"
            f"<input type=\"hidden\" name=\"year_min\" value=\"{yr_min_str}\">"
            f"<input type=\"hidden\" name=\"year_max\" value=\"{yr_max_str}\">"
            f"<input type=\"hidden\" name=\"conversation_id\" value=\"{esc(conv_id)}\">"
            "<input type=\"text\" name=\"label\" placeholder=\"Label\u2026\" "
            "maxlength=\"200\" style=\"width:140px\">"
            "<button type=\"submit\">Save</button>"
            "</form></details>"
        )
        rows_html += (
            "<tr>"
            f"<td style=\"white-space:nowrap;color:#6c757d;font-size:11px\">{esc(ts)}</td>"
            f"<td title=\"{esc(qtext)}\">{esc(qtrun)}</td>"
            f"<td><span class=\"src-label\">{esc(bkt)}</span></td>"
            f"<td style=\"text-align:center\">{esc(yr_str)}</td>"
            f"<td style=\"white-space:nowrap\">"
            f"<a href=\"{rerun_url}\" class=\"action-link\">&#8617; Re-run</a>"
            f"&nbsp;{save_form}"
            "</td></tr>"
        )
    if not rows_html:
        rows_html = (
            "<tr><td colspan=\"5\" style=\"text-align:center;color:#6c757d;"
            "padding:1rem\">No queries found.</td></tr>"
        )

    # -- Pagination controls -----------------------------------------------
    pagination = ""
    if has_prev or has_next:
        pagination = "<div class=\"pagination\">"
        if has_prev:
            pagination += (
                f"<a href=\"/history?offset={prev_offset}&limit={limit}{base_qs}\">"
                "&laquo; Previous</a> "
            )
        shown_end = min(offset + limit, total)
        pagination += f"<span>{offset + 1}\u2013{shown_end} of {total}</span>"
        if has_next:
            pagination += (
                f" <a href=\"/history?offset={next_offset}&limit={limit}{base_qs}\">"
                "Next &raquo;</a>"
            )
        pagination += "</div>"

    # -- Saved search rows -------------------------------------------------
    saved_rows_html = ""
    for s in saved:
        slabel = esc(s.get("label") or "") or "<em style=\"color:#adb5bd\">\u2014</em>"
        sqtext = s.get("query_text") or ""
        sqtrun = sqtext[:100] + ("\u2026" if len(sqtext) > 100 else "")
        sbkt = s.get("bucket") or ""
        syr_min = s.get("year_min")
        syr_max = s.get("year_max")
        if syr_min and syr_max:
            syr_str = f"{syr_min}\u2013{syr_max}"
        elif syr_min:
            syr_str = str(syr_min)
        elif syr_max:
            syr_str = str(syr_max)
        else:
            syr_str = "\u2014"
        screated_by = s.get("created_by") or ""
        sts = (s.get("created_at") or "")[:16]
        saved_id = s.get("saved_id") or ""
        srerun_url = esc("/?q=" + _urlparse.quote(sqtext))
        saved_rows_html += (
            "<tr>"
            f"<td>{slabel}</td>"
            f"<td title=\"{esc(sqtext)}\">{esc(sqtrun)}</td>"
            f"<td style=\"text-align:center\"><span class=\"src-label\">{esc(sbkt)}</span></td>"
            f"<td style=\"text-align:center\">{esc(syr_str)}</td>"
            f"<td style=\"white-space:nowrap;font-size:11px;color:#6c757d\">"
            f"{esc(screated_by)}<br>{esc(sts)}</td>"
            f"<td style=\"white-space:nowrap\">"
            f"<a href=\"{srerun_url}\" class=\"action-link\">&#8617; Re-run</a> "
            f"<form method=\"POST\" action=\"/history/saved/{esc(saved_id)}/delete\" "
            f"style=\"display:inline\">"
            f"<button type=\"submit\" class=\"del-btn\" "
            f"onclick=\"return confirm('Delete this saved search?')\">&#128465; Delete</button>"
            f"</form>"
            "</td></tr>"
        )
    if not saved_rows_html:
        saved_rows_html = (
            "<tr><td colspan=\"6\" style=\"text-align:center;color:#6c757d;"
            "padding:1rem\">No saved searches yet.</td></tr>"
        )

    # -- Bucket selector options -------------------------------------------
    def _opt(val: str, label: str) -> str:
        sel = ' selected' if bucket == val else ''
        return f"<option value=\"{val}\"{sel}>{label}</option>"

    bucket_opts = (
        _opt("", "All types")
        + _opt("conversational", "Conversational")
        + _opt("analytical", "Analytical")
        + _opt("hybrid", "Hybrid")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Query History \u2014 Turnbull GraphRAG</title>
{_STATS_CSS}
<style>
.action-link{{color:#0d6efd;text-decoration:none;font-size:12px}}
.action-link:hover{{text-decoration:underline}}
.del-btn{{font-size:12px;padding:2px 6px;border:1px solid #dc3545;border-radius:4px;
         background:transparent;color:#dc3545;cursor:pointer}}
.del-btn:hover{{background:#dc3545;color:#fff}}
.save-details{{display:inline-block;font-size:12px;vertical-align:middle}}
.save-details summary{{color:#fd7e14;cursor:pointer;font-size:12px;display:inline;list-style:none}}
.save-details summary:hover{{text-decoration:underline}}
.save-form{{display:flex;gap:4px;align-items:center;margin-top:4px;flex-wrap:wrap}}
.save-form input[type=text]{{font-size:12px;padding:2px 6px;border:1px solid #ced4da;border-radius:4px}}
.save-form button{{font-size:12px;padding:2px 8px;border:none;border-radius:4px;
                   background:#fd7e14;color:#fff;cursor:pointer}}
.pagination{{margin-top:.75rem;font-size:13px}}
.pagination a{{color:#0d6efd;text-decoration:none;margin:0 4px}}
.filter-bar{{display:flex;gap:.75rem;align-items:flex-end;flex-wrap:wrap;margin-bottom:1rem}}
.filter-bar label{{font-size:12px;color:#6c757d;display:block;margin-bottom:2px}}
.filter-bar input[type=text]{{font-size:13px;padding:4px 8px;border:1px solid #ced4da;border-radius:4px}}
.filter-bar select{{font-size:13px;padding:4px 8px;border:1px solid #ced4da;border-radius:4px}}
.filter-bar button{{font-size:13px;padding:4px 10px;border:none;border-radius:4px;
                    background:#0d6efd;color:#fff;cursor:pointer}}
</style>
</head>
<body style="background:#f8f9fa;padding:2rem">
<div style="max-width:960px;margin:0 auto">
<a class="nav-back" href="/">&#8592; Back to Chat</a>
<h1 style="font-size:1.4rem;margin-bottom:1.5rem">Query History &amp; Saved Searches</h1>

<div class="stats-section">
  <h2>Saved Searches</h2>
  <div class="table-wrap">
  <table class="stats">
    <thead><tr>
      <th>Label</th><th>Query</th><th>Type</th><th>Years</th>
      <th>Saved by / When</th><th>Actions</th>
    </tr></thead>
    <tbody>{saved_rows_html}</tbody>
  </table>
  </div>
</div>

<div class="stats-section">
  <h2>Recent Queries</h2>
  <form class="filter-bar" method="GET" action="/history">
    <div>
      <label>Search query text</label>
      <input type="text" name="q" value="{esc(q)}"
             placeholder="Filter by keyword\u2026" style="width:220px">
    </div>
    <div>
      <label>Type</label>
      <select name="bucket">{bucket_opts}</select>
    </div>
    <button type="submit">Filter</button>
    <a href="/history"
       style="font-size:13px;color:#6c757d;text-decoration:none;align-self:flex-end;padding:4px 0">
      Clear
    </a>
  </form>
  <div class="table-wrap">
  <table class="stats">
    <thead><tr>
      <th>When</th><th>Query</th><th>Type</th><th>Years</th><th>Actions</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
  {pagination}
</div>
</div>
</body>
</html>"""


def _render_stats_html(
    overview: dict,
    doc_types: list[dict],
    claim_types: list[dict],
    entity_types: list[dict],
    temporal: list[dict],
    confidence: dict,
) -> str:
    import html as _html

    def esc(v: object) -> str:
        return _html.escape(str(v))

    total_docs = overview.get("total_docs") or 0
    earliest = overview.get("earliest_year") or "—"
    latest = overview.get("latest_year") or "—"
    total_pages = overview.get("total_pages") or 0
    donor_restricted = overview.get("donor_restricted_count") or 0

    total_claims = confidence.get("total_claims") or 0
    avg_conf = confidence.get("avg_confidence")
    high = confidence.get("high_count") or 0
    medium = confidence.get("medium_count") or 0
    low = confidence.get("low_count") or 0
    uncertain_epistemic = confidence.get("uncertain_epistemic_count") or 0
    total_entities = sum(r.get("count", 0) for r in entity_types)

    date_span = f"{earliest} – {latest}" if earliest != "—" else "—"

    # Summary cards
    cards = (
        f'<div class="stat-cards">'
        f'<div class="stat-card"><div class="value">{esc(total_docs)}</div>'
        f'<div class="label">Documents</div></div>'
        f'<div class="stat-card"><div class="value">{esc(total_claims)}</div>'
        f'<div class="label">Active Claims</div></div>'
        f'<div class="stat-card"><div class="value">{esc(total_entities)}</div>'
        f'<div class="label">Entities</div></div>'
        f'<div class="stat-card"><div class="value">{esc(date_span)}</div>'
        f'<div class="label">Date Span</div></div>'
        f'<div class="stat-card"><div class="value">{esc(total_pages)}</div>'
        f'<div class="label">Total Pages</div></div>'
        f'</div>'
    )

    # Document type table
    dt_rows = "".join(
        f"<tr><td>{esc(r.get('doc_type','(unknown)'))}</td><td>{esc(r.get('count',0))}</td></tr>"
        for r in doc_types
    )
    doc_type_section = (
        '<div class="stats-section"><h2>Document Types</h2>'
        '<table class="stats"><thead><tr><th>Type</th><th>Count</th></tr></thead>'
        f"<tbody>{dt_rows}</tbody></table></div>"
    )

    # Claim type table with confidence
    ct_rows = ""
    for r in claim_types:
        avg = r.get("avg_confidence")
        avg_str = f"{avg:.2f}" if avg is not None else "—"
        tier_cls = _conf_tier_class(avg)
        ct_rows += (
            f"<tr><td>{esc(r.get('claim_type','(unknown)'))}</td>"
            f"<td>{esc(r.get('count',0))}</td>"
            f'<td class="{tier_cls}">{esc(avg_str)}</td></tr>'
        )
    claim_type_section = (
        '<div class="stats-section"><h2>Claim Type Distribution</h2>'
        '<table class="stats"><thead><tr><th>Claim Type</th><th>Count</th>'
        '<th>Avg Confidence</th></tr></thead>'
        f"<tbody>{ct_rows}</tbody></table></div>"
    )

    # Temporal coverage — fill in gap years
    temporal_section = ""
    if temporal:
        year_map = {int(r["year"]): int(r["doc_count"]) for r in temporal if r.get("year") is not None}
        min_year = min(year_map)
        max_year = max(year_map)
        span = max_year - min_year
        # Group into decades if span > 30 years
        if span > 30:
            decade_map: dict[int, int] = {}
            for y, c in year_map.items():
                d = (y // 10) * 10
                decade_map[d] = decade_map.get(d, 0) + c
            temp_rows = ""
            for d in range((min_year // 10) * 10, max_year + 10, 10):
                cnt = decade_map.get(d, 0)
                gap_cls = ' class="gap-row"' if cnt == 0 else ""
                temp_rows += f"<tr{gap_cls}><td>{d}s</td><td>{cnt if cnt else '(none)'}</td></tr>"
        else:
            temp_rows = ""
            for y in range(min_year, max_year + 1):
                cnt = year_map.get(y, 0)
                gap_cls = ' class="gap-row"' if cnt == 0 else ""
                temp_rows += f"<tr{gap_cls}><td>{y}</td><td>{cnt if cnt else '(none)'}</td></tr>"
        temporal_section = (
            '<div class="stats-section"><h2>Temporal Coverage</h2>'
            '<table class="stats"><thead><tr><th>Year/Decade</th>'
            '<th>Documents</th></tr></thead>'
            f"<tbody>{temp_rows}</tbody></table>"
            "<p style='font-size:0.8rem;color:#6c757d'>Greyed rows = coverage gaps.</p></div>"
        )

    # Entity type table
    et_rows = "".join(
        f"<tr><td>{esc(r.get('entity_type','(unknown)'))}</td><td>{esc(r.get('count',0))}</td></tr>"
        for r in entity_types
    )
    entity_section = (
        '<div class="stats-section"><h2>Entity Categories</h2>'
        '<table class="stats"><thead><tr><th>Entity Type</th><th>Count</th></tr></thead>'
        f"<tbody>{et_rows}</tbody></table></div>"
    )

    # Confidence profile
    safe_total = max(total_claims, 1)
    conf_avg_str = f"{avg_conf:.2f}" if avg_conf is not None else "—"
    conf_rows = (
        f'<tr><td class="tier-high">HIGH (&ge;0.85)</td>'
        f"<td>{high}</td><td>{high*100//safe_total}%</td></tr>"
        f'<tr><td class="tier-medium">MEDIUM (0.70–0.84)</td>'
        f"<td>{medium}</td><td>{medium*100//safe_total}%</td></tr>"
        f'<tr><td class="tier-low">LOW (&lt;0.70)</td>'
        f"<td>{low}</td><td>{low*100//safe_total}%</td></tr>"
    )
    confidence_section = (
        '<div class="stats-section"><h2>Confidence Profile</h2>'
        f"<p><strong>Average extraction confidence:</strong> {esc(conf_avg_str)} "
        f"&nbsp;|&nbsp; <strong>Epistemic 'uncertain' flags:</strong> {esc(uncertain_epistemic)}</p>"
        '<table class="stats"><thead><tr><th>Tier</th><th>Claims</th>'
        '<th>Share</th></tr></thead>'
        f"<tbody>{conf_rows}</tbody></table></div>"
    )

    # Governance note
    gov_note = ""
    if donor_restricted:
        gov_note = (
            f'<p style="color:#6c757d;font-size:0.85rem">&#9432; {esc(donor_restricted)} document(s) '
            "carry donor reproduction restrictions.</p>"
        )

    return (
        "<!DOCTYPE html><html lang='en'><head>"
        "<title>Collection Statistics</title>"
        f"{_STATS_CSS}</head><body style='font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem'>"
        "<a class='nav-back' href='/'>&#8592; Back to Query Interface</a>"
        "<h1>Collection Statistics</h1>"
        f"{cards}{gov_note}"
        f"{doc_type_section}{claim_type_section}{temporal_section}{entity_section}{confidence_section}"
        "<p style='margin-top:2rem;font-size:0.9rem'>"
        "<a href='/gaps' style='color:#0d6efd'>&#8594; View Gap Analysis</a></p>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Gap detection helpers and HTML renderer
# ---------------------------------------------------------------------------

def _compute_temporal_gaps(temporal_rows: list[dict]) -> list[dict]:
    """Return consecutive sparse-year periods that fall between active coverage."""
    if not temporal_rows:
        return []
    year_data = {row["year"]: row.get("doc_count", 0) for row in temporal_rows}
    all_years = sorted(year_data)
    if len(all_years) < 2:
        return []
    min_year, max_year = all_years[0], all_years[-1]
    full = {y: year_data.get(y, 0) for y in range(min_year, max_year + 1)}
    active = [v for v in full.values() if v > 0]
    if not active:
        return []
    median_active = _statistics.median(active)
    gap_threshold = max(1, median_active * 0.2)

    gaps: list[dict] = []
    current_gap: list[int] = []
    for y in range(min_year, max_year + 1):
        if full[y] <= gap_threshold:
            current_gap.append(y)
        else:
            if current_gap:
                before_docs = full.get(current_gap[0] - 1, 0)
                after_docs = full[y]
                if before_docs > gap_threshold or after_docs > gap_threshold:
                    gaps.append({"start": current_gap[0], "end": current_gap[-1],
                                 "years_count": len(current_gap),
                                 "before_docs": before_docs, "after_docs": after_docs})
            current_gap = []
    if current_gap:
        before_docs = full.get(current_gap[0] - 1, 0)
        if before_docs > gap_threshold:
            gaps.append({"start": current_gap[0], "end": current_gap[-1],
                         "years_count": len(current_gap),
                         "before_docs": before_docs, "after_docs": 0})
    return gaps


_RESTRICTED_ACCESS_LEVELS = frozenset({"restricted", "indigenous_restricted", "staff_only"})


def _safe_log_claim_id(claim_id: str, access_level: str) -> str:
    """Return an opaque stable hash for claims from restricted documents.

    Preserves the ability to see access patterns (e.g. same claim retrieved
    N times) without exposing the actual claim_id of sensitive content in the
    conversation log.

    Hash length: 12 hex chars = 48 bits of SHA-256.  This is a deliberate
    tradeoff — 48 bits is sufficient to distinguish individual claims within
    a single institution's corpus (<<2^24 claims expected) while keeping log
    rows compact.  It is NOT collision-resistant at scale; do not use these
    hashed IDs as primary keys or for cryptographic purposes.
    """
    if access_level in _RESTRICTED_ACCESS_LEVELS:
        return "redacted_" + _hashlib.sha256(claim_id.encode()).hexdigest()[:12]
    return claim_id


def _read_query_signal_gaps(conv_log_db: str, limit: int = 10) -> list[dict]:
    """Return query buckets with high frequency but thin claim context from conversation log."""
    try:
        con = _sqlite3.connect(conv_log_db, check_same_thread=False)
        try:
            rows = con.execute(
                """
                SELECT c.bucket,
                       COUNT(DISTINCT c.conversation_id) AS query_count,
                       AVG(CAST(r.claims_in_context AS REAL)) AS avg_claims
                FROM conversation c
                JOIN retrieval_event r ON c.conversation_id = r.conversation_id
                GROUP BY c.bucket
                HAVING avg_claims < 5.0
                ORDER BY query_count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            con.close()
        return [{"bucket": r[0], "query_count": r[1], "avg_claims": round(r[2], 1)} for r in rows]
    except Exception:
        return []


def _render_gaps_html(
    temporal: list[dict],
    entity_depth: list[dict],
    geo_coverage: list[dict],
    claim_types: list[dict],
    conv_gaps: list[dict],
) -> str:
    def esc(v: object) -> str:
        return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # --- 1. Temporal coverage gaps ---
    temporal_gaps = _compute_temporal_gaps(temporal)
    if temporal_gaps:
        gap_rows = ""
        for g in temporal_gaps:
            label = f"{g['start']}" if g["start"] == g["end"] else f"{g['start']}–{g['end']} ({g['years_count']} years)"
            context = f"Adjacent coverage: {g['before_docs']} doc(s) before, {g['after_docs']} doc(s) after."
            gap_rows += f"<li><strong>{esc(label)}</strong> — {esc(context)}</li>"
        temporal_section = (
            "<div class='stats-section'><h2>Temporal Coverage Gaps</h2>"
            f"<ul style='line-height:1.9'>{gap_rows}</ul>"
            "<p style='font-size:0.85rem'><a href='/stats'>&#8594; View full temporal coverage</a></p>"
            "</div>"
        )
    else:
        year_span = f"{temporal[0]['year']}–{temporal[-1]['year']}" if temporal else "unknown"
        temporal_section = (
            "<div class='stats-section'><h2>Temporal Coverage Gaps</h2>"
            f"<p>No significant temporal gaps found across the covered range ({esc(year_span)}).</p>"
            "<p style='font-size:0.85rem'><a href='/stats'>&#8594; View full temporal coverage</a></p>"
            "</div>"
        )

    # --- 2. Entity depth gaps ---
    def _depth_badge(count: int) -> str:
        if count <= 1:
            return "<span style='background:#f8d7da;color:#721c24;border-radius:3px;padding:1px 5px;font-size:11px'>1 claim</span>"
        return "<span style='background:#fff3cd;color:#856404;border-radius:3px;padding:1px 5px;font-size:11px'>2–3 claims</span>"

    if entity_depth:
        # Group by entity_type
        by_type: dict[str, list[dict]] = {}
        for row in entity_depth:
            by_type.setdefault(row["entity_type"], []).append(row)
        type_blocks = ""
        for etype, rows in sorted(by_type.items()):
            tbody = "".join(
                f"<tr><td>{esc(r['name'])}</td><td>{_depth_badge(r['primary_claim_count'])}</td></tr>"
                for r in rows
            )
            type_blocks += (
                f"<h3 style='font-size:0.95rem;margin:1rem 0 0.3rem'>{esc(etype)}</h3>"
                "<table class='stats'><thead><tr><th>Entity</th><th>Primary Claims</th></tr></thead>"
                f"<tbody>{tbody}</tbody></table>"
            )
        entity_section = (
            "<div class='stats-section'><h2>Entity Depth Gaps</h2>"
            "<p style='font-size:0.85rem;color:#6c757d'>Entities with &#8804;3 primary-subject claims — "
            "known to the collection but without substantive documentation.</p>"
            f"{type_blocks}</div>"
        )
    else:
        entity_section = (
            "<div class='stats-section'><h2>Entity Depth Gaps</h2>"
            "<p>No thinly-covered entities found (all entities have &gt;3 primary claims).</p></div>"
        )

    # --- 3. Geographic coverage gaps ---
    if geo_coverage:
        tbody = "".join(
            f"<tr><td>{esc(r['name'])}</td><td>{esc(r['entity_type'])}</td>"
            f"<td>{_depth_badge(r['location_specific_claims'])}</td></tr>"
            for r in geo_coverage
        )
        geo_section = (
            "<div class='stats-section'><h2>Geographic Coverage Gaps</h2>"
            "<p style='font-size:0.85rem;color:#6c757d'>Places referenced with few geographically-specific claims "
            "(OCCURRED_AT or LOCATION_FOCUS relationships).</p>"
            "<table class='stats'><thead><tr><th>Place</th><th>Type</th><th>Geo Claims</th></tr></thead>"
            f"<tbody>{tbody}</tbody></table></div>"
        )
    else:
        geo_section = (
            "<div class='stats-section'><h2>Geographic Coverage Gaps</h2>"
            "<p>No thinly-covered places found.</p></div>"
        )

    # --- 4. Topical balance ---
    total_claims = sum(r.get("count", 0) for r in claim_types)
    actual_shares: dict[str, float] = {}
    unclassified_count = 0
    if total_claims > 0:
        for r in claim_types:
            ct = r.get("claim_type") or ""
            actual_shares[ct] = r.get("count", 0) / total_claims
            if ct == "unclassified_assertion":
                unclassified_count = r.get("count", 0)

    unclassified_banner = ""
    if total_claims > 0 and unclassified_count / total_claims > 0.10:
        unclassified_banner = (
            "<p style='background:#fff3cd;color:#856404;padding:0.5rem 0.75rem;border-radius:4px;font-size:0.85rem'>"
            f"&#9888; {round(unclassified_count/total_claims*100)}% of claims are unclassified assertions — "
            "this may indicate extraction difficulty in certain domain areas.</p>"
        )

    topic_rows = ""
    for claim_type, expected in sorted(_EXPECTED_CLAIM_SHARES.items(), key=lambda x: -x[1]):
        actual = actual_shares.get(claim_type, 0.0)
        actual_pct = round(actual * 100, 1)
        expected_pct = round(expected * 100, 1)
        if actual < expected * 0.25:
            status = "<span style='color:#dc3545;font-weight:600'>gap</span>"
        elif actual < expected * 0.50:
            status = "<span style='color:#fd7e14;font-weight:600'>thin</span>"
        else:
            status = "<span style='color:#198754'>ok</span>"
        topic_rows += (
            f"<tr><td>{esc(claim_type)}</td><td>{actual_pct}%</td>"
            f"<td>{expected_pct}%</td><td>{status}</td></tr>"
        )

    topic_section = (
        "<div class='stats-section'><h2>Topical Balance</h2>"
        "<p style='font-size:0.85rem;color:#6c757d'>Actual claim-type share vs. expected profile "
        "for a wildlife refuge annual-report corpus.</p>"
        f"{unclassified_banner}"
        "<table class='stats'><thead><tr><th>Claim Type</th><th>Actual</th>"
        "<th>Expected</th><th>Status</th></tr></thead>"
        f"<tbody>{topic_rows}</tbody></table></div>"
    )

    # --- 5. Query signal (conditional) ---
    if conv_gaps:
        tbody = "".join(
            f"<tr><td>{esc(r['bucket'])}</td><td>{esc(r['query_count'])}</td>"
            f"<td>{esc(r['avg_claims'])}</td></tr>"
            for r in conv_gaps
        )
        conv_section = (
            "<div class='stats-section'><h2>Query Signal Gaps</h2>"
            "<p style='font-size:0.85rem;color:#6c757d'>Topic buckets queried frequently but consistently "
            "returning fewer than 5 claims — suggesting low coverage rather than failed retrieval.</p>"
            "<table class='stats'><thead><tr><th>Query Bucket</th><th>Query Count</th>"
            "<th>Avg Claims Retrieved</th></tr></thead>"
            f"<tbody>{tbody}</tbody></table></div>"
        )
    else:
        conv_section = ""

    return (
        "<!DOCTYPE html><html lang='en'><head>"
        "<title>Gap Analysis</title>"
        f"{_STATS_CSS}</head>"
        "<body style='font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem'>"
        "<a class='nav-back' href='/'>&#8592; Back to Query Interface</a>"
        "&nbsp;&nbsp;<a class='nav-back' href='/stats'>&#8592; Collection Stats</a>"
        "<h1>Gap Analysis</h1>"
        "<p style='color:#6c757d'>What the collection should document but doesn&#8217;t — "
        "thin coverage, absent periods, and underrepresented topics.</p>"
        f"{temporal_section}{entity_section}{geo_section}{topic_section}{conv_section}"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Relationship mapping page
# ---------------------------------------------------------------------------

_MAP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relationship Map</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.29.2/cytoscape.min.js"
        integrity="sha512-yi5TwB0WBpzqlJXNLURNMtpFXJt4yxJhkOG8yqkVQYWhfMkAoDF93rZ/KjfoN1gADGr5uKXvr5/Bw6CC03YWpA=="
        crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<style>
:root{--bg:#f8f9fa;--card:#fff;--border:#dee2e6;--header-bg:#2c5f2e;--header-fg:#fff;
      --detail-bg:#fff;--muted:#6c757d;--link:#0d6efd}
@media(prefers-color-scheme:dark){
  :root{--bg:#1a1a1a;--card:#242424;--border:#444;--header-bg:#1a3a1c;--header-fg:#e8e8e8;
        --detail-bg:#242424;--muted:#adb5bd;--link:#6ea8fe}
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:var(--bg);color:inherit;min-height:100vh}
header{background:var(--header-bg);color:var(--header-fg);padding:0.6rem 1.2rem;
       display:flex;align-items:center;justify-content:space-between;gap:0.5rem}
header h1{font-size:1.05rem;font-weight:600}
.nav-links{display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap}
.nav-links a{color:var(--header-fg);font-size:13px;text-decoration:none;
             border:1px solid rgba(255,255,255,0.25);border-radius:6px;
             padding:4px 10px;white-space:nowrap}
.main{max-width:1100px;margin:1.5rem auto;padding:0 1rem}
.search-wrap{position:relative;max-width:480px;margin-bottom:1rem}
#entity-input{width:100%;padding:0.55rem 0.8rem;border:1px solid var(--border);
              border-radius:6px;font-size:0.95rem;background:var(--card);color:inherit}
.autocomplete-list{position:absolute;top:100%;left:0;right:0;background:var(--card);
                   border:1px solid var(--border);border-radius:0 0 6px 6px;
                   list-style:none;max-height:260px;overflow-y:auto;z-index:100;display:none}
.autocomplete-list li{padding:0.45rem 0.8rem;cursor:pointer;font-size:0.9rem;
                      display:flex;justify-content:space-between;align-items:center}
.autocomplete-list li:hover{background:var(--bg)}
.et-badge{font-size:11px;background:#e9ecef;color:#495057;border-radius:3px;padding:1px 5px}
.layout{display:grid;grid-template-columns:1fr 320px;gap:1rem}
@media(max-width:700px){.layout{grid-template-columns:1fr}}
#cy{width:100%;height:560px;border:1px solid var(--border);border-radius:8px;
    background:var(--card)}
#cy.empty{display:flex;align-items:center;justify-content:center;color:var(--muted);
          font-size:0.9rem}
.detail-panel{background:var(--detail-bg);border:1px solid var(--border);border-radius:8px;
              padding:1rem;overflow-y:auto;max-height:560px}
.detail-panel h3{font-size:0.95rem;margin-bottom:0.75rem;word-break:break-word}
.detail-panel .rel-tags{display:flex;flex-wrap:wrap;gap:0.3rem;margin-bottom:0.75rem}
.rel-tag{font-size:11px;background:#e9ecef;color:#495057;border-radius:3px;padding:2px 6px}
.detail-panel blockquote{font-size:0.82rem;color:var(--muted);border-left:3px solid var(--border);
                         padding-left:0.6rem;margin:0.4rem 0;line-height:1.5}
.detail-panel .claim-ids{font-size:11px;color:var(--muted);margin-top:0.5rem;word-break:break-all}
.detail-placeholder{color:var(--muted);font-size:0.85rem;line-height:1.6}
.legend{display:flex;flex-wrap:wrap;gap:0.4rem;margin:0.5rem 0 1rem}
.legend-item{display:flex;align-items:center;gap:0.3rem;font-size:12px}
.legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
#status-bar{font-size:0.82rem;color:var(--muted);margin-bottom:0.5rem;min-height:1.2em}
</style>
</head>
<body>
<header>
  <h1>Turnbull GraphRAG</h1>
  <div class="nav-links">
    <a href="/">&#8592; Query Interface</a>
    <a href="/stats">&#128202; Collection Stats</a>
    <a href="/gaps">&#128270; Gap Analysis</a>
  </div>
</header>
<div class="main">
  <h2 style="font-size:1.15rem;margin-bottom:0.75rem">Relationship Map</h2>
  <p style="font-size:0.88rem;color:var(--muted);margin-bottom:1rem">
    Search for an entity to explore how it connects to others across the collection.
    Click an edge to see supporting source sentences. Click a node to re-centre the graph.
  </p>
  <div class="search-wrap">
    <input id="entity-input" type="text" placeholder="Search entities by name (e.g. Trumpeter Swan)…" autocomplete="off">
    <ul id="autocomplete-list" class="autocomplete-list"></ul>
  </div>
  <div class="legend" id="legend"></div>
  <div id="status-bar"></div>
  <div class="layout">
    <div id="cy" class="empty">Search for an entity above to begin.</div>
    <div class="detail-panel" id="detail-panel">
      <h3>Connection Details</h3>
      <div class="detail-placeholder" id="detail-content">
        Click an edge to see the source sentences supporting that connection,
        or click a node to explore its neighborhood.
      </div>
    </div>
  </div>
</div>
<script>
(function() {
'use strict';

var ENTITY_COLORS = {
  'Species':      '#198754',
  'Person':       '#0d6efd',
  'Organization': '#6f42c1',
  'Place':        '#795548',
  'Refuge':       '#795548',
  'Habitat':      '#0dcaf0',
  'Activity':     '#fd7e14',
  'SurveyMethod': '#20c997',
  'Event':        '#e83e8c',
};
var DEFAULT_COLOR = '#6c757d';

function entityColor(et) { return ENTITY_COLORS[et] || DEFAULT_COLOR; }
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                  .replace(/"/g,'&quot;');
}

// Build legend
var legendEl = document.getElementById('legend');
Object.keys(ENTITY_COLORS).forEach(function(et) {
  if (et === 'Refuge') return; // same colour as Place, skip duplicate
  legendEl.innerHTML += '<span class="legend-item">'
    + '<span class="legend-dot" style="background:' + ENTITY_COLORS[et] + '"></span>'
    + esc(et) + '</span>';
});

// ── Autocomplete ──────────────────────────────────────────────────────────
var input = document.getElementById('entity-input');
var listEl = document.getElementById('autocomplete-list');
var debounceTimer;

input.addEventListener('input', function() {
  clearTimeout(debounceTimer);
  var q = this.value.trim();
  if (q.length < 2) { listEl.style.display = 'none'; return; }
  debounceTimer = setTimeout(function() { fetchSuggestions(q); }, 300);
});
document.addEventListener('click', function(e) {
  if (!e.target.closest('.search-wrap')) listEl.style.display = 'none';
});
input.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') listEl.style.display = 'none';
});

async function fetchSuggestions(q) {
  try {
    var res = await fetch('/api/entities/search?q=' + encodeURIComponent(q) + '&limit=15');
    var items = await res.json();
    renderSuggestions(items);
  } catch(err) { listEl.style.display = 'none'; }
}

function renderSuggestions(items) {
  if (!items.length) { listEl.style.display = 'none'; return; }
  listEl.innerHTML = items.map(function(it) {
    return '<li data-id="' + esc(it.entity_id) + '" data-name="' + esc(it.name) + '">'
      + esc(it.name)
      + '<span class="et-badge" style="background:' + entityColor(it.entity_type)
      + ';color:#fff">' + esc(it.entity_type) + '</span></li>';
  }).join('');
  listEl.style.display = 'block';
  listEl.querySelectorAll('li').forEach(function(li) {
    li.addEventListener('click', function() {
      selectEntity(li.dataset.id, li.dataset.name);
    });
  });
}

function selectEntity(id, name) {
  input.value = name;
  listEl.style.display = 'none';
  loadGraph(id);
}

// ── Graph loading & rendering ─────────────────────────────────────────────
var cy = null;
var statusBar = document.getElementById('status-bar');

async function loadGraph(entityId) {
  statusBar.textContent = 'Loading neighborhood…';
  var cyEl = document.getElementById('cy');
  cyEl.classList.remove('empty');
  cyEl.textContent = '';
  try {
    var res = await fetch('/api/entity/' + encodeURIComponent(entityId) + '/neighborhood?limit=15');
    if (!res.ok) { statusBar.textContent = 'Error: ' + res.status; return; }
    var data = await res.json();
    renderGraph(data);
    statusBar.textContent = data.center.name + ' — ' + (data.nodes.length - 1)
      + ' connected entities, ' + data.edges.length + ' edges';
    showCenterDetail(data.center);
  } catch(err) {
    statusBar.textContent = 'Failed to load: ' + err.message;
  }
}

function renderGraph(data) {
  if (cy) cy.destroy();
  var style = [
    { selector: 'node', style: {
        'label': 'data(label)',
        'background-color': function(ele) { return entityColor(ele.data('entity_type')); },
        'border-width': function(ele) { return ele.data('is_center') ? 3 : 1; },
        'border-color': function(ele) { return ele.data('is_center') ? '#ffc107' : '#fff'; },
        'width': function(ele) { return ele.data('is_center') ? 60 : 36; },
        'height': function(ele) { return ele.data('is_center') ? 60 : 36; },
        'font-size': 11,
        'text-valign': 'bottom',
        'text-margin-y': 4,
        'text-max-width': 90,
        'text-wrap': 'wrap',
        'color': '#333',
        'text-background-color': 'rgba(255,255,255,0.75)',
        'text-background-opacity': 1,
        'text-background-padding': 2,
    }},
    { selector: 'edge', style: {
        'width': function(ele) { return Math.max(1, Math.min(8, ele.data('weight'))); },
        'line-color': '#adb5bd',
        'target-arrow-color': '#adb5bd',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'opacity': 0.8,
    }},
    { selector: 'edge:selected, edge.highlighted', style: {
        'line-color': '#fd7e14', 'target-arrow-color': '#fd7e14', 'opacity': 1,
    }},
    { selector: 'node:selected, node.highlighted', style: {
        'border-color': '#fd7e14', 'border-width': 3,
    }},
  ];
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: data.nodes.concat(data.edges),
    style: style,
    layout: { name: 'cose', animate: false, padding: 40, nodeRepulsion: 8000,
               idealEdgeLength: 120, gravity: 0.3 },
    userZoomingEnabled: true,
    userPanningEnabled: true,
  });
  cy.on('tap', 'edge', handleEdgeTap);
  cy.on('tap', 'node', handleNodeTap);
}

// ── Detail panel ──────────────────────────────────────────────────────────
var detailPanel = document.getElementById('detail-panel');
var detailContent = document.getElementById('detail-content');
var detailTitle = detailPanel.querySelector('h3');

function showCenterDetail(center) {
  detailTitle.textContent = center.name;
  detailContent.innerHTML =
    '<p style="font-size:0.85rem;margin-bottom:0.5rem">'
    + '<strong>Type:</strong> ' + esc(center.entity_type || '—') + '<br>'
    + '<strong>Claims:</strong> ' + esc(center.claim_count || 0) + '</p>'
    + '<p class="detail-placeholder">Click an edge to see connection evidence, '
    + 'or a neighbor node to explore its neighborhood.</p>';
}

function handleEdgeTap(evt) {
  cy.elements().removeClass('highlighted');
  evt.target.addClass('highlighted');
  var d = evt.target.data();
  detailTitle.textContent = esc(d.source_label) + ' \u2194 ' + esc(d.target_label);
  var relTags = (d.relationship_types || []).map(function(r) {
    return '<span class="rel-tag">' + esc(r) + '</span>';
  }).join('');
  var sentences = (d.sample_sentences || []).map(function(s) {
    return '<blockquote>' + esc(s) + '</blockquote>';
  }).join('');
  var claimIds = (d.sample_claim_ids || []).length
    ? '<div class="claim-ids">Claims: ' + d.sample_claim_ids.map(esc).join(', ') + '</div>'
    : '';
  detailContent.innerHTML =
    '<p style="font-size:0.82rem;margin-bottom:0.5rem">'
    + '<strong>Co-occurrences:</strong> ' + esc(d.weight) + ' claim(s)</p>'
    + (relTags ? '<div class="rel-tags">' + relTags + '</div>' : '')
    + (sentences || '<p class="detail-placeholder">No source sentences recorded.</p>')
    + claimIds;
}

function handleNodeTap(evt) {
  var nodeData = evt.target.data();
  if (nodeData.is_center) {
    // Just show info for center
    detailTitle.textContent = esc(nodeData.label);
    detailContent.innerHTML =
      '<p style="font-size:0.85rem"><strong>Type:</strong> ' + esc(nodeData.entity_type || '—')
      + '<br><strong>Claims:</strong> ' + esc(nodeData.claim_count || 0) + '</p>';
    return;
  }
  // Drill down: reload graph centred on this node
  input.value = nodeData.label;
  loadGraph(nodeData.id);
}

})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str = "neo4j",
    neo4j_trust: str = "system",
    neo4j_ca_cert: str | None = None,
    anthropic_api_key: str | None = None,
    max_tokens: int = 1000,
    domain_dir: str | None = None,
    annotation_db_path: str | None = None,
) -> FastAPI:
    """Construct and return the FastAPI retrieval application.

    All parameters fall back to environment variables so the app can be
    started from the CLI without explicit arguments.
    """
    from graphrag_pipeline.shared.logging_config import setup_logging
    setup_logging()
    _log = logging.getLogger(__name__)

    # Resolve connection parameters from env if not provided explicitly.
    uri = neo4j_uri or os.environ.get("NEO4J_URI", "")
    user = neo4j_user or os.environ.get("NEO4J_USER", "")
    password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "")
    database = neo4j_database or os.environ.get("NEO4J_DATABASE", "neo4j")
    trust = neo4j_trust or os.environ.get("NEO4J_TRUST", "system")
    ca_cert = neo4j_ca_cert or os.environ.get("NEO4J_CA_CERT")
    api_key = anthropic_api_key or os.environ.get("Anthropic_API_Key") or os.environ.get("ANTHROPIC_API_KEY")

    # Shared pipeline components (initialised at startup, closed at shutdown).
    state: dict[str, Any] = {}

    from graphrag_pipeline.retrieval.web.rate_limit import TokenBucketLimiter
    _query_limiter = TokenBucketLimiter(
        max_calls=int(os.environ.get("QUERY_RATE_LIMIT_MAX", "20")),
        period_seconds=float(os.environ.get("QUERY_RATE_LIMIT_PERIOD", "60")),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[misc]
        executor = Neo4jQueryExecutor(
            uri=uri,
            user=user,
            password=password,
            database=database,
            trust_mode=trust,
            ca_cert_path=ca_cert,
        )
        executor.ensure_schema()
        corpus_rows = executor.run(CORPUS_STATS_QUERY, {})
        state["corpus_stats"] = corpus_rows[0] if corpus_rows else {"total_paragraphs": 0, "total_documents": 0}
        state["executor"] = executor
        state["query_builder"] = CypherQueryBuilder(executor)
        _ann_db = annotation_db_path or os.environ.get("ANNOTATION_DB")
        _annotation_store = None
        if _ann_db:
            from graphrag_pipeline.ingest.annotation import AnnotationStore
            _annotation_store = AnnotationStore(_ann_db)
            state["annotation_store"] = _annotation_store
        state["assembler"] = ProvenanceContextAssembler(executor, annotation_store=_annotation_store)
        state["gateway"] = EntityResolutionGateway()
        synthesis_ctx: str | None = None
        if domain_dir:
            from ...shared.resource_loader import load_domain_profile
            synthesis_ctx = load_domain_profile(Path(domain_dir)).get("synthesis_context")
        state["synthesis"] = SynthesisEngine(
            api_key=api_key,
            max_tokens=max_tokens,
            synthesis_context=synthesis_ctx,
            timeout=float(os.environ.get("SYNTHESIS_TIMEOUT", "60")),
        )
        conv_log_path = Path(os.environ.get("CONV_LOG_DB", "data/conversation_log.db"))
        state["conv_logger"] = ConversationLogger(conv_log_path)
        from graphrag_pipeline.retrieval.conversation_log import QueryHistoryStore as _QHSInit
        state["history_store"] = _QHSInit(conv_log_path)
        write_audit_path = os.environ.get("WRITE_AUDIT_DB", "data/write_audit.db")
        from .write_audit_log import WriteAuditLogger
        state["write_audit"] = WriteAuditLogger(write_audit_path)
        yield
        executor.close()
        if state.get("annotation_store"):
            state["annotation_store"].close()
        if state.get("history_store"):
            state["history_store"].close()

    app = FastAPI(
        title="Turnbull GraphRAG Retrieval API",
        description="Natural-language query interface over the Turnbull refuge archive graph.",
        lifespan=lifespan,
    )

    # Auth router (login/logout/me/user management at /auth/...)
    from fastapi import Request
    from fastapi.responses import RedirectResponse as _RedirectResponse
    from graphrag_pipeline.auth.setup import is_setup_needed

    users_db_path = os.environ.get("USERS_DB", "data/users.db")
    app.include_router(create_auth_router(users_db_path), prefix="/auth")

    @app.exception_handler(NeedsLoginException)
    async def _needs_login_handler(request: Request, exc: NeedsLoginException):
        return _RedirectResponse(url=exc.redirect_url, status_code=303)

    from fastapi import HTTPException as _HTTPException
    from fastapi.responses import JSONResponse as _JSONResponse

    @app.exception_handler(Exception)
    async def _internal_error_handler(request: Request, exc: Exception) -> _JSONResponse:
        # Re-raise HTTPException so FastAPI's built-in handler produces the
        # correct status code and detail — only swallow truly unhandled errors.
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

    import uuid as _uuid

    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.middleware("http")
    async def _setup_guard(request: Request, call_next):
        if not request.url.path.startswith("/auth/setup") and request.url.path != "/health":
            if is_setup_needed(users_db_path):
                return _RedirectResponse(url="/auth/setup", status_code=303)
        return await call_next(request)

    # ------------------------------------------------------------------
    # GET /health — liveness + Neo4j connectivity check
    # NOTE: This endpoint MUST remain unauthenticated. It is used by uptime
    # monitors and load balancers that have no session credentials. Do not
    # add require_user or any auth dependency here.
    # ------------------------------------------------------------------
    @app.get("/health", include_in_schema=True)
    def health():
        ts = datetime.now(timezone.utc).isoformat()
        executor = state.get("executor")
        if executor is None:
            return _JSONResponse(
                status_code=503,
                content={"status": "starting", "neo4j": "not_ready", "timestamp": ts},
            )
        try:
            executor.run("RETURN 1 AS ok", {})
            neo4j_status = "connected"
        except Exception:
            neo4j_status = "unavailable"
        neo4j_ok = neo4j_status == "connected"
        return _JSONResponse(
            status_code=200 if neo4j_ok else 503,
            content={"status": "ok" if neo4j_ok else "degraded", "neo4j": neo4j_status, "timestamp": ts},
        )

    # ------------------------------------------------------------------
    # GET / — Chat UI
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def chat_ui(user: UserContext = Depends(require_user)) -> HTMLResponse:
        return HTMLResponse(_CHAT_HTML)

    # ------------------------------------------------------------------
    # POST /query
    # ------------------------------------------------------------------
    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, request: Request, user: UserContext = Depends(require_user)) -> QueryResponse:
        # Rate limiting: admins are exempt; all other users are limited to
        # QUERY_RATE_LIMIT_MAX requests per QUERY_RATE_LIMIT_PERIOD seconds.
        if user.role != "admin":
            _rate_key = user.user_id or user.institution_id
            if not _query_limiter.is_allowed(_rate_key):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Maximum 20 requests per minute.",
                    headers={"Retry-After": "60"},
                )

        gateway: EntityResolutionGateway = state["gateway"]
        assembler: ProvenanceContextAssembler = state["assembler"]
        builder: CypherQueryBuilder = state["query_builder"]
        engine: SynthesisEngine = state["synthesis"]

        created_at = datetime.now(timezone.utc).isoformat()
        # Always generate a unique per-turn conversation_id so every turn gets
        # its own row in the conversation + retrieval_event tables.  Previously
        # this was set to req.session_id, which collapsed all turns onto the
        # same primary key — the INSERT OR IGNORE in _write_record then silently
        # dropped entity resolution data for turns 2+ in a session.
        # session_id is stored separately in LogRecord for grouping multi-turn
        # sessions in the analytics queries.
        conversation_id = make_conversation_id(req.text, created_at)
        # Assumes symmetric history (alternating user/assistant pairs), which holds
        # for the browser client (history only accumulates on success). Non-browser
        # clients sending asymmetric history will get an approximate turn_number.
        turn_number = len(req.conversation_history) // 2 + 1

        # Layer 0: classify intent.
        intent = classify_query(
            text=req.text,
            year_range=req.year_range,
            entity_hints=req.entity_hints or [],
        )
        if req.mode != "auto":
            # Caller overrides bucket.
            intent = intent.__class__(  # type: ignore[call-arg]
                bucket=req.mode if req.mode != "auto" else intent.bucket,
                classifier_confidence=intent.classifier_confidence,
                entities=intent.entities,
                year_min=intent.year_min,
                year_max=intent.year_max,
                claim_types=intent.claim_types,
            )

        # Layer 1: entity resolution.
        entity_ctx = gateway.resolve(
            surface_forms=intent.entities,
            entity_hints=req.entity_hints,
        )
        _log.debug("entities extracted: %s", intent.entities)
        _log.debug("resolved: %s", [(r.surface_form, r.entity_id, r.resolution_confidence) for r in entity_ctx.resolved])
        _log.debug("unresolved: %s", entity_ctx.unresolved)

        is_hybrid = intent.bucket == "hybrid"
        analytical_result = None

        # Layer 2A: analytical path.
        if intent.bucket in ("analytical", "hybrid"):
            # Run species trend for the first resolved species entity, if any.
            species_entities = [e for e in entity_ctx.resolved if e.entity_type == "Species"]
            habitat_entities = [e for e in entity_ctx.resolved if e.entity_type == "Habitat"]

            if species_entities:
                analytical_result = builder.species_trend(
                    species_id=species_entities[0].entity_id,
                    year_min=intent.year_min,
                    year_max=intent.year_max,
                    permitted_levels=user.permitted_levels,
                    institution_id=user.institution_id,
                )
            elif habitat_entities:
                analytical_result = builder.habitat_conditions(
                    habitat_id=habitat_entities[0].entity_id,
                    year_min=intent.year_min,
                    year_max=intent.year_max,
                    permitted_levels=user.permitted_levels,
                    institution_id=user.institution_id,
                )

        # Layer 2B: conversational path.
        blocks, context_text = assembler.assemble(
            query_text=req.text,
            entity_context=entity_ctx,
            year_min=intent.year_min,
            year_max=intent.year_max,
            is_hybrid=is_hybrid,
            permitted_levels=user.permitted_levels,
            institution_id=user.institution_id,
        )

        # Build coverage stats before synthesis.
        corpus = state["corpus_stats"]
        retrieval_stats = RetrievalStats(
            candidates_retrieved=assembler._last_candidate_count,
            ocr_dropped=assembler._last_ocr_dropped,
            claims_in_context=assembler._last_context_count,
            paragraphs_in_context=len({b.paragraph_id for b in blocks}),
            documents_in_context=len({b.doc_title for b in blocks}),
            corpus_total_paragraphs=int(corpus.get("total_paragraphs", 0)),
            corpus_total_documents=int(corpus.get("total_documents", 0)),
        )

        # Layer 3: synthesis.
        synthesis_available = True
        try:
            result: SynthesisResult = engine.synthesise(
                query=req.text,
                provenance_blocks=blocks,
                context_text=context_text,
                analytical_result=analytical_result,
                conversation_history=[t.model_dump() for t in req.conversation_history],
            )
        except Exception:
            # API unavailable, rate-limited, or response unparseable.
            # Return retrieved provenance blocks with a fallback answer so the
            # archivist can read source sentences directly without a 500 error.
            synthesis_available = False
            result = SynthesisResult(
                answer=(
                    "AI synthesis is currently unavailable. "
                    "Retrieved source sentences are provided via supporting_claim_ids for direct review."
                ),
                confidence_assessment="",
                supporting_claim_ids=[b.claim_id for b in blocks],
                caveats=["AI synthesis was unavailable at query time. Source provenance blocks were retrieved successfully."],
            )

        conv_logger: ConversationLogger | None = state.get("conv_logger")
        if conv_logger is not None:
            cited_ids = set(result.supporting_claim_ids)
            conv_logger.enqueue(LogRecord(
                conversation_id=conversation_id,
                query_text=req.text,
                bucket=intent.bucket,
                classifier_confidence=intent.classifier_confidence,
                year_min=intent.year_min,
                year_max=intent.year_max,
                retrieval_path="entity_anchored" if entity_ctx.resolved else "fulltext",
                created_at=created_at,
                entity_ids_resolved=[e.entity_id for e in entity_ctx.resolved],
                entity_types_resolved=[e.entity_type for e in entity_ctx.resolved],
                candidates_retrieved=retrieval_stats.candidates_retrieved,
                ocr_dropped=retrieval_stats.ocr_dropped,
                claims_in_context=retrieval_stats.claims_in_context,
                session_id=req.session_id,
                turn_number=turn_number,
                request_id=getattr(request.state, "request_id", None),
                claim_interactions=[
                    ClaimInteraction(
                        claim_id=_safe_log_claim_id(b.claim_id, b.access_level),
                        claim_type=b.claim_type,
                        traversal_rel_types=b.traversal_rel_types,
                        was_cited=b.claim_id in cited_ids,
                        extraction_confidence=b.extraction_confidence,
                    )
                    for b in blocks
                ],
            ))

        # Count quarantined claims for this institution to include in response.
        quarantined_claims_count = 0
        try:
            _executor: Neo4jQueryExecutor = state["executor"]
            _qrows = _executor.run(
                COUNT_QUARANTINED_CLAIMS_QUERY,
                {
                    "institution_id": user.institution_id,
                    "permitted_levels": user.permitted_levels,
                },
            )
            quarantined_claims_count = int((_qrows[0].get("quarantined_count") or 0) if _qrows else 0)
        except Exception:
            pass

        final_caveats = list(result.caveats)
        if quarantined_claims_count > 0:
            final_caveats.append(
                f"{quarantined_claims_count} potentially relevant claim(s) are currently under "
                "sensitivity review and have been excluded from this answer."
            )

        # Build per-claim detail for uncertainty surfacing in the UI.
        blocks_by_id = {b.claim_id: b for b in blocks}
        claims_detail = [
            {
                "claim_id": cid,
                "confidence": blocks_by_id[cid].extraction_confidence if cid in blocks_by_id else None,
                "epistemic_status": blocks_by_id[cid].epistemic_status if cid in blocks_by_id else "unknown",
                "claim_type": blocks_by_id[cid].claim_type if cid in blocks_by_id else "",
                "doc_id": blocks_by_id[cid].doc_id if cid in blocks_by_id else "",
                "doc_title": blocks_by_id[cid].doc_title if cid in blocks_by_id else "",
            }
            for cid in result.supporting_claim_ids
        ]

        return QueryResponse(
            answer=result.answer,
            confidence_assessment=result.confidence_assessment,
            supporting_claim_ids=result.supporting_claim_ids,
            caveats=final_caveats,
            min_extraction_confidence=result.min_extraction_confidence,
            analytical_result=(
                {
                    "query_name": analytical_result.query_name,
                    "columns": analytical_result.columns,
                    "rows": analytical_result.rows,
                }
                if analytical_result
                else None
            ),
            ambiguous_entities=entity_ctx.ambiguous,
            retrieval_stats=dataclasses.asdict(retrieval_stats),
            synthesis_available=synthesis_available,
            quarantined_claims_count=quarantined_claims_count,
            supporting_claims_detail=claims_detail,
            user_can_annotate=user.role in ("admin", "archivist"),
        )

    # ------------------------------------------------------------------
    # POST /query/provenance
    # ------------------------------------------------------------------
    @app.post("/query/provenance")
    def query_provenance(req: ProvenanceRequest, user: UserContext = Depends(require_user)) -> dict[str, Any]:
        """Return the full raw provenance chain for a known *claim_id*.

        Does not invoke the synthesis engine — useful for citation
        verification and the review web interface.
        """
        assembler: ProvenanceContextAssembler = state["assembler"]
        blocks = assembler.chain_for_claim(
            req.claim_id,
            permitted_levels=user.permitted_levels,
            institution_id=user.institution_id,
        )
        if not blocks:
            raise HTTPException(status_code=404, detail=f"No provenance found for claim_id={req.claim_id!r}")
        return {
            "claim_id": req.claim_id,
            "provenance_blocks": [
                {
                    "doc_title": b.doc_title,
                    "doc_date_start": b.doc_date_start,
                    "doc_date_end": b.doc_date_end,
                    "page_number": b.page_number,
                    "paragraph_id": b.paragraph_id,
                    "claim_id": b.claim_id,
                    "claim_type": b.claim_type,
                    "extraction_confidence": b.extraction_confidence,
                    "epistemic_status": b.epistemic_status,
                    "source_sentence": b.source_sentence,
                    "observation_type": b.observation_type,
                    "species_name": b.species_name,
                    "year": b.year,
                    "measurements": b.measurements,
                }
                for b in blocks
            ],
            "serialised_context": "\n\n".join(_serialise_block(b) for b in blocks),
        }

    # ------------------------------------------------------------------
    # GET /stats  — collection statistics dashboard
    # ------------------------------------------------------------------
    @app.get("/stats", response_class=HTMLResponse, include_in_schema=False)
    def collection_stats(user: UserContext = Depends(require_user)) -> str:
        executor: Neo4jQueryExecutor = state["executor"]
        params = {"institution_id": user.institution_id, "permitted_levels": user.permitted_levels}
        overview = (executor.run(STATS_DOC_OVERVIEW_QUERY, params) or [{}])[0]
        doc_types = executor.run(STATS_DOC_TYPE_QUERY, params) or []
        claim_types = executor.run(STATS_CLAIM_TYPE_QUERY, params) or []
        entity_types = executor.run(STATS_ENTITY_TYPE_QUERY, params) or []
        temporal = executor.run(STATS_TEMPORAL_COVERAGE_QUERY, params) or []
        confidence = (executor.run(STATS_CONFIDENCE_DISTRIBUTION_QUERY, params) or [{}])[0]
        return _render_stats_html(overview, doc_types, claim_types, entity_types, temporal, confidence)

    # ------------------------------------------------------------------
    # GET /gaps  — collection gap analysis dashboard
    # ------------------------------------------------------------------
    @app.get("/gaps", response_class=HTMLResponse, include_in_schema=False)
    def collection_gaps(user: UserContext = Depends(require_admin)) -> str:
        executor: Neo4jQueryExecutor = state["executor"]
        params = {"institution_id": user.institution_id, "permitted_levels": user.permitted_levels}
        thin_params = {**params, "thin_threshold": 3, "limit": 50}

        temporal = executor.run(GAP_TEMPORAL_DENSITY_QUERY, params) or []
        entity_depth = executor.run(GAP_ENTITY_DEPTH_QUERY, thin_params) or []
        geo_coverage = executor.run(GAP_GEOGRAPHIC_COVERAGE_QUERY, thin_params) or []
        claim_types = executor.run(STATS_CLAIM_TYPE_QUERY, params) or []

        conv_gaps: list[dict] = []
        conv_log_db = os.environ.get("CONV_LOG_DB", "")
        if conv_log_db and os.path.exists(conv_log_db):
            conv_gaps = _read_query_signal_gaps(conv_log_db)

        return _render_gaps_html(temporal, entity_depth, geo_coverage, claim_types, conv_gaps)

    # ------------------------------------------------------------------
    # GET /map  — relationship mapping explorer
    # ------------------------------------------------------------------
    @app.get("/map", response_class=HTMLResponse, include_in_schema=False)
    def relationship_map(user: UserContext = Depends(require_user)) -> str:
        return _MAP_HTML

    # ------------------------------------------------------------------
    # GET /api/entities/search  — entity autocomplete
    # ------------------------------------------------------------------
    @app.get("/api/entities/search")
    def entity_search(
        q: str = Query("", min_length=0),
        limit: int = Query(15, le=50),
        user: UserContext = Depends(require_user),
    ) -> list[dict]:
        if not q.strip():
            return []
        executor: Neo4jQueryExecutor = state["executor"]
        return executor.run(ENTITY_SEARCH_QUERY, {"query": q.strip(), "limit": limit}) or []

    # ------------------------------------------------------------------
    # GET /api/entity/{entity_id}/neighborhood  — graph neighborhood data
    # ------------------------------------------------------------------
    @app.get("/api/entity/{entity_id}/neighborhood")
    def entity_neighborhood(
        entity_id: str,
        limit: int = Query(15, le=30),
        user: UserContext = Depends(require_user),
    ) -> dict[str, Any]:
        if not _ENTITY_ID_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity_id format")
        executor: Neo4jQueryExecutor = state["executor"]
        params = {"institution_id": user.institution_id, "permitted_levels": user.permitted_levels}

        center_rows = executor.run(ENTITY_DETAIL_QUERY, {**params, "entity_id": entity_id}) or [{}]
        center = center_rows[0] if center_rows else {}
        if not center.get("entity_id"):
            raise HTTPException(status_code=404, detail="Entity not found")

        neighbors = executor.run(
            ENTITY_NEIGHBORHOOD_QUERY, {**params, "entity_id": entity_id, "limit": limit}
        ) or []

        nodes: list[dict] = [{"data": {
            "id": center["entity_id"], "label": center["name"],
            "entity_type": center.get("entity_type", ""),
            "claim_count": center.get("claim_count", 0),
            "is_center": True,
        }}]
        edges: list[dict] = []
        for n in neighbors:
            nodes.append({"data": {
                "id": n["entity_id"], "label": n["name"],
                "entity_type": n.get("entity_type", ""),
                "claim_count": 0,
                "is_center": False,
            }})
            edges.append({"data": {
                "id": f"{entity_id}__{n['entity_id']}",
                "source": entity_id, "target": n["entity_id"],
                "weight": n["co_occurrence_count"],
                "relationship_types": n.get("relationship_types") or [],
                "sample_sentences": n.get("sample_sentences") or [],
                "sample_claim_ids": n.get("sample_claim_ids") or [],
                "source_label": center["name"],
                "target_label": n["name"],
            }})

        return {"nodes": nodes, "edges": edges, "center": center}

    # ------------------------------------------------------------------
    # GET /documents  — admin: list all documents including soft-deleted
    # ------------------------------------------------------------------
    @app.get("/documents")
    def list_documents(user: UserContext = Depends(require_admin)) -> dict[str, Any]:
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(LIST_DOCUMENTS_QUERY, {"institution_id": user.institution_id})
        return {"institution_id": user.institution_id, "documents": rows}

    # ------------------------------------------------------------------
    # GET /document/{doc_id}/notes-panel  — HTMX partial: archivist notes
    # POST /document/{doc_id}/notes       — save/update a document note
    # ------------------------------------------------------------------
    @app.get("/document/{doc_id}/notes-panel")
    async def get_notes_panel(
        doc_id: str,
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> HTMLResponse:
        if not _DOC_ID_RE.match(doc_id):
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
        store = state.get("annotation_store")
        if store is None:
            return HTMLResponse(_render_notes_panel_unavailable(doc_id))
        current = store.get_current_note(doc_id)
        history = store.get_note_history(doc_id, limit=5)
        return HTMLResponse(_render_notes_panel(doc_id, current, history))

    @app.post("/document/{doc_id}/notes")
    async def post_document_note(
        doc_id: str,
        note_text: str = Form(...),
        user: UserContext = Depends(require_archivist_or_admin),
    ) -> HTMLResponse:
        if not _DOC_ID_RE.match(doc_id):
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
        stripped = note_text.strip()
        if not stripped:
            return HTMLResponse(_render_notes_panel_error(doc_id, "Note cannot be empty."))
        if len(stripped) > 2000:
            return HTMLResponse(
                _render_notes_panel_error(doc_id, "Note exceeds 2000 characters.")
            )
        store = state.get("annotation_store")
        if store is None:
            raise HTTPException(status_code=503, detail="Annotation store not configured.")
        store.upsert_note(doc_id, stripped, created_by=user.identity)
        current = store.get_current_note(doc_id)
        history = store.get_note_history(doc_id, limit=5)
        return HTMLResponse(_render_notes_panel(doc_id, current, history))

    # ------------------------------------------------------------------
    # GET /history             — query history & saved searches page
    # POST /history/saved      — save a search
    # POST /history/saved/{id}/delete  — delete a saved search
    # GET /api/history         — JSON history list
    # GET /api/history/saved   — JSON saved searches
    # ------------------------------------------------------------------
    from graphrag_pipeline.retrieval.conversation_log import QueryHistoryStore as _QHS

    @app.get("/history", response_class=HTMLResponse, include_in_schema=False)
    def query_history_page(
        q: str = Query(""),
        bucket: str = Query(""),
        limit: int = Query(50, le=200),
        offset: int = Query(0, ge=0),
        user: UserContext = Depends(require_user),
    ) -> HTMLResponse:
        store: _QHS | None = state.get("history_store")
        if store is None:
            return HTMLResponse(
                "<p style='font-family:sans-serif;padding:2rem'>"
                "Query history not available. Set the <code>CONV_LOG_DB</code> "
                "environment variable to enable it.</p>"
            )
        rows = store.list_queries(limit=limit, offset=offset, q=q, bucket=bucket)
        total = store.count_queries(q=q, bucket=bucket)
        saved = store.get_saved_searches(created_by=user.identity)
        return HTMLResponse(_render_history_html(rows, total, saved, q, bucket, limit, offset))

    @app.post("/history/saved", include_in_schema=False)
    async def save_query(
        query_text: str = Form(...),
        label: str = Form(""),
        bucket: str = Form(""),
        year_min: str = Form(""),
        year_max: str = Form(""),
        conversation_id: str = Form(""),
        user: UserContext = Depends(require_user),
    ):
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        _yr_min: int | None = None
        _yr_max: int | None = None
        try:
            _yr_min = int(year_min) if year_min.strip() else None
        except ValueError:
            pass
        try:
            _yr_max = int(year_max) if year_max.strip() else None
        except ValueError:
            pass
        store.save_search(
            query_text=query_text.strip()[:2000],
            label=label.strip()[:200],
            bucket=bucket,
            year_min=_yr_min,
            year_max=_yr_max,
            created_by=user.identity,
            conversation_id=conversation_id.strip() or None,
        )
        return _RedirectResponse(url="/history", status_code=303)

    @app.post("/history/saved/{saved_id}/delete", include_in_schema=False)
    async def delete_saved_query(
        saved_id: str,
        user: UserContext = Depends(require_user),
    ):
        if not _SAVED_ID_RE.match(saved_id):
            raise HTTPException(status_code=400, detail="Invalid saved_id format")
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        store.delete_saved_search(saved_id, created_by=user.identity)
        return _RedirectResponse(url="/history", status_code=303)

    @app.get("/api/history")
    def api_history(
        q: str = Query(""),
        bucket: str = Query(""),
        limit: int = Query(50, le=200),
        offset: int = Query(0, ge=0),
        user: UserContext = Depends(require_user),
    ) -> dict[str, Any]:
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        rows = store.list_queries(limit=limit, offset=offset, q=q, bucket=bucket)
        total = store.count_queries(q=q, bucket=bucket)
        return {"total": total, "offset": offset, "limit": limit, "queries": rows}

    @app.get("/api/history/saved")
    def api_history_saved(
        user: UserContext = Depends(require_user),
    ) -> list[dict[str, Any]]:
        store: _QHS | None = state.get("history_store")
        if store is None:
            raise HTTPException(status_code=503, detail="History store not configured.")
        return store.get_saved_searches(created_by=user.identity)

    # ------------------------------------------------------------------
    # DELETE /document/{doc_id}  — admin: soft-delete a document
    # ------------------------------------------------------------------
    @app.delete("/document/{doc_id}")
    def delete_document(
        doc_id: str,
        confirm: bool = Query(default=False),
        user: UserContext = Depends(require_admin),
    ) -> dict[str, Any]:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Set ?confirm=true to acknowledge this soft-deletion. The document can be restored.",
            )
        executor: Neo4jQueryExecutor = state["executor"]
        deleted_at = datetime.now(timezone.utc).isoformat()
        deleted_by = user.identity
        rows = executor.run(SOFT_DELETE_DOCUMENT_QUERY, {
            "doc_id": doc_id,
            "institution_id": user.institution_id,
            "deleted_at": deleted_at,
            "deleted_by": deleted_by,
        })
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id!r} not found in institution {user.institution_id!r}",
            )
        write_audit: Any = state.get("write_audit")
        if write_audit is not None:
            write_audit.log(
                event_type="soft_delete",
                doc_id=doc_id,
                doc_title=rows[0].get("title"),
                institution_id=user.institution_id,
                performed_by=deleted_by,
            )
        return {"status": "deleted", "doc_id": doc_id, "deleted_at": deleted_at, "deleted_by": deleted_by}

    # ------------------------------------------------------------------
    # POST /document/{doc_id}/restore  — admin: restore a soft-deleted document
    # ------------------------------------------------------------------
    @app.post("/document/{doc_id}/restore")
    def restore_document(
        doc_id: str,
        user: UserContext = Depends(require_admin),
    ) -> dict[str, Any]:
        executor: Neo4jQueryExecutor = state["executor"]
        rows = executor.run(RESTORE_DOCUMENT_QUERY, {
            "doc_id": doc_id,
            "institution_id": user.institution_id,
        })
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id!r} not found in institution {user.institution_id!r}",
            )
        restored_at = datetime.now(timezone.utc).isoformat()
        performed_by = user.identity
        write_audit: Any = state.get("write_audit")
        if write_audit is not None:
            write_audit.log(
                event_type="restore",
                doc_id=doc_id,
                doc_title=rows[0].get("title"),
                institution_id=user.institution_id,
                performed_by=performed_by,
            )
        return {"status": "restored", "doc_id": doc_id, "restored_at": restored_at}

    # ------------------------------------------------------------------
    # /v1/ API versioning
    # Register canonical versioned paths for all JSON API routes.  The
    # unversioned paths above remain active (include_in_schema=False) so
    # the browser UI and existing integrations continue to work unchanged.
    # ------------------------------------------------------------------
    from fastapi import APIRouter as _APIRouter
    v1 = _APIRouter(prefix="/v1")
    v1.add_api_route("/query", query, methods=["POST"], response_model=QueryResponse)
    v1.add_api_route("/query/provenance", query_provenance, methods=["POST"])
    v1.add_api_route("/api/entities/search", entity_search, methods=["GET"])
    v1.add_api_route("/api/entity/{entity_id}/neighborhood", entity_neighborhood, methods=["GET"])
    v1.add_api_route("/documents", list_documents, methods=["GET"])
    v1.add_api_route("/document/{doc_id}", delete_document, methods=["DELETE"])
    v1.add_api_route("/document/{doc_id}/restore", restore_document, methods=["POST"])
    v1.add_api_route("/document/{doc_id}/notes-panel", get_notes_panel, methods=["GET"])
    v1.add_api_route("/document/{doc_id}/notes", post_document_note, methods=["POST"])
    v1.add_api_route("/api/history", api_history, methods=["GET"])
    v1.add_api_route("/api/history/saved", api_history_saved, methods=["GET"])
    app.include_router(v1)

    # Mark the original unversioned paths as deprecated in the schema.
    for _route in app.routes:
        if hasattr(_route, "path") and hasattr(_route, "include_in_schema"):
            if getattr(_route, "path", None) in {
                "/query", "/query/provenance", "/api/entities/search",
                "/api/entity/{entity_id}/neighborhood", "/documents",
                "/document/{doc_id}", "/document/{doc_id}/restore",
            } and not getattr(_route, "deprecated", False):
                _route.deprecated = True  # type: ignore[attr-defined]
                _route.include_in_schema = False  # type: ignore[attr-defined]

    return app
