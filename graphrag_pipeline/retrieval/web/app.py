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
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "fastapi/pydantic not installed. Install with: pip install -e .[retrieval]"
    )

from ..classifier import classify_query
from ..context_assembler import ProvenanceContextAssembler, _serialise_block
from ...graph.cypher import CORPUS_STATS_QUERY
from ..entity_gateway import EntityResolutionGateway
from ..executor import Neo4jQueryExecutor
from ..models import RetrievalStats, SynthesisResult
from ..query_builder import CypherQueryBuilder
from ..synthesis import SynthesisEngine


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

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
#settings-toggle {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.25);
  color: var(--header-fg);
  border-radius: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 13px;
  transition: background 0.15s;
}
#settings-toggle:hover { background: rgba(255,255,255,0.1); }

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
</style>
</head>
<body>

<header>
  <h1>Turnbull GraphRAG</h1>
  <button id="settings-toggle">&#9881; Settings</button>
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

  if (data.supporting_claim_ids && data.supporting_claim_ids.length > 0) {
    html += '<details><summary>Sources (' + data.supporting_claim_ids.length + ' claims)</summary>'
      + '<p class="claim-ids">' + data.supporting_claim_ids.map(esc).join(', ') + '</p></details>';
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
      body: JSON.stringify({text: text, mode: mode, year_range: year_range, entity_hints: entityHints}),
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

  document.getElementById('send-btn').addEventListener('click', sendMessage);

  appendBubble('assistant',
    '<p>Hello! Ask me anything about the Turnbull National Wildlife Refuge archive. '
    + 'Use <strong>\u2699 Settings</strong> to adjust mode, year range, or entity hints. '
    + '<em>Shift+Enter</em> for a new line.</p>'
  );
});
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
) -> FastAPI:
    """Construct and return the FastAPI retrieval application.

    All parameters fall back to environment variables so the app can be
    started from the CLI without explicit arguments.
    """
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
        state["assembler"] = ProvenanceContextAssembler(executor)
        state["gateway"] = EntityResolutionGateway()
        state["synthesis"] = SynthesisEngine(api_key=api_key, max_tokens=max_tokens)
        yield
        executor.close()

    app = FastAPI(
        title="Turnbull GraphRAG Retrieval API",
        description="Natural-language query interface over the Turnbull refuge archive graph.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # GET / — Chat UI
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def chat_ui() -> HTMLResponse:
        return HTMLResponse(_CHAT_HTML)

    # ------------------------------------------------------------------
    # POST /query
    # ------------------------------------------------------------------
    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        gateway: EntityResolutionGateway = state["gateway"]
        assembler: ProvenanceContextAssembler = state["assembler"]
        builder: CypherQueryBuilder = state["query_builder"]
        engine: SynthesisEngine = state["synthesis"]

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
                )
            elif habitat_entities:
                analytical_result = builder.habitat_conditions(
                    habitat_id=habitat_entities[0].entity_id,
                    year_min=intent.year_min,
                    year_max=intent.year_max,
                )

        # Layer 2B: conversational path.
        blocks, context_text = assembler.assemble(
            query_text=req.text,
            entity_context=entity_ctx,
            year_min=intent.year_min,
            year_max=intent.year_max,
            is_hybrid=is_hybrid,
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
        result: SynthesisResult = engine.synthesise(
            query=req.text,
            provenance_blocks=blocks,
            context_text=context_text,
            analytical_result=analytical_result,
        )

        return QueryResponse(
            answer=result.answer,
            confidence_assessment=result.confidence_assessment,
            supporting_claim_ids=result.supporting_claim_ids,
            caveats=result.caveats,
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
        )

    # ------------------------------------------------------------------
    # POST /query/provenance
    # ------------------------------------------------------------------
    @app.post("/query/provenance")
    def query_provenance(req: ProvenanceRequest) -> dict[str, Any]:
        """Return the full raw provenance chain for a known *claim_id*.

        Does not invoke the synthesis engine — useful for citation
        verification and the review web interface.
        """
        assembler: ProvenanceContextAssembler = state["assembler"]
        blocks = assembler.chain_for_claim(req.claim_id)
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

    return app
