"""FastAPI application for the document ingestion UI.

Archivists drag PDFs (or pre-converted JSON files) onto a browser page,
watch per-document progress in real time via HTMX polling, and get an
extraction summary when processing completes.

Launch via the CLI:
    graphrag ingest-serve --out-dir out/ --port 8789
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded HTML
# ---------------------------------------------------------------------------

_CSS = """
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f8f9fa; color: #212529; margin: 0; padding: 0;
  }
  .navbar {
    background: #2c3e50; color: #fff; padding: 0.75rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .navbar a { color: #adb5bd; text-decoration: none; font-size: 0.875rem; }
  .navbar a:hover { color: #fff; }
  .navbar-brand { font-weight: 600; font-size: 1rem; color: #fff; }
  .container { max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; }
  h1 { font-size: 1.5rem; font-weight: 600; margin: 0 0 0.25rem; }
  .subtitle { color: #6c757d; font-size: 0.9rem; margin: 0 0 1.5rem; }
  .card {
    background: #fff; border: 1px solid #dee2e6; border-radius: 8px;
    padding: 1.5rem; margin-bottom: 1.5rem;
  }
  /* Drop zone */
  #drop-zone {
    border: 2px dashed #adb5bd; border-radius: 8px; padding: 3rem 2rem;
    text-align: center; cursor: pointer; transition: border-color 0.15s, background 0.15s;
  }
  #drop-zone.drag-over { border-color: #0d6efd; background: #e9f0ff; }
  #drop-zone svg { display: block; margin: 0 auto 1rem; color: #adb5bd; }
  #drop-zone p { margin: 0 0 0.5rem; font-size: 1rem; color: #495057; }
  #drop-zone .hint { font-size: 0.8rem; color: #6c757d; margin: 0; }
  #file-input { display: none; }
  #file-list { margin-top: 1rem; }
  .file-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #e9ecef; border-radius: 4px; padding: 0.25rem 0.6rem;
    font-size: 0.8rem; margin: 0.25rem 0.25rem 0 0;
  }
  .file-chip .remove { cursor: pointer; color: #6c757d; line-height: 1; }
  .file-chip .remove:hover { color: #dc3545; }
  /* Buttons */
  .btn {
    display: inline-block; padding: 0.5rem 1.25rem; border-radius: 5px;
    font-size: 0.9rem; font-weight: 500; cursor: pointer; border: none;
    text-decoration: none; transition: opacity 0.15s;
  }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: #0d6efd; color: #fff; }
  .btn-primary:hover:not(:disabled) { background: #0b5ed7; }
  .btn-secondary { background: #6c757d; color: #fff; }
  .btn-secondary:hover { background: #5c636a; }
  /* Status badges */
  .badge {
    display: inline-block; padding: 0.2em 0.6em; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; white-space: nowrap;
  }
  .badge-queued    { background: #e9ecef; color: #495057; }
  .badge-converting{ background: #fff3cd; color: #664d03; }
  .badge-extracting{ background: #cfe2ff; color: #084298; }
  .badge-completed { background: #d1e7dd; color: #0a3622; }
  .badge-failed    { background: #f8d7da; color: #58151c; }
  /* Progress */
  .progress { height: 8px; background: #dee2e6; border-radius: 4px; overflow: hidden; }
  .progress-bar { height: 100%; background: #0d6efd; transition: width 0.3s; }
  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  th { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 2px solid #dee2e6;
       color: #6c757d; font-weight: 600; white-space: nowrap; }
  td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #f1f3f5; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  .error-detail { font-size: 0.75rem; color: #842029; margin-top: 0.2rem;
                  font-family: monospace; word-break: break-word; }
  /* Summary box */
  .summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem; margin-top: 1rem;
  }
  .summary-stat { text-align: center; }
  .summary-stat .number { font-size: 2rem; font-weight: 700; color: #0d6efd; }
  .summary-stat .label { font-size: 0.8rem; color: #6c757d; }
  .alert { padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem; }
  .alert-danger { background: #f8d7da; border: 1px solid #f5c2c7; color: #842029; }
  .alert-success{ background: #d1e7dd; border: 1px solid #badbcc; color: #0a3622; }
</style>
"""

_UPLOAD_HTML = (
    "<!doctype html><html lang='en'><head>"
    "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>Ingest Documents — GraphRAG</title>"
    + _CSS +
    "</head><body>"
    "<nav class='navbar'>"
    "<span class='navbar-brand'>GraphRAG</span>"
    "<a href='/'>Ingest</a>"
    "</nav>"
    "<div class='container'>"
    "<h1>Ingest Documents</h1>"
    "<p class='subtitle'>Drag PDF or JSON files onto the zone below, then click Upload.</p>"
    "<div class='card'>"
    "<form id='upload-form' method='post' action='/ingest' enctype='multipart/form-data'>"
    "<div id='drop-zone' onclick='document.getElementById(\"file-input\").click()'>"
    "<svg width='48' height='48' fill='none' stroke='currentColor' stroke-width='1.5' viewBox='0 0 24 24'>"
    "<path stroke-linecap='round' stroke-linejoin='round' "
    "d='M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5'/>"
    "</svg>"
    "<p>Drop PDF or JSON files here, or click to browse</p>"
    "<p class='hint'>Accepted formats: .pdf, .json</p>"
    "</div>"
    "<input id='file-input' type='file' name='files' multiple accept='.pdf,.json'>"
    "<div id='file-list'></div>"
    "<div style='margin-top:1.25rem;display:flex;gap:0.75rem;align-items:center'>"
    "<button type='submit' class='btn btn-primary' id='submit-btn' disabled>"
    "Upload &amp; Process"
    "</button>"
    "<span id='file-count' style='font-size:0.85rem;color:#6c757d'></span>"
    "</div>"
    "</form>"
    "</div>"
    "</div>"
    "<script>"
    "(function(){"
    "const zone=document.getElementById('drop-zone');"
    "const inp=document.getElementById('file-input');"
    "const list=document.getElementById('file-list');"
    "const btn=document.getElementById('submit-btn');"
    "const counter=document.getElementById('file-count');"
    "let selectedFiles=[];"
    "function renderFiles(){"
    "list.innerHTML='';"
    "selectedFiles.forEach(function(f,i){"
    "const chip=document.createElement('span');"
    "chip.className='file-chip';"
    "chip.innerHTML=f.name+' <span class=\"remove\" data-i=\"'+i+'\">&times;</span>';"
    "list.appendChild(chip);"
    "});"
    "list.querySelectorAll('.remove').forEach(function(el){"
    "el.addEventListener('click',function(e){"
    "e.stopPropagation();"
    "selectedFiles.splice(parseInt(this.dataset.i),1);"
    "syncInput();renderFiles();"
    "});"
    "});"
    "btn.disabled=selectedFiles.length===0;"
    "counter.textContent=selectedFiles.length>0?selectedFiles.length+' file(s) selected':'';"
    "}"
    "function syncInput(){"
    "const dt=new DataTransfer();"
    "selectedFiles.forEach(function(f){dt.items.add(f);});"
    "inp.files=dt.files;"
    "}"
    "inp.addEventListener('change',function(){"
    "Array.from(this.files).forEach(function(f){"
    "if(!selectedFiles.find(function(x){return x.name===f.name;})){"
    "selectedFiles.push(f);"
    "}"
    "});"
    "syncInput();renderFiles();"
    "});"
    "zone.addEventListener('dragover',function(e){e.preventDefault();zone.classList.add('drag-over');});"
    "zone.addEventListener('dragleave',function(){zone.classList.remove('drag-over');});"
    "zone.addEventListener('drop',function(e){"
    "e.preventDefault();zone.classList.remove('drag-over');"
    "Array.from(e.dataTransfer.files).forEach(function(f){"
    "if(!selectedFiles.find(function(x){return x.name===f.name;})){"
    "selectedFiles.push(f);"
    "}"
    "});"
    "syncInput();renderFiles();"
    "});"
    "})()"
    "</script>"
    "</body></html>"
)


def _status_page(job: dict[str, Any], docs: list[dict[str, Any]]) -> str:
    job_id = job["job_id"]
    total = job["total_docs"]
    done = job["completed_docs"] + job["failed_docs"]
    pct = int(done / total * 100) if total else 0
    is_running = job["status"] == "running"

    created = (job.get("created_at") or "")[:19].replace("T", " ")

    status_label = (
        "<span class='badge badge-extracting'>Running</span>"
        if is_running
        else "<span class='badge badge-completed'>Complete</span>"
        if job["failed_docs"] == 0
        else "<span class='badge badge-failed'>Completed with errors</span>"
    )

    # Fragment div: includes polling trigger only while job is running
    poll_attrs = (
        f" hx-get='/ingest/{job_id}/status'"
        " hx-trigger='every 2s'"
        " hx-swap='outerHTML'"
        if is_running
        else ""
    )

    rows_html = _status_rows(docs)

    summary_html = ""
    if not is_running:
        total_claims = sum((d.get("claims_count") or 0) for d in docs)
        total_mentions = sum((d.get("mention_count") or 0) for d in docs)
        failed = job["failed_docs"]
        summary_html = (
            "<div class='card' style='margin-top:0'>"
            "<h2 style='font-size:1.1rem;margin:0 0 1rem'>Extraction summary</h2>"
            "<div class='summary-grid'>"
            f"<div class='summary-stat'><div class='number'>{job['completed_docs']}</div>"
            f"<div class='label'>Documents ingested</div></div>"
            f"<div class='summary-stat'><div class='number'>{total_claims}</div>"
            f"<div class='label'>Claims extracted</div></div>"
            f"<div class='summary-stat'><div class='number'>{total_mentions}</div>"
            f"<div class='label'>Mentions resolved</div></div>"
            + (
                f"<div class='summary-stat'><div class='number' style='color:#dc3545'>{failed}</div>"
                f"<div class='label'>Failed (flag for follow-up)</div></div>"
                if failed
                else ""
            )
            + "</div>"
            + (
                "<p style='margin:1rem 0 0;font-size:0.85rem;color:#6c757d'>"
                "Failed documents are listed above with error details. "
                "Re-upload them individually once the issue is resolved."
                "</p>"
                if failed
                else ""
            )
            + "</div>"
        )

    return (
        "<!doctype html><html lang='en'><head>"
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Job {job_id[:8]}… — GraphRAG Ingest</title>"
        + _CSS +
        "<script src='https://unpkg.com/htmx.org@1.9.12' "
        "integrity='sha384-ujb1lZYygJmzgSwoxRggbCHcjc0rB2uodhFl9m+DEk9HNyNnBj5ld7DLUmKpZ3Q' "
        "crossorigin='anonymous'></script>"
        "</head><body>"
        "<nav class='navbar'>"
        "<span class='navbar-brand'>GraphRAG</span>"
        "<a href='/'>&#8592; New batch</a>"
        "</nav>"
        "<div class='container'>"
        f"<h1>Ingest job <code style='font-size:1rem'>{job_id[:12]}…</code></h1>"
        f"<p class='subtitle'>Started {created} &nbsp;·&nbsp; {total} document(s) &nbsp;·&nbsp; {status_label}</p>"
        "<div class='card'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem'>"
        f"<span style='font-size:0.85rem;color:#6c757d'>{done} of {total} processed</span>"
        f"<span style='font-size:0.85rem;color:#6c757d'>{pct}%</span>"
        "</div>"
        f"<div class='progress'><div class='progress-bar' style='width:{pct}%'></div></div>"
        "</div>"
        "<div class='card'>"
        "<h2 style='font-size:1.1rem;margin:0 0 1rem'>Documents</h2>"
        f"<div id='doc-status'{poll_attrs}>"
        "<table><thead><tr>"
        "<th>File</th><th>Status</th><th>Claims</th><th>Mentions</th><th>Notes</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
        "</div>"
        + summary_html +
        "</div>"
        "</body></html>"
    )


def _status_fragment(job: dict[str, Any], docs: list[dict[str, Any]]) -> str:
    """HTMX-targeted fragment that replaces #doc-status on each poll."""
    job_id = job["job_id"]
    is_running = job["status"] == "running"

    poll_attrs = (
        f" hx-get='/ingest/{job_id}/status'"
        " hx-trigger='every 2s'"
        " hx-swap='outerHTML'"
        if is_running
        else ""
    )

    rows_html = _status_rows(docs)
    return (
        f"<div id='doc-status'{poll_attrs}>"
        "<table><thead><tr>"
        "<th>File</th><th>Status</th><th>Claims</th><th>Mentions</th><th>Notes</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
    )


_STATUS_BADGE = {
    "queued":     "<span class='badge badge-queued'>Queued</span>",
    "converting": "<span class='badge badge-converting'>Converting</span>",
    "extracting": "<span class='badge badge-extracting'>Extracting</span>",
    "completed":  "<span class='badge badge-completed'>Done</span>",
    "failed":     "<span class='badge badge-failed'>Failed</span>",
}


def _status_rows(docs: list[dict[str, Any]]) -> str:
    rows = []
    for d in docs:
        status = d.get("status", "queued")
        badge = _STATUS_BADGE.get(status, f"<span class='badge badge-queued'>{status}</span>")
        claims = d.get("claims_count")
        mentions = d.get("mention_count")
        claims_cell = str(claims) if claims is not None else "—"
        mentions_cell = str(mentions) if mentions is not None else "—"
        notes = ""
        if d.get("error_message"):
            msg = str(d["error_message"])[:200]
            notes = f"<div class='error-detail'>{msg}</div>"
        rows.append(
            f"<tr>"
            f"<td><code style='font-size:0.8rem'>{d['filename']}</code></td>"
            f"<td>{badge}</td>"
            f"<td>{claims_cell}</td>"
            f"<td>{mentions_cell}</td>"
            f"<td>{notes}</td>"
            f"</tr>"
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    out_dir: str,
    db_path: str = "data/ingest_jobs.db",
    users_db_path: str = "data/users.db",
    review_out_dir: str | None = None,
) -> Any:
    """Create and return the ingest FastAPI application.

    Requires ``fastapi``, ``uvicorn``, and ``python-multipart`` to be installed.
    """
    from graphrag_pipeline.logging_config import setup_logging
    setup_logging()

    try:
        from fastapi import BackgroundTasks, Depends, FastAPI, Request, UploadFile
        from fastapi.responses import HTMLResponse, RedirectResponse as _Redirect
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for ingest-serve. "
            "Install with: pip install -e .[ingest]"
        ) from exc

    from graphrag_pipeline.auth.dependencies import (
        NeedsLoginException,
        require_archivist_or_admin,
        require_login,
    )
    from graphrag_pipeline.auth.models import UserContext
    from graphrag_pipeline.auth.router import create_auth_router
    from graphrag_pipeline.auth.setup import is_setup_needed
    from graphrag_pipeline.ingest.pdf_converter import ConversionError, convert_pdf_to_json
    from graphrag_pipeline.ingest.store import IngestStore
    from graphrag_pipeline.pipeline import _process_single_document

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    store = IngestStore(db_path)

    app = FastAPI(title="GraphRAG Ingest UI")
    app.include_router(create_auth_router(users_db_path), prefix="/auth")

    @app.exception_handler(NeedsLoginException)
    async def _needs_login(request: Request, exc: NeedsLoginException):
        return _Redirect(url=exc.redirect_url, status_code=303)

    from fastapi import HTTPException as _HTTPException
    from fastapi.responses import JSONResponse as _JSONResponse

    @app.exception_handler(Exception)
    async def _internal_error(request: Request, exc: Exception) -> _JSONResponse:
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
        exempt = request.url.path.startswith("/auth/setup") or request.url.path == "/health"
        if not exempt and is_setup_needed(users_db_path):
            return _Redirect(url="/auth/setup", status_code=303)
        return await call_next(request)

    # ------------------------------------------------------------------
    # Background job processor
    # ------------------------------------------------------------------

    async def _run_job(
        job_id: str,
        staged_files: list[tuple[Path, str]],  # (file_path, original_filename)
        staging_dir: Path,
    ) -> None:
        """Process each uploaded file through PDF conversion then the pipeline."""
        for file_path, orig_name in staged_files:
            try:
                if file_path.suffix.lower() == ".pdf":
                    store.update_document_status(job_id, orig_name, "converting")
                    json_path = await asyncio.to_thread(
                        convert_pdf_to_json, file_path, staging_dir
                    )
                else:
                    # Pre-converted JSON — skip the conversion step.
                    json_path = file_path

                store.update_document_status(job_id, orig_name, "extracting")
                result = await asyncio.to_thread(
                    _process_single_document,
                    str(json_path),
                    str(out_dir_path),
                    review_out_dir,
                )

                quality = result.get("quality", {})
                store.update_document_status(
                    job_id,
                    orig_name,
                    "completed",
                    output_dir=result.get("structure_output"),
                    claims_count=quality.get("claim_count"),
                    mention_count=quality.get("mention_count"),
                )

            except ConversionError as exc:
                _log.error("PDF conversion failed for %r: %s", orig_name, exc)
                store.update_document_status(
                    job_id, orig_name, "failed",
                    error_message=f"Conversion error: {exc}",
                )
            except Exception as exc:
                _log.error("Extraction failed for %r: %s", orig_name, exc, exc_info=True)
                store.update_document_status(
                    job_id, orig_name, "failed",
                    error_message=str(exc),
                )

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health", include_in_schema=True)
    def health():
        from datetime import datetime as _dt, timezone as _tz
        from fastapi.responses import JSONResponse as _JSONResponse
        ts = _dt.now(_tz.utc).isoformat()
        try:
            store._connect().close()
            db_status = "connected"
        except Exception:
            db_status = "unavailable"
        db_ok = db_status == "connected"
        return _JSONResponse(
            status_code=200 if db_ok else 503,
            content={
                "status": "ok" if db_ok else "degraded",
                "ingest_db": db_status,
                "timestamp": ts,
            },
        )

    # ------------------------------------------------------------------
    # GET / — upload UI
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(_user: UserContext = Depends(require_login)) -> str:
        return _UPLOAD_HTML

    # ------------------------------------------------------------------
    # POST /ingest — receive uploads, create job, kick off background task
    # ------------------------------------------------------------------

    @app.post("/ingest")
    async def create_ingest_job(
        files: list[UploadFile],
        background_tasks: BackgroundTasks,
        _user: UserContext = Depends(require_archivist_or_admin),
    ):
        if not files:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="No files uploaded.")

        # Validate extensions up front.
        allowed = {".pdf", ".json"}
        bad = [f.filename for f in files if Path(f.filename or "").suffix.lower() not in allowed]
        if bad:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type(s): {', '.join(bad)}. Only .pdf and .json are accepted.",
            )

        staging_dir = out_dir_path / ".ingest_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        staged: list[tuple[Path, str]] = []
        for upload in files:
            orig_name = upload.filename or "unknown"
            dest = staging_dir / orig_name
            dest.write_bytes(await upload.read())
            staged.append((dest, orig_name))

        filenames = [orig for _, orig in staged]
        job_id = store.create_job(str(out_dir_path), filenames)

        background_tasks.add_task(_run_job, job_id, staged, staging_dir)

        return _Redirect(url=f"/ingest/{job_id}", status_code=303)

    # ------------------------------------------------------------------
    # GET /ingest/{job_id} — full status page
    # ------------------------------------------------------------------

    @app.get("/ingest/{job_id}", response_class=HTMLResponse)
    async def job_status_page(
        job_id: str,
        _user: UserContext = Depends(require_login),
    ) -> str:
        job = store.get_job(job_id)
        if job is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Job not found.")
        docs = store.list_documents(job_id)
        return _status_page(job, docs)

    # ------------------------------------------------------------------
    # GET /ingest/{job_id}/status — HTMX polling fragment
    # ------------------------------------------------------------------

    @app.get("/ingest/{job_id}/status", response_class=HTMLResponse)
    async def job_status_fragment(
        job_id: str,
        _user: UserContext = Depends(require_login),
    ) -> str:
        job = store.get_job(job_id)
        if job is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Job not found.")
        docs = store.list_documents(job_id)
        return _status_fragment(job, docs)

    return app
