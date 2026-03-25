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

_TEMPLATES_DIR = Path(__file__).parent / "templates"

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
    from graphrag_pipeline.shared.logging_config import setup_logging
    setup_logging()

    try:
        from fastapi import BackgroundTasks, Depends, FastAPI, Request, UploadFile
        from fastapi.responses import HTMLResponse, RedirectResponse as _Redirect
        from fastapi.templating import Jinja2Templates
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for ingest-serve. "
            "Install with: pip install -e .[ingest]"
        ) from exc

    _templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

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
    from graphrag_pipeline.ingest.pipeline import _process_single_document

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
    async def index(request: Request, _user: UserContext = Depends(require_login)):
        return _templates.TemplateResponse("upload.html", {"request": request})

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
        request: Request,
        _user: UserContext = Depends(require_login),
    ):
        job = store.get_job(job_id)
        if job is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Job not found.")
        docs = store.list_documents(job_id)
        total = job["total_docs"]
        done = job["completed_docs"] + job["failed_docs"]
        pct = int(done / total * 100) if total else 0
        is_running = job["status"] == "running"
        created = (job.get("created_at") or "")[:19].replace("T", " ")
        total_claims = sum((d.get("claims_count") or 0) for d in docs)
        total_mentions = sum((d.get("mention_count") or 0) for d in docs)
        return _templates.TemplateResponse("status_page.html", {
            "request": request,
            "job_id": job_id,
            "job": job,
            "docs": docs,
            "total": total,
            "done": done,
            "pct": pct,
            "is_running": is_running,
            "created": created,
            "total_claims": total_claims,
            "total_mentions": total_mentions,
        })

    # ------------------------------------------------------------------
    # GET /ingest/{job_id}/status — HTMX polling fragment
    # ------------------------------------------------------------------

    @app.get("/ingest/{job_id}/status", response_class=HTMLResponse)
    async def job_status_fragment(
        job_id: str,
        request: Request,
        _user: UserContext = Depends(require_login),
    ):
        job = store.get_job(job_id)
        if job is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Job not found.")
        docs = store.list_documents(job_id)
        is_running = job["status"] == "running"
        return _templates.TemplateResponse("fragments/status_fragment.html", {
            "request": request,
            "job_id": job_id,
            "is_running": is_running,
            "docs": docs,
        })

    return app
