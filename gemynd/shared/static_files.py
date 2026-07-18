"""Shared static-asset mount for the three web apps.

Every vendored front-end asset (htmx, Cytoscape, the two web fonts) is served
from this package rather than a CDN. The pilot runs behind an institutional
firewall and the system is pitched as airgap-capable; a blocked CDN fetch
fails *silently*, which would leave htmx-driven modals, notes panels and
progress polling dead with no error message anywhere on the page.

Mount once per app via :func:`mount_static`, then reference assets as
``/static/vendor/htmx.min.js`` and ``/static/fonts.css``.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent.parent / "shared_static"


def mount_static(app: FastAPI, *, path: str = "/static") -> None:
    """Mount the vendored static assets at *path* (idempotent per app)."""
    app.mount(path, StaticFiles(directory=str(STATIC_DIR)), name="static")
