"""Resolve the originating client IP for rate-limit and audit purposes.

Honours X-Forwarded-For only when the direct peer is in the trusted-proxy
list (TRUSTED_PROXY_IPS, comma-separated).  Default empty list => trust
nothing — the direct peer is always returned, which is safe in local dev
and bare-metal deploys where there is no proxy in front of the app.
"""
from __future__ import annotations

import os

from fastapi import Request


def client_ip(request: Request) -> str:
    direct = request.client.host if request.client else ""
    trusted = {
        p.strip()
        for p in os.environ.get("TRUSTED_PROXY_IPS", "").split(",")
        if p.strip()
    }
    if direct in trusted:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    return direct
