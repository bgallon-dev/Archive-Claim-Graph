"""Centralised logging configuration for the gemynd package.

Call setup_logging() once at application startup (in the lifespan handler
or CLI entry point).  Never call it at module import time — importing this
module must be side-effect free so tests can import pipeline code without
forcing log configuration.
"""
from __future__ import annotations

import logging
import os
import re
import time


# Matches the canonical 3-part JWT shape (eyJ-prefixed base64url segments
# joined by dots).  Greedy enough to catch tokens embedded in larger strings
# (e.g. cookie headers); narrow enough that ordinary log content won't match.
# False-positive cost is low: the only failure mode is a redacted log line
# that confuses debugging.  Don't try to "fix" the rare false positive on a
# password starting with "eyJ" and dots — over-redaction is the safe default.
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")


def _redact(value):
    return _JWT_RE.sub("[REDACTED-JWT]", value) if isinstance(value, str) else value


class JWTRedactionFilter(logging.Filter):
    """Strip JWT-shaped substrings from log records before emission."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _redact(record.msg)
        # record.args is a tuple for positional %s formatting and a dict for
        # named %(key)s formatting — handle both shapes.
        if isinstance(record.args, tuple):
            record.args = tuple(_redact(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: _redact(v) for k, v in record.args.items()}
        return True


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger with a structured formatter.

    Parameters
    ----------
    level:
        Override log level string (e.g. "DEBUG", "INFO").  Falls back to the
        LOG_LEVEL environment variable, then to "INFO".
    """
    resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime  # UTC timestamps

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(JWTRedactionFilter())

    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)
