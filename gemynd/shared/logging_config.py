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
from logging.handlers import RotatingFileHandler
from pathlib import Path


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


def setup_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Configure the root logger with a structured formatter.

    Parameters
    ----------
    level:
        Override log level string (e.g. "DEBUG", "INFO").  Falls back to the
        LOG_LEVEL environment variable, then to "INFO".
    log_file:
        Path to a rotating log file.  Falls back to the LOG_FILE environment
        variable; when neither is set, logs go to stderr only.

    Notes
    -----
    Set ``LOG_FILE`` in any deployment you expect to debug after the fact.
    Stderr dies with the terminal, so without it a report of "it broke on
    Tuesday" has no forensics behind it at all.  Rotation is capped
    (``LOG_MAX_BYTES`` x ``LOG_BACKUP_COUNT``, default 5 x 10 MB) so logs
    can't fill the disk.
    """
    resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime  # UTC timestamps

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    resolved_file = log_file or os.environ.get("LOG_FILE")
    if resolved_file:
        try:
            path = Path(resolved_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(
                    str(path),
                    maxBytes=int(os.environ.get("LOG_MAX_BYTES", 10 * 1024 * 1024)),
                    backupCount=int(os.environ.get("LOG_BACKUP_COUNT", 5)),
                    encoding="utf-8",
                )
            )
        except OSError:
            # An unwritable log path must not prevent the server from starting;
            # stderr logging still applies and the failure is reported there.
            logging.getLogger(__name__).warning(
                "Could not open log file %r — continuing with stderr only",
                resolved_file, exc_info=True,
            )

    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(JWTRedactionFilter())
        root.addHandler(handler)
    root.setLevel(numeric_level)
