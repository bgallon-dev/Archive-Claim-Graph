"""Centralised logging configuration for the graphrag_pipeline package.

Call setup_logging() once at application startup (in the lifespan handler
or CLI entry point).  Never call it at module import time — importing this
module must be side-effect free so tests can import pipeline code without
forcing log configuration.
"""
from __future__ import annotations

import logging
import os
import time


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

    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)
