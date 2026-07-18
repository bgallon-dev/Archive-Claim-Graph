"""Logs must survive the terminal that started the process.

setup_logging() previously installed only a StreamHandler, so everything went
to stderr and died with the shell. A tester reporting "it broke Tuesday" left
no forensics at all. LOG_FILE adds rotating file output alongside stderr.
"""
from __future__ import annotations

import logging

import pytest

from gemynd.shared.logging_config import setup_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    """Root logger is process-global; put it back for the rest of the suite."""
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
        if hasattr(h, "close"):
            h.close()
    for h in saved:
        root.addHandler(h)
    root.setLevel(level)


def _file_handlers():
    from logging.handlers import RotatingFileHandler

    return [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]


class TestLogFile:
    def test_no_file_handler_by_default(self, monkeypatch):
        monkeypatch.delenv("LOG_FILE", raising=False)
        setup_logging()
        assert not _file_handlers()

    def test_writes_to_configured_file(self, tmp_path, monkeypatch):
        log_path = tmp_path / "gemynd.log"
        monkeypatch.setenv("LOG_FILE", str(log_path))
        setup_logging()
        logging.getLogger("gemynd.test").warning("canary-message")
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_path.exists()
        assert "canary-message" in log_path.read_text(encoding="utf-8")

    def test_parent_directory_is_created(self, tmp_path, monkeypatch):
        log_path = tmp_path / "nested" / "dir" / "gemynd.log"
        monkeypatch.setenv("LOG_FILE", str(log_path))
        setup_logging()
        logging.getLogger("gemynd.test").warning("hello")
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_path.exists()

    def test_stderr_handler_is_kept_alongside_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_FILE", str(tmp_path / "g.log"))
        setup_logging()
        handlers = logging.getLogger().handlers
        assert any(type(h) is logging.StreamHandler for h in handlers)
        assert len(_file_handlers()) == 1

    def test_rotation_is_bounded(self, tmp_path, monkeypatch):
        """Unbounded logs on a small disk is its own outage."""
        monkeypatch.setenv("LOG_FILE", str(tmp_path / "g.log"))
        monkeypatch.setenv("LOG_MAX_BYTES", "2048")
        monkeypatch.setenv("LOG_BACKUP_COUNT", "2")
        setup_logging()
        handler = _file_handlers()[0]
        assert handler.maxBytes == 2048
        assert handler.backupCount == 2

    def test_unwritable_path_does_not_prevent_startup(self, tmp_path, monkeypatch):
        """A bad log path must never be the reason the server won't boot."""
        blocker = tmp_path / "afile"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setenv("LOG_FILE", str(blocker / "nested" / "g.log"))
        setup_logging()  # must not raise
        logging.getLogger("gemynd.test").warning("still logging")

    def test_jwt_redaction_applies_to_file_output(self, tmp_path, monkeypatch):
        """The redaction filter must cover the new handler, not just stderr."""
        log_path = tmp_path / "g.log"
        monkeypatch.setenv("LOG_FILE", str(log_path))
        setup_logging()
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abc-DEF_123"
        logging.getLogger("gemynd.test").warning("auth header: %s", token)
        for h in logging.getLogger().handlers:
            h.flush()
        contents = log_path.read_text(encoding="utf-8")
        assert token not in contents
        assert "[REDACTED-JWT]" in contents
