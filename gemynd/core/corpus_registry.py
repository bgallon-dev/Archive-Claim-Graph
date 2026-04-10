"""YAML-backed corpus registry.

Maps corpus identifiers to their resource directory paths and metadata.
Enables multi-corpus support by providing a single lookup for corpus
configuration.

Usage::

    from gemynd.core.corpus_registry import load_registry, get_corpus

    entries = load_registry()
    entry = get_corpus("turnbull")
    resources = resolve_resources_dir("turnbull")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path("data/corpus_registry.yaml")


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """A registered corpus with its resource directory."""

    corpus_id: str
    display_name: str
    resources_dir: str
    description: str = ""
    created_at: str = ""
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "display_name": self.display_name,
            "resources_dir": self.resources_dir,
            "description": self.description,
            "created_at": self.created_at,
            "is_default": self.is_default,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusEntry:
        return cls(
            corpus_id=data["corpus_id"],
            display_name=data["display_name"],
            resources_dir=data["resources_dir"],
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            is_default=data.get("is_default", False),
        )


def _registry_path(path: Path | None = None) -> Path:
    return path or _DEFAULT_REGISTRY_PATH


def load_registry(path: Path | None = None) -> list[CorpusEntry]:
    """Load all corpus entries from the registry YAML file."""
    import yaml

    rp = _registry_path(path)
    if not rp.exists():
        _log.warning("Corpus registry not found at %s; returning empty list", rp)
        return []
    with rp.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data or not isinstance(data.get("corpora"), list):
        return []
    return [CorpusEntry.from_dict(entry) for entry in data["corpora"]]


def _save_registry(entries: list[CorpusEntry], path: Path | None = None) -> None:
    """Write the full registry back to YAML."""
    import yaml

    rp = _registry_path(path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    payload = {"corpora": [e.to_dict() for e in entries]}
    with rp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)


def get_corpus(corpus_id: str, path: Path | None = None) -> CorpusEntry:
    """Look up a single corpus by ID. Raises ``KeyError`` if not found."""
    for entry in load_registry(path):
        if entry.corpus_id == corpus_id:
            return entry
    raise KeyError(f"Corpus {corpus_id!r} not found in registry")


def get_default_corpus(path: Path | None = None) -> CorpusEntry:
    """Return the corpus marked ``is_default: true``.

    Raises ``KeyError`` if no default is set.
    """
    entries = load_registry(path)
    for entry in entries:
        if entry.is_default:
            return entry
    if entries:
        return entries[0]
    raise KeyError("No corpora registered")


def list_corpora(path: Path | None = None) -> list[CorpusEntry]:
    """Return all registered corpora."""
    return load_registry(path)


def register_corpus(entry: CorpusEntry, path: Path | None = None) -> None:
    """Append a new corpus to the registry. Raises if ID already exists."""
    entries = load_registry(path)
    for existing in entries:
        if existing.corpus_id == entry.corpus_id:
            raise ValueError(f"Corpus {entry.corpus_id!r} already registered")
    entries.append(entry)
    _save_registry(entries, path)
    _log.info("Registered corpus %r", entry.corpus_id)


def set_default(corpus_id: str, path: Path | None = None) -> None:
    """Mark a corpus as the default, clearing the flag on all others."""
    entries = load_registry(path)
    found = False
    updated: list[CorpusEntry] = []
    for entry in entries:
        if entry.corpus_id == corpus_id:
            found = True
            updated.append(CorpusEntry(
                corpus_id=entry.corpus_id,
                display_name=entry.display_name,
                resources_dir=entry.resources_dir,
                description=entry.description,
                created_at=entry.created_at,
                is_default=True,
            ))
        elif entry.is_default:
            updated.append(CorpusEntry(
                corpus_id=entry.corpus_id,
                display_name=entry.display_name,
                resources_dir=entry.resources_dir,
                description=entry.description,
                created_at=entry.created_at,
                is_default=False,
            ))
        else:
            updated.append(entry)
    if not found:
        raise KeyError(f"Corpus {corpus_id!r} not found in registry")
    _save_registry(updated, path)
    _log.info("Set default corpus to %r", corpus_id)


def resolve_resources_dir(corpus_id: str, path: Path | None = None) -> Path:
    """Return the resolved ``Path`` for a corpus's resources directory."""
    entry = get_corpus(corpus_id, path)
    return Path(entry.resources_dir)


def utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
