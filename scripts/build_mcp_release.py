"""Build a small, secret-free, reproducible MCP application archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    Path("gemynd"),
    Path("deploy/gemynd-mcp.service"),
    Path("deploy/nginx-kettle-mcp.conf"),
    Path("deploy/kettle.env.example"),
    Path("deploy/publish_snapshot.sh"),
    Path("deploy/rollback_snapshot.sh"),
    Path("deploy/cleanup-snapshots.sh"),
    Path("deploy/backup.sh"),
    Path("deploy/restore-test.sh"),
    Path("deploy/prometheus-alerts.yml"),
    Path("deploy/uptime-monitor.example.yml"),
    Path("scripts/validate_evidence_snapshot.py"),
    Path("docs/mcp.md"),
    Path("docs/MCP-RUNBOOK.md"),
    Path("docs/production-readiness-checklist.md"),
    Path("deploy/kettle-corpus-registry.yaml"),
    Path("pyproject.toml"),
    Path("README.md"),
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _copy_release_tree(destination: Path) -> list[Path]:
    copied: list[Path] = []
    for relative in INCLUDE:
        source = REPO_ROOT / relative
        if source.is_dir():
            for item in sorted(source.rglob("*")):
                if not item.is_file():
                    continue
                item_relative = item.relative_to(REPO_ROOT)
                if EXCLUDED_PARTS & set(item_relative.parts):
                    continue
                if item.suffix in EXCLUDED_SUFFIXES:
                    continue
                target = destination / item_relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, target)
                copied.append(item_relative)
        elif source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(relative)
        else:
            raise FileNotFoundError(f"required release input is missing: {relative}")
    return sorted(set(copied))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_archive(source: Path, archive: Path, root_name: str) -> None:
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as tar,
    ):
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source)
            info = tar.gettarinfo(str(path), arcname=str(Path(root_name) / relative))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                tar.addfile(info, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Build a marked development archive from a dirty working tree.",
    )
    args = parser.parse_args()
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise SystemExit("refusing to build a production release from a dirty working tree")
    root_name = f"gemynd-mcp-{commit[:12]}"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{root_name}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="gemynd-mcp-release-") as temp:
        release_root = Path(temp) / root_name
        release_root.mkdir()
        copied = _copy_release_tree(release_root)
        manifest = {
            "format": "gemynd-mcp-application-release-v1",
            "code_commit": commit,
            "dirty": dirty,
            "files": {relative.as_posix(): _sha256(release_root / relative) for relative in copied},
        }
        (release_root / "RELEASE-MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_archive(release_root, archive, root_name)
    checksum = _sha256(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": checksum,
                "size_bytes": archive.stat().st_size,
                "code_commit": commit,
                "dirty": dirty,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
