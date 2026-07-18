"""Front-end assets must be served locally, never fetched from a CDN.

The pilot runs behind an institutional firewall and the system is pitched as
airgap-capable. A blocked CDN fetch fails *silently* — htmx never loads, so
document modals, notes panels and ingest progress polling all stop working
with no error message rendered anywhere. These tests keep that from
regressing the next time someone pastes a <script src="https://..."> tag.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from gemynd.shared.static_files import STATIC_DIR, mount_static

_REPO_ROOT = Path(__file__).parent.parent
_TEMPLATE_ROOTS = [
    _REPO_ROOT / "gemynd" / "shared_templates",
    _REPO_ROOT / "gemynd" / "auth" / "templates",
    _REPO_ROOT / "gemynd" / "retrieval" / "web" / "templates",
    _REPO_ROOT / "gemynd" / "ingest" / "web" / "templates",
    _REPO_ROOT / "gemynd" / "review" / "web" / "templates",
]

# Hosts that previously served this project's front-end assets.
_CDN_HOSTS = re.compile(
    r"https?://(?:[\w.-]*\.)?"
    r"(unpkg\.com|cdnjs\.cloudflare\.com|jsdelivr\.net|"
    r"fonts\.googleapis\.com|fonts\.gstatic\.com)",
    re.IGNORECASE,
)


def _all_templates() -> list[Path]:
    found: list[Path] = []
    for root in _TEMPLATE_ROOTS:
        if root.exists():
            found.extend(root.rglob("*.html"))
    return found


class TestVendoredAssetsExist:
    def test_static_dir_exists(self):
        assert STATIC_DIR.is_dir(), f"missing static dir: {STATIC_DIR}"

    @pytest.mark.parametrize(
        "rel", ["vendor/htmx.min.js", "vendor/cytoscape.min.js", "fonts.css"]
    )
    def test_asset_present_and_nonempty(self, rel):
        asset = STATIC_DIR / rel
        assert asset.is_file(), f"missing vendored asset: {rel}"
        assert asset.stat().st_size > 0, f"empty vendored asset: {rel}"

    def test_fonts_css_references_only_local_files(self):
        css = (STATIC_DIR / "fonts.css").read_text(encoding="utf-8")
        assert not _CDN_HOSTS.search(css), "fonts.css still points at a remote host"
        # Every referenced woff2 must actually be on disk.
        for rel in re.findall(r"url\(([^)]+)\)", css):
            rel = rel.strip("'\"")
            assert rel.startswith("/static/"), f"non-local font url: {rel}"
            target = STATIC_DIR / rel[len("/static/"):]
            assert target.is_file(), f"fonts.css references missing file: {rel}"

    def test_at_least_one_font_file_shipped(self):
        assert list((STATIC_DIR / "fonts").glob("*.woff2")), "no woff2 files vendored"


class TestNoTemplateUsesACDN:
    def test_templates_were_discovered(self):
        """Guard against the sweep below passing vacuously."""
        assert len(_all_templates()) > 10

    def test_no_template_references_a_cdn(self):
        offenders: list[str] = []
        for tpl in _all_templates():
            for i, line in enumerate(tpl.read_text(encoding="utf-8").splitlines(), 1):
                if _CDN_HOSTS.search(line):
                    offenders.append(f"{tpl.relative_to(_REPO_ROOT)}:{i}: {line.strip()}")
        assert not offenders, "templates must use /static/, not a CDN:\n" + "\n".join(offenders)


class TestMountStatic:
    def test_mount_registers_static_route(self):
        fastapi = pytest.importorskip("fastapi")
        app = fastapi.FastAPI()
        mount_static(app)
        assert any(getattr(r, "name", None) == "static" for r in app.routes)

    def test_mounted_assets_are_served(self):
        fastapi = pytest.importorskip("fastapi")
        testclient = pytest.importorskip("fastapi.testclient")
        app = fastapi.FastAPI()
        mount_static(app)
        client = testclient.TestClient(app)
        for rel in ("vendor/htmx.min.js", "fonts.css"):
            resp = client.get(f"/static/{rel}")
            assert resp.status_code == 200, f"/static/{rel} -> {resp.status_code}"
            assert resp.content
