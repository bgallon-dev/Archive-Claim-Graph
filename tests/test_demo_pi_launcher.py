from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pi_launcher_targets_direct_newspapers_database() -> None:
    launcher = (ROOT / "demo-pi.cmd").read_text(encoding="utf-8")

    assert 'bolt://pi5node1.local:7687' in launcher
    assert '-Neo4jDatabase "newspapers"' in launcher
    assert '-EnvOverlay "%~dp0.env.pi"' in launcher
    assert "neo4j://" not in launcher


def test_pi_credentials_are_ignored_and_not_committed() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    example = (ROOT / ".env.pi.example").read_text(encoding="utf-8")

    assert ".env.pi" in gitignore
    assert "replace-with-pi-neo4j-password" in example
    assert "bb86wp7umkb2adfpc96a" not in example


def test_demo_preflight_uses_configured_neo4j_endpoint() -> None:
    script = (ROOT / "scripts" / "demo_serve.ps1").read_text(encoding="utf-8")

    assert "[string] $Neo4jUri" in script
    assert "[string] $Neo4jDatabase" in script
    assert "[string] $EnvOverlay" in script
    assert "Test-NetConnection -ComputerName $neo4jHost -Port $neo4jPort" in script
    assert '$env:NEO4J_URI = $Neo4jUri' in script
    assert '$env:NEO4J_DATABASE = $Neo4jDatabase' in script


def test_local_clone_skips_expensive_path_cache_warmup() -> None:
    launcher = (ROOT / "demo-local-pi5.cmd").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "demo_serve.ps1").read_text(encoding="utf-8")

    assert 'bolt://127.0.0.1:7689' in launcher
    assert "-SkipPathCacheWarmup" in launcher
    assert "[switch] $SkipPathCacheWarmup" in script
    assert "$env:GEMYND_SKIP_PATH_CACHE_WARMUP" in script


def test_demo_defaults_use_api_synthesis() -> None:
    script = (ROOT / "scripts" / "demo_serve.ps1").read_text(encoding="utf-8")

    assert "[int]    $MaxTokens = 24000" in script
    assert '$env:SYNTHESIS_PROVIDER = "anthropic"' in script
    assert "Anthropic API synthesis configured" in script
    assert "local-inference-disabled" not in script


def test_demo_configures_fallback_providers() -> None:
    script = (ROOT / "scripts" / "demo_serve.ps1").read_text(encoding="utf-8")

    # opencode primary -> Anthropic fallback; anthropic primary -> opencode fallback.
    assert '$env:SYNTHESIS_FALLBACK_PROVIDERS = "opencode"' in script
    assert '$env:SYNTHESIS_FALLBACK_PROVIDERS = "anthropic"' in script
    # Both are set from the same $Provider branch.
    assert script.index('$env:SYNTHESIS_PROVIDER = "opencode"') < script.index(
        '$env:SYNTHESIS_FALLBACK_PROVIDERS = "anthropic"'
    )


def test_demo_opencode_provider_branch_present() -> None:
    script = (ROOT / "scripts" / "demo_serve.ps1").read_text(encoding="utf-8")
    launcher = (ROOT / "demo-opencode.cmd").read_text(encoding="utf-8")

    assert '[ValidateSet("anthropic", "opencode")]' in script
    assert '$env:SYNTHESIS_PROVIDER = "opencode"' in script
    assert "opencode CLI synthesis configured" in script
    assert "-Provider opencode" in launcher
