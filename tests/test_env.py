import os

from graphrag_pipeline.shared.env import load_dotenv


_ENV_KEYS = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")


def _save_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _ENV_KEYS}


def _restore_env(saved: dict[str, str | None]) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_load_dotenv_reads_values(tmp_path) -> None:
    saved = _save_env()
    try:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "# comment",
                    "NEO4J_URI=neo4j+s://example.databases.neo4j.io",
                    "NEO4J_USER=neo4j",
                    "NEO4J_PASSWORD='secret value'",
                    "export NEO4J_DATABASE=neo4j",
                ]
            ),
            encoding="utf-8",
        )
        for k in _ENV_KEYS:
            os.environ.pop(k, None)

        loaded = load_dotenv(env_file)
        assert loaded["NEO4J_URI"].startswith("neo4j+s://")
        assert os.environ["NEO4J_USER"] == "neo4j"
        assert os.environ["NEO4J_PASSWORD"] == "secret value"
        assert os.environ["NEO4J_DATABASE"] == "neo4j"
    finally:
        _restore_env(saved)


def test_load_dotenv_does_not_override_existing(tmp_path) -> None:
    saved = _save_env()
    try:
        env_file = tmp_path / ".env"
        env_file.write_text("NEO4J_USER=from-file", encoding="utf-8")
        os.environ["NEO4J_USER"] = "from-env"

        loaded = load_dotenv(env_file, override=False)
        assert "NEO4J_USER" not in loaded
        assert os.environ["NEO4J_USER"] == "from-env"
    finally:
        _restore_env(saved)
