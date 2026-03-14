import os

from graphrag_pipeline.env import load_dotenv


def test_load_dotenv_reads_values(tmp_path) -> None:
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
    os.environ.pop("NEO4J_URI", None)
    os.environ.pop("NEO4J_USER", None)
    os.environ.pop("NEO4J_PASSWORD", None)
    os.environ.pop("NEO4J_DATABASE", None)

    loaded = load_dotenv(env_file)
    assert loaded["NEO4J_URI"].startswith("neo4j+s://")
    assert os.environ["NEO4J_USER"] == "neo4j"
    assert os.environ["NEO4J_PASSWORD"] == "secret value"
    assert os.environ["NEO4J_DATABASE"] == "neo4j"


def test_load_dotenv_does_not_override_existing(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("NEO4J_USER=from-file", encoding="utf-8")
    os.environ["NEO4J_USER"] = "from-env"

    loaded = load_dotenv(env_file, override=False)
    assert "NEO4J_USER" not in loaded
    assert os.environ["NEO4J_USER"] == "from-env"
