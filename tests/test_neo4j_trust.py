from __future__ import annotations

import gemynd.ingest.graph.writer as writer_mod
from tests.conftest import TEST_ENTITY_LABELS


class _DummyDriver:
    def __init__(self) -> None:
        self.verified = False

    def verify_connectivity(self) -> None:
        self.verified = True

    def close(self) -> None:
        return None


class _DummyGraphDatabase:
    last_call: dict[str, object] = {}

    @staticmethod
    def driver(uri, auth=None, **kwargs):
        _DummyGraphDatabase.last_call = {
            "uri": uri,
            "auth": auth,
            "kwargs": kwargs,
        }
        return _DummyDriver()


class _DummyNeo4j:
    @staticmethod
    def TrustAll():
        return "TRUST_ALL"

    @staticmethod
    def TrustSystemCAs():
        return "TRUST_SYSTEM"

    @staticmethod
    def TrustCustomCAs(path):
        return ("TRUST_CUSTOM", path)


def test_neo4j_writer_uses_trust_all_for_base_scheme(monkeypatch) -> None:
    monkeypatch.setattr(writer_mod, "GraphDatabase", _DummyGraphDatabase)
    monkeypatch.setattr(writer_mod, "neo4j_pkg", _DummyNeo4j)

    writer_mod.Neo4jGraphWriter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="pw",
        trust_mode="all",
        entity_labels=TEST_ENTITY_LABELS,
    )
    kwargs = _DummyGraphDatabase.last_call["kwargs"]
    assert kwargs["trusted_certificates"] == "TRUST_ALL"


def test_neo4j_writer_requires_ca_for_custom(monkeypatch) -> None:
    monkeypatch.setattr(writer_mod, "GraphDatabase", _DummyGraphDatabase)
    monkeypatch.setattr(writer_mod, "neo4j_pkg", _DummyNeo4j)

    try:
        writer_mod.Neo4jGraphWriter(
            uri="neo4j://localhost:7687",
            user="neo4j",
            password="pw",
            trust_mode="custom",
            ca_cert_path=None,
            entity_labels=TEST_ENTITY_LABELS,
        )
    except ValueError as exc:
        assert "NEO4J_CA_CERT" in str(exc)
    else:
        raise AssertionError("Expected ValueError when custom trust mode has no CA certificate path.")


def test_neo4j_writer_ignores_trust_overrides_for_plus_ssc(monkeypatch) -> None:
    monkeypatch.setattr(writer_mod, "GraphDatabase", _DummyGraphDatabase)
    monkeypatch.setattr(writer_mod, "neo4j_pkg", _DummyNeo4j)

    writer_mod.Neo4jGraphWriter(
        uri="bolt+ssc://localhost:7687",
        user="neo4j",
        password="pw",
        trust_mode="custom",
        ca_cert_path="C:/tmp/ca.crt",
        entity_labels=TEST_ENTITY_LABELS,
    )
    kwargs = _DummyGraphDatabase.last_call["kwargs"]
    assert "trusted_certificates" not in kwargs
