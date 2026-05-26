"""Neo4j connection helpers — single source of construction for retrieval executors."""
from __future__ import annotations

from typing import TYPE_CHECKING

from gemynd.shared.settings import Settings

if TYPE_CHECKING:
    from gemynd.retrieval.executor import Neo4jQueryExecutor


def build_neo4j_executor(
    settings: Settings,
    *,
    entity_labels: frozenset[str] | None = None,
) -> "Neo4jQueryExecutor":
    from gemynd.retrieval.executor import Neo4jQueryExecutor

    return Neo4jQueryExecutor(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        trust_mode=settings.neo4j_trust,
        ca_cert_path=settings.neo4j_ca_cert,
        entity_labels=entity_labels,
    )
