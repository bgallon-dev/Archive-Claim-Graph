"""Neo4j query executor — thin session wrapper for the retrieval layer.

Mirrors the driver initialisation pattern used in Neo4jGraphWriter exactly
so that the same connection parameters and TLS trust modes work unchanged.
"""
from __future__ import annotations

from typing import Any

from gemynd.ingest.graph.writer import _build_driver_kwargs, _format_connection_error

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None  # type: ignore[assignment]


class Neo4jQueryExecutor:
    """Execute read-only Cypher queries against Neo4j.

    Parameters
    ----------
    uri, user, password, database:
        Connection details — identical semantics to Neo4jGraphWriter.
    trust_mode:
        One of ``"system"`` (default), ``"all"``, or ``"custom"``.
    ca_cert_path:
        Path to CA certificate when *trust_mode* is ``"custom"``.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        trust_mode: str = "system",
        ca_cert_path: str | None = None,
        *,
        entity_labels: frozenset[str] | None = None,
    ) -> None:
        if GraphDatabase is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "neo4j package is not installed. Install with: pip install -e .[retrieval]"
            )
        from ..core.graph.cypher import (
            INDEX_STATEMENTS,
            build_constraint_statements,
        )

        driver_kwargs = _build_driver_kwargs(uri=uri, trust_mode=trust_mode, ca_cert_path=ca_cert_path)
        self._driver = GraphDatabase.driver(uri, auth=(user, password), **driver_kwargs)
        self._database = database
        self._uri = uri
        # entity_labels is optional at construction so read-only callers
        # (duplicate-hash check, graph-entity fetch, ad-hoc queries) don't
        # need to load a full DomainConfig just to run a SELECT. It becomes
        # required only when ensure_schema() is called.
        self._entity_labels: frozenset[str] | None = (
            frozenset(entity_labels) if entity_labels is not None else None
        )
        if self._entity_labels is not None:
            self._schema_statements = (
                build_constraint_statements(self._entity_labels) + INDEX_STATEMENTS
            )
        else:
            self._schema_statements = ()

        try:
            self._driver.verify_connectivity()
        except Exception as exc:  # pragma: no cover - requires neo4j runtime
            message = _format_connection_error(uri, exc)
            raise RuntimeError(message) from exc

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute *cypher* and return results as a list of plain dicts.

        Each dict maps column name to value exactly as Neo4j returns it.
        Node objects are converted to their property dicts via ``data()``.
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, params or {})
            rows: list[dict[str, Any]] = []
            for record in result:
                row: dict[str, Any] = {}
                for key, value in record.items():
                    # Neo4j node/relationship objects expose .data(); plain
                    # Python types (int, str, list, None) pass through as-is.
                    if hasattr(value, "data"):
                        row[key] = dict(value.data())
                    elif isinstance(value, list):
                        row[key] = [
                            dict(v.data()) if hasattr(v, "data") else v for v in value
                        ]
                    else:
                        row[key] = value
                rows.append(row)
            return rows

    def ensure_schema(self) -> None:
        """Apply schema constraints and indexes to the connected database.

        All statements use IF NOT EXISTS so this is idempotent — safe to
        call on every server startup.
        """
        if self._entity_labels is None:
            raise RuntimeError(
                "ensure_schema() requires entity_labels to be passed at "
                "construction time; load a DomainConfig and pass "
                "entity_labels=config.entity_labels"
            )
        with self._driver.session(database=self._database) as session:
            for stmt in self._schema_statements:
                session.run(stmt)

    def close(self) -> None:
        self._driver.close()
