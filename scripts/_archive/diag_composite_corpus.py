"""Diagnostic: does the web app's boot-time composite actually include Spokane?

Loads the composite exactly the way the retrieval web app does, then runs
CORPUS_STATS_QUERY with the same parameters, plus a raw per-institution
breakdown. Run while the server is up OR down — Neo4j handles concurrent reads.
"""
from __future__ import annotations

from gemynd.shared.env import load_dotenv

load_dotenv()

from gemynd.core.domain_config import load_all_registered_corpora, merge_domain_configs
from gemynd.core.graph.cypher import CORPUS_STATS_QUERY
from gemynd.retrieval.executor import Neo4jQueryExecutor
from gemynd.shared.settings import Settings


def main() -> None:
    comp = merge_domain_configs(load_all_registered_corpora())
    print("composite institution_ids:", comp.institution_ids)
    print("member count:", len(comp.members))
    for m in comp.members:
        print(f"  - {m.institution_id}  (anchor={m.anchor_entity_id})")
    print("seed_entities (union):", len(comp.seed_entities))

    s = Settings.from_env()
    ex = Neo4jQueryExecutor(
        uri=s.neo4j_uri,
        user=s.neo4j_user,
        password=s.neo4j_password,
        database=s.neo4j_database,
        entity_labels=comp.entity_labels,
    )
    try:
        print("\n--- CORPUS_STATS_QUERY with composite list ---")
        print(ex.run(CORPUS_STATS_QUERY, {"institution_ids": comp.institution_ids}))

        print("\n--- CORPUS_STATS_QUERY with ['spokane_newspaper'] only ---")
        print(ex.run(CORPUS_STATS_QUERY, {"institution_ids": ["spokane_newspaper"]}))

        print("\n--- CORPUS_STATS_QUERY with ['turnbull'] only ---")
        print(ex.run(CORPUS_STATS_QUERY, {"institution_ids": ["turnbull"]}))

        print("\n--- Raw per-institution breakdown (all live docs) ---")
        rows = ex.run(
            "MATCH (d:Document) WHERE d.deleted_at IS NULL "
            "RETURN d.institution_id AS inst, count(d) AS n ORDER BY n DESC",
            {},
        )
        for row in rows:
            print(f"  {row['inst']}: {row['n']}")
    finally:
        ex.close()


if __name__ == "__main__":
    main()
