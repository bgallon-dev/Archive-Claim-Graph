"""Merge the duplicate Turnbull refuge anchor nodes so temporal/anchor queries work.

Symptom this fixes
------------------
Year-only and "about the refuge" queries returned confidently-wrong results (e.g.
"what was observed in 1955?" returned 1938 content). Root cause: two ``:Refuge``
nodes share ``refuge_id='turnbull_refuge'`` but have different names/normalized
forms and split the corpus:

    refuge_16ba48a60b3fcc4d  name="Turnbull"         2 docs   <- config/seed anchor
    refuge_83a540d8bc35b099  name="Turnbull Refuge"  108 docs <- ingest config-drift dup

Retrieval resolves the anchor from ``DomainConfig`` to the *seed* node ("Turnbull"),
which only has 2 docs, so the anchor-temporal query finds almost nothing and the
cascade falls back to that 2-doc/1938 slice. The 108-doc node was created by an
earlier ingest run whose anchor was named "Turnbull Refuge".

Fix: reroute every incoming relationship from the duplicate(s) onto the seed/config
anchor node, then delete the emptied duplicate. This aligns the graph with the
*current* config, so future ingests stay consistent and no config change is needed.

Safety
------
* Dry-run by default; mutates only with ``--execute``.
* Backs up every rerouted edge (source business key + rel type + props) to JSON first.
* Reroutes with ``MERGE`` (dedups, preserves props via ``ON CREATE SET``).
* Verifies the duplicate has **zero** relationships before a plain ``DELETE`` (which
  errors rather than silently dropping data if any edge was missed).

Usage::

    python scripts/fix_refuge_anchor.py                      # dry-run report
    python scripts/fix_refuge_anchor.py --execute            # apply
    python scripts/fix_refuge_anchor.py --execute --backup out/refuge_merge_backup.json

Environment fallbacks: NEO4J_URI, NEO4J_USERNAME/NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("fix_refuge_anchor")

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from gemynd.shared.env import load_dotenv  # noqa: E402


def _connect():
    load_dotenv(_project_root / ".env")
    from neo4j import GraphDatabase

    logging.getLogger("neo4j").setLevel(logging.ERROR)
    uri = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER") or "neo4j"
    pw = os.environ.get("NEO4J_PASSWORD")
    db = os.environ.get("NEO4J_DATABASE") or "neo4j"
    drv = GraphDatabase.driver(uri, auth=(user, pw))
    return drv, db


def _rows(session, q, **kw):
    return [r.data() for r in session.run(q, **kw)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refuge-id", default="turnbull_refuge",
                    help="Shared refuge_id of the split anchor nodes (default: turnbull_refuge).")
    ap.add_argument("--survivor-name", default="Turnbull",
                    help="name/normalized_form of the node to KEEP (the config/seed anchor). Default: Turnbull.")
    ap.add_argument("--execute", action="store_true", help="Apply the merge (default is dry-run).")
    ap.add_argument("--backup", default=str(_project_root / "out" / "refuge_merge_backup.json"),
                    help="Where to write the reversibility backup JSON (on --execute).")
    args = ap.parse_args(argv)

    drv, db = _connect()
    try:
        with drv.session(database=db) as s:
            group = _rows(s, """
                MATCH (e:Entity {refuge_id:$rid})
                OPTIONAL MATCH (d:Document)-[:ABOUT_REFUGE]->(e)
                RETURN e.entity_id AS id, e.name AS name, e.normalized_form AS norm,
                       count(DISTINCT d) AS docs
                ORDER BY docs DESC
            """, rid=args.refuge_id)
            if len(group) < 2:
                _log.info("Found %d node(s) with refuge_id=%r; nothing to merge.", len(group), args.refuge_id)
                return 0

            want = args.survivor_name.strip().lower()
            survivors = [g for g in group if (g["name"] or "").strip().lower() == want
                         or (g["norm"] or "").strip().lower() == want]
            if len(survivors) != 1:
                _log.error("Expected exactly one survivor matching %r; found %d. Nodes: %s",
                           args.survivor_name, len(survivors), group)
                return 2
            survivor = survivors[0]
            dups = [g for g in group if g["id"] != survivor["id"]]
            _log.info("Survivor (KEEP): %s  name=%r  docs=%d", survivor["id"], survivor["name"], survivor["docs"])
            for d in dups:
                _log.info("Duplicate (MERGE->survivor then DELETE): %s  name=%r  docs=%d",
                          d["id"], d["name"], d["docs"])

            # Guard: neither survivor nor dups should have outgoing rels (anchor nodes are sinks).
            for g in group:
                out = _rows(s, "MATCH (e:Entity {entity_id:$id})-[r]->() RETURN count(r) AS n", id=g["id"])[0]["n"]
                if out:
                    _log.error("Node %s has %d OUTGOING rels; unexpected for an anchor. Aborting.", g["id"], out)
                    return 2

            plan = {}
            for d in dups:
                types = [r["t"] for r in _rows(s, "MATCH ()-[r]->(e:Entity {entity_id:$id}) RETURN DISTINCT type(r) AS t", id=d["id"])]
                counts = {t: _rows(s, f"MATCH ()-[r:{t}]->(e:Entity {{entity_id:$id}}) RETURN count(r) AS n", id=d["id"])[0]["n"] for t in types}
                plan[d["id"]] = counts
                _log.info("  %s inbound edges to move: %s", d["id"], counts)

            if not args.execute:
                _log.info("DRY-RUN complete. Re-run with --execute to apply.")
                return 0

            # --- Execute ---
            backup = {"refuge_id": args.refuge_id, "survivor": survivor, "duplicates": []}
            for d in dups:
                edges = _rows(s, """
                    MATCH (src)-[r]->(dup:Entity {entity_id:$id})
                    RETURN type(r) AS rel_type, properties(r) AS rel_props, labels(src) AS src_labels,
                           coalesce(src.entity_id, src.doc_id, src.document_id, src.claim_id,
                                    src.mention_id, src.observation_id, src.event_id, src.paragraph_id) AS src_key,
                           properties(src).institution_id AS src_inst
                """, id=d["id"])
                node_props = _rows(s, "MATCH (e:Entity {entity_id:$id}) RETURN properties(e) AS p, labels(e) AS l", id=d["id"])
                backup["duplicates"].append({"id": d["id"], "node": node_props, "inbound_edges": edges})

            Path(args.backup).parent.mkdir(parents=True, exist_ok=True)
            Path(args.backup).write_text(json.dumps(backup, indent=2), encoding="utf-8")
            _log.info("Wrote reversibility backup: %s", args.backup)

            for d in dups:
                for t in plan[d["id"]]:
                    moved = _rows(s, (
                        f"MATCH (src)-[r:{t}]->(dup:Entity {{entity_id:$dup}}) "
                        f"MATCH (canon:Entity {{entity_id:$canon}}) "
                        f"MERGE (src)-[r2:{t}]->(canon) ON CREATE SET r2 = properties(r) "
                        f"DELETE r RETURN count(*) AS n"
                    ), dup=d["id"], canon=survivor["id"])[0]["n"]
                    _log.info("  moved %-20s %d  (%s -> %s)", t, moved, d["id"], survivor["id"])
                remaining = _rows(s, "MATCH (dup:Entity {entity_id:$id})-[r]-() RETURN count(r) AS n", id=d["id"])[0]["n"]
                if remaining:
                    _log.error("Duplicate %s still has %d rels; NOT deleting. Investigate.", d["id"], remaining)
                    return 2
                s.run("MATCH (dup:Entity {entity_id:$id}) DELETE dup", id=d["id"])
                _log.info("Deleted duplicate node %s", d["id"])

            docs = _rows(s, "MATCH (d:Document)-[:ABOUT_REFUGE]->(c:Entity {entity_id:$id}) RETURN count(d) AS n", id=survivor["id"])[0]["n"]
            left = _rows(s, "MATCH (e:Entity {refuge_id:$rid}) RETURN e.entity_id AS id, e.name AS name", rid=args.refuge_id)
            _log.info("DONE. Survivor now anchors %d docs. Remaining refuge nodes: %s", docs, left)
            return 0
    finally:
        drv.close()


if __name__ == "__main__":
    raise SystemExit(main())
