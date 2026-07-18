"""
diagnose_doc_extraction.py — Drill into one document's extraction profile.

Given a doc_id (or a title substring), pull a single-doc report explaining
*why* the claim extractor produced fewer claims than peer documents would
suggest. Looks at:

  * Doc metadata (title, year, total chars, paragraph count, page count).
  * Claim type distribution.
  * Paragraph stats: how many have 0 claims? what's their length profile
    compared to claim-bearing paragraphs?
  * OCR quality proxy: clean_text/raw_ocr_text length ratio per paragraph
    (low ratio means lots of text was scrubbed by normalize_text — usually
    a sign of OCR garbage).
  * Quarantined claim count.
  * Sample 0-claim paragraphs that have substantial text — these are the
    ones the extractor most plausibly missed signal on.

Usage:
    python tools/diagnose_doc_extraction.py --doc-id doc_960f6801aeac413d
    python tools/diagnose_doc_extraction.py --title-like "1963 Narrativereport"
    python tools/diagnose_doc_extraction.py --doc-ids doc_a,doc_b,doc_c

Requires NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (loaded from .env).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from gemynd.shared.env import load_dotenv
from gemynd.shared.settings import Settings
from gemynd.ingest.graph import build_driver_kwargs as _build_driver_kwargs

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("neo4j package is not installed. Install with: pip install neo4j")


META_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
OPTIONAL MATCH (d)-[:HAS_PAGE]->(pg:Page)
WITH d, count(DISTINCT pg) AS page_count
OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)-[:HAS_PARAGRAPH]->(p:Paragraph)
WITH d, page_count, count(DISTINCT p) AS paragraph_count,
     sum(coalesce(p.char_count, 0)) AS total_chars
RETURN d.doc_id          AS doc_id,
       d.title           AS title,
       d.report_year     AS report_year,
       d.institution_id  AS institution_id,
       page_count,
       paragraph_count,
       total_chars
"""

CLAIM_TYPES_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
      -[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)
      -[:HAS_PARAGRAPH]->(:Paragraph)-[:HAS_CLAIM]->(c:Claim)
RETURN c.claim_type                                  AS claim_type,
       c.quarantine_status                           AS quarantine_status,
       count(*)                                      AS n
ORDER BY n DESC
"""

PARAGRAPHS_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
      -[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)
      -[:HAS_PARAGRAPH]->(p:Paragraph)
OPTIONAL MATCH (p)-[:HAS_CLAIM]->(c:Claim)
WHERE c.quarantine_status IS NULL OR c.quarantine_status <> 'quarantined'
WITH p, count(DISTINCT c) AS active_claims
RETURN p.paragraph_id                                AS paragraph_id,
       coalesce(p.char_count, 0)                     AS char_count,
       size(coalesce(p.raw_ocr_text, ''))            AS raw_chars,
       size(coalesce(p.clean_text, ''))              AS clean_chars,
       p.clean_text                                  AS clean_text,
       active_claims
"""


def fetch(settings: Settings, query: str, **params):
    driver_kwargs = _build_driver_kwargs(
        uri=settings.neo4j_uri,
        trust_mode=settings.neo4j_trust,
        ca_cert_path=settings.neo4j_ca_cert,
    )
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        **driver_kwargs,
    )
    rows: list[dict] = []
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(query, **params)
        for record in result:
            rows.append(dict(record))
    driver.close()
    return rows


def resolve_doc_ids(
    settings: Settings, doc_id: str | None, title_like: str | None, doc_ids: str | None
) -> list[str]:
    if doc_id:
        return [doc_id]
    if doc_ids:
        return [d.strip() for d in doc_ids.split(",") if d.strip()]
    if title_like:
        rows = fetch(
            settings,
            "MATCH (d:Document) WHERE toLower(d.title) CONTAINS toLower($q) "
            "RETURN d.doc_id AS doc_id, d.title AS title",
            q=title_like,
        )
        if not rows:
            sys.exit(f"No documents found with title containing {title_like!r}")
        if len(rows) > 1:
            print(f"Multiple matches for {title_like!r}:")
            for r in rows:
                print(f"  {r['doc_id']:35s}  {r['title']}")
            print("Pass --doc-id to pick one.")
            sys.exit(1)
        return [rows[0]["doc_id"]]
    sys.exit("Pass --doc-id, --doc-ids, or --title-like")


def diagnose(settings: Settings, doc_id: str, sample_n: int = 5) -> None:
    meta_rows = fetch(settings, META_QUERY, doc_id=doc_id)
    if not meta_rows:
        print(f"[!] No document found with doc_id={doc_id}")
        return
    meta = meta_rows[0]

    claim_rows = fetch(settings, CLAIM_TYPES_QUERY, doc_id=doc_id)
    paras = fetch(settings, PARAGRAPHS_QUERY, doc_id=doc_id)

    print("=" * 80)
    print(f"DIAGNOSE: {meta['title']}")
    print(f"  doc_id={meta['doc_id']}")
    print(
        f"  year={meta['report_year']}  institution={meta['institution_id']}  "
        f"pages={meta['page_count']}  paragraphs={meta['paragraph_count']}  "
        f"chars={meta['total_chars']:,}"
    )
    print("=" * 80)

    active_total = sum(r["n"] for r in claim_rows if r["quarantine_status"] != "quarantined")
    quarantined_total = sum(r["n"] for r in claim_rows if r["quarantine_status"] == "quarantined")
    print(f"\nClaims: {active_total} active, {quarantined_total} quarantined")

    print("\nClaim type distribution (active claims):")
    grouped: dict[str, int] = {}
    for r in claim_rows:
        if r["quarantine_status"] != "quarantined":
            grouped[r["claim_type"]] = grouped.get(r["claim_type"], 0) + r["n"]
    for ctype, n in sorted(grouped.items(), key=lambda kv: -kv[1]):
        print(f"  {ctype:30s} {n}")

    if not paras:
        print("\n[!] No paragraphs found.")
        return

    zero = [p for p in paras if p["active_claims"] == 0]
    nonzero = [p for p in paras if p["active_claims"] > 0]

    print(f"\nParagraphs: {len(paras)} total, {len(zero)} with 0 claims, {len(nonzero)} with >=1")
    if paras:
        zero_chars = [p["char_count"] for p in zero]
        nz_chars = [p["char_count"] for p in nonzero]
        if zero_chars:
            print(
                f"  0-claim paragraphs: median chars={int(statistics.median(zero_chars))}, "
                f"mean={int(statistics.mean(zero_chars))}, max={max(zero_chars)}, "
                f"sum={sum(zero_chars):,}"
            )
        if nz_chars:
            print(
                f"  claim-bearing:      median chars={int(statistics.median(nz_chars))}, "
                f"mean={int(statistics.mean(nz_chars))}, max={max(nz_chars)}, "
                f"sum={sum(nz_chars):,}"
            )

    # OCR quality: ratio of clean/raw per paragraph. Low ratios = lots was
    # scrubbed by normalize_text, often OCR garbage.
    ratios = []
    for p in paras:
        raw = p["raw_chars"]
        clean = p["clean_chars"]
        if raw > 0:
            ratios.append(clean / raw)
    if ratios:
        print(
            f"\nOCR cleanliness (clean_text / raw_ocr_text length per paragraph):"
        )
        print(
            f"  median={statistics.median(ratios):.3f}  "
            f"mean={statistics.mean(ratios):.3f}  "
            f"min={min(ratios):.3f}  max={max(ratios):.3f}"
        )
        very_low = sum(1 for r in ratios if r < 0.5)
        if very_low:
            print(
                f"  {very_low} paragraphs have <50% retention — strong signal of OCR garbage"
            )

    # Special char density on the 0-claim paragraphs as a tiebreaker.
    if zero:
        substantial = [
            p for p in zero if p["char_count"] >= 200 and p["clean_text"]
        ]
        substantial.sort(key=lambda p: -p["char_count"])
        print(
            f"\n0-claim paragraphs with >=200 chars: {len(substantial)} "
            f"(extractor missed prose-length text in these)"
        )
        for i, p in enumerate(substantial[:sample_n]):
            text = p["clean_text"] or ""
            preview = text[:240].replace("\n", " ")
            non_alnum = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
            non_alnum_pct = non_alnum / max(len(text), 1)
            print(
                f"\n  [{i+1}] paragraph_id={p['paragraph_id']}  "
                f"chars={p['char_count']}  non_alnum_pct={non_alnum_pct:.1%}"
            )
            print(f"      {preview}{'...' if len(text) > 240 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id")
    parser.add_argument("--doc-ids", help="Comma-separated list of doc_ids")
    parser.add_argument("--title-like")
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(_REPO_ROOT / ".env")
    settings = Settings.from_env()
    if not settings.neo4j_uri:
        sys.exit("NEO4J_URI not set.")

    ids = resolve_doc_ids(settings, args.doc_id, args.title_like, args.doc_ids)
    for i, did in enumerate(ids):
        if i > 0:
            print("\n")
        diagnose(settings, did, sample_n=args.sample)


if __name__ == "__main__":
    main()
