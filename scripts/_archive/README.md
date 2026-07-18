# Archived scripts

Spent one-off / diagnostic scripts moved here by the 2026-07-08 system audit
(see `docs/system-audit-2026-07-08.md`). They are **not** imported by anything
and are kept only for reference. Safe to delete this whole folder.

| File | Why archived |
|------|--------------|
| `ingest_newspapers.py` | Points `INPUT_DIR`/`DOMAIN_DIR` at `<Gemynd>/Newspapers`, which no longer exists (fails fast via `SystemExit`). Superseded by `scripts/ingest_tutorstructure.py`, the live newspaper-ingest driver. |
| `fix_entity_labels.py` | Spent one-time Neo4j migration that repaired an old `:Entity`/`:Place` node split the writer no longer produces. Label coercion (e.g. Refuge→Place) is now config-driven via `domain_config.categorial_label_map` (`ingest/graph/writer.py`). |
| `diag_composite_corpus.py` | One-off "does the boot-time composite include Spokane?" debug script with hardcoded `spokane_newspaper`/`turnbull` institution ids. Use `tools/run_diagnostics.py` for corpus-membership checks. |
| `_audit_smoke.py` | Repo-root scratch harness that verified the domain-agnosticism refactor removed hardcoded Turnbull labels/concepts. The refactor has landed; assertions belong in `tests/` if still wanted. |
| `_corpus_stats.py` | Repo-root scratch stats over a hardcoded `Desktop/Tutorstructure/corpus` path. |
| `_resmoke_minting.py` | Repo-root scratch that re-measured a single minting change against disposable `out/` smoke dirs. |

The last three were previously loose at the repo root (`Desktop/Gemynd/`),
outside version control; they are now tracked here.
