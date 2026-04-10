# review

Anti-pattern detection, proposal CRUD, and patch validation. Quality assurance layer for the extraction pipeline.

## Architecture

- `detect.py` — orchestrates all four detector families, validates patch specs, routes proposals through auto-accept/suppress/review triage, upserts to `ReviewStore`.
- `detectors/` — four families: `ocr_entity.py` (duplicate/corruption), `junk_mention.py` (header/boilerplate/OCR garbage), `builder_repair.py` (missing species focus, missing location, method over-trigger), `sensitivity_monitor.py` (PII, Indigenous cultural material, living persons).
- `store.py` — SQLite-backed proposal store with full revision history.
- `actions.py` — accept, reject, defer, edit, split proposal actions.
- `patch_spec.py` — typed mini-language for patch operations. `validate_patch_spec()` enforces schema v1.
- `monitor.py` — background sensitivity scanner against live Neo4j claims.

## Controlled vocabularies

All controlled vocabularies (`PROPOSAL_TYPES`, `ISSUE_CLASSES`, `PROPOSAL_STATUSES`, `REVIEW_TIERS`) are `frozenset` constants in `models.py`. Adding a new detector output type requires adding entries here.

## Routing thresholds (`detect.py`)

- Auto-accept: >= 0.90 confidence, `suppress_mention` proposals with `ocr_garbage` or `short_generic_token` reasons only.
- Auto-suppress: below 0.50 confidence — proposals are not generated.
- Everything else goes to `needs_review` queue.

## Priority scoring

`_compute_priority_score()` ranks `merge_entities`/`create_alias` highest because wrong merges have corpus-wide consequences.

## Sensitivity proposals

`quarantine_claim`, `quarantine_document`, `restrict_permanently` follow special handling. `indigenous_restricted` quarantine clearance requires tribal consultation.

## Cross-module sync

`ALLOWED_CLAIM_ENTITY_RELATIONS` in `patch_spec.py` must stay in sync with `core/claim_contract.py`.

## Testing

Test detectors independently with synthetic bundles. `test_review_store.py` uses temp SQLite. `test_review_patch_spec.py` validates schema enforcement.
