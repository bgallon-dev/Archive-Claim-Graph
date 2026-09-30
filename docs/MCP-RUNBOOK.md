# Kettle MCP deployment and incident runbook

## Deploy

1. Commit and test the release, then run `python scripts/build_mcp_release.py`. Verify the archive against its companion SHA-256 file. On the VM, extract with `tar -xzf <archive> -C /opt/gemynd --strip-components=1`, create `/opt/gemynd/.venv`, and install with `/opt/gemynd/.venv/bin/pip install '/opt/gemynd[mcp,x402]'`.
2. Upload the evidence snapshot separately into a non-active staging directory. Run `python scripts/validate_evidence_snapshot.py <snapshot> --minimum-records <expected floor>`.
3. Create `/etc/gemynd/kettle.env` with mode `600`; never commit or package populated secrets.
4. Install `deploy/gemynd-mcp.service`. Publish with `sudo RESTART_SERVICE=1 deploy/publish_snapshot.sh <snapshot> /opt/gemynd/snapshots`, then enable the unit.
5. Install `deploy/nginx-kettle-mcp.conf`, provide the certificate, run `nginx -t`, then reload Nginx. Expose only ports 80/443; do not install or expose Neo4j.
6. Create a Coinbase CDP API key, set `KETTLE_X402_PAY_TO` to the receiving EVM address, keep `KETTLE_X402_NETWORK=eip155:84532`, and set `CDP_API_KEY_ID` plus `CDP_API_KEY_SECRET`. Enable x402 only after these values and the public base URL are installed. The older `KETTLE_MCP_REQUIRE_PAYMENT` verifier applies only to paid calls made through `/mcp`; it is not needed for the public `/paid/*` routes.
7. Check `https://<host>/health`, `/ready`, and `/public/capabilities`; record the snapshot ID, record count, x402 network, institutions, and release commit. An unpaid POST to `/paid/find-evidence` must return 402 with a `Payment-Required` header before attempting a funded Base Sepolia purchase.

Keep enough free disk for the incoming, current, and previous uncompressed
snapshots plus 20% headroom. `publish_snapshot.sh` uses reflink copies when the
filesystem supports them. On a constrained same-filesystem deployment, set
`SNAPSHOT_PUBLISH_MODE=move` to consume the validated inbox directory without
creating a second uncompressed copy. Compress with zstd for transport, but
extract the SQLite file before activation. For a Neo4j-free deployment, run nightly
application backups with `DUMP_GRAPH=0`; retain the immutable snapshot in
durable object storage instead of copying it into every daily backup.

The unit sets `KETTLE_MCP_REGISTRY_PATH` to
`deploy/kettle-corpus-registry.yaml`, which the release archive ships. `data/`
is gitignored, so update that copy whenever `data/corpus_registry.yaml`
changes. Without a registry, public coverage always reports `none`, and the
agent instructions omit the corpus names.

## Capacity planning

A 2026-09-26 local baseline (Windows workstation, 10,000-row v2 snapshot, mixed
probe/search/research calls) sustained about 165 requests/s with zero errors.
At 50 concurrent clients, `search_evidence` p95 was 600 ms, and `/health` stayed under
30 ms. Throughput is bounded by the single serialized read connection per
process. Repeat this on the rented server with the production snapshot before
checking the performance items in the readiness checklist.

A 2026-09-10 sample of 10,000 `spokane_newspaper` claims produced a 19.7 MB
serving snapshot (1,972 bytes per row) and a 3.9 MB zstd transfer artifact
(390 bytes per row). The graph contained 29,401,846 total `Claim` nodes, so a
linear all-claim ceiling is about 54 GiB uncompressed or 10.7 GiB in transit,
versus roughly 175.5 GiB for the Neo4j store. The production export should be
smaller because it excludes non-public, restricted, deleted, stale-run, and
unlinked claims. Re-measure from a larger representative sample before choosing
the VM disk; this sample is a planning estimate, not a quota.

## Incident response

The service emits one-line JSON logs to journald. Filter by `event`,
`request_id`, or `tool` (for example, `journalctl -u gemynd-mcp -o cat`).
Each RPC record includes method, tenant, HTTP status, latency, and—on a
handled failure—`error_code`. Request bodies, credentials, receipts, and
stack traces are intentionally excluded from normal records.

- **Unavailable:** check `/health`, `journalctl -u gemynd-mcp`, disk space, and `systemctl status`; restart once after capturing logs.
- **Not ready:** inspect the reported backend and snapshot ID; do not serve traffic until the snapshot is restored or rolled back.
- **Payment error:** set `KETTLE_X402_ENABLED=false`, restart to stop public paid traffic, preserve any legacy payment ledger, and check CDP credentials plus facilitator reachability. Never delete ledger rows.
- **High error rate or disk alert:** disable new traffic, capture a backup, and investigate before rotating or deleting data.
- **Suspected credential leak:** revoke the affected API key/verifier token immediately, issue a replacement, and review redacted logs.

## Backup and restore

Run `deploy/backup.sh` nightly. Before launch and quarterly thereafter, run
`deploy/restore-test.sh /var/backups/gemynd/<timestamp>` and record the result.
For a real restore, stop the service, restore SQLite databases from the backup,
restore the graph if required, run integrity checks, then start and verify both
health endpoints.

Snapshot cleanup is dry-run by default. Review `deploy/cleanup-snapshots.sh`
output, then schedule it with `DRY_RUN=0` after validating the retention
period; it refuses paths outside `/opt/gemynd/snapshots` and never removes
the active `current` target.
