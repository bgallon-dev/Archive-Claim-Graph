# Neo4j-Free Evidence Service Production Checklist

This checklist covers the work required to deploy the indexed evidence service as an always-on Bazaar offering without a serving-time Neo4j dependency.

## Already completed

- [x] Evidence-only MCP tools implemented
- [x] Cache graph connected to the retrieval service
- [x] Full index build completed
- [x] 1,037,506 entities validated
- [x] 64,349,565 adjacencies validated
- [x] Zero missing or extra indexed relationships
- [x] Per-shard checksums generated
- [x] MCP and routing tests passing
- [x] Basic API authentication supported

## Release blockers

- [x] Package claim text and complete provenance into the standalone index format
- [x] Remove the serving-time Neo4j dependency from the standalone MCP path
- [x] Define a versioned evidence-packet schema
- [x] Include document, collection, page, date, source URL, and evidence fingerprint fields where available
- [x] Add a snapshot ID and content checksum to the manifest
- [x] Generate a hashed manifest for each release
- [x] Verify the snapshot checksum when loading
- [x] Refuse to serve incomplete or corrupted snapshots
- [x] Test a clean server path using only a published snapshot
- [x] Disable unsupported arbitrary relationship tracing

## Snapshot deployment

- [x] Store snapshots in immutable, versioned directories
- [x] Upload new snapshots without modifying the active version
- [x] Validate a snapshot before activation
- [x] Switch versions atomically through a `current` pointer
- [x] Keep previous snapshots available for rollback
- [x] Document the rollback procedure
- [x] Prevent partial uploads from becoming active
- [x] Add automated storage cleanup with retention rules (dry-run by default)

## Query safety and quality

- [x] Define exactly which query types Bazaar v1 supports: coverage probing,
      bounded evidence search, source-linked subject research, and evidence
      packets; arbitrary graph traversal and synthesized answers are excluded
- [x] Add maximum result and execution-time limits
- [x] Fail closed when provenance is missing
- [x] Never return an unsupported claim ID as evidence
- [x] Create a golden evaluation set for evidence search
- [x] Test identity ambiguity and similarly named entities
- [x] Test dates, locations, document filters, and pagination
- [x] Test zero-result and malformed queries
- [x] Set minimum top-result precision and 100% provenance-completeness thresholds in the golden tests

## Performance

- [ ] Measure cold-start time on the intended 4 GB server
- [ ] Measure steady-state memory usage
- [ ] Test 10, 25, and 50 concurrent requests
- [ ] Establish p50, p95, and p99 latency targets
- [ ] Test high-degree entities and maximum-size packets
- [ ] Test service behavior when a shard is unavailable
- [ ] Confirm the service remains responsive during snapshot activation
- [ ] Add caching for repeated coverage and evidence queries

### Suggested initial targets

- [ ] p95 evidence search under 1 second
- [ ] p95 evidence packet under 2 seconds
- [ ] Error rate below 1%
- [ ] Provenance completeness at 100%
- [ ] Memory usage below 75% of the server limit

## Security and payments

- [ ] Put the service behind HTTPS (nginx configuration is ready; certificate deployment remains)
- [x] Store API keys and payment-verifier secrets outside the repository
- [x] Add API-key rotation
- [x] Add rate limiting per customer
- [x] Derive authenticated tenant identity from the API key instead of trusting
      the caller-supplied tenant header
- [x] Add request and payment idempotency keys
- [x] Verify payment before releasing paid packet content
- [x] Record price, payment receipt digest, and snapshot ID together
- [x] Prevent the same successful payment from being charged twice
- [x] Keep wallet signing isolated from the query process
- [x] Redact secrets and payment credentials from errors and responses

## Operations

- [x] Add `/health` and `/ready` endpoints
- [x] Report the active snapshot ID through readiness
- [x] Add structured request, latency, and error logs
- [x] Add disk, memory, error-rate, and availability alert rules (deploy/prometheus-alerts.yml)
- [x] Configure automatic restart after crashes
- [x] Configure start-on-boot
- [x] Back up usage records and payment receipts
- [ ] Perform and document a real restore test (run deploy/restore-test.sh against a production backup)
- [x] Write a short deployment and incident runbook
- [x] Add external uptime monitoring configuration (deploy/uptime-monitor.example.yml; provider activation pending)

## CI and release gate

- [ ] Make lint failures blocking
- [ ] Enable type checking for the serving and index packages
- [ ] Run index-format compatibility tests in CI
- [ ] Test corrupted and truncated fixtures
- [ ] Test upgrade and rollback between two snapshot versions
- [x] Add a reproducible, secret-free deployment artifact builder
- [x] Record the code commit used to build each snapshot
- [ ] Require all production checks before publishing

## Bazaar v1 go-live gate

The service is ready to deploy when:

- [ ] The server works with Neo4j completely offline
- [ ] Every packet contains verifiable provenance
- [ ] Snapshot integrity is checked at startup
- [ ] Load testing passes on the rented server
- [ ] HTTPS, authentication, and rate limiting are active
- [ ] Payment retries cannot create duplicate charges
- [ ] Monitoring, backup, and rollback have each been tested
- [ ] Only validated query tools are advertised in Bazaar

## Recommended implementation order

1. Build the standalone provenance bundle.
2. Add the checksum-verifying loader.
3. Add immutable, atomic snapshot releases and rollback.
4. Run load, corruption, recovery, and security tests.
5. Complete Bazaar payment integration and the production deployment.
