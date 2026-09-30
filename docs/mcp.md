# Kettle research MCP server

The Kettle adapter exposes a small, stateless MCP-over-HTTP surface plus
optional public Coinbase x402 routes for the research marketplace MVP. x402 is
disabled by default, so ordinary local runs record metered usage but do not
move money.

## Install and run

From the `Archive-Claim-Graph` directory:

```powershell
py -m pip install -e ".[mcp]"
gemynd-mcp
```

If Windows reports that the Scripts directory is not on `PATH`, run the
executable by its reported full path or add that directory to your user PATH.

The server listens on `http://127.0.0.1:8789/mcp`. It accepts MCP JSON-RPC
requests and exposes `tools/list`, `tools/call`, `server/discover`, and a
health check at `/health`.
The `/ready` endpoint reports whether a usable evidence backend is loaded and
includes the active snapshot ID and record count when configured.

## Configure evidence

Use either a local JSON/JSONL fixture or an existing Gemynd evidence-capable API:

```powershell
$env:KETTLE_MCP_EVIDENCE_PATH = "C:\path\to\evidence.jsonl"
# or: $env:KETTLE_API_BASE_URL = "http://127.0.0.1:8000"
# Optional future dedicated route: $env:KETTLE_API_EVIDENCE_PATH = "/v1/evidence"
$env:KETTLE_MCP_API_KEY = "replace-with-a-local-key"
gemynd-mcp
```

For a small fixture, build an immutable JSONL snapshot:

```powershell
py scripts/build_evidence_snapshot.py `
  --input C:\path\to\evidence.jsonl `
  --output snapshots\2026-08-27 `
  --snapshot-id 2026-08-27
$env:KETTLE_MCP_SNAPSHOT_PATH = "C:\path\to\snapshots\2026-08-27"
gemynd-mcp
```

For a production corpus, export a compact SQLite/FTS snapshot directly from
the authoritative local Neo4j database. Scoping is mandatory unless the
operator explicitly chooses `--all-institutions`:

```powershell
py -m pip install -e ".[snapshot]"
py scripts/export_evidence_snapshot.py `
  --institution-id spokane_newspaper `
  --output snapshots\spokane-2026-09-09 `
  --snapshot-id spokane-2026-09-09
py scripts/validate_evidence_snapshot.py snapshots\spokane-2026-09-09 `
  --institution-id spokane_newspaper `
  --minimum-records 1000
```

The v2 SQLite format stores canonical evidence once and uses an
external-content FTS index, avoiding the second full text copy used by v1.
Checksums are streamed so startup memory does not scale with snapshot size.
The export contains only public, unrestricted documents from each document's
latest extraction run. Production exports also refuse a dirty source tree;
`--allow-dirty` exists only for explicitly marked development snapshots.

The server verifies the snapshot format, record count, required evidence
fields, and SHA-256 checksum before serving any records. A snapshot directory
must be newly built; publish it by switching the configured path atomically.
Set `KETTLE_MCP_RATE_LIMIT` and `KETTLE_MCP_RATE_WINDOW` to tune the per-tenant
HTTP request budget (defaults: 120 requests per 60 seconds).

For Linux deployment, install `deploy/gemynd-mcp.service`, publish releases
with `deploy/publish_snapshot.sh`, and use `deploy/rollback_snapshot.sh` to
switch back to a retained snapshot. Both scripts validate before switching.
Because the MCP keeps the verified SQLite file open, publish with
`RESTART_SERVICE=1` (as an account allowed to restart the unit) or restart
`gemynd-mcp` immediately afterward.
On a space-constrained VM, upload the compressed artifact, extract it into an
inbox directory on the snapshot filesystem, and set `SNAPSHOT_PUBLISH_MODE=move`
when publishing. The source directory is consumed, avoiding another full copy.

Each evidence row can contain `claim`, `excerpt`, `year`, `subjects`,
`relationships`, and a nested `source` object. The adapter normalizes rows into
stable evidence and provenance objects. Set `KETTLE_MCP_REGISTRY_PATH` to use a
different corpus registry.

The MCP response never forwards a generated answer. The adapter requests the
current `/v1/query` compatibility route with `evidence_only: true`; the
retrieval API must include `supporting_claims_detail`. A dedicated evidence-only
route can be selected later with `KETTLE_API_EVIDENCE_PATH`.

## Tool funnel

1. `coverage_probe` is free and should be called first. Over `/mcp` it searches
   the archive and reports `coverage_status` from the records it finds:
   `strong` (at least three records contain most of the question's content
   terms), `partial` (one or two do), `weak` (records match only some terms),
   or `none`. It also returns `query_terms`, `evidence_found`,
   `strong_matches`, the corpus year span, `known_gaps`, and three
   `sample_evidence` claims. The unauthenticated `/public/coverage` route
   never searches the archive; it returns a registry-based estimate marked
   `coverage_basis: registry_estimate`.
2. `search_evidence` is a free, compact preview.
3. `research_subject` and `build_evidence_packet` are paid-style operations
   with receipts and an in-memory daily cap. Arbitrary relationship tracing is
   intentionally disabled; source-linked relationships remain in
   `research_subject` results.

Search is lexical. Function words ("who", "was", "tell me") are ignored,
records containing every content term rank first, and records matching only
some terms fill the rest of the page. Each evidence object's `matched_terms`
shows which terms it contains, so agents can discard off-topic records.

Responses are shaped for agent context budgets. Each evidence object appears
once, in `evidence_objects`. `timeline` lists `{date, evidence_id}`
references, and a packet's `supporting_evidence` and `contradictory_evidence`
are evidence ID lists. `excerpt` is empty when it would repeat `claim`.
`time` has day precision when the source has a date. Newspaper sources whose
collection is a known LCCN also carry a readable `source.publication`, and
the `document` label uses that title. The `document` filter matches the title
as well. Unknown argument names and badly typed values return an
`invalid_arguments` tool error rather than being ignored.

## Connecting agents

The endpoint speaks MCP streamable HTTP (JSON responses; no SSE stream, so
`GET /mcp` returns 405). At `initialize` it returns instructions that describe
the loaded corpus and the probe → search → paid-research workflow.

Claude Code:

```bash
claude mcp add --transport http kettle https://<host>/mcp \
  --header "Authorization: Bearer <client-api-key>"
```

Clients that take a JSON server list (Claude Desktop connectors, Cursor, and
others) use the same URL and header:

```json
{
  "mcpServers": {
    "kettle": {
      "type": "http",
      "url": "https://<host>/mcp",
      "headers": { "Authorization": "Bearer <client-api-key>" }
    }
  }
}
```

Issue each client its own key in `KETTLE_MCP_API_KEYS`: tenant identity, rate
limits, and the daily cap are all derived from the key.

Set `KETTLE_MCP_REQUIRE_PAYMENT=true` to make paid tools called through `/mcp` fail closed unless the
request supplies both `Idempotency-Key` and `X-Kettle-Payment-Receipt` headers.
Configure `KETTLE_PAYMENT_VERIFIER_URL` to point at the isolated payment
verifier; the MCP process has no wallet-key setting and never signs payments.

Successful authorizations are written to `KETTLE_PAYMENT_LEDGER`. The ledger
enforces unique idempotency keys and unique payment receipts and records the
tenant, operation, price, provider reference, and active snapshot ID. Responses
include only the receipt SHA-256 digest, never the opaque receipt itself.

Production deployments can rotate client keys without downtime by placing the
old and new keys together in `KETTLE_MCP_API_KEYS`, then removing the old key
after clients have migrated. Store the populated environment file at
`/etc/gemynd/kettle.env` with mode `600`; `deploy/kettle.env.example` contains
placeholders only. TLS termination is defined in `deploy/nginx-kettle-mcp.conf`.
When API keys are configured, rate-limit and payment tenant identity is derived
from the authenticated key; a client-supplied `X-Kettle-Tenant` value is not
trusted.

## Coinbase x402 and Bazaar routes

Install the optional payment dependencies with `pip install -e ".[mcp,x402]"`.
The public funnel begins with three unauthenticated, rate-limited operations:

- `GET /public/capabilities` — when and why an agent should use Kettle
- `POST /public/coverage` — whether the corpus fits a historical question
- `POST /public/estimate` — the recommended paid resource and exact price

The paid interface has four fixed-price USDC resources so each Bazaar listing
has one unambiguous price and a bounded evidence budget:

- `POST /paid/find-evidence` — $0.03, up to 10 evidence objects
- `POST /paid/research-subject` — $0.06, up to 40 evidence objects
- `POST /paid/investigate-subject` — $0.15, up to 100 evidence objects
- `POST /paid/compile-evidence-packet` — $0.25, up to 100 evidence objects

Each accepts a JSON object containing `question` and optional evidence limits,
time range, document, location, and offset. The 402 response carries x402 v2
payment requirements and Bazaar discovery metadata. Coinbase CDP verifies the
buyer before Kettle runs, then settles the payment before the response body is
released. Failed verification does no research work; failed settlement does
not disclose the generated result.

Configure Base Sepolia first:

```text
KETTLE_X402_ENABLED=true
KETTLE_X402_NETWORK=eip155:84532
KETTLE_X402_PAY_TO=0x<receiving-address>
KETTLE_X402_PUBLIC_BASE_URL=https://<public-host>
CDP_API_KEY_ID=<cdp-key-id>
CDP_API_KEY_SECRET=<cdp-key-secret>
```

The application refuses to start when x402 is enabled with an invalid address,
unsupported network, incomplete credentials, or malformed public URL. Keep the
network on Base Sepolia until the end-to-end funded test and launch review are
complete. The receiving server needs no wallet private key or seed phrase.

For an unpaid smoke test, POST `{"question":"Spokane riverfront leaders"}` to
`/paid/find-evidence` and confirm HTTP 402 plus a `Payment-Required` response
header. A successful funded test additionally returns `Payment-Response` with
the Base Sepolia settlement transaction. `/health` and `/ready` report whether
x402 is enabled and its selected network without exposing the receiving address
or CDP credentials.

## Build the application release

Do not copy the working directory: it may contain Neo4j exports, benchmarks,
logs, local SQLite state, and `.env`. Commit the release first, then build the
secret-free application archive:

```powershell
py scripts/build_mcp_release.py
```

The builder refuses a dirty tree, includes only runtime/deployment files, emits
a deterministic `tar.gz`, and writes a companion SHA-256 file. Upload that
archive and the separately versioned evidence snapshot. Never put a production
snapshot or populated secret file in Git.
