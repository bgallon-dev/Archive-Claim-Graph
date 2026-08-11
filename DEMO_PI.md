# Pi newspapers demo

This profile runs the existing Gemynd query demo on this Windows machine while
streaming graph queries from the `newspapers` Neo4j database on `Pi5Node1`.
The 30 GB database stays on the Pi; only query results cross the network.

## One-time setup

From the `Archive-Claim-Graph` folder:

```powershell
Copy-Item .env.pi.example .env.pi
notepad .env.pi
```

Put the Pi's **Neo4j** username and password in `.env.pi`. These are separate
from the Pi's SSH login. Use a read-only Neo4j account for a demo environment.
The real file is ignored by Git.

The main `.env` remains the source for the application's JWT and Anthropic API
settings. The Pi overlay changes only the Neo4j login.

## Start

Double-click `demo-pi.cmd`, or run:

```powershell
.\demo-pi.cmd -SkipFirewall
```

The launcher performs preflight checks, verifies Anthropic API synthesis is
configured, creates or rotates the public-tier demo login, and prints the
browser URL.
Omit `-SkipFirewall` when other devices on the same LAN need access; the first
run may need an Administrator terminal to add the inbound firewall rule.

Useful options from the normal demo launcher also work:

```powershell
.\demo-pi.cmd -Port 8790
.\demo-pi.cmd -KeepPassword
.\demo-pi.cmd -SingleCorpus
```

## Connection details

- Bolt URI: `bolt://pi5node1.local:7687`
- Database: `newspapers`
- Web UI: served locally on port `8788` by default
- Neo4j Browser: `http://pi5node1.local:7474`

The direct `bolt://` scheme is intentional. The Pi currently advertises a
stale routing address (`192.168.0.111`), so `neo4j://` can redirect clients to
an unreachable endpoint.

## Safety boundary

The web demo creates a `readonly` application user and exposes public-tier
claims only. The Neo4j credential in `.env.pi` should independently be granted
read-only database access, so the database still rejects writes even if the
application layer is bypassed.
