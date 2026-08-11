# Local Pi 5 newspapers clone

The Pi 5 `newspapers` database is restored into an isolated Neo4j Enterprise
2026.06 instance. It does not replace or modify the databases managed by
Neo4j Desktop.

## Connection

- Bolt: `bolt://127.0.0.1:7689`
- Neo4j Browser: `http://127.0.0.1:7475`
- Database: `newspapers`
- Credentials: the `NEO4J_USER` and `NEO4J_PASSWORD` already stored in `.env`

The server binds only to loopback. Cluster, routing, and backup listeners are
either moved away from Neo4j Desktop's ports or disabled.

## Start the database

```powershell
.\scripts\start_pi5_clone.ps1
```

The launcher is idempotent: if the clone is already listening, it exits
successfully without starting a second process.

## Start the query demo

Double-click `demo-local-pi5.cmd`, or run:

```powershell
.\demo-local-pi5.cmd -SkipFirewall
```

This large clone skips the optional entity-degree cache warmup at web startup.
Path search remains available with uniform weights, and startup avoids a
multi-minute aggregation over the 56-million-relationship graph.

The local database clone still uses Anthropic API synthesis. Topical questions
use a bounded full-text candidate set before graph expansion, with the broader
claim-type scan retained as a fallback.

The database runtime is using Neo4j's 30-day Enterprise evaluation license
because the source graph uses the Enterprise-only block store format.

## Local storage

- Runtime and live store:
  `%USERPROFILE%\.Neo4jDesktop2\Data\dbmss\pi5-newspapers-enterprise`
- Immutable verified backup:
  `%USERPROFILE%\.Neo4jDesktop2\Backups\Pi5Node1\newspapers-2026-07-25T18-31-59.backup`
- Backup SHA-256:
  `9a2b2035e7b4d552f2cfdc734b91ee3be1bc72a377d6ad5c82255cf88e2210c1`
