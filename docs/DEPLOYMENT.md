# Deploying Gemynd for a limited tester pilot

Scope: a **retrieval-only** pilot. A handful of named institutional testers
query an existing graph. Ingest and review are deliberately **not** exposed —
that single decision removes most of the operational risk (stranded jobs,
unbounded uploads, in-process PDF parsing, the review app's blocked event
loop) and makes the graph read-only for the pilot's duration.

If you later add ingest, re-read the "Not covered" section at the bottom
first — several things that are fine for a read-only pilot are not fine once
testers can write.

---

## 1. Prerequisites

- Linux host (the systemd unit assumes it), Python 3.11+, Neo4j 5.x
- A DNS name and TLS certificate, or an internal CA — see step 5
- Disk: the graph plus `data/`, `out/`, and logs. **Check free space before
  you start**; a full disk takes down Neo4j and SQLite together, and the
  failure mode is confusing.

## 2. Install

```bash
sudo useradd --system --home /opt/gemynd --shell /usr/sbin/nologin gemynd
sudo mkdir -p /opt/gemynd /var/log/gemynd
sudo chown gemynd:gemynd /opt/gemynd /var/log/gemynd

sudo -u gemynd git clone <repo> /opt/gemynd
cd /opt/gemynd
sudo -u gemynd python3 -m venv .venv
sudo -u gemynd .venv/bin/pip install -e '.[retrieval]'
```

## 3. Configure

Copy `.env.example` to `.env` and set at minimum:

| Setting | Notes |
|---|---|
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Use a **least-privilege** account, not `neo4j` admin. The app performs index/constraint DDL at startup, so it needs schema privileges on its own database — nothing more. |
| `JWT_SECRET_KEY` | 48+ random chars. The app refuses to start without it rather than falling back to a default. |
| `ANTHROPIC_API_KEY` | Or configure the local-model seam (`SYNTHESIS_LLM_BASE_URL`). |
| `COOKIE_SECURE` | `true` — **but only once TLS is actually terminating.** See step 5. |
| `TRUSTED_PROXY_IPS` | The reverse proxy's IP. See step 5 — omitting this quietly breaks rate limiting. |
| `LOG_FILE` | e.g. `/var/log/gemynd/app.log`. Without it, logs die with the terminal. |
| `FEEDBACK_DB` | Defaults to `data/feedback.db`. Where tester reports land. |

```bash
sudo chown gemynd:gemynd /opt/gemynd/.env
sudo chmod 600 /opt/gemynd/.env
```

> **Working directory is load-bearing.** Every path — `.env`, `data/*.db`,
> `out/` — resolves relative to the process's cwd. Started from the wrong
> directory, the app creates a *new, empty* `data/users.db`, which re-triggers
> first-run setup and makes every existing account appear to vanish. The
> systemd unit sets `WorkingDirectory=/opt/gemynd` for exactly this reason.

## 4. Supervisor

```bash
sudo cp deploy/gemynd-query.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gemynd-query
systemctl status gemynd-query
```

`Restart=always` is the point. Without a supervisor there is no
restart-on-crash and no start-on-boot, so the first unhandled error or reboot
is a permanent outage that nobody notices until a tester emails.

Neo4j is intentionally `After=` and not `Requires=`: the app starts degraded
when the database is unreachable and reconnects on its own, so a Neo4j
restart no longer takes the app down with it.

## 5. TLS and the reverse proxy

```bash
sudo cp deploy/nginx-gemynd.conf /etc/nginx/sites-available/gemynd
sudo ln -s /etc/nginx/sites-available/gemynd /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

Two coupled settings that cause confusing failures if you get them wrong:

- **`COOKIE_SECURE=true` without TLS makes login fail silently.** The browser
  accepts the response, drops the cookie, and bounces the user back to the
  login page with no error. If testers report "login does nothing", check this
  first.
- **`TRUSTED_PROXY_IPS` unset behind a proxy collapses rate limiting into one
  bucket.** Every request appears to come from the proxy, so the whole
  institution shares a single login-attempt allowance — one person fumbling a
  password can lock out everyone.

Authentication is a JWT in a cookie. Do not run a pilot over plain HTTP on a
shared network.

## 6. Backups

```bash
sudo cp deploy/backup.sh /opt/gemynd/deploy/
sudo chmod +x /opt/gemynd/deploy/backup.sh
sudo -u gemynd crontab -e
# 15 2 * * *  /opt/gemynd/deploy/backup.sh >> /var/log/gemynd/backup.log 2>&1
```

`data/*.db` is the irreplaceable part — users, document notes, annotations,
saved searches, query history, tester feedback. None of it is in git and none
of it can be regenerated. The graph can be rebuilt from `out/` bundles, but
that costs hours to weeks of LLM work.

The script uses `sqlite3 .backup`, not `cp`, because the databases are in WAL
mode and live: a plain copy can capture a torn page or miss committed
transactions still sitting in the `-wal` file.

**Restore, and test it before the pilot:**

```bash
sudo systemctl stop gemynd-query
sudo -u gemynd cp /var/backups/gemynd/<timestamp>/data/*.db /opt/gemynd/data/
# graph, if needed:
sudo systemctl stop neo4j
sudo neo4j-admin database load neo4j --from-path=/var/backups/gemynd/<timestamp> --overwrite-destination=true
sudo systemctl start neo4j gemynd-query
```

A backup nobody has restored is not yet a backup.

## 7. First run

Starting with no users triggers a one-time setup flow. The token is written
to **stderr only** (`journalctl -u gemynd-query`), is mode 0600 on disk, and
is deleted after use.

```bash
journalctl -u gemynd-query -n 50 | grep -i setup
```

Create the admin account, then add testers. **Give every tester the
`researcher` or `readonly` role — never `admin`.** Role assignment is what
scopes the pilot: `/review`, `/ingest`, `/users`, `/gaps` and the tester
feedback list are all admin-gated routes inside the same app, so a non-admin
account simply never sees them.

## 8. Verify before handing it to anyone

```bash
curl -sS https://gemynd.example.org/health          # {"status":"ok","neo4j":"connected"}
systemctl is-enabled gemynd-query                    # enabled  (survives reboot)
sudo systemctl restart neo4j && sleep 20 && curl -sS https://gemynd.example.org/health
```

That last one is the important one: the app should recover on its own, not
crash-loop. Then, in a browser, as a **non-admin** account:

- [ ] Log in over HTTPS; the padlock is real
- [ ] Run a query and get an answer with citation pills
- [ ] Click a citation — the source document opens at the highlighted sentence
- [ ] Open **Help** and confirm it describes *your* corpus sensibly
- [ ] Submit a **Report a problem**, then confirm it appears at `/feedback` as admin
- [ ] Confirm the sidebar shows **no** Admin section
- [ ] Sign out; confirm you're returned to the login page
- [ ] Disconnect from the internet and reload — the UI must still render
      (all assets are self-hosted; this catches a firewalled asset regression)

## 9. Watching it during the pilot

```bash
journalctl -u gemynd-query -f                       # live
tail -f /var/log/gemynd/app.log                     # if LOG_FILE is set
sqlite3 /opt/gemynd/data/feedback.db \
  "SELECT created_at, category, message FROM feedback ORDER BY created_at DESC LIMIT 20;"
```

Poll `/health` from whatever uptime checker you have. It is not just a
readout: a poll against a degraded instance *triggers* the reconnect attempt,
so it is also the recovery mechanism.

Check `/feedback` regularly. An empty list early on more often means testers
haven't found the control than that nothing is wrong — worth asking in person.

---

## Not covered (deliberately)

These are known and accepted for a retrieval-only pilot. Revisit before
widening scope:

- **Ingest and review are not exposed.** Both have open issues: unsanitised
  upload filenames (path traversal), no upload size cap, jobs stranded in
  `running` after a crash with no reaper, in-process PDF parsing, and the
  review app serialising every request on the event loop with a permanently
  degraded `/health`.
- **Logout does not revoke the token.** It clears the cookie; the JWT stays
  valid until `exp` (24h default). The `token_version` mechanism exists and is
  used for password change and deactivation, but not wired to logout.
- **No graph schema version marker.** If the expected graph shape changes
  under testers, nothing detects the mismatch — queries just return less.
- **No LLM spend cap.** `shared/token_tracker.py` records usage and warns, but
  nothing enforces a budget. Watch cost manually, or route synthesis at a
  local model via the `SYNTHESIS_LLM_*` seam.
- **Single process, no workers.** Fine for a handful of concurrent testers —
  the expensive paths (Neo4j I/O, synthesis) already run in the threadpool.
  Revisit before opening it up more widely.
