# Serving Gemynd beyond your laptop

Three levels, from "other devices on the same network can reach it in 5 minutes" to a
proper always-on server. Pick by how the demo/pilot will actually be used.

`gemynd query-serve` binds to `127.0.0.1` (localhost only) by default. Everything below
is about safely changing *where* it listens and *how long* it stays up.

---

## Level 1 — LAN demo from this Windows laptop (fastest)

Other devices on the **same network** reach the app at `http://<laptop-ip>:8788`. Good for
a demo where you bring the laptop (a booth, a conference room, the museum's guest network).

```powershell
# normal shell — if the firewall port is already open:
.\deploy\serve-lan.ps1

# elevated (Admin) shell — also opens the Windows Firewall port:
.\deploy\serve-lan.ps1 -OpenFirewall
```

The helper binds `0.0.0.0`, prints the reachable `http://<ip>:8788` URLs, and starts the
server. Under the hood it's just:

```powershell
py -m gemynd.cli query-serve --host 0.0.0.0 --port 8788
```

**Before you rely on it:**
- Keep **Neo4j** running (the app degrades without it), and **LM Studio** if you use the
  local model — or point synthesis at Anthropic (see the demo notes).
- This is **plain HTTP**. Keep `COOKIE_SECURE=false` (its current value) or login silently
  fails over HTTP. Only do this on a **trusted** network — the auth cookie is not encrypted
  in transit.
- It stops when you close the shell or the laptop sleeps. That's fine for an attended demo,
  not for a pilot people use on their own time.

**Caveats:** guest/enterprise Wi-Fi often has *client isolation* that blocks device-to-device
traffic — test from a second device beforehand. If it fails, use a phone hotspot or a small
switch/router you control.

---

## Level 2 — Always-on server (the pilot)

For a pilot people use unattended, you need something that survives reboots and sleep, and
(once it's reachable beyond a trusted LAN) **TLS**. Three real options:

### 2a. On-prem Linux box at the museum — *the product (recommended)*
The commercial model is **on-premises** (`docs/commercial/PRICING.md`): Gemynd runs inside
the institution's boundary so nothing leaves the building. `docs/DEPLOYMENT.md` covers this
path end-to-end: `systemd` (restart-on-crash, start-on-boot), `nginx` + TLS reverse proxy,
nightly backups, first-run setup, role scoping, and a pre-handover checklist — and the
`deploy/` assets (`gemynd-query.service`, `nginx-gemynd.conf`, `backup.sh`) target exactly
it. Delivered either onto the museum's own Linux server (if they have IT) or as a
**Kettle appliance** — a pre-configured mini-server you install on their network and support
remotely (the "archive-in-a-box" turnkey option for institutions without a sysadmin). The
app itself is OS-agnostic Python; the appliance is the clean way to give a small museum
on-prem without asking them to run Linux.

### 2b. Cloud VM — *for remote demos only, not the pilot*
A small cloud VM (e.g. [provider] 2 vCPU / 8 GB + Neo4j) is handy purely as a **stable
demo URL you can show remotely**. It is **not** the customer deployment — the on-prem thesis
means customer data stays on customer premises. Use it to rehearse and to demo, then deliver
on-prem. TLS via nginx + Let's Encrypt or **Caddy** (automatic HTTPS).

### 2c. Windows-native on-prem — *if the museum mandates on-prem Windows*
Runnable, but it's a second deployment path to maintain (the `deploy/` assets are Linux):
- Run `py -m gemynd.cli query-serve --host 0.0.0.0 --port 8788` as a **Windows Service**
  via [NSSM](https://nssm.cc/) (or a Task Scheduler "at startup" task), so it survives
  logout/reboot. Install **Neo4j** as its Windows service too.
- Put **[Caddy for Windows](https://caddyserver.com/)** in front for automatic HTTPS — this
  also fixes the cookie problem (see below) cleanly on an internal hostname.

---

## Level 3 — Reachable beyond a trusted LAN (TLS is now mandatory)

The moment the app is reachable over anything but a trusted local network, switch to HTTPS:

1. Terminate **TLS** at a reverse proxy (nginx per `DEPLOYMENT.md`, or Caddy for auto-HTTPS).
2. Set **`COOKIE_SECURE=true`** in `.env` — **but only once TLS is actually terminating**,
   or login breaks silently.
3. Set **`TRUSTED_PROXY_IPS`** to the proxy's IP — otherwise every request looks like it
   comes from the proxy and the whole institution shares one rate-limit bucket (one person
   fumbling a password locks out everyone).
4. Never run auth over plain HTTP on a shared network.

`DEPLOYMENT.md` §5 covers this proxy/TLS wiring in detail.

---

## Recommendation

Hosting model is decided: **on-premises** (the product thesis — data never leaves the
institution).

- **This week's demo:** Level 1 (`serve-lan.ps1`) if you're driving it on-site; optionally a
  throwaway cloud VM (**2b**) only as a stable *demo* URL.
- **First pilot:** **2a** — on-prem at the museum, delivered on their Linux server or (for a
  museum without IT) as a **Kettle appliance**, using the existing `DEPLOYMENT.md` runbook.
  Go **2c** (Windows-native) only if a customer mandates on-prem Windows.

Next build step for the on-prem path: decide **customer-server vs. Kettle appliance** as the
standard delivery, and spec the appliance (GPU for acceptable local-model latency) — see
`docs/commercial/PRICING.md` §3.
