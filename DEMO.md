# Museum demo — operator card

Serves the query UI to phones, tablets, and laptops on the same local network.

## On the day

In the project folder, **right-click `demo.cmd` → Run as administrator.**

That is the whole procedure. Administrator is needed the first time so the
firewall rule can be added; without it the server runs and works on this
machine, but every other device times out, because Windows blocks inbound
connections by default. Once the rule exists, a plain double-click is fine.

> Use `demo.cmd`, not the `.ps1` directly. PowerShell's default execution
> policy is `Restricted` and will refuse to run the script with a
> "running scripts is disabled on this system" error. `demo.cmd` passes
> `-ExecutionPolicy Bypass` for that one invocation and changes nothing on
> the machine. If you prefer the script directly, either use
> `powershell -ExecutionPolicy Bypass -File .\scripts\demo_serve.ps1`, or
> allow local scripts permanently for your account (no admin needed):
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Then confirm from a phone before you rely on it.** Open the printed URL on a
second device on the same wifi. That is the only check that actually proves
visitors can connect; testing in a browser on the serving machine does not
traverse the firewall and will pass even when everything else is blocked.

The script prints a URL and a password. Write them on a card:

```
    http://<your-lan-ip>:8788
    demo@museum.local / <printed password>
```

Visitors open the URL on any device on the same wifi. Stop with `Ctrl+C`.

## Before the audience arrives

**Ask one throwaway question first.** Measured timings on this machine:

| | |
|---|---|
| First query after startup | slower (caches warming) |
| Every query after that | **~60s** |
| Broad cross-corpus questions | still the slowest |

Roughly 56s of that ~60s is now the Anthropic synthesis call; database time is
about 3s. Before the `claim_run_id` index (see below) a query took ~100s.

That first slow answer is the difference between a demo that feels alive and
one that looks hung. Warm it up before anyone is watching.

Narrow, entity-anchored questions are the ones that land — *"What happened at
Turnbull in the 1930s?"* returns a cited answer with ~20 supporting claims.
Open-ended sweeps across every corpus are the slowest and the vaguest.

## Phones

The UI is responsive below 820px. Two drawers, both in the top bar:

- **Hamburger, top-left** — the main navigation.
- **Speech-bubble icon, top-right** — past conversations. As a fixed 260px
  column it swallowed two thirds of a phone screen, so on small screens it
  slides in over the chat instead. Tap a conversation and it closes itself.

Both close on tap-outside or Escape. Desktop is unchanged — the conversation
column stays permanently visible above 820px.

**Refreshing does not cancel an answer.** `/query` runs in a server-side
threadpool, so a disconnect cannot stop it — the answer still lands ~100s
later. The page now remembers an in-flight question and picks it back up
automatically after a reload, showing *"Still working on this — your refresh
did not cancel it"* until the answer arrives. Tell visitors: if you reload,
just wait, it will reappear.

The one thing to avoid is asking the same question over and over — each attempt
starts another ~100-second job on the same laptop while the earlier ones are
still running.

## Featured questions + instant answers (curator)

The empty chat screen can show a row of starter questions so visitors aren't
staring at a blank box, and a featured question the curator has *captured* comes
back **instantly** instead of running the ~100s pipeline. Same store does both.

Curate from a terminal (the running server is not involved in writes):

```powershell
# 1. add starter questions (shown as suggestions; still run live when tapped)
python -m gemynd.cli featured add "What animals are most prevalent at Turnbull?"
python -m gemynd.cli featured add "Why was the refuge established?"

# 2. ask each one once in the UI so a good answer lands in the log, then:
python -m gemynd.cli featured capture "What animals are most prevalent at Turnbull?"
#    -> that question now answers instantly

python -m gemynd.cli featured list      # [cached] = answers instantly
python -m gemynd.cli featured remove "..."
```

Matching is exact on a normalised form (case/whitespace/trailing punctuation),
so **tapping a suggestion always hits**; a typed paraphrase just runs live. It
can never serve a *wrong* cached answer — worst case is a miss.

**Needs a server restart to take effect** (it's Python, not a template): restart
with `-KeepPassword` so the demo card stays valid. Seed the store *before*
restarting, ideally during the cache warm-up, so the featured answers are ready
when the doors open. An empty store = today's behaviour (no suggestions, no
instant answers).

## What visitors can and cannot see

The demo account is `readonly`, which resolves to `permitted_levels = ['public']`.
The Cypher access gate filters on `d.access_level IN $permitted_levels`, so
`staff_only`, `restricted`, and `indigenous_restricted` material is excluded at
query time — a kiosk credential never clears tribal-consultation content.

Corpus breadth is a **separate** axis, set by `DEMO_ALL_CORPORA=true` (the
script sets it). That widens *which corpora* are readable without touching the
sensitivity tier: 5,400 documents across Turnbull + Spokane instead of 458
Turnbull-only. Use `-SingleCorpus` to show only the home corpus.

> Worth knowing: every document currently in the graph is `access_level='public'`.
> The gate is correct and enforced, but today it is filtering nothing, because
> nothing above `public` has been ingested yet. The protection matters the
> moment restricted material lands — it is not doing work for you right now.

## Flags

| Flag | Use |
|---|---|
| `-Port 8790` | Another port (default 8788) |
| `-Password "cedar-lantern-42"` | Pin a password instead of generating one |
| `-KeepPassword` | Restart without rotating the password |
| `-SingleCorpus` | Home corpus only |
| `-SkipFirewall` | Don't touch firewall rules |

Flags work through the wrapper too: `demo.cmd -Port 8790 -SingleCorpus`

Reprint the credentials without starting the server:

```powershell
python scripts\demo_user.py
```

## When something breaks

**"Port is already in use"** — a server from an earlier run is still up. The
script prints the PID; `Stop-Process -Id <pid> -Force`. Note it may run as
`python3.11`, not `python.exe`.

**Login bounces back to the login page** — `COOKIE_SECURE` is true somewhere.
Over plain HTTP the browser silently drops a `Secure` cookie, so the session
never sticks. The script forces it false; if you launched the server by hand,
that is the cause.

**Visitors get a timeout, but it works on this machine** — the firewall rule is
missing (run elevated once), or devices are on a different SSID / guest network
with client isolation. Guest wifi that blocks device-to-device traffic will
defeat this no matter what the server does — no server-side setting can fix it;
you need a network that permits device-to-device traffic, or your own hotspot.

The script scopes the firewall rule to the profile Windows assigned the current
network. If you set it up on one network and demo on another with a different
classification, the script detects the mismatch and re-scopes the rule when run
elevated. Remove it when you are done:

```powershell
Remove-NetFirewallRule -DisplayName "Gemynd demo (TCP 8788)"
```

**Neo4j not reachable** — start Neo4j Desktop before the script.

**Every question fails, but the UI loads fine** — check that this machine has
outbound internet. Neo4j and LM Studio are local, but synthesis calls
`api.anthropic.com`, so a venue network with no internet access means retrieval
succeeds and every answer fails. Test it on site before the doors open:

```powershell
python -c "import requests; print(requests.get('https://api.anthropic.com', timeout=10).status_code)"
```

Any response (even 404) means outbound HTTPS works. A timeout means you need a
phone hotspot for the serving laptop — the visitors' devices only need to reach
the laptop, not the internet.

## Live answer streaming (on by default)

Instead of ~60s of spinner, sentences appear as the model writes them — first
one at about 20-25s, the rest arriving over the next 30s. Total time is
unchanged; what changes is that the visitor can start reading.

While it works, the indicator names each step it is actually doing — reading
the question, searching the archive, gathering records, weighing evidence,
planning, writing — so the pause reads as deliberation rather than a stall. The
elapsed-time counter stays hidden until 30s, at which point the wait is long
enough to be worth admitting.

Needs a server restart to pick up (it is Python, not a template). To turn it
off for a browser, untick **"Stream as written"** in the chat settings panel,
or open `?stream=0`.

If anything goes wrong — endpoint missing, connection dropped — the page falls
back to the normal request automatically, so enabling it cannot cost an answer.
Sentences are streamed only once complete and PII-redacted, never half-written.

## If you want it faster still

The database side is done — two indexes (`claim_run_id`, `run_timestamp`) took
retrieval from 47s to 3s per query. They are in `core/graph/cypher.py`, so any
new deployment creates them automatically; the live database already has them.

What is left is the synthesis call (~56s). One tunable knob, no code change:

```powershell
$env:SYNTHESIS_CONTEXT_CHAR_BUDGET = "60000"   # default 250000
```

Measured effect: synthesis 56s → 32s, but the answer got shorter (1,805 chars
citing 8 claims, versus 3,032 citing 22). That is a real quality cost, so it is
left at the default. Try 120000 for a middle ground if you want to trade some
depth for speed.

## The security tradeoff, stated plainly

This serves **plain HTTP with no TLS**. Anyone on the same network can read the
traffic, including the demo password. That is an accepted tradeoff here because
the only credential exposed is a throwaway read-only login that clears nothing
above `public`, and it is rotated on every launch.

Do **not** reuse this script to expose `review-serve` or `ingest-serve` — those
are write surfaces. Do not run it on a network you do not control, and do not
log in with your admin account on a device connected to the demo network.
