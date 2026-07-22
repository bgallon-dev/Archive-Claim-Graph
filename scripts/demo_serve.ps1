<#
.SYNOPSIS
    One-command launcher for the museum demo: serves the query UI on the local
    network with a read-only account.

.DESCRIPTION
    Runs the preflight checks that otherwise fail confusingly on the day, then
    starts the query server bound to all interfaces and prints the URL and
    credentials to read off to visitors.

    Two environment settings do the load-bearing work:

      COOKIE_SECURE=false   The apps default this to TRUE. Over plain HTTP a
                            Secure cookie is dropped by the browser, so login
                            silently bounces back to the login page. This is
                            the single most confusing failure mode; the script
                            sets it explicitly rather than trusting .env.

      DEMO_ALL_CORPORA=true Lets the read-only account read across all
                            registered corpora. This widens corpus breadth
                            ONLY — the sensitivity tier stays role-derived, so
                            the account still sees just 'public' claims.

.NOTES
    Plain HTTP, no TLS. Acceptable here because the only credential on the wire
    is a throwaway read-only login that clears nothing above 'public'. Do not
    reuse this script to expose the review or ingest UIs, which are write
    surfaces, or to serve anything off a network you do not control.

.EXAMPLE
    .\scripts\demo_serve.ps1
.EXAMPLE
    .\scripts\demo_serve.ps1 -Port 8080 -Password "cedar-lantern-2026"
#>
[CmdletBinding()]
param(
    [int]    $Port = 8788,
    [string] $Email = "demo@museum.local",
    [string] $Password = "",
    # Passthrough to --max-tokens. Left at the server default: measured answers
    # land around 2700 characters, nowhere near either cap, so lowering this
    # does NOT speed the demo up (6000 and 24000 both timed ~99s on the same
    # question). It only matters if you see answers truncated mid-sentence.
    [int]    $MaxTokens = 24000,
    [switch] $SingleCorpus,
    [switch] $KeepPassword,
    [switch] $SkipFirewall
)

$ErrorActionPreference = "Stop"
# Test-NetConnection draws a progress bar that overwrites the preflight [ok]
# lines already on screen, so the operator loses the results they are meant to
# be reading. -WarningAction does not suppress it; this does.
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Write-Step($msg) { Write-Host "  $msg" -ForegroundColor DarkGray }
function Write-Ok  ($msg) { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Write-Bad ($msg) { Write-Host "  [x]  $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "Gemynd museum demo - preflight" -ForegroundColor Cyan
Write-Host ""

# --- 1. Environment file -----------------------------------------------------
if (-not (Test-Path ".env")) {
    Write-Bad ".env not found in $root"
    Write-Host "       Copy .env.example to .env and fill in NEO4J_PASSWORD + JWT_SECRET_KEY."
    exit 1
}
Write-Ok ".env present"

# JWT_SECRET_KEY is read lazily at first login, so a missing/short value fails
# at the worst possible moment - in front of an audience. Check it now.
$envLines = Get-Content ".env" | Where-Object { $_ -match "^\s*JWT_SECRET_KEY\s*=" }
if (-not $envLines) {
    Write-Bad "JWT_SECRET_KEY is not set in .env"
    Write-Host '       Generate one:  python -c "import secrets; print(secrets.token_hex(32))"'
    exit 1
}
$jwt = ($envLines | Select-Object -First 1) -replace "^\s*JWT_SECRET_KEY\s*=\s*", "" -replace '^"|"$', ''
if ($jwt.Trim().Length -lt 32) {
    Write-Bad "JWT_SECRET_KEY is shorter than the required 32 characters"
    exit 1
}
Write-Ok "JWT_SECRET_KEY looks valid"

# --- 2. Python + uvicorn -----------------------------------------------------
$probe = python -c "import uvicorn, fastapi, gemynd; print('ok')"
if ($LASTEXITCODE -ne 0) {
    Write-Bad "Python dependencies missing"
    Write-Host "       Run:  python -m pip install -e .[retrieval]"
    exit 1
}
Write-Ok "python + uvicorn + gemynd importable"

# --- 3. Port availability ----------------------------------------------------
# A stale server from an earlier run keeps the socket, and uvicorn's bind error
# scrolls past under the startup logs - the banner still prints, so it looks
# like everything worked until nobody can connect.
$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    $holder = Get-Process -Id ($listening | Select-Object -First 1).OwningProcess -ErrorAction SilentlyContinue
    Write-Bad "Port $Port is already in use by $($holder.ProcessName) (PID $($holder.Id))"
    Write-Host "       Stop it first:  Stop-Process -Id $($holder.Id) -Force"
    Write-Host "       ...or pick another port:  .\scripts\demo_serve.ps1 -Port 8790"
    exit 1
}
Write-Ok "Port $Port is free"

# --- 4. Neo4j ----------------------------------------------------------------
$bolt = Test-NetConnection -ComputerName 127.0.0.1 -Port 7687 -InformationLevel Quiet -WarningAction SilentlyContinue
if (-not $bolt) {
    Write-Bad "Neo4j is not answering on 127.0.0.1:7687"
    Write-Host "       Start Neo4j Desktop (or the container) before the demo."
    exit 1
}
Write-Ok "Neo4j reachable on 127.0.0.1:7687"

# --- 5. Demo account ---------------------------------------------------------
if ($KeepPassword) {
    Write-Step "Reusing the existing demo account password (-KeepPassword)."
    $shownPassword = "(unchanged - use the password from the last run)"
} else {
    $userArgs = @("scripts/demo_user.py", "--email", $Email)
    if ($Password -ne "") { $userArgs += @("--password", $Password) }
    $userOut = python @userArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Bad "Could not create or reset the demo account"
        $userOut | ForEach-Object { Write-Host "       $_" }
        exit 1
    }
    $pwLine = $userOut | Where-Object { $_ -match "^\s*password:" } | Select-Object -First 1
    $shownPassword = ($pwLine -replace "^\s*password:\s*", "").Trim()
    Write-Ok "Demo account ready (readonly, public-tier claims only)"
}

# --- 6. LAN address ----------------------------------------------------------
# The adapter carrying the default gateway is the one visitors can reach.
$cfg = Get-NetIPConfiguration |
    Where-Object { $null -ne $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq "Up" } |
    Select-Object -First 1
if ($null -eq $cfg) {
    Write-Bad "No network adapter with a default gateway is up."
    Write-Host "       Connect this machine to the museum network first."
    exit 1
}
$lanIp = $cfg.IPv4Address.IPAddress
$adapter = $cfg.InterfaceAlias
Write-Ok "LAN address $lanIp on '$adapter'"

# --- 7. Firewall -------------------------------------------------------------
if ($SkipFirewall) {
    Write-Step "Skipping the firewall rule (-SkipFirewall)."
} else {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $isAdmin = ([Security.Principal.WindowsPrincipal]$identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    $ruleName = "Gemynd demo (TCP $Port)"
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

    # Scope to the profile Windows actually assigned THIS network. Hardcoding
    # 'Private' silently fails when the venue wifi registers as Public, which
    # looks identical to a broken server.
    # NetworkCategory is an enum whose Public member is 0 - which PowerShell
    # treats as FALSY. A `-not $profileName` guard here silently rewrites
    # "Public" to "Private" on exactly the networks where getting this right
    # matters most, so test against $null explicitly.
    $profileName = (Get-NetConnectionProfile -InterfaceAlias $adapter -ErrorAction SilentlyContinue |
                    Select-Object -First 1).NetworkCategory
    if ($null -eq $profileName) { $profileName = "Private" } else { $profileName = [string]$profileName }

    $manualCmd = "New-NetFirewallRule -DisplayName '$ruleName' -Direction Inbound " +
                 "-Action Allow -Protocol TCP -LocalPort $Port -Profile $profileName"

    if ($existing) {
        # A rule scoped to the profile of some *other* network does not apply
        # here. Reporting "already present" in that case is the exact silent
        # failure this script exists to prevent, so check coverage.
        $covered = ($existing.Profile -match "Any") -or ($existing.Profile -match $profileName)
        if ($covered) {
            Write-Ok "Firewall rule '$ruleName' present and covers the '$profileName' profile"
        } elseif ($isAdmin) {
            Set-NetFirewallRule -DisplayName $ruleName -Profile $profileName
            Write-Ok "Firewall rule re-scoped from '$($existing.Profile)' to '$profileName'"
        } else {
            Write-Warn "Firewall rule exists but is scoped to '$($existing.Profile)', and this"
            Write-Host "       network is '$profileName'. It will NOT let visitors through."
            Write-Host "       Re-run this script as Administrator, or run:"
            Write-Host ""
            Write-Host "         Set-NetFirewallRule -DisplayName '$ruleName' -Profile $profileName"
            Write-Host ""
        }
    } elseif (-not $isAdmin) {
        Write-Warn "Not running as Administrator - cannot add the firewall rule."
        Write-Host "       Other devices WILL time out until you re-run this script from an"
        Write-Host "       elevated PowerShell, or run this once in an admin window:"
        Write-Host ""
        Write-Host "         $manualCmd"
        Write-Host ""
    } else {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $Port -Profile $profileName | Out-Null
        Write-Ok "Firewall opened on TCP $Port for the '$profileName' profile"
    }

    if ($profileName -eq "Public") {
        Write-Warn "This network is classed Public - the port is open to everyone on it."
        Write-Host "       Remove the rule after the demo:"
        Write-Host "         Remove-NetFirewallRule -DisplayName '$ruleName'"
    }
}

# --- 8. Serve ----------------------------------------------------------------
# Set in the process environment so they beat .env (the loader does not
# override already-set variables).
$env:COOKIE_SECURE = "false"
if ($SingleCorpus) {
    $env:DEMO_ALL_CORPORA = "false"
    $corpusNote = "home corpus only"
} else {
    $env:DEMO_ALL_CORPORA = "true"
    $corpusNote = "all registered corpora"
}

$url = "http://${lanIp}:${Port}"
Write-Host ""
Write-Host ("=" * 62) -ForegroundColor Cyan
Write-Host "  READY - open this on any device on the same network" -ForegroundColor Cyan
Write-Host ("=" * 62) -ForegroundColor Cyan
Write-Host ""
Write-Host "    $url" -ForegroundColor White
Write-Host ""
Write-Host "    email:     $Email"
Write-Host "    password:  $shownPassword"
Write-Host ""
Write-Host "    Showing:   $corpusNote, public-tier claims only"
Write-Host "    Transport: plain HTTP (no TLS) - local network only"
Write-Host ""
Write-Host "  Measured: ~100s per answer; the FIRST query after startup runs" -ForegroundColor DarkGray
Write-Host "  ~190s while caches warm. Ask one throwaway question before the" -ForegroundColor DarkGray
Write-Host "  audience arrives. Broad cross-corpus questions are slowest (~5m)." -ForegroundColor DarkGray
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor DarkGray
Write-Host ""

python -m gemynd.cli query-serve --host 0.0.0.0 --port $Port --max-tokens $MaxTokens
