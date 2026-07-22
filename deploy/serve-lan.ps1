<#
.SYNOPSIS
  Serve the Gemynd query app to your local network (LAN) from this Windows machine.

.DESCRIPTION
  Binds the query server to 0.0.0.0 so other devices on the same network can reach it
  at http://<this-machine-ip>:<port>. Intended for a demo on a TRUSTED network.

  This serves over plain HTTP. Do not use on a shared/untrusted network, and do not
  expose it to the internet this way — see docs/SERVING-ON-NETWORK.md for the TLS path.

.PARAMETER Port
  Port to bind (default 8788).

.PARAMETER OpenFirewall
  Add a Windows Firewall inbound rule for the port. Requires an elevated (Admin) shell.

.EXAMPLE
  # From a normal shell (firewall already allows the port):
  .\deploy\serve-lan.ps1

.EXAMPLE
  # From an ELEVATED shell, also open the firewall port:
  .\deploy\serve-lan.ps1 -OpenFirewall
#>
param(
    [int]$Port = 8788,
    [switch]$OpenFirewall
)

$ErrorActionPreference = "Stop"

# Repo root = this script's parent directory's parent (deploy/ -> repo root)
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot"

# Python launcher check
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) { throw "Python launcher not found. Install Python 3.11+ (the 'py' launcher)." }
$pyExe = $py.Source

# Optional firewall rule
if ($OpenFirewall) {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
              ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Warning "-OpenFirewall needs an elevated (Admin) shell. Skipping the firewall rule."
    } else {
        $ruleName = "Gemynd Query $Port"
        $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
        if (-not $existing) {
            New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
                -Protocol TCP -LocalPort $Port -Profile Private | Out-Null
            Write-Host "Added firewall rule '$ruleName' (TCP $Port, Private profile)."
        } else {
            Write-Host "Firewall rule '$ruleName' already exists."
        }
    }
}

# Show reachable LAN addresses
Write-Host ""
Write-Host "Reachable from other devices on this network at:" -ForegroundColor Cyan
Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    ForEach-Object { Write-Host ("  http://{0}:{1}" -f $_.IPAddress, $Port) -ForegroundColor Green }
Write-Host ""
Write-Host "Plain HTTP on a trusted network only. Keep Neo4j and (if used) LM Studio running." -ForegroundColor Yellow
Write-Host "Ctrl+C to stop."
Write-Host ""

# Launch the query server bound to all interfaces.
# (No console script is installed, so invoke the CLI module directly from the repo root.)
& $pyExe -m gemynd.cli query-serve --host 0.0.0.0 --port $Port
