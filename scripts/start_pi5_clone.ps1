[CmdletBinding()]
param(
    [int] $TimeoutSeconds = 45
)

$ErrorActionPreference = "Stop"

$demoHome = Join-Path $env:USERPROFILE ".Neo4jDesktop2\Data\dbmss\pi5-newspapers-enterprise"
$neo4j = Join-Path $demoHome "bin\neo4j.bat"
$stdout = Join-Path $demoHome "logs\console.out.log"
$stderr = Join-Path $demoHome "logs\console.err.log"

if (-not (Test-Path -LiteralPath $neo4j)) {
    throw "The Pi 5 Neo4j clone is not installed at $demoHome"
}

$listener = Get-NetTCPConnection -State Listen -LocalPort 7689 -ErrorAction SilentlyContinue
if ($listener) {
    Write-Host "Pi 5 newspapers clone is already available at bolt://127.0.0.1:7689"
    exit 0
}

$java = Get-ChildItem -LiteralPath (Join-Path $env:USERPROFILE ".Neo4jDesktop2\Cache\runtime") -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "bin\java.exe") } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $java) {
    throw "No Java runtime was found in the Neo4j Desktop cache."
}

$env:JAVA_HOME = $java.FullName
Start-Process -FilePath $neo4j `
    -ArgumentList @("console") `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    Start-Sleep -Seconds 1
    $listener = Get-NetTCPConnection -State Listen -LocalPort 7689 -ErrorAction SilentlyContinue
} until ($listener -or (Get-Date) -ge $deadline)

if (-not $listener) {
    Write-Host "Neo4j did not open Bolt port 7689. Recent log output:" -ForegroundColor Red
    Get-Content -LiteralPath $stdout -Tail 30 -ErrorAction SilentlyContinue
    Get-Content -LiteralPath $stderr -Tail 30 -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "Pi 5 newspapers clone started." -ForegroundColor Green
Write-Host "Bolt:    bolt://127.0.0.1:7689"
Write-Host "Browser: http://127.0.0.1:7475"

