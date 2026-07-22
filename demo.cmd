@echo off
REM Museum demo launcher.
REM
REM Right-click this file -> "Run as administrator" (needed once, so the
REM firewall rule can be added; after that a normal double-click is fine).
REM
REM Exists because PowerShell's default execution policy is Restricted, which
REM refuses to run demo_serve.ps1 at all. -ExecutionPolicy Bypass applies to
REM THIS invocation only - it changes nothing on the machine.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo_serve.ps1" %*

REM Keep the window open if launched by double-click, so a preflight failure
REM is readable instead of vanishing.
if errorlevel 1 (
    echo.
    echo The demo server exited with an error. Read the message above.
    pause
)
