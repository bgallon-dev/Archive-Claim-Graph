@echo off
REM Museum demo launcher using opencode as the synthesis provider.
REM
REM Starts `opencode serve` automatically and serves the query UI with
REM SYNTHESIS_PROVIDER=opencode. Install opencode and run `opencode auth login`
REM before using this launcher.
REM
REM Right-click -> "Run as administrator" is needed only for the first run so
REM the firewall rule can be added; after that a normal double-click is fine.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo_serve.ps1" %* -Provider opencode -StartOpencodeServer -SkipPathCacheWarmup

if errorlevel 1 (
    echo.
    echo The opencode demo server exited with an error. Read the message above.
    pause
)
