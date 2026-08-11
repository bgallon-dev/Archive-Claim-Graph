@echo off
REM Start the isolated local clone, then run the existing read-only demo UI.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_pi5_clone.ps1"
if errorlevel 1 (
    echo.
    echo The local Pi 5 database clone could not be started.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo_serve.ps1" ^
  -Neo4jUri "bolt://127.0.0.1:7689" ^
  -Neo4jDatabase "newspapers" ^
  -SkipPathCacheWarmup %*

if errorlevel 1 (
    echo.
    echo The local Pi 5 demo exited with an error. Read the message above.
    pause
)
