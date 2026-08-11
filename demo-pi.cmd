@echo off
REM Newspapers demo backed by Neo4j on Pi5Node1.
REM
REM First copy .env.pi.example to .env.pi and add the Pi's Neo4j credentials.
REM The SSH password is not a Neo4j credential and must not be used here.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo_serve.ps1" ^
  -EnvOverlay "%~dp0.env.pi" ^
  -Neo4jUri "bolt://pi5node1.local:7687" ^
  -Neo4jDatabase "newspapers" %*

if errorlevel 1 (
    echo.
    echo The Pi demo exited with an error. Read the message above.
    pause
)
