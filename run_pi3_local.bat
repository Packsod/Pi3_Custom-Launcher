@echo off
setlocal EnableExtensions

REM Local-only Gradio server
set PI3_SHARE=0
set PI3_SERVER_NAME=127.0.0.1
set PI3_SERVER_PORT=7860

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv\Scripts\python.exe was not found.
    pause
    exit /b 1
)

venv\Scripts\python.exe run_pi3_local.py

pause
