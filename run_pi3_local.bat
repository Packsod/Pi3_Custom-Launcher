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

REM -------------------------------------------------------------------
REM 检查并自动新建所需的权重文件夹结构
REM -------------------------------------------------------------------
if not exist "weights\geocalib" (
    echo Creating missing directory: weights\geocalib
    mkdir "weights\geocalib"
)

if not exist "weights\Pi3" (
    echo Creating missing directory: weights\Pi3
    mkdir "weights\Pi3"
)

if not exist "weights\Pi3X" (
    echo Creating missing directory: weights\Pi3X
    mkdir "weights\Pi3X"
)
REM -------------------------------------------------------------------

venv\Scripts\python.exe run_pi3_local.py

pause
