@echo off
rem ============================================================
rem  Air drawing - one-click launcher (draw.py)
rem  Pure ASCII script; project path resolved at runtime via %~dp0
rem  Usage: double-click, or "run_draw.bat --hand right --brush 8"
rem
rem  Interpreter lookup order (same as run.bat):
rem    1) .venv inside this folder
rem    2) .venv in the parent folder
rem    3) CV_PYTHON environment variable
rem    4) python on PATH
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY if defined CV_PYTHON set "PY=%CV_PYTHON%"
if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo [error] Python not found.
    echo   Install Python 3.11 - 3.13 from https://www.python.org/downloads/
    echo   then double-click setup.bat once.
    pause
    exit /b 1
)

"%PY%" -c "import mediapipe, cv2, numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo [error] Interpreter lacks required packages:
    echo     %PY%
    echo.
    if exist "%~dp0setup.bat" (
        echo   Double-click setup.bat once - it installs everything.
    ) else (
        echo   Install them with:  "%PY%" -m pip install -r requirements.txt
    )
    pause
    exit /b 1
)

"%PY%" draw.py %*
if errorlevel 1 pause
