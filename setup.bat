@echo off
rem ============================================================
rem  Hand gesture control - FIRST-TIME SETUP (run once)
rem  Creates .venv next to this file and installs dependencies.
rem  Requires Python 3.11 - 3.13 (mediapipe has no wheels for 3.14 yet).
rem  Pure ASCII on purpose: cmd.exe code pages mangle non-ASCII batch files.
rem ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ==========================================
echo   Hand gesture control - first-time setup
echo ==========================================
echo.

rem ---------- 1. find a python interpreter ----------
rem 优先挑 mediapipe 有轮子的版本（3.13 / 3.12 / 3.11），
rem 最后才退回 py -3 / python —— 很多机器上 py -3 是 3.14，装不了 mediapipe。
set "PY="
for %%V in (3.13 3.12 3.11) do (
    if not defined PY (
        py -%%V -V >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    py -3 -V >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo [error] Python not found.
    echo.
    echo   Install Python 3.11 - 3.13 from https://www.python.org/downloads/
    echo   During install, tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('%PY% -V 2^>^&1') do set "PYVER=%%v"
echo [info] found Python %PYVER%

%PY% -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)"
if errorlevel 1 (
    echo.
    echo [error] Python %PYVER% is not supported by mediapipe.
    echo.
    echo   mediapipe has wheels for Python 3.10 - 3.13 only.
    echo   Install Python 3.13 from:
    echo       https://www.python.org/downloads/release/python-3139/
    echo   then run setup.bat again ^(it will pick 3.13 automatically^).
    echo.
    pause
    exit /b 1
)

rem ---------- 2. create the virtual environment ----------
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [1/4] creating .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [error] failed to create .venv
        pause
        exit /b 1
    )
) else (
    "%~dp0.venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [1/4] existing .venv is broken, recreating it ...
        %PY% -m venv --clear .venv
        if errorlevel 1 (
            echo [error] failed to repair .venv
            pause
            exit /b 1
        )
    ) else (
        echo [1/4] .venv already exists, reusing it
    )
)
set "VPY=%~dp0.venv\Scripts\python.exe"

echo [2/4] upgrading pip ...
"%VPY%" -m pip install --upgrade pip -q

rem ---------- 3. install dependencies ----------
echo [3/4] installing dependencies - this may take a few minutes ...
echo        mediapipe / opencv / numpy / pillow, about 200 MB in total
echo.
rem 忽略本机 pip.ini 里配的国内镜像：mediapipe 在清华/阿里云都没有 wheel,
rem 而 extra-index-url 还可能直接抛 IncompleteRead, 连 Collecting 都到不了。
rem PIP_CONFIG_FILE=NUL 只跳过配置文件, HTTPS_PROXY 等环境变量仍然生效。
set "PIP_CONFIG_FILE=NUL"
"%VPY%" -m pip install --index-url https://pypi.org/simple --timeout 120 -r requirements.txt
set "PIP_CONFIG_FILE="
if errorlevel 1 (
    echo.
    echo [error] install failed.
    echo.
    echo   mediapipe is NOT on the Tsinghua mirror, it must come from
    echo   the official index. If your network is slow or blocked, set a
    echo   proxy and rerun setup.bat, for example:
    echo.
    echo       set HTTPS_PROXY=http://127.0.0.1:7890
    echo       setup.bat
    echo.
    pause
    exit /b 1
)

rem ---------- 4. download and verify models ----------
echo [4/4] checking MediaPipe models ...
"%VPY%" download_models.py
if errorlevel 1 (
    echo.
    echo [error] model download failed.
    echo   Check network access to storage.googleapis.com and rerun setup.bat.
    echo   Existing valid model files are kept; only missing files are downloaded.
    pause
    exit /b 1
)

rem ---------- 5. verify ----------
"%VPY%" -c "import mediapipe, cv2, numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo [error] dependencies installed but import check failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   Done. Ways to start:
echo.
echo     run.bat --test    recognize only, sends NO keys (try this first)
echo     run.bat           full control (HUD window)
echo     run_draw.bat      draw in the air with your index finger
echo ==========================================
pause
