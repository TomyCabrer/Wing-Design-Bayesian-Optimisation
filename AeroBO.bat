@echo off
REM Double-click this to start AeroBO on Windows.
REM
REM It is the whole install: there is nothing to have installed first, not
REM even Python. It fetches uv (Astral's installer) into this folder, builds
REM a private environment beside it, and starts the app in your browser.
REM
REM Deleting this folder deletes everything it ever put on the machine.

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "APP_DIR=%CD%"
set "VENV=%APP_DIR%\.venv-app"
set "UV_DIR=%APP_DIR%\.aerobo-tools"
set "CORE=%APP_DIR%\installer\requirements-core.txt"
set "BO=%APP_DIR%\installer\requirements-bo.txt"
set "STAMP=%VENV%\.aerobo-stamp"

REM Everything uv writes goes inside this folder - its wheel cache, any Python
REM it downloads, its receipt - so that deleting the folder is the whole
REM uninstall. The installs below also pass --no-cache: the cache is a
REM download accelerator for a step that happens once, and it measured 872 MB.
set "UV_CACHE_DIR=%UV_DIR%\cache"
set "UV_PYTHON_INSTALL_DIR=%UV_DIR%\python"

echo.
echo   AeroBO
echo.

REM ------------------------------------------------------------------ uv ---
set "UV="
if exist "%UV_DIR%\uv.exe" set "UV=%UV_DIR%\uv.exe"
if not defined UV for %%I in (uv.exe) do if not "%%~$PATH:I"=="" set "UV=%%~$PATH:I"
if not defined UV (
  echo   first run - fetching the installer ^(uv, ~35 MB^)
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$env:UV_INSTALL_DIR='%UV_DIR%'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
  if exist "%UV_DIR%\uv.exe" set "UV=%UV_DIR%\uv.exe"
)
if not defined UV goto :nouv

REM ---------------------------------------------------------------- venv ---
set "REINSTALL=0"
if /I "%~1"=="--reinstall" set "REINSTALL=1"
if not exist "%VENV%\Scripts\python.exe" (
  echo   building the environment ^(Python 3.11^)
  "%UV%" venv --python 3.11 "%VENV%" >nul || goto :novenv
  set "REINSTALL=1"
)

REM The stamp records which lock files this environment was built from, so a
REM second launch starts straight away instead of re-checking the network.
set "LOCK_ID="
for /f "usebackq delims=" %%H in (`certutil -hashfile "%CORE%" SHA256 ^| findstr /r "^[0-9a-f]"`) do set "LOCK_ID=%%H"
if not exist "%STAMP%" set "REINSTALL=1"
if exist "%STAMP%" (
  set /p SAVED=<"%STAMP%"
  if not "!SAVED!"=="!LOCK_ID!" set "REINSTALL=1"
)

if "!REINSTALL!"=="1" (
  echo   installing what the app needs ^(a few hundred MB, once^)
  "%UV%" pip install --no-cache --python "%VENV%\Scripts\python.exe" -r "%CORE%" || goto :nocore
  REM Best effort. If PyTorch will not install, the app still opens; only the
  REM Bayesian optimiser is off, and launch.py's preflight says so.
  "%UV%" pip install --no-cache --python "%VENV%\Scripts\python.exe" -r "%BO%" 2>nul
  if errorlevel 1 (
    echo   note: no PyTorch for this machine - Bayesian optimisation will be off.
    echo         Everything else, including the GA and SLSQP searches, still runs.
  )
  >"%STAMP%" echo !LOCK_ID!
)

REM ----------------------------------------------------------------- run ---
"%VENV%\Scripts\python.exe" "%APP_DIR%\launch.py" %*
goto :eof

:nouv
echo.
echo   AeroBO could not start: could not download uv.
echo   Check the network connection and try again.
echo.
pause
exit /b 1

:novenv
echo.
echo   AeroBO could not start: could not create the Python environment.
echo.
pause
exit /b 1

:nocore
echo.
echo   AeroBO could not start: the install failed. The message above says why.
echo.
pause
exit /b 1
