@echo off
setlocal
title M.A.X.I.K. plant health
cd /d "%~dp0plant_cv"

rem Python 3.13 / 3.12 / 3.11 are the versions the packages are known to install on.
set "PY="
for %%V in (3.13 3.12 3.11) do if not defined PY (py -%%V --version >nul 2>nul && set "PY=py -%%V")
if not defined PY (python --version >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python was not found.
  echo Install Python 3.12 or 3.13 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment ^(first run only^)...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create the environment.
    pause
    exit /b 1
  )
)

if not exist ".venv\.maxik_installed" (
  echo Installing packages ^(first run only, 1-2 minutes, needs internet^)...
  ".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements-server.txt
  if errorlevel 1 (
    echo Installation failed. Check the internet connection and run this file again.
    echo If you have a very new Python, install 3.12 or 3.13 instead.
    pause
    exit /b 1
  )
  echo done> ".venv\.maxik_installed"
)

if not defined MAXIK_NO_BROWSER set "MAXIK_OPEN_BROWSER=1"
".venv\Scripts\python.exe" webapp.py
echo.
echo The server stopped.
pause
