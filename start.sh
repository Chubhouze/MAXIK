#!/usr/bin/env sh
# M.A.X.I.K. one-command start for Linux / macOS:  sh start.sh
set -e
cd "$(dirname "$0")/plant_cv"

PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install Python 3.12 or 3.13 (https://www.python.org/downloads/)." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating the Python environment (first run only)..."
  "$PY" -m venv .venv || { echo "On Debian/Ubuntu run: sudo apt install python3-venv" >&2; exit 1; }
fi

if [ ! -f .venv/.maxik_installed ]; then
  echo "Installing packages (first run only, 1-2 minutes, needs internet)..."
  .venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements-server.txt
  touch .venv/.maxik_installed
fi

[ -n "${MAXIK_NO_BROWSER:-}" ] || export MAXIK_OPEN_BROWSER=1
exec .venv/bin/python webapp.py
