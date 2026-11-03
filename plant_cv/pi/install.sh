#!/usr/bin/env bash
# M.A.X.I.K. installer for the robot's Raspberry Pi (64-bit Raspberry Pi OS, Bookworm or newer).
#   ./install.sh            camera + analysis on the Pi or on a server; adds the `maxik` command
#   ./install.sh --server   also serve the web page from this Pi (http://<pi-address>:5055), started at boot
# Run it from the unpacked folder as the normal user; it asks for sudo when it needs it. Safe to re-run.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# Package layout: install.sh next to maxik_cv/. Repository layout (git clone): plant_cv/pi/install.sh.
if [ ! -d "$DIR/maxik_cv" ] && [ -d "$DIR/../maxik_cv" ]; then DIR="$(cd "$DIR/.." && pwd)"; fi
WITH_SERVER=0
for arg in "$@"; do
  case "$arg" in
    --server) WITH_SERVER=1 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see ./install.sh --help)" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -eq 0 ]; then
  echo "Run as the normal user (e.g. pi), without sudo: the script asks for sudo itself." >&2
  exit 1
fi
if [ "$(uname -m)" != "aarch64" ]; then
  echo "Needs 64-bit Raspberry Pi OS (aarch64); this system is $(uname -m). onnxruntime has no 32-bit ARM build." >&2
  exit 1
fi

echo "== 1/4 System packages (sudo)"
sudo apt-get update
# numpy, OpenCV and Pillow come from apt: they are built for the Pi and match the picamera2 package.
sudo apt-get install -y python3-venv python3-pip python3-numpy python3-opencv python3-pil libgomp1
sudo apt-get install -y python3-picamera2 \
  || echo "WARNING: python3-picamera2 is not available - the camera will not work (analysis of files still will)."

echo "== 2/4 Python environment"
# --system-site-packages: picamera2/libcamera cannot be installed with pip, the venv uses the apt ones.
python3 -m venv --system-site-packages "$DIR/.venv"
PY="$DIR/.venv/bin/python"
# pip must not replace the system numpy that picamera2 and OpenCV were built against.
"$PY" -c "import numpy; print('numpy==' + numpy.__version__)" > "$DIR/.venv/constraints.txt"
"$PY" -m pip install --disable-pip-version-check -c "$DIR/.venv/constraints.txt" -r "$DIR/requirements-pi.txt"
if [ "$WITH_SERVER" -eq 1 ]; then
  "$PY" -m pip install --disable-pip-version-check -c "$DIR/.venv/constraints.txt" "flask>=3.0" "waitress>=3.0"
fi

echo "== 3/4 Settings and the maxik command"
[ -f "$DIR/robot.conf" ] || cp "$DIR/pi/robot.conf.example" "$DIR/robot.conf"
chmod +x "$DIR/pi/maxik"
sudo ln -sf "$DIR/pi/maxik" /usr/local/bin/maxik
if [ "$WITH_SERVER" -eq 1 ]; then
  sed "s|@USER@|$USER|g; s|@DIR@|$DIR|g" "$DIR/pi/maxik-server.service" \
    | sudo tee /etc/systemd/system/maxik-server.service > /dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable maxik-server
  sudo systemctl restart maxik-server
fi

echo "== 4/4 Self-test on the sample photos"
"$PY" "$DIR/scripts/selftest.py"

echo
echo "Installed. Settings: $DIR/robot.conf (where photos are analysed, server address, token)."
echo "  maxik camera      test shot from the camera"
echo "  maxik photo       photograph one plant and analyse it"
echo "  maxik patrol --bed bed-1 --step 30 --frames 20    pass along a bed, camera looking down"
echo "  maxik help        all commands"
if [ "$WITH_SERVER" -eq 1 ]; then
  echo "Web page: http://$(hostname -I | awk '{print $1}'):5055/tomato   (service: sudo systemctl status maxik-server)"
fi
