"""Build the Raspberry Pi package: one .tar.gz with the code, models and installer the robot needs (no training
scripts, datasets or checkpoints). Shell files get Unix line endings and the executable bit, so the archive
works on the Pi even when built on Windows.

    python scripts/build_pi_package.py                  # -> dist/MAXIK-pi-YYYY-MM-DD.tar.gz
    python scripts/build_pi_package.py --out robot.tar.gz

On the Pi:  tar xzf MAXIK-pi-*.tar.gz && cd maxik && ./install.sh
"""
import argparse
import io
import tarfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "maxik"

# What the server image ships (see Dockerfile) plus the robot scripts and the installer.
FILES = [
    "webapp.py", "dashboard_ui.py", "bed_ui.py", "requirements-pi.txt", "DATA_COLLECTION_GUIDE.md",
    "scripts/send_to_server.py", "scripts/patrol_bed.py", "scripts/capture_pi.py", "scripts/selftest.py",
    "models_synthetic/plant_gate_trained.onnx", "models_synthetic/plant_gate_trained.json",
    "models_synthetic/plant_health.onnx", "models_synthetic/plant_health.json",
    "models_tomato_field/tomato_health.onnx", "models_tomato_field/tomato_health.json",
    "models_leaf/leaf_detector.onnx", "models_leaf/leaf_detector.json",
    "models_species/species.onnx", "models_species/species.json",
    "models_bed/bed_detector.onnx", "models_bed/bed_detector.json",
    "pi/maxik", "pi/robot.conf.example", "pi/maxik-server.service", "pi/README.md",
]
TREES = ["maxik_cv", "samples"]
RENAMED = {"pi/install.sh": "install.sh", "pi/README.md": "README.md"}  # on top, where people look first
EXECUTABLE = {"install.sh", "pi/maxik"}
UNIX_TEXT = {".sh", ".service", ".example", ""}  # "" = pi/maxik (no extension)


def collect() -> list[Path]:
    paths = [ROOT / f for f in FILES] + [ROOT / "pi" / "install.sh"]
    for tree in TREES:
        paths += [p for p in sorted((ROOT / tree).rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    return list(dict.fromkeys(paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / f"MAXIK-pi-{date.today()}.tar.gz")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.out, "w:gz") as tar:
        for path in collect():
            rel = path.relative_to(ROOT).as_posix()
            name = RENAMED.get(rel, rel)
            data = path.read_bytes()
            if path.suffix in UNIX_TEXT and path.parent.name == "pi":
                data = data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(f"{PREFIX}/{name}")
            info.size, info.mtime = len(data), int(time.time())
            info.mode = 0o755 if name in EXECUTABLE else 0o644
            tar.addfile(info, io.BytesIO(data))
    print(f"{args.out}  {args.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
