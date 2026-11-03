"""Robot patrol along a bed with a downward-looking camera: take a frame every `--step` cm (or every
`--interval` seconds), analyse it - on the server (/api/bed) or on the robot itself (--local) - and print
what was found as it drives. Frames are stored per bed and position, see /beds/<bed> on the server.

    python3 scripts/patrol_bed.py --url http://SERVER:5055 --token ... --bed bed-1 --step 30 --frames 20
    python3 scripts/patrol_bed.py --local --bed bed-1 --step 30 --frames 20          # no network
    python3 scripts/patrol_bed.py --url ... --bed demo --files "photos/*.jpg" --step 30   # replay saved photos

Position: the robot's navigation should supply the real distance; until then `--start` + n x `--step`
is used (set --step to how far the robot moves between frames). With --position-file the current
position in cm is read from that file before every frame (written by the navigation code).
"""
import argparse
import glob
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from send_to_server import capture_frame, encode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def post_bed(url, jpeg, token, params, timeout=60) -> dict:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url.rstrip("/") + "/api/bed?" + query, data=jpeg, method="POST",
                                 headers={"Content-Type": "image/jpeg"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def local_analyzer():
    sys.path.insert(0, str(ROOT))
    from maxik_cv.tomato.bed import analyze_bed
    from maxik_cv.tomato.history import TomatoHistory
    from maxik_cv.tomato.leaf_detector import LeafDetector
    model = ROOT / "models_bed" / "bed_detector.onnx"
    if not model.exists():
        sys.exit(f"--local needs {model}")
    det, hist = LeafDetector(model), TomatoHistory(ROOT / "robot_history.db")

    def run(frame, bed, pos, sensors):
        report = analyze_bed(frame, det, sensors).to_dict()
        report.update({"bed": bed, "position": pos, "frameId": hist.add_bed_frame(bed, pos, report, frame)})
        return report
    return run


def line(pos, r) -> str:
    damaged = [p for p in r["plants"] if p["kind"] == "crop" and (p["affectedFraction"] or 0) >= 0.10]
    worst = ", ".join(f"#{p['index']} {p['affectedFraction']:.0%}" for p in sorted(damaged, key=lambda p: -p["affectedFraction"])[:3])
    return (f"{pos if pos is not None else '?':>7} cm | crops {r['crops']:3d} (damaged {r['damagedCrops']}{': ' + worst if worst else ''})"
            f" | weeds {r['weeds']:3d} ({r['weedCover']:.0%} cover)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--token")
    ap.add_argument("--bed", required=True)
    ap.add_argument("--local", action="store_true", help="analyse on the robot (models_bed/bed_detector.onnx)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until Ctrl+C or files run out)")
    ap.add_argument("--start", type=float, default=0.0, help="position of the first frame, cm")
    ap.add_argument("--step", type=float, help="cm the robot moves between frames")
    ap.add_argument("--position-file", help="file the navigation writes the current position (cm) into")
    ap.add_argument("--interval", type=float, default=0.0, help="seconds to wait between frames")
    ap.add_argument("--files", help="replay these photos instead of the camera (glob)")
    ap.add_argument("--soil", type=float)
    ap.add_argument("--temp", type=float)
    ap.add_argument("--humidity", type=float)
    args = ap.parse_args()
    if not args.local and not args.url:
        ap.error("--url is required unless --local is given")
    sensors = {k: v for k, v in {"soil_moisture": args.soil, "temperature": args.temp, "humidity": args.humidity}.items()
               if v is not None}
    files = sorted(glob.glob(args.files)) if args.files else None
    run_local = local_analyzer() if args.local else None

    n, totals = 0, {"crops": 0, "weeds": 0, "damagedCrops": 0}
    try:
        while (not args.frames or n < args.frames) and (files is None or n < len(files)):
            if args.position_file:
                pos = float(Path(args.position_file).read_text().strip())
            else:
                pos = args.start + n * args.step if args.step is not None else None
            frame = cv2.imread(files[n]) if files else capture_frame()
            if run_local:
                r = run_local(frame, args.bed, pos, sensors)
            else:
                try:
                    r = post_bed(args.url, encode(frame, 1920), args.token, {"bed": args.bed, "position": pos, **sensors})
                except (urllib.error.URLError, TimeoutError) as e:
                    print(f"{pos} cm: server unreachable ({e}) - frame skipped", file=sys.stderr)
                    n += 1
                    continue
            print(line(pos, r), flush=True)
            for k in totals:
                totals[k] += r[k]
            n += 1
            if args.interval:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    print(f"pass finished: {n} frames, {totals['crops']} crop plants seen ({totals['damagedCrops']} with visible damage), "
          f"{totals['weeds']} weeds. Frames overlap, so a plant can be counted in two frames.")


if __name__ == "__main__":
    main()
