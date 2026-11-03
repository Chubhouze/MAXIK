"""Robot -> server: capture one photo on the Raspberry Pi (or take an existing file), POST it to
the analysis server (webapp.py, /api/tomato) and print the JSON result. stdlib + OpenCV only.

    python3 scripts/send_to_server.py --url http://192.168.1.50:5055 --token SECRET --point row1-plant3 --soil 0.4
    python3 scripts/send_to_server.py --url http://192.168.1.50:5055 --file leaf.jpg
    python3 scripts/send_to_server.py --url http://192.168.1.50:5055 --resend
    python3 scripts/send_to_server.py --local --point row1-plant3            # analyse on the robot, no network
    python3 scripts/send_to_server.py --url ... --fallback-local             # server if reachable, else robot

If the server is unreachable the photo is kept in outbox/ (nothing is lost); --resend retries them.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2

OUTBOX = Path("outbox")
ROOT = Path(__file__).resolve().parents[1]


def build_local_inspector():
    """The same models as the server (webapp.py): field-trained disease model if present, plus the optional
    crop check and leaf detector (whole-plant mode)."""
    sys.path.insert(0, str(ROOT))
    from maxik_cv.tomato import OnnxTomatoHealthClassifier, PlantDetector, TomatoInspector
    from maxik_cv.tomato.leaf_detector import LeafDetector
    from maxik_cv.tomato.species_classifier import OnnxSpeciesClassifier

    gate = ROOT / "models_synthetic" / "plant_gate_trained.onnx"
    health = next((p for p in (ROOT / "models_tomato_field" / "tomato_health.onnx",
                               ROOT / "models_tomato" / "tomato_health.onnx") if p.exists()), None)
    if not gate.exists() or health is None:
        sys.exit(f"local analysis needs {gate} and models_tomato_field/tomato_health.onnx")
    species, leaves = ROOT / "models_species" / "species.onnx", ROOT / "models_leaf" / "leaf_detector.onnx"
    return TomatoInspector(PlantDetector.from_model_path(gate),
                           species_classifier=OnnxSpeciesClassifier(species) if species.exists() else None,
                           health_classifier=OnnxTomatoHealthClassifier(health),
                           leaf_detector=LeafDetector(leaves) if leaves.exists() else None)


def analyze_locally(frame, point: str | None, sensors: dict) -> dict:
    """Run the whole pipeline on the robot itself (ONNX Runtime on CPU, no network). History goes to
    robot_history.db next to the code, so trends still work offline."""
    inspector = build_local_inspector()  # also puts the package on sys.path
    from maxik_cv.tomato.history import TomatoHistory
    from maxik_cv.tomato.service import analyze

    result, inspection_id = analyze(inspector, TomatoHistory(ROOT / "robot_history.db"), frame, point, sensors)
    return {**result.to_dict(), "inspectionId": inspection_id, "analyzedOn": "robot"}


def capture_frame():
    """Same lock-AE/AWB approach as capture_pi.py: auto white balance shifts leaf colours between shots."""
    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (2304, 1296), "format": "RGB888"}))
    cam.start()
    time.sleep(2)
    md = cam.capture_metadata()
    cam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": md["ExposureTime"],
                      "AnalogueGain": md["AnalogueGain"], "ColourGains": md["ColourGains"]})
    time.sleep(0.5)
    frame = cam.capture_array()  # BGR order, OpenCV-ready
    cam.stop()
    return frame


def encode(frame, max_side: int) -> bytes:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:  # the server downsizes anyway; sending less saves Wi-Fi time
        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def post(url: str, jpeg: bytes, token: str | None, timeout: float, params: dict | None = None) -> dict:
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
    req = urllib.request.Request(url.rstrip("/") + "/api/tomato" + (f"?{query}" if query else ""),
                                 data=jpeg, method="POST", headers={"Content-Type": "image/jpeg"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="server base URL, e.g. http://192.168.1.50:5055 (not needed with --local)")
    ap.add_argument("--token", help="API token, if the server was started with MAXIK_API_TOKEN")
    ap.add_argument("--file", help="send this image instead of capturing")
    ap.add_argument("--resend", action="store_true", help="send everything waiting in outbox/ and exit")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--point", help="plant/point ID; the server keeps a history per point and reports the trend")
    ap.add_argument("--soil", type=float, help="soil moisture, 0..1")
    ap.add_argument("--temp", type=float, help="air temperature, degC")
    ap.add_argument("--humidity", type=float, help="relative humidity, 0..1")
    ap.add_argument("--local", action="store_true", help="analyse on this machine, do not contact the server")
    ap.add_argument("--fallback-local", action="store_true",
                    help="if the server is unreachable, analyse locally AND keep the photo in outbox/ for later upload")
    args = ap.parse_args()
    params = {"point": args.point, "soil_moisture": args.soil, "temperature": args.temp, "humidity": args.humidity}
    sensors = {k: v for k, v in params.items() if k != "point" and v is not None}
    if not args.url and not args.local:
        ap.error("--url is required unless --local is given")

    if args.resend:
        failed = 0
        for f in sorted(OUTBOX.glob("*.jpg")):
            meta = f.with_suffix(".json")  # point/sensors saved next to the photo when it was queued
            try:
                sent = json.loads(meta.read_text()) if meta.exists() else {}
                print(f.name, json.dumps(post(args.url, f.read_bytes(), args.token, args.timeout, sent), ensure_ascii=False))
                f.unlink()
                meta.unlink(missing_ok=True)
            except (urllib.error.URLError, TimeoutError) as e:
                failed += 1
                print(f"{f.name}: still unreachable ({e})", file=sys.stderr)
        sys.exit(1 if failed else 0)

    frame = cv2.imread(args.file) if args.file else capture_frame()
    if frame is None:
        sys.exit(f"could not read {args.file}")
    if args.local:
        print(json.dumps(analyze_locally(frame, args.point, sensors), indent=2, ensure_ascii=False))
        return
    jpeg = encode(frame, args.max_side)

    try:
        print(json.dumps(post(args.url, jpeg, args.token, args.timeout, params), indent=2, ensure_ascii=False))
    except urllib.error.HTTPError as e:  # server answered with an error (bad token, undecodable image)
        sys.exit(f"server rejected the photo: HTTP {e.code} {e.read().decode(errors='replace')}")
    except (urllib.error.URLError, TimeoutError) as e:
        OUTBOX.mkdir(exist_ok=True)
        stamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        path = OUTBOX / f"{stamp}.jpg"
        path.write_bytes(jpeg)
        (OUTBOX / f"{stamp}.json").write_text(json.dumps({k: v for k, v in params.items() if v is not None}))
        if args.fallback_local:
            print(f"server unreachable ({e}); analysing on the robot, photo kept in {path} for later upload",
                  file=sys.stderr)
            print(json.dumps(analyze_locally(frame, args.point, sensors), indent=2, ensure_ascii=False))
            return
        sys.exit(f"server unreachable ({e}); photo kept in {path} - retry later with --resend")


if __name__ == "__main__":
    main()
