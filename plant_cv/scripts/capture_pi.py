"""Raspberry Pi: capture a dataset / inspection photo with FIXED exposure and white balance.

Auto white balance shifts leaf colours between shots, which breaks yellow/brown detection -
so AWB and exposure are locked after a short warm-up.
    python3 scripts/capture_pi.py --point p1 --label healthy --plant basil03
    python3 scripts/capture_pi.py --point p1 --inspect --model models/plant_health.onnx
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True, help="inspection point id, e.g. p1")
    ap.add_argument("--label", help="class folder for dataset collection (healthy/dryness/yellowing/weak_damage)")
    ap.add_argument("--plant", default="plant", help="plant id, used to split train/test by plant")
    ap.add_argument("--inspect", action="store_true", help="run the pipeline right after capture")
    ap.add_argument("--model", default="models/plant_health.onnx")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (2304, 1296), "format": "RGB888"}))
    cam.start()
    time.sleep(2)  # let AE/AWB converge once
    md = cam.capture_metadata()
    cam.set_controls({"AeEnable": False, "AwbEnable": False,
                      "ExposureTime": md["ExposureTime"], "AnalogueGain": md["AnalogueGain"],
                      "ColourGains": md["ColourGains"]})
    time.sleep(0.5)
    frame = cam.capture_array()  # RGB888 in picamera2 is actually BGR order -> OpenCV-ready
    cam.stop()

    import cv2

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    folder = Path(args.out) / (args.label or "unlabeled")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{args.plant}_{args.point}_{ts}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("saved", path)

    if args.inspect:
        from maxik_cv.history import InspectionHistory
        from maxik_cv.pipeline import PlantInspector

        sensors = {}  # TODO: fill from the sensor module (soil_moisture 0..1, temperature, humidity, light)
        hist = InspectionHistory()
        report = PlantInspector(args.model).inspect(frame, sensors, hist.last(args.point))
        hist.add(args.point, ts, str(path), report)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
