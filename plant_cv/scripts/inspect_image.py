"""Run the pipeline on one image or a folder; saves a debug overlay next to the JSON report.

    python scripts/inspect_image.py photo.jpg --model models/plant_health.onnx --soil 0.2 --point p3
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxik_cv.history import InspectionHistory  # noqa: E402
from maxik_cv.pipeline import PlantInspector  # noqa: E402
from maxik_cv.segment import overlay, plant_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="models/plant_health.onnx")
    ap.add_argument("--soil", type=float, help="soil moisture 0..1")
    ap.add_argument("--point", help="inspection point id (enables history comparison)")
    ap.add_argument("--db", default="inspections.db")
    ap.add_argument("--out", default="reports/inspect")
    args = ap.parse_args()

    insp = PlantInspector(args.model)
    hist = InspectionHistory(args.db) if args.point else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p = Path(args.path)
    files = sorted(p.iterdir()) if p.is_dir() else [p]
    for f in files:
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        sensors = {"soil_moisture": args.soil} if args.soil is not None else {}
        prev = hist.last(args.point) if hist else None
        report = insp.inspect(img, sensors, prev)
        if hist:
            hist.add(args.point, datetime.now().isoformat(timespec="seconds"), str(f), report)
        cv2.imwrite(str(out / f"{f.stem}_mask.jpg"), overlay(img, plant_mask(img)))
        (out / f"{f.stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
        print(f"{f.name}: {report['label']} ({report['confidence']:.0%})  {'; '.join(report['reasons'])}")


if __name__ == "__main__":
    main()
