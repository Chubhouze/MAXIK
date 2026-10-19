"""TRL 4 validation: accuracy on the held-out test set, confusion matrix, error log, unknown-threshold sweep.

    python scripts/evaluate.py --data data/split/test --model models/plant_health.onnx
    python scripts/evaluate.py --data data/split/test --rules      # rule-based baseline, no model
Target from the application: >= 75% correct classification on the controlled dataset.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxik_cv import CLASSES, UNKNOWN  # noqa: E402
from maxik_cv.pipeline import PlantInspector  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET = 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/split/test")
    ap.add_argument("--model", default="models/plant_health.onnx")
    ap.add_argument("--rules", action="store_true", help="ignore model, evaluate rules only")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    insp = PlantInspector(None if args.rules else args.model, unknown_threshold=0.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls_dir in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            r = insp.inspect(img)
            rows.append({"file": str(f), "true": cls_dir.name, "pred": r["label"],
                         "conf": r["confidence"], "reasons": "; ".join(r["reasons"])})

    labels = CLASSES + [UNKNOWN]
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), int)
    for r in rows:
        cm[idx[r["true"]], idx[r["pred"]]] += 1

    acc = np.mean([r["true"] == r["pred"] for r in rows])
    print(f"\nimages: {len(rows)}   accuracy: {acc:.1%}   target {TARGET:.0%}: {'PASS' if acc >= TARGET else 'FAIL'}\n")
    print("confusion matrix (rows = true, cols = predicted)")
    print(" " * 13 + "".join(f"{c[:10]:>11s}" for c in labels))
    for i, c in enumerate(labels):
        if cm[i].sum():
            print(f"{c:13s}" + "".join(f"{v:11d}" for v in cm[i]))

    print("\nper class:")
    for i, c in enumerate(CLASSES):
        tp, fn, fp = cm[i, i], cm[i].sum() - cm[i, i], cm[:, i].sum() - cm[i, i]
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        print(f"  {c:13s} precision {prec:.2f}  recall {rec:.2f}  n={cm[i].sum()}")

    print("\nunknown threshold sweep (coverage = share of images given a definite answer):")
    sweep = []
    for th in np.arange(0.3, 0.95, 0.05):
        kept = [r for r in rows if r["conf"] >= th]
        cov = len(kept) / len(rows)
        a = np.mean([r["true"] == r["pred"] for r in kept]) if kept else 0
        sweep.append({"threshold": round(th, 2), "coverage": round(cov, 3), "accuracy": round(float(a), 3)})
        print(f"  th={th:.2f}  coverage {cov:6.1%}  accuracy on answered {a:6.1%}")

    with open(out / "predictions.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    (out / "metrics.json").write_text(json.dumps({
        "n": len(rows), "accuracy": float(acc), "target": TARGET, "labels": labels,
        "confusion_matrix": cm.tolist(), "threshold_sweep": sweep}, indent=2), encoding="utf8")
    errors = [r for r in rows if r["true"] != r["pred"]]
    print(f"\n{len(errors)} errors logged to {out / 'predictions.csv'} (for the error analysis section)")


if __name__ == "__main__":
    main()
