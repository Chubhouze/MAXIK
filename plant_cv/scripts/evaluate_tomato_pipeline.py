"""End-to-end evaluation of the tomato pipeline (quality gate -> plant gate -> classifier ->
confidence gate -> severity) on an ImageFolder split whose folder names are PlantVillage labels.
Reports what the user would actually see: correct / uncertain / rejected / confidently wrong, the
severity spread per class, and per-image latency. Optionally degrades the photos (--degrade) to check
that bad frames are rejected instead of answered wrongly.

    python scripts/evaluate_tomato_pipeline.py --per-class 40
    python scripts/evaluate_tomato_pipeline.py --per-class 40 --degrade gauss3
    python scripts/evaluate_tomato_pipeline.py --data path/to/own_photos --per-class 100   # folders named like the labels

Writes reports/tomato_pipeline_eval.json (add --out to change).
"""
import argparse
import collections
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maxik_cv.tomato import OnnxTomatoHealthClassifier, PlantDetector, TomatoInspector  # noqa: E402
from maxik_cv.tomato.class_mapping import DATASET_LABEL_TO_CONDITION  # noqa: E402

# PlantDoc (real field photos, CC-BY-4.0) names its folders differently; "Tomato Target Spot" has no
# folder there, so that class can't be checked on it.
PLANTDOC_MAP = {
    "Tomato leaf": "Tomato___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
}

DEGRADE = {
    "none": lambda i: i,
    "gauss1.5": lambda i: cv2.GaussianBlur(i, (0, 0), 1.5),
    "gauss3": lambda i: cv2.GaussianBlur(i, (0, 0), 3),
    "motion9": lambda i: cv2.filter2D(i, -1, _hline(9)),
    "motion21": lambda i: cv2.filter2D(i, -1, _hline(21)),
    "dark": lambda i: (i * 0.1).astype(np.uint8),
}


def _hline(k: int) -> np.ndarray:
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    return ker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "tomato_split" / "test"))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--degrade", default="none", choices=sorted(DEGRADE))
    ap.add_argument("--labelmap", default="plantvillage", choices=["plantvillage", "plantdoc"],
                    help="how the folder names map to classes (plantdoc: PlantDoc's own folder names)")
    ap.add_argument("--gate", default=str(ROOT / "models_synthetic" / "plant_gate_trained.onnx"))
    ap.add_argument("--model", default=str(ROOT / "models_tomato" / "tomato_health.onnx"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "tomato_pipeline_eval.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    insp = TomatoInspector(PlantDetector.from_model_path(args.gate),
                           health_classifier=OnnxTomatoHealthClassifier(args.model))
    degrade = DEGRADE[args.degrade]
    rows, times = {}, []
    print(f"degradation: {args.degrade}\n{'class':46s}{'n':>4s}{'right':>7s}{'unsure':>8s}{'reject':>8s}{'wrong':>7s}   severity L/M/H")
    for cls in sorted(os.listdir(args.data)):
        label = PLANTDOC_MAP.get(cls) if args.labelmap == "plantdoc" else cls
        if label not in DATASET_LABEL_TO_CONDITION:
            print(f"skipping folder {cls!r}: no matching class")
            continue
        want = DATASET_LABEL_TO_CONDITION[label].value
        files = sorted(glob.glob(os.path.join(args.data, cls, "*")))
        step = max(1, len(files) // args.per_class)
        c = collections.Counter()
        sev = collections.Counter()
        for f in files[::step][:args.per_class]:
            img = degrade(cv2.imread(f))
            t = time.time()
            r = insp.inspect(img)
            times.append(time.time() - t)
            c["n"] += 1
            if not r.plant_detected or r.status_label == "IMAGE QUALITY ISSUE":
                c["reject"] += 1
            elif r.condition.value == want:
                c["right"] += 1
            elif r.condition.value == "TOMATO_UNCERTAIN":
                c["unsure"] += 1
            else:
                c["wrong"] += 1
            sev[r.severity.value] += 1
        rows[cls] = {**c, "severity": dict(sev)}
        print(f"{cls:46s}{c['n']:4d}{c['right']:7d}{c['unsure']:8d}{c['reject']:8d}{c['wrong']:7d}"
              f"   {sev['LOW']}/{sev['MODERATE']}/{sev['HIGH']}")

    tot = collections.Counter()
    for v in rows.values():
        for k in ("n", "right", "unsure", "reject", "wrong"):
            tot[k] += v.get(k, 0)
    answered = tot["right"] + tot["wrong"]
    print(f"\nTOTAL n={tot['n']}: right {tot['right']} ({tot['right'] / tot['n']:.1%}), uncertain {tot['unsure']}, "
          f"rejected {tot['reject']}, confidently wrong {tot['wrong']}")
    if answered:
        print(f"accuracy when the pipeline gives a diagnosis: {tot['right'] / answered:.1%} ({answered} diagnoses)")
    print(f"latency per image: median {np.median(times) * 1000:.0f} ms, p95 {np.percentile(times, 95) * 1000:.0f} ms")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"degrade": args.degrade, "total": dict(tot), "per_class": rows}, indent=2),
                              encoding="utf8")


if __name__ == "__main__":
    main()
