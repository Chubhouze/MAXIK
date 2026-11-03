"""How the tomato pipeline does on REAL-WORLD photos (PlantDoc test split, prepared by
prepare_plantdoc.py) - the honest counterpart of evaluate_tomato_pipeline.py, which uses
PlantVillage's lab photos.

Two levels:
- leaves: every tomato leaf box of the test photos, cropped, given to the disease classifier alone.
- photos: every whole tomato test photo through the full pipeline (quality -> plant gate ->
  classifier [-> leaf detector + per-leaf analysis when --whole-plant]), compared with the photo's
  boxes: "diseased" if any box is a disease, and the most frequent box label.

    python scripts/evaluate_field.py
    python scripts/evaluate_field.py --model models_tomato/tomato_health.onnx --whole-plant
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maxik_cv.tomato import OnnxTomatoHealthClassifier, PlantDetector, TomatoInspector  # noqa: E402
from maxik_cv.tomato.class_mapping import DATASET_LABEL_TO_CONDITION  # noqa: E402
from maxik_cv.tomato.schema import Condition  # noqa: E402

DATA = ROOT.parent / "data"
# PlantDoc box label -> PlantVillage-style label used by our models (PlantDoc has no target spot, and
# only 2 spider-mite boxes, all in train)
PLANTDOC_TO_LABEL = {
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


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    # same preference as webapp.py: the field-trained model if present, else the lab-only one
    default_model = next((p for p in (ROOT / "models_tomato_field" / "tomato_health.onnx",
                                      ROOT / "models_tomato" / "tomato_health.onnx") if p.exists()), None)
    ap.add_argument("--model", default=str(default_model))
    ap.add_argument("--gate", default=str(ROOT / "models_synthetic" / "plant_gate_trained.onnx"))
    ap.add_argument("--detector", default=str(ROOT / "models_leaf" / "leaf_detector.onnx"))
    ap.add_argument("--whole-plant", action="store_true", help="enable the leaf detector stage")
    ap.add_argument("--species", default=str(ROOT / "models_species" / "species.onnx"),
                    help="crop classifier; used if the file exists (else every plant is assumed to be tomato)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "reports" / "field_eval.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    clf = OnnxTomatoHealthClassifier(args.model)
    detector = None
    if args.whole_plant:
        from maxik_cv.tomato.leaf_detector import LeafDetector
        detector = LeafDetector(args.detector)
    species = None
    if Path(args.species).exists():
        from maxik_cv.tomato.species_classifier import OnnxSpeciesClassifier
        species = OnnxSpeciesClassifier(args.species)
    print(f"model {Path(args.model).parent.name}/{Path(args.model).name}; species model: {'yes' if species else 'no'}; "
          f"leaf detector: {'yes' if detector else 'no'}")
    insp = TomatoInspector(PlantDetector.from_model_path(args.gate), species_classifier=species,
                           health_classifier=clf, leaf_detector=detector)

    boxes = [r for r in csv.DictReader(open(DATA / "plantdoc_od" / "split_boxes.csv", encoding="utf8"))
             if r["split"] == args.split and r["class"] in PLANTDOC_TO_LABEL]
    report = {"model": Path(args.model).name, "whole_plant": args.whole_plant}

    # ---- leaf level -------------------------------------------------------------------------
    per = collections.defaultdict(lambda: [0, 0])
    binary_ok = n = 0
    cache = {}
    for r in boxes:
        f = r["file"]
        if f not in cache:
            cache = {f: cv2.imread(str(DATA / "plantdoc_od" / f))}
        img = cache[f]
        x0, y0, x1, y1 = (int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue
        mx, my = int(0.1 * (x1 - x0)), int(0.1 * (y1 - y0))
        crop = img[max(0, y0 - my):y1 + my, max(0, x0 - mx):x1 + mx]
        want = DATASET_LABEL_TO_CONDITION[PLANTDOC_TO_LABEL[r["class"]]]
        got = clf.classify(crop).condition
        per[want.value][0] += int(got == want)
        per[want.value][1] += 1
        binary_ok += int((got == Condition.TOMATO_HEALTHY) == (want == Condition.TOMATO_HEALTHY))
        n += 1
    acc = sum(v[0] for v in per.values()) / max(n, 1)
    print(f"LEAVES ({n} real-world tomato leaf crops): exact condition {acc:.1%}, healthy-vs-diseased {binary_ok / max(n, 1):.1%}")
    for c, (ok, tot) in sorted(per.items()):
        print(f"   {c:34s} {ok:3d}/{tot:<3d} {ok / tot:.0%}")
    report["leaves"] = {"n": n, "exact": acc, "binary": binary_ok / max(n, 1), "per_class": dict(per)}

    # ---- photo level ------------------------------------------------------------------------
    by_photo = collections.defaultdict(list)
    gt_boxes = collections.defaultdict(list)
    for r in boxes:
        by_photo[r["file"]].append(DATASET_LABEL_TO_CONDITION[PLANTDOC_TO_LABEL[r["class"]]])
        gt_boxes[r["file"]].append(tuple(int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")))
    c = collections.Counter()
    for f, labels in by_photo.items():
        res = insp.inspect(cv2.imread(str(DATA / "plantdoc_od" / f)))
        c["gt_leaves"] += len(labels)
        if res.mode == "whole_plant":
            # end-to-end per leaf: each true leaf matched to a found leaf (IoU >= 0.5), then its diagnosis
            c["whole_plant_photos"] += 1
            used = set()
            for gb, gl in zip(gt_boxes[f], labels):
                best, bi = 0.0, None
                for i, leaf in enumerate(res.leaves):
                    if i not in used and (v := iou(gb, leaf["box"])) > best:
                        best, bi = v, i
                if bi is not None and best >= 0.5:
                    used.add(bi)
                    c["leaves_found"] += 1
                    c["leaf_diag_ok"] += int(res.leaves[bi]["condition"] == gl.value)
        diseased_truth = any(l != Condition.TOMATO_HEALTHY for l in labels)
        majority = collections.Counter(labels).most_common(1)[0][0]
        c["photos"] += 1
        if not res.plant_detected:
            c["not_plant" if res.status_label != "IMAGE QUALITY ISSUE" else "quality_reject"] += 1
            continue
        if res.status_label == "IMAGE QUALITY ISSUE":
            c["quality_reject"] += 1
            continue
        c["analysed"] += 1
        c["uncertain"] += int(res.condition == Condition.TOMATO_UNCERTAIN)
        said_diseased = res.condition not in (Condition.TOMATO_HEALTHY, Condition.TOMATO_UNCERTAIN)
        if res.condition != Condition.TOMATO_UNCERTAIN:
            c["answered"] += 1
            c["binary_ok"] += int(said_diseased == diseased_truth)
            c["exact_ok"] += int(res.condition in set(labels))
            c["majority_ok"] += int(res.condition == majority)
    a = max(c["answered"], 1)
    print(f"\nPHOTOS ({c['photos']} real-world tomato photos, whole image through the pipeline"
          f"{' + leaf detector' if args.whole_plant else ''}):")
    print(f"   rejected as not a plant {c['not_plant']}, rejected for image quality {c['quality_reject']}, "
          f"analysed {c['analysed']} (uncertain {c['uncertain']}, answered {c['answered']})")
    print(f"   of the answered: healthy-vs-diseased right {c['binary_ok'] / a:.1%}, condition present on the photo "
          f"{c['exact_ok'] / a:.1%}, matches the most common label {c['majority_ok'] / a:.1%}")
    if args.whole_plant:
        print(f"   whole-plant mode used on {c['whole_plant_photos']} photos; true leaves found (IoU>=0.5): "
              f"{c['leaves_found']}/{c['gt_leaves']} = {c['leaves_found'] / max(c['gt_leaves'], 1):.1%}; "
              f"diagnosis right on found leaves: {c['leaf_diag_ok'] / max(c['leaves_found'], 1):.1%}")
    report["photos"] = dict(c)
    report["ccmt"] = evaluate_ccmt(insp, args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf8")


CCMT_EXACT = {"healthy": Condition.TOMATO_HEALTHY, "septoria leaf spot": Condition.TOMATO_SEPTORIA_LEAF_SPOT,
              "leaf curl": Condition.TOMATO_YELLOW_LEAF_CURL_VIRUS}


def evaluate_ccmt(insp, split: str) -> dict:
    """Whole farm photos of tomato plants from CCMT (Ghana) through the full pipeline. All five CCMT
    tomato classes count for healthy-vs-diseased; only the three that exist in our model count for the
    exact condition ("leaf blight" and "verticillium wilt" have no counterpart)."""
    out = {}
    root = DATA / "ccmt_split" / split
    if not root.exists():
        return out
    tot = collections.Counter()
    print(f"\nCCMT FARM PHOTOS ({split} split, whole photo through the pipeline):")
    for d in sorted(root.glob("Tomato___*")):
        name = d.name.split("___")[1]
        c = collections.Counter()
        for f in sorted(d.glob("*")):
            res = insp.inspect(cv2.imread(str(f)))
            c["n"] += 1
            if not res.plant_detected or res.status_label == "IMAGE QUALITY ISSUE":
                c["rejected"] += 1
                continue
            if res.condition == Condition.UNSUPPORTED_PLANT:
                c["unsupported"] += 1
                continue
            if res.condition == Condition.TOMATO_UNCERTAIN:
                c["uncertain"] += 1
                continue
            c["answered"] += 1
            said_sick = res.condition != Condition.TOMATO_HEALTHY
            c["binary_ok"] += int(said_sick == (name != "healthy"))
            if name in CCMT_EXACT:
                c["exact_n"] += 1
                c["exact_ok"] += int(res.condition == CCMT_EXACT[name])
            c["whole_plant"] += int(res.mode == "whole_plant")
        a = max(c["answered"], 1)
        exact = f", exact {c['exact_ok'] / max(c['exact_n'], 1):.0%}" if c["exact_n"] else ""
        print(f"   {name:20s} n {c['n']:3d}: rejected {c['rejected']:3d}, not-tomato {c['unsupported']:3d}, "
              f"uncertain {c['uncertain']:3d}, answered {c['answered']:3d} -> healthy-vs-diseased "
              f"{c['binary_ok'] / a:.0%}{exact} (whole-plant mode on {c['whole_plant']})")
        out[name] = dict(c)
        tot.update(c)
    a = max(tot["answered"], 1)
    print(f"   TOTAL {tot['n']} photos: rejected {tot['rejected']}, not-tomato {tot['unsupported']}, uncertain "
          f"{tot['uncertain']}; of the answered healthy-vs-diseased {tot['binary_ok'] / a:.1%}, exact condition "
          f"(3 mappable classes) {tot['exact_ok'] / max(tot['exact_n'], 1):.1%}")
    out["total"] = dict(tot)
    return out


if __name__ == "__main__":
    main()
