"""Clean PlantDoc (downloaded by download_plantdoc.py) into a leak-free split and leaf crops.

1. Leakage: 35 of PlantDoc's 236 test photos are near-duplicates of train photos (perceptual hash;
   47 train photos involved). Those train photos are dropped so the official test split stays
   complete and honest.
2. Validation: 10% of the remaining train photos, split by near-duplicate cluster (dedup_split.py).
3. Boxes: rescaled where the CSV's image size differs from the real file (90 boxes), clipped to the
   image (4 boxes); boxes under 20 px are skipped for crops.

Writes data/plantdoc_od/split_boxes.csv (split,file,width,height,class,xmin,ymin,xmax,ymax in real
pixels) and data/plantdoc_leaves/<split>/<class>/*.jpg (each leaf box with 10% margin), the latter
for training/evaluating leaf classifiers on real-world photos.

    python scripts/prepare_plantdoc.py
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_split import HAMMING_THRESHOLD, ahash, cluster_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT.parent / "data" / "plantdoc_od"
LEAVES = ROOT.parent / "data" / "plantdoc_leaves"
MIN_CROP = 20
SEED = 0


def class_dir(name: str) -> str:
    return name.strip().replace(" ", "_")


def main():
    rows = {s: list(csv.DictReader(open(D / f"boxes_{s}.csv", encoding="utf8"))) for s in ("train", "test")}
    files = {s: sorted({r["file"] for r in rows[s]}) for s in rows}
    hashes = {f: ahash(D / f) for s in files for f in files[s]}

    leaked = {tr for tr in files["train"] for te in files["test"]
              if bin(hashes[tr] ^ hashes[te]).count("1") <= HAMMING_THRESHOLD}
    train_files = [f for f in files["train"] if f not in leaked]
    print(f"dropped {len(leaked)} train photos that duplicate test photos")

    clusters = cluster_files([D / f for f in train_files])
    random.Random(SEED).shuffle(clusters)
    val_target = 0.1 * len(train_files)
    val_files = set()
    for c in clusters:
        if len(val_files) >= val_target:
            break
        val_files.update(str(p.relative_to(D)).replace("\\", "/") for p in c)
    split_of = {f: ("val" if f in val_files else "train") for f in train_files}
    split_of.update({f: "test" for f in files["test"]})

    sizes = {}
    out_rows, crops = [], defaultdict(int)
    for r in rows["train"] + rows["test"]:
        f = r["file"]
        if f not in split_of:
            continue
        if f not in sizes:
            img = cv2.imread(str(D / f))
            sizes[f] = img.shape[:2]
        h, w = sizes[f]
        cw, ch = float(r["width"]), float(r["height"])
        sx, sy = (w / cw, h / ch) if cw > 0 and ch > 0 else (1.0, 1.0)  # some rows have size 0: boxes are in real px
        x0, x1 = sorted((max(0.0, float(r["xmin"]) * sx), min(w - 1.0, float(r["xmax"]) * sx)))
        y0, y1 = sorted((max(0.0, float(r["ymin"]) * sy), min(h - 1.0, float(r["ymax"]) * sy)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        out_rows.append([split_of[f], f, w, h, r["class"].strip(), round(x0), round(y0), round(x1), round(y1)])

    with open(D / "split_boxes.csv", "w", newline="", encoding="utf8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["split", "file", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
        wr.writerows(out_rows)

    cache = {}
    for i, (split, f, w, h, cls, x0, y0, x1, y1) in enumerate(out_rows):
        if x1 - x0 < MIN_CROP or y1 - y0 < MIN_CROP:
            continue
        if f not in cache:
            cache = {f: cv2.imread(str(D / f))}
        mx, my = int(0.1 * (x1 - x0)), int(0.1 * (y1 - y0))
        crop = cache[f][max(0, y0 - my):min(h, y1 + my), max(0, x0 - mx):min(w, x1 + mx)]
        dst = LEAVES / split / class_dir(cls) / f"{Path(f).stem[:60]}_{i}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crops[split] += 1

    for s in ("train", "val", "test"):
        photos = len({row[1] for row in out_rows if row[0] == s})
        boxes = sum(1 for row in out_rows if row[0] == s)
        print(f"{s:5s} photos {photos:5d}  boxes {boxes:5d}  leaf crops {crops[s]:5d}")


if __name__ == "__main__":
    main()
