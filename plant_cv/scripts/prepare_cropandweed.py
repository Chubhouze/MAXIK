"""Prepare CropAndWeed (download_cropandweed.py) for the top-down bed detector: every plant -> "crop",
"weed" or ignored, a split by recording session, and images downsized to 1280 px.

- crop: the official CropOrWeed2 crop list (maize, sugar beet, soybean, sunflower, potato, pea, beans,
  pumpkin) PLUS garden vegetables/herbs the official variant drops (courgette, radish, asparagus,
  parsley, hemp, sage, chives, garlic) - a garden-bed robot needs those.
- weed: the official CropOrWeed2 weed list plus the remaining wild species.
- ignored: the "Vegetation" fallback (tiny or ambiguous plants) and plants under 24 px at full
  resolution - too small to say anything about their health.
- split: whole recording sessions go to train / val / test (80/10/10 of the images), so frames of the
  same bed never appear on both sides.

    python scripts/prepare_cropandweed.py   -> data/cropandweed_prepared/{images/*.jpg, split_boxes.csv}
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "data" / "cropandweed"
OUT = ROOT.parent / "data" / "cropandweed_prepared"
LONG_SIDE = 1280
MIN_PLANT_PX = 24

OFFICIAL_CROP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 94, 24, 18, 13, 26, 27, 15]
GARDEN_CROP = [14, 16, 17, 19, 20, 23, 25, 92, 93]
OFFICIAL_WEED = [31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86, 32, 29, 33, 37, 49, 30, 44, 66, 87, 89, 91, 61, 79, 34,
                 41, 52, 35, 36, 78, 38, 39, 71, 72, 88, 42, 45, 70, 47, 51, 54, 58, 60, 80, 83, 96, 22, 63, 85, 56,
                 57, 64, 77, 50, 59, 67, 76]
OTHER_WILD = [21, 28, 40, 43, 46, 53, 55, 73, 82, 90, 95, 97, 98, 99]
KIND = {i: "crop" for i in OFFICIAL_CROP + GARDEN_CROP} | {i: "weed" for i in OFFICIAL_WEED + OTHER_WILD}


def main():
    images = {p.stem: p for p in (SRC / "images").glob("*")}
    ann = {p.stem: p for p in (SRC / "bboxes").glob("*.csv") if p.stem in images}
    print(f"{len(images)} images, {len(ann)} with annotations")

    by_session = defaultdict(list)
    for stem in sorted(ann):
        by_session[stem[:8]].append(stem)          # e.g. "ave-0035"
    sessions = sorted(by_session)
    random.Random(0).shuffle(sessions)
    split_of, n, total = {}, 0, len(ann)
    for s in sessions:
        split = "train" if n < 0.8 * total else "val" if n < 0.9 * total else "test"
        for stem in by_session[s]:
            split_of[stem] = split
        n += len(by_session[s])

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows, stats = [], Counter()
    for stem in sorted(ann):
        img = cv2.imread(str(images[stem]))
        if img is None:
            stats["unreadable"] += 1
            continue
        h, w = img.shape[:2]
        s = LONG_SIDE / max(h, w)
        nw, nh = round(w * s), round(h * s)
        dst = OUT / "images" / f"{stem}.jpg"
        if not dst.exists():
            cv2.imwrite(str(dst), cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
        kept = 0
        for r in csv.reader(open(ann[stem])):
            if not r:
                continue
            x0, y0, x1, y1, label = (int(v) for v in r[:5])
            kind = KIND.get(label)
            if kind is None:
                stats["ignored_vegetation_or_unknown"] += 1
                continue
            if min(x1 - x0, y1 - y0) < MIN_PLANT_PX:
                stats["ignored_too_small"] += 1
                continue
            rows.append([split_of[stem], f"images/{stem}.jpg", nw, nh, kind,
                         round(x0 * s, 1), round(y0 * s, 1), round(x1 * s, 1), round(y1 * s, 1)])
            stats[f"{split_of[stem]}_{kind}"] += 1
            kept += 1
        if not kept:  # bare soil / only ignored plants: keep the photo as a "nothing here" example
            rows.append([split_of[stem], f"images/{stem}.jpg", nw, nh, "", "", "", "", ""])
        stats[f"{split_of[stem]}_photos"] += 1

    with open(OUT / "split_boxes.csv", "w", newline="", encoding="utf8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["split", "file", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
        wr.writerows(rows)
    for k, v in sorted(stats.items()):
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
