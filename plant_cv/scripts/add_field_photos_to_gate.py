"""Add REAL-WORLD plant photos to the plant-gate dataset (data/plant_gate_split) as "plant" positives:
every PlantDoc photo (13 crops, plants/branches in the wild) and a crop- and class-balanced sample of
CCMT farm photos (cashew, cassava, maize, tomato). Each keeps the split it already has in those
datasets, so the gate's test set contains real-world photos no model was trained on.

Why: the gate had only seen lab leaves and flower photos as plants, and rejected 4-6% of real-world
plant photos (PlantDoc test 96.3%, CCMT test 95.4% recognised) while rejecting 0% of non-plants.

    python scripts/add_field_photos_to_gate.py
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
OUT = DATA / "plant_gate_split"
CCMT_PER_CLASS = {"train": 220, "val": 40, "test": 40}   # 22 classes -> ~4.8k / 0.9k / 0.9k photos


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            dst.hardlink_to(src)
        except OSError:
            import shutil
            shutil.copy2(src, dst)


def main():
    added = defaultdict(int)
    split_of = {r["file"]: r["split"] for r in csv.DictReader(open(DATA / "plantdoc_od" / "split_boxes.csv", encoding="utf8"))}
    for f, split in split_of.items():
        src = DATA / "plantdoc_od" / f
        link(src, OUT / split / "plant" / f"plantdoc_{src.name}")
        added[(split, "plantdoc")] += 1

    rng = random.Random(0)
    for split, n in CCMT_PER_CLASS.items():
        for d in sorted((DATA / "ccmt_split" / split).iterdir()):
            files = sorted(d.glob("*"))
            for f in rng.sample(files, min(n, len(files))):
                link(f, OUT / split / "plant" / f"ccmt_{d.name.replace(' ', '_')}_{f.name}")
                added[(split, "ccmt")] += 1

    for split in ("train", "val", "test"):
        print(f"{split:5s} +{added[(split, 'plantdoc')]} PlantDoc, +{added[(split, 'ccmt')]} CCMT -> plant "
              f"{len(list((OUT / split / 'plant').glob('*')))}, not_plant {len(list((OUT / split / 'not_plant').glob('*')))}")


if __name__ == "__main__":
    main()
