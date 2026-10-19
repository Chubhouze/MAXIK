"""Fold more PlantVillage photos into the plant-gate dataset (data/plant_gate_split) as "plant"
positives: all 10 tomato classes (broader disease/pest visual diversity than before) plus a
healthy sample from 6 other crops (apple, corn, grape, potato, bell pepper, strawberry) so the
gate recognizes plants in general, not just tomato-leaf-shaped or flower-shaped ones.

Uses dedup_split.py per class so near-duplicate photo bursts don't leak between train/val/test
of the GATE dataset either.

    python scripts/add_crops_to_gate_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_split import dedup_split

RAW = Path(r"C:\Users\chub\Desktop\MAXIK\data\PlantVillage-Dataset\raw\color")
OUT = Path(r"C:\Users\chub\Desktop\MAXIK\data\plant_gate_split")
CLASSES = [
    "Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Bacterial_spot", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Apple___healthy", "Corn_(maize)___healthy", "Grape___healthy", "Potato___healthy",
    "Pepper,_bell___healthy", "Strawberry___healthy",
]
SEED = 0


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            dst.hardlink_to(src)
        except OSError:
            import shutil
            shutil.copy2(src, dst)


def main():
    counts = {"train": 0, "val": 0, "test": 0}
    for cls in CLASSES:
        files = sorted((RAW / cls).glob("*"))
        train, val, test = dedup_split(files, seed=SEED)
        for split, fs in [("train", train), ("val", val), ("test", test)]:
            safe_cls = cls.replace("/", "_").replace(",", "")
            for f in fs:
                link(f, OUT / split / "plant" / f"{safe_cls}_{f.name}")
            counts[split] += len(fs)
        print(f"{cls:45s} train={len(train)} val={len(val)} test={len(test)}")

    for split in ["train", "val", "test"]:
        n_plant = len(list((OUT / split / "plant").glob("*")))
        n_not_plant = len(list((OUT / split / "not_plant").glob("*")))
        print(f"{split:5s} plant={n_plant} not_plant={n_not_plant}")


if __name__ == "__main__":
    main()
