"""Fold the PlantVillage tomato photos into the existing plant-gate dataset as additional
"plant" positives (data/plant_gate_split, built by prepare_plant_gate_dataset.py).

Why: the gate was trained on Flowers102 (whole blooming flowers) vs Imagenette. That generalizes
well to tomato leaf photos (98.8% recall tested directly), but misses a few severely necrotic/
blighted leaves - exactly the case a plant-health robot most needs to still recognize as "plant,
but sick" rather than reject. Adding real diseased-leaf photos as positives should close that gap.

    python scripts/add_tomato_to_gate_dataset.py
"""
import random
from pathlib import Path

RAW = Path(r"C:\Users\chub\Desktop\MAXIK\data\PlantVillage-Dataset\raw\color")
OUT = Path(r"C:\Users\chub\Desktop\MAXIK\data\plant_gate_split")
CLASSES = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold"]
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
    random.seed(SEED)
    counts = {"train": 0, "val": 0, "test": 0}
    for cls in CLASSES:
        files = sorted((RAW / cls).glob("*"))
        random.shuffle(files)
        n = len(files)
        n_train, n_val = int(n * 0.8), int(n * 0.1)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, fs in splits.items():
            for f in fs:
                link(f, OUT / split / "plant" / f"tomato_{cls}_{f.name}")
            counts[split] += len(fs)

    for split in ["train", "val", "test"]:
        n_plant = len(list((OUT / split / "plant").glob("*")))
        n_not_plant = len(list((OUT / split / "not_plant").glob("*")))
        print(f"{split:5s} plant={n_plant} not_plant={n_not_plant} (added {counts[split]} tomato photos)")


if __name__ == "__main__":
    main()
