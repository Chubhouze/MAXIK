"""Build an ImageFolder train/val/test split from the PlantVillage tomato subset (all 10 classes),
using dedup_split.py so near-duplicate photo bursts stay on one side of the split instead of
leaking between train and test (see dedup_split.py docstring for why).

    python scripts/prepare_tomato_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_split import dedup_split

RAW = Path(r"C:\Users\chub\Desktop\MAXIK\data\PlantVillage-Dataset\raw\color")
OUT = Path(r"C:\Users\chub\Desktop\MAXIK\data\tomato_split")
CLASSES = [
    "Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Bacterial_spot", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
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
    for cls in CLASSES:
        files = sorted((RAW / cls).glob("*"))
        train, val, test = dedup_split(files, seed=SEED)
        for split, fs in [("train", train), ("val", val), ("test", test)]:
            for f in fs:
                link(f, OUT / split / cls / f.name)
            print(f"{split:5s} {cls:45s} {len(fs)}")


if __name__ == "__main__":
    main()
