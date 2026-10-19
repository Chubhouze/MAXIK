"""Build an ImageFolder-style plant / not_plant dataset from the downloaded raw sources, using
hardlinks (no extra disk space, same NTFS volume) instead of copying.

Positive (plant): Oxford Flowers102 (8189 photos, all flowering plants).
Negative (not_plant): Imagenette (13395 photos, 10 ImageNet classes, none plant-related).

    python scripts/prepare_plant_gate_dataset.py
"""
import random
from pathlib import Path

from torchvision.datasets import Flowers102

RAW = Path(r"C:\Users\chub\Desktop\MAXIK\data\plant_gate_raw")
OUT = Path(r"C:\Users\chub\Desktop\MAXIK\data\plant_gate_split")
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

    # --- plant: Flowers102 official splits (train/val used for train, test -> val+test here) ---
    counts = {"train": 0, "val": 0, "test": 0}
    for split in ["train", "val", "test"]:
        ds = Flowers102(root=str(RAW), split=split, download=False)
        files = ds._image_files
        random.shuffle(files)
        # test split (6149 imgs) is the largest -> carve out our val set from it too.
        if split == "test":
            n_val = 800
            for f in files[:n_val]:
                link(f, OUT / "val" / "plant" / f.name)
                counts["val"] += 1
            for f in files[n_val:]:
                link(f, OUT / "test" / "plant" / f.name)
                counts["test"] += 1
        else:
            for f in files:
                link(f, OUT / "train" / "plant" / f.name)
                counts["train"] += 1

    # --- not_plant: Imagenette train -> train, val -> split into val/test ---
    for f in (RAW / "imagenette2-160" / "train").rglob("*.JPEG"):
        link(f, OUT / "train" / "not_plant" / f.name)
        counts["train"] += 1

    val_files = list((RAW / "imagenette2-160" / "val").rglob("*.JPEG"))
    random.shuffle(val_files)
    half = len(val_files) // 2
    for f in val_files[:half]:
        link(f, OUT / "val" / "not_plant" / f.name)
        counts["val"] += 1
    for f in val_files[half:]:
        link(f, OUT / "test" / "not_plant" / f.name)
        counts["test"] += 1

    for split in ["train", "val", "test"]:
        for cls in ["plant", "not_plant"]:
            n = len(list((OUT / split / cls).glob("*")))
            print(f"{split:5s} {cls:10s} {n}")


if __name__ == "__main__":
    main()
