"""Split the raw CCMT photos (download_ccmt.py) into train/val/test (80/10/10) per class, keeping
near-duplicate photo bursts on one side (dedup_split.py) - field photos are often taken in quick
series of the same plant. Hardlinks, no extra disk space.

    python scripts/prepare_ccmt.py      -> data/ccmt_split/<split>/<Crop>___<class>/*
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_split import dedup_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "data" / "ccmt" / "raw"
OUT = ROOT.parent / "data" / "ccmt_split"


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            dst.hardlink_to(src)
        except OSError:
            import shutil
            shutil.copy2(src, dst)


def readable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.convert("L")
        return True
    except OSError:
        return False


def main():
    total = {"train": 0, "val": 0, "test": 0}
    broken = []
    for class_dir in sorted(p for p in RAW.glob("*/*") if p.is_dir()):
        files = sorted(class_dir.glob("*"))
        good = [f for f in files if readable(f)]  # a few source files are truncated/corrupt
        broken += [f for f in files if f not in set(good)]
        train, val, test = dedup_split(good, seed=0)
        name = f"{class_dir.parent.name}___{class_dir.name}"
        for split, fs in (("train", train), ("val", val), ("test", test)):
            for f in fs:
                link(f, OUT / split / name / f.name)
            total[split] += len(fs)
        print(f"{name:32s} train {len(train):5d}  val {len(val):4d}  test {len(test):4d}", flush=True)
    print("total", total, "| unreadable source files skipped:", len(broken))
    for f in broken:
        print("   broken:", f.relative_to(RAW))


if __name__ == "__main__":
    main()
