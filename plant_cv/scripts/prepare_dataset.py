"""Split data/raw/<class>/*.jpg into data/split/{train,val,test}/<class>/.

Photos of the SAME plant must not end up in both train and test (otherwise accuracy is inflated).
Name files as <plantid>_<anything>.jpg (e.g. basil03_2026-10-01_p2.jpg) - splitting is done by plant id.
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--dst", default="data/split")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    rng = random.Random(args.seed)

    for cls_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        groups = defaultdict(list)
        for f in cls_dir.iterdir():
            if f.suffix.lower() in EXT:
                groups[f.stem.split("_")[0]].append(f)
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, round(n * args.test)) if n >= 3 else 0
        n_val = max(1, round(n * args.val)) if n >= 3 else 0
        parts = {"test": keys[:n_test], "val": keys[n_test:n_test + n_val], "train": keys[n_test + n_val:]}
        counts = {}
        for split, ks in parts.items():
            out = dst / split / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            files = [f for k in ks for f in groups[k]]
            for f in files:
                shutil.copy2(f, out / f.name)
            counts[split] = len(files)
        print(f"{cls_dir.name:14s} plants={n:4d}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
