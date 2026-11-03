"""Download the CropAndWeed dataset (top-down field photos, every plant boxed: 74 crop and weed
species, ~8k images, ~112k plants), ~11.5 GB, streamed and extracted on the fly.

Steininger et al., "The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and
Weed Manipulation", WACV 2023. https://github.com/cropandweed/cropandweed-dataset
LICENCE: non-commercial purposes only (academic research, teaching, publications, personal
experimentation) - fine for a non-profit project, not for a commercial product.

    python scripts/download_cropandweed.py      -> data/cropandweed/{images,bboxes,labelIds,params}
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "data" / "cropandweed"
URL = "https://vitro-testing.com/wp-content/uploads/2022/12/{}.tar"
PARTS = ["cropandweed_annotations", "cropandweed_images1of4", "cropandweed_images2of4",
         "cropandweed_images3of4", "cropandweed_images4of4"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        done = OUT / f".{part}.done"
        if done.exists():
            print(part, "already extracted", flush=True)
            continue
        print("downloading + extracting", part, flush=True)
        req = urllib.request.Request(URL.format(part), headers={"User-Agent": "Mozilla/5.0 (M.A.X.I.K.)"})
        with urllib.request.urlopen(req, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|") as tar:
            n = 0
            for member in tar:
                if member.name.startswith("/") or ".." in Path(member.name).parts:   # never write outside OUT
                    continue
                tar.extract(member, OUT, filter="data")
                n += 1
                if n % 1000 == 0:
                    print(f"   {n} files", flush=True)
        done.write_text("ok")
        print(f"   {part}: {n} files", flush=True)
    imgs = list((OUT / "images").glob("*")) if (OUT / "images").exists() else []
    print("images:", len(imgs), flush=True)


if __name__ == "__main__":
    sys.exit(main())
