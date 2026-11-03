"""Download the PlantDoc object-detection dataset (real-world photos of plants/branches, one box per
leaf with its crop and disease), CC BY 4.0:
Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection", CODS-COMAD 2020,
https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset

Writes data/plantdoc_od/{TRAIN,TEST}/ and boxes_{train,test}.csv. Many original file names are web
URLs containing characters Windows can't store (? & = ...), so files are saved under a safe name and
the boxes_*.csv files point at the local name.

    python scripts/download_plantdoc.py               # all 13 crops (~2.5k photos)
    python scripts/download_plantdoc.py --tomato-only
"""
import argparse
import csv
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "data" / "plantdoc_od"
BASE = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Object-Detection-Dataset/master/"


def safe_name(name: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*&=%#]+', "_", name)
    return stem if stem.lower().endswith((".jpg", ".jpeg", ".png")) else stem + ".jpg"


def fetch(url: str, dst: Path) -> str | None:
    if dst.exists() and dst.stat().st_size > 0:
        return None
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            dst.write_bytes(r.read())
        return None
    except Exception as e:  # report and keep going: the CSVs list some images that aren't in the repo
        return f"{dst.name}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tomato-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("train_labels.csv", "test_labels.csv", "LICENSE.txt"):
        if not (OUT / name).exists():
            fetch(BASE + name, OUT / name)

    jobs, tables = [], {}
    for split, folder in (("train", "TRAIN"), ("test", "TEST")):
        rows = [r for r in csv.DictReader(open(OUT / f"{split}_labels.csv", encoding="utf8"))
                if not args.tomato_only or r["class"].lower().startswith("tomato")]
        (OUT / folder).mkdir(exist_ok=True)
        names = sorted({r["filename"] for r in rows})
        jobs += [(BASE + folder + "/" + urllib.parse.quote(n), OUT / folder / safe_name(n)) for n in names]
        tables[split] = (folder, rows)

    with ThreadPoolExecutor(8) as pool:
        errors = [e for e in pool.map(lambda j: fetch(*j), jobs) if e]

    for split, (folder, rows) in tables.items():
        with open(OUT / f"boxes_{split}.csv", "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
            for r in rows:
                local = OUT / folder / safe_name(r["filename"])
                if local.exists():
                    w.writerow([f"{folder}/{local.name}", r["width"], r["height"], r["class"],
                                r["xmin"], r["ymin"], r["xmax"], r["ymax"]])
    size = sum(p.stat().st_size for _, p in jobs if p.exists())
    print(f"{len(jobs) - len(errors)} of {len(jobs)} images in {OUT} ({size / 1e6:.0f} MB)")
    for e in errors:
        print("  missing:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
