"""Download the RAW part of the CCMT dataset (field photos from farms in Ghana, validated by plant
virologists): 25,220 images, 4 crops (cashew, cassava, maize, tomato), 22 classes, ~1.3 GB, CC BY 4.0.
The augmented copies (~7 GB) are skipped - training does its own augmentation.

Mensah, P. K. et al. (2023). CCMT: Dataset for crop pest and disease detection. Data in Brief 49, 109306.
Mendeley Data, V1, doi:10.17632/bwh3zbpkpv.1

    python scripts/download_ccmt.py            -> data/ccmt/raw/<Crop>/<class>/<file>
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "data" / "ccmt"
API = "https://data.mendeley.com/public-api/datasets/bwh3zbpkpv"
HEADERS = {"User-Agent": "Mozilla/5.0 (M.A.X.I.K. dataset download)"}  # the default Python agent gets HTTP 403


def open_url(url: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def get_json(url: str, cache: Path):
    if not cache.exists():
        with open_url(url, 300) as r:
            cache.write_bytes(r.read())
    return json.loads(cache.read_text(encoding="utf8"))


def fetch(job) -> str | None:
    url, dst, size = job
    if dst.exists() and dst.stat().st_size == size:
        return None
    try:
        with open_url(url, 120) as r:
            data = r.read()
        dst.write_bytes(data)
        return None if len(data) == size else f"{dst}: size {len(data)} != {size}"
    except Exception as e:
        return f"{dst}: {e}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = get_json(API, OUT / "meta.json")
    folders = {f["id"]: f for f in get_json(API + "/folders/1?version=1", OUT / "folders.json")}

    def path(fid):
        parts = []
        while fid in folders:
            parts.append(folders[fid]["name"])
            fid = folders[fid].get("parent_id")
        return list(reversed(parts))

    jobs = []
    for f in meta["files"]:
        p = path(f["folder_id"])  # ["Raw Data", "CCMT Dataset", crop, class]
        if len(p) == 4 and p[0] == "Raw Data":
            dst = OUT / "raw" / p[2] / p[3] / f["filename"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((f["content_details"]["download_url"], dst, f["size"]))
    print(f"{len(jobs)} raw files, {sum(j[2] for j in jobs) / 1e9:.2f} GB", flush=True)

    errors, done = [], 0
    with ThreadPoolExecutor(12) as pool:
        for err in pool.map(fetch, jobs):
            done += 1
            if err:
                errors.append(err)
            if done % 2000 == 0:
                print(f"  {done}/{len(jobs)} (failed so far: {len(errors)})", flush=True)
                if done == 2000 and len(errors) > 1000:
                    sys.exit(f"most downloads fail, stopping. First error: {errors[0]}")
    print(f"downloaded {len(jobs) - len(errors)} of {len(jobs)}; failed {len(errors)}")
    for e in errors[:20]:
        print("  ", e, file=sys.stderr)


if __name__ == "__main__":
    main()
