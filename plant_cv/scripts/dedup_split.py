"""Near-duplicate-aware train/val/test split.

PlantVillage (and similar datasets) contain bursts of near-identical photos of the same leaf
(same angle, lighting, a few pixels apart). A naive random split puts near-duplicates on both
sides of train/test, which inflates reported accuracy without the model actually generalizing
better - the same failure mode scripts/prepare_dataset.py already avoids for our own captures by
splitting "by plant". Here there's no plant ID, so we approximate it: compute a perceptual hash
per image, cluster near-identical hashes together, and keep each whole cluster on one side of the
split - never splitting near-duplicates between train and test.

Hashing is a coarse 8x8 average-hash (64 bits) with bucketing on the top 16 bits as a fast
approximate nearest-neighbour index - good enough to catch near-duplicate bursts without an
O(n^2) full comparison over thousands of images.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HASH_SIZE = 8
BUCKET_BITS = 16
HAMMING_THRESHOLD = 4


def ahash(path: Path) -> int:
    img = Image.open(path).convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def cluster_files(files: list[Path]) -> list[list[Path]]:
    hashes = [(f, ahash(f)) for f in files]
    buckets: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for f, h in hashes:
        buckets[h >> (64 - BUCKET_BITS)].append((f, h))

    parent = {f: f for f, _ in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for bucket in buckets.values():
        for i in range(len(bucket)):
            fi, hi = bucket[i]
            for j in range(i + 1, len(bucket)):
                fj, hj = bucket[j]
                if bin(hi ^ hj).count("1") <= HAMMING_THRESHOLD:
                    union(fi, fj)

    clusters: dict[Path, list[Path]] = defaultdict(list)
    for f, _ in hashes:
        clusters[find(f)].append(f)
    return list(clusters.values())


def dedup_split(files: list[Path], seed: int = 0, train_frac: float = 0.8, val_frac: float = 0.1):
    """Returns (train_files, val_files, test_files), keeping near-duplicate clusters intact."""
    import random
    clusters = cluster_files(files)
    rng = random.Random(seed)
    rng.shuffle(clusters)

    n_total = len(files)
    train, val, test = [], [], []
    n_train_target, n_val_target = n_total * train_frac, n_total * val_frac
    for cluster in clusters:
        if len(train) < n_train_target:
            train.extend(cluster)
        elif len(val) < n_val_target:
            val.extend(cluster)
        else:
            test.extend(cluster)
    return train, val, test
