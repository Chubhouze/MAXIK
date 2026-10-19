"""Add procedurally generated NON-plant images to the plant-gate dataset (data/plant_gate_split).

Why: the gate was trained against Imagenette objects only. Measured afterwards on harder inputs it
called 60/60 images of per-pixel colour noise "a plant" (high-frequency texture looked like foliage
to it) and ~8% of stripes/checkerboards. Real-world non-plant scenes are textured too, so the gate
should have seen texture-like negatives. These are NOT a substitute for real photographs of soil,
walls, pots, floors - those are still missing and the gate's behaviour on them is unmeasured.

    python scripts/add_synthetic_negatives.py
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(r"C:\Users\chub\Desktop\MAXIK\data\plant_gate_split")
SIDE = 256
COUNTS = {"train": 1200, "val": 150, "test": 150}


def rand_color(rng):
    return rng.integers(0, 256, 3)


def pixel_noise(rng):
    kind = rng.integers(0, 3)
    if kind == 0:
        return rng.integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8)
    if kind == 1:  # gaussian noise around a colour
        img = rand_color(rng) + rng.normal(0, rng.uniform(15, 90), (SIDE, SIDE, 3))
        return np.clip(img, 0, 255).astype(np.uint8)
    img = np.full((SIDE, SIDE, 3), rand_color(rng), np.uint8)  # salt and pepper
    m = rng.random((SIDE, SIDE)) < rng.uniform(0.02, 0.3)
    img[m] = rng.integers(0, 256, (int(m.sum()), 3))
    return img


def multiscale_noise(rng):
    img = np.zeros((SIDE, SIDE, 3), np.float32)
    for sigma in rng.choice([1, 2, 4, 8, 16, 32], size=rng.integers(1, 4), replace=False):
        n = cv2.GaussianBlur(rng.random((SIDE, SIDE, 3)).astype(np.float32), (0, 0), float(sigma))
        img += (n - n.mean()) / (n.std() + 1e-6) * rng.uniform(20, 60)
    return np.clip(img + rand_color(rng), 0, 255).astype(np.uint8)


def stripes(rng):
    w = int(rng.integers(3, 40))
    c1, c2 = rand_color(rng), rand_color(rng)
    big = np.zeros((SIDE * 2, SIDE * 2, 3), np.uint8)
    for x in range(0, SIDE * 2, w * 2):
        big[:, x:x + w], big[:, x + w:x + 2 * w] = c1, c2
    rot = cv2.getRotationMatrix2D((SIDE, SIDE), float(rng.uniform(0, 180)), 1.0)
    return cv2.warpAffine(big, rot, (SIDE * 2, SIDE * 2))[SIDE // 2:SIDE // 2 + SIDE, SIDE // 2:SIDE // 2 + SIDE]


def checker(rng):
    w = int(rng.integers(4, 48))
    g = ((np.indices((SIDE, SIDE)).sum(0) // w) % 2).astype(bool)
    img = np.zeros((SIDE, SIDE, 3), np.uint8)
    img[g], img[~g] = rand_color(rng), rand_color(rng)
    return img


def gradient(rng):
    t = np.linspace(0, 1, SIDE, dtype=np.float32)[None, :, None]
    a, b = rand_color(rng).astype(np.float32), rand_color(rng).astype(np.float32)
    img = np.repeat(a + (b - a) * t, SIDE, axis=0)
    img = np.rot90(img, int(rng.integers(0, 4)))
    return np.clip(img + rng.normal(0, rng.uniform(0, 25), img.shape), 0, 255).astype(np.uint8)


def shapes(rng):
    img = np.full((SIDE, SIDE, 3), rand_color(rng), np.uint8)
    for _ in range(int(rng.integers(5, 40))):
        c = tuple(int(v) for v in rand_color(rng))
        if rng.random() < 0.5:
            cv2.circle(img, tuple(int(v) for v in rng.integers(0, SIDE, 2)), int(rng.integers(3, 60)), c, -1)
        else:
            p = rng.integers(0, SIDE, 2)
            cv2.rectangle(img, tuple(int(v) for v in p), tuple(int(v) for v in p + rng.integers(5, 90, 2)), c, -1)
    return img


GENERATORS = [pixel_noise, multiscale_noise, stripes, checker, gradient, shapes]


def main():
    rng = np.random.default_rng(12345)
    for split, n in COUNTS.items():
        folder = OUT / split / "not_plant"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            gen = GENERATORS[i % len(GENERATORS)]
            cv2.imwrite(str(folder / f"synth_{gen.__name__}_{i:04d}.jpg"), gen(rng), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{split:5s} added {n}, not_plant now {len(list(folder.glob('*')))}")


if __name__ == "__main__":
    main()
