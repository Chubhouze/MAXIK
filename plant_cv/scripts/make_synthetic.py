"""Generate a tiny synthetic dataset (drawn 'plants' on soil) - ONLY to smoke-test the code end to end.
Do not report accuracy on it; real validation must use Pi-camera photos of real plants."""
import argparse
from pathlib import Path

import cv2
import numpy as np

COLORS = {  # BGR leaf colour ranges per class
    "healthy": [(40, 150, 40), (70, 190, 60)],
    "dryness": [(30, 70, 110), (40, 100, 140)],
    "yellowing": [(40, 200, 210), (60, 220, 235)],
    "weak_damage": [(50, 140, 60), (60, 160, 70)],
}


def draw(cls: str, rng: np.random.Generator, size=320):
    img = np.full((size, size, 3), (40, 60, 90), np.uint8)  # soil
    img = cv2.add(img, rng.integers(0, 25, img.shape, dtype=np.uint8))
    cx, cy = size // 2 + rng.integers(-30, 30), size // 2 + rng.integers(-30, 30)
    n_leaves = rng.integers(2, 4) if cls == "weak_damage" else rng.integers(5, 9)
    scale = 0.5 if cls == "weak_damage" else 1.0
    for _ in range(n_leaves):
        ang = rng.uniform(0, 360)
        L = int(rng.uniform(50, 90) * scale)
        W = int(L * rng.uniform(0.3, 0.45))
        dx, dy = np.cos(np.radians(ang)) * L * 0.6, np.sin(np.radians(ang)) * L * 0.6
        base = np.array(COLORS["healthy"][rng.integers(0, 2)])
        if cls != "healthy" and rng.random() < 0.7:
            base = np.array(COLORS[cls][rng.integers(0, 2)])
        color = tuple(int(c) for c in np.clip(base + rng.integers(-10, 10, 3), 0, 255))
        cv2.ellipse(img, (int(cx + dx), int(cy + dy)), (L // 2 + 10, W // 2 + 4), ang, 0, 360, color, -1)
    if cls == "weak_damage":  # holes
        for _ in range(rng.integers(2, 5)):
            cv2.circle(img, (int(cx + rng.integers(-30, 30)), int(cy + rng.integers(-30, 30))), 6, (40, 60, 90), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_synthetic")
    ap.add_argument("--per-class", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for cls in COLORS:
        d = Path(args.out) / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            cv2.imwrite(str(d / f"syn{i:03d}_{cls}.jpg"), draw(cls, rng))
    print("done", args.out)


if __name__ == "__main__":
    main()
