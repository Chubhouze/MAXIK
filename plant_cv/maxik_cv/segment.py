"""Plant segmentation and explainable colour indicators (OpenCV + NumPy only, runs on Raspberry Pi).

Idea: separate plant pixels from soil/background, then measure what share of the plant is
green / yellow / brown. These numbers are human-readable and work even without a trained model.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class ColorStats:
    plant_fraction: float   # plant pixels / all pixels (proxy for plant size in frame)
    green_ratio: float      # share of plant pixels that are green
    yellow_ratio: float     # share of plant pixels that are yellow / pale
    brown_ratio: float      # share of plant pixels that are brown / necrotic (dry)
    mean_saturation: float  # low saturation on leaves -> faded / wilting look
    exg_mean: float         # mean Excess-Green index on plant pixels (vigour proxy)

    def to_dict(self):
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


# HSV ranges for OpenCV (H: 0..179). Tune on your own photos with scripts/tune_hsv.py.
GREEN = ((35, 40, 30), (90, 255, 255))
YELLOW = ((18, 60, 80), (34, 255, 255))
BROWN = ((5, 50, 20), (18, 255, 200))
SOIL_LAB_DIST = 18.0  # min Lab distance from median soil colour for a brown pixel to count as dry leaf


def plant_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of plant tissue (green + yellow + brown leaves), soil/background removed."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img = bgr.astype(np.float32) / 255.0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    s = b + g + r + 1e-6
    exg = 2 * (g / s) - (r / s) - (b / s)  # Excess Green index

    green = cv2.inRange(hsv, *GREEN) > 0
    yellow = cv2.inRange(hsv, *YELLOW) > 0
    brown = cv2.inRange(hsv, *BROWN) > 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leafy = (((green & (exg > 0.05)) | yellow).astype(np.uint8)) * 255
    leafy = cv2.morphologyEx(leafy, cv2.MORPH_OPEN, kernel)

    # Soil is brown too, so brown alone can't seed the mask: seed from green/yellow leaf tissue and
    # accept brown (dry) pixels only in the immediate neighbourhood of that tissue.
    r = max(3, int(0.03 * min(bgr.shape[:2])))
    near = cv2.dilate(leafy, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    # ...and only if the pixel differs from the soil colour (estimated from the background far from leaves).
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    far = cv2.dilate(leafy, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 1, 4 * r + 1))) == 0
    if far.sum() > 0.05 * far.size:
        soil = np.median(lab[far], axis=0)
        brown &= np.linalg.norm(lab - soil, axis=2) > SOIL_LAB_DIST
    mask = ((leafy > 0) | (brown & near)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = max(50, int(0.001 * mask.size))
    keep_ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_area)
    keep_ids = keep_ids[keep_ids != 0]
    return np.isin(labels, keep_ids).astype(np.uint8) * 255


def color_stats(bgr: np.ndarray, mask: np.ndarray | None = None) -> ColorStats:
    if mask is None:
        mask = plant_mask(bgr)
    m = mask > 0
    n = int(m.sum())
    if n == 0:
        return ColorStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = (cv2.inRange(hsv, *GREEN) > 0) & m
    yellow = (cv2.inRange(hsv, *YELLOW) > 0) & m
    brown = (cv2.inRange(hsv, *BROWN) > 0) & m

    img = bgr.astype(np.float32) / 255.0
    s = img.sum(axis=2) + 1e-6
    exg = 2 * img[..., 1] / s - img[..., 2] / s - img[..., 0] / s

    return ColorStats(
        plant_fraction=n / m.size,
        green_ratio=green.sum() / n,
        yellow_ratio=yellow.sum() / n,
        brown_ratio=brown.sum() / n,
        mean_saturation=float(hsv[..., 1][m].mean()) / 255.0,
        exg_mean=float(exg[m].mean()),
    )


def crop_to_plant(bgr: np.ndarray, mask: np.ndarray, pad: float = 0.1) -> np.ndarray:
    """Square crop around the plant so the CNN sees the plant, not the greenhouse."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return bgr
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    h, w = bgr.shape[:2]
    side = min(side, h, w)
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return bgr[y0:y0 + side, x0:x0 + side]


def overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Debug visualisation: plant area tinted, useful for the validation report."""
    vis = bgr.copy()
    tint = np.zeros_like(bgr)
    tint[..., 1] = 255
    m = mask > 0
    vis[m] = cv2.addWeighted(bgr, 0.6, tint, 0.4, 0)[m]
    return vis
