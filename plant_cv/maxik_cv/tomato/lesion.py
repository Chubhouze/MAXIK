"""Visible-damage estimator: what share of the leaf area is NOT healthy green (spots, necrosis,
chlorosis, mould). Pure colour segmentation (OpenCV/NumPy, runs on the Pi) - no ML, so it works for
any condition and is explainable: the returned masks can be drawn over the photo.

How: take the plant mask from maxik_cv.segment, close small gaps and fill holes so dark lesions
sitting INSIDE the leaf (which the plant mask itself drops, since black/very dark pixels are in no
colour band) count as leaf; then affected = leaf region minus healthy-green pixels.

It's an approximate, uncalibrated visual proxy, not a biological measurement. It sees colour only:
it cannot tell a lesion from a shadow, a pale-green variety, or an unrelated leaf in the background.
Measured on the PlantVillage test split (scripts-free experiment, 10 classes): healthy leaves
median 2% / p90 6%; early blight median 14% / p90 63%; septoria 11% / 67%; late blight 11% / 48%.
It is nearly blind to diseases whose damage is not a colour change or is tiny (target spot, mosaic
virus, mites, bacterial spot, leaf curl: medians 0-3% like healthy) - so a LOW severity does NOT
mean a plant is fine when the classifier names one of those.
Returns fraction=None when the leaf region is too small for the number to mean anything.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ..segment import GREEN, plant_mask

MAX_SIDE = 640
MIN_LEAF_FRACTION = 0.02       # leaf region smaller than this share of the frame -> no estimate
HEALTHY_EXG_MIN = 0.02         # excess-green floor so grey-green pixels don't count as healthy


@dataclass
class LesionAnalysis:
    fraction: float | None     # affected share of leaf area, 0..1
    leaf_mask: np.ndarray      # uint8 0/255, at analysis resolution
    affected_mask: np.ndarray  # uint8 0/255, at analysis resolution


def analyze_affected_area(bgr: np.ndarray) -> LesionAnalysis:
    h, w = bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr

    mask = plant_mask(small)
    k = max(5, int(0.02 * min(small.shape[:2])) | 1)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf = np.zeros_like(mask)
    cv2.drawContours(leaf, contours, -1, 255, thickness=cv2.FILLED)
    # Measure inside the leaf, not on its rim: edge pixels blend with the background and read as
    # "damage" on perfectly healthy leaves (median 8% -> 2% on healthy PlantVillage test photos).
    r = max(3, int(0.03 * min(small.shape[:2])))
    leaf = cv2.erode(leaf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

    leaf_px = int((leaf > 0).sum())
    if leaf_px < MIN_LEAF_FRACTION * leaf.size:
        return LesionAnalysis(None, leaf, np.zeros_like(leaf))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    img = small.astype(np.float32) / 255.0
    total = img.sum(axis=2) + 1e-6
    exg = 2 * img[..., 1] / total - img[..., 2] / total - img[..., 0] / total
    healthy = (cv2.inRange(hsv, *GREEN) > 0) & (exg > HEALTHY_EXG_MIN)

    affected = ((leaf > 0) & ~healthy).astype(np.uint8) * 255
    affected = cv2.morphologyEx(affected, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop 1-2px edge noise
    return LesionAnalysis(float((affected > 0).sum()) / leaf_px, leaf, affected)
