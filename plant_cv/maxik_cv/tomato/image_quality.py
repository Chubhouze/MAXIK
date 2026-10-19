"""ImageQualityAnalyzer: rule-based pre-checks, no ML (spec section 8).

Split in two, because "is the plant too small" only means something once we know a plant is
there at all:
- analyze_basic_quality() runs BEFORE plant detection: brightness/blur, independent of content.
- flag_plant_too_small() runs AFTER PlantDetector confirms a plant, using the same colour-based
  segmentation (maxik_cv.segment) the rest of the codebase already relies on. Running this before
  detection would misreport genuine NOT_PLANT photos as a "blurry/too small" quality issue instead
  of correctly reporting no plant found.
"""
import cv2
import numpy as np

from ..segment import plant_mask
from .schema import ImageQualityIssue, QualityReport

DARK_THRESHOLD = 0.30     # 95th-percentile brightness below this -> underexposed everywhere
BRIGHT_THRESHOLD = 0.85   # 5th-percentile brightness above this -> washed out everywhere
# Sharpness is measured at the scale the classifier actually sees (256 px), so only blur that is
# visible to the network counts. Calibrated on PlantVillage test photos with synthetic blur
# (not on robot-camera photos - recheck once real frames exist):
#   sharp photos            p5 = 50, median 109
#   smooth-but-sharp leaves 34-49   (must pass: classified correctly at ~100%)
#   gaussian s=3 / motion 21px  median ~26-30
# Found with the first model (no blur augmentation): blurred input did NOT lower its confidence - at
# s=3 it was confidently wrong 81% of the time - so the confidence gate could not catch it and this
# gate was the only protection. The current model is trained with gaussian + motion blur and is no
# longer confidently wrong on blurred frames (0 of 800 degraded test frames; at s=3 / 21px the gate
# rejects 65-76% and every frame it lets through is classified correctly). The gate stays as a safety
# net and for the user-facing "retake the photo" message.
SHARPNESS_HARD = 30.0     # below: reject, the frame is too blurry to analyse
SHARPNESS_SOFT = 55.0     # below: analyse, but warn that accuracy may be reduced
SHARPNESS_SIDE = 256
PLANT_TOO_SMALL_THRESHOLD = 0.03
ANALYSIS_MAX_SIDE = 800   # brightness/segmentation resolution


def _normalize(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = ANALYSIS_MAX_SIDE / max(h, w)
    return cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr


def analyze_basic_quality(bgr: np.ndarray) -> QualityReport:
    small = _normalize(bgr)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    brightness = float(v.mean()) / 255.0  # reported for the UI only
    # Judged on percentiles, not the mean: a well-lit leaf on a black/dark-soil background has a
    # very low mean but is a perfectly usable photo; a genuinely underexposed frame is dark everywhere.
    v_high = float(np.percentile(v, 95)) / 255.0
    v_low = float(np.percentile(v, 5)) / 255.0

    sharpness = sharpness_score(bgr)

    issues = []
    if v_high < DARK_THRESHOLD:
        issues.append(ImageQualityIssue.IMAGE_TOO_DARK)
    if v_low > BRIGHT_THRESHOLD:
        issues.append(ImageQualityIssue.IMAGE_TOO_BRIGHT)
    if sharpness < SHARPNESS_HARD:
        issues.append(ImageQualityIssue.IMAGE_BLURRY)

    return QualityReport(ok=len(issues) == 0, issues=issues, brightness=brightness, sharpness=sharpness)


def sharpness_score(bgr: np.ndarray) -> float:
    """Gradient energy of the strongest 25% of pixels, in the WEAKER of the x/y directions, at 256 px.
    Looking at both directions catches motion blur (which keeps edges across the motion, so a plain
    Laplacian-variance score stays high while the picture is unusable); looking only at the strongest
    pixels ignores flat background, so a leaf on black/soil isn't judged by its empty surroundings."""
    h, w = bgr.shape[:2]
    s = SHARPNESS_SIDE / max(h, w)
    g = cv2.cvtColor(cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)

    def top_quarter(grad: np.ndarray) -> float:
        a = np.abs(grad).ravel()
        return float(np.partition(a, -max(1, a.size // 4))[-max(1, a.size // 4):].mean())

    return min(top_quarter(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)),
               top_quarter(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)))


def flag_plant_too_small(bgr: np.ndarray, report: QualityReport) -> QualityReport:
    small = _normalize(bgr)
    plant_fraction = float((plant_mask(small) > 0).mean())
    issues = list(report.issues)
    if plant_fraction < PLANT_TOO_SMALL_THRESHOLD:
        issues.append(ImageQualityIssue.PLANT_TOO_SMALL)
    return QualityReport(ok=len(issues) == 0, issues=issues, brightness=report.brightness,
                          sharpness=report.sharpness, plant_area_fraction=plant_fraction)
