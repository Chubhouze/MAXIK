"""Whole-plant analysis: diagnose every leaf the LeafDetector found, then summarise the plant.

Plant-level answer:
- condition: the disease with the most total confidence across the leaves that show one; HEALTHY if
  no leaf shows a disease and most leaves were confidently healthy; otherwise UNCERTAIN;
- affected_fraction: visible damage, area-weighted over all analysed leaves (same lesion.py measure as
  for a single leaf);
- severity: the worse of two approximate bands - visible damage (LOW <10% / MODERATE <30% / HIGH) and
  the share of leaves with a problem (LOW <25% / MODERATE <50% / HIGH) - so a plant whose leaves are
  nearly all affected, each only lightly, is not reported as LOW.
Pure logic on per-leaf results; no ML or IO here except the per-leaf calls in analyze_leaves.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .class_mapping import CONDITION_TO_STATUS_LABEL
from .confidence import apply_confidence_gate
from .lesion import analyze_affected_area
from .schema import Condition, Severity
from .severity import estimate_severity

MAX_LEAVES = 30          # analyse at most the 30 largest leaves (time on a Raspberry Pi)
MIN_LEAF_PX = 20
LEAF_SHARE_BANDS = (0.25, 0.50)
_RANK = {Severity.UNKNOWN: -1, Severity.LOW: 0, Severity.MODERATE: 1, Severity.HIGH: 2}


@dataclass
class LeafResult:
    box: tuple[int, int, int, int]
    detector_score: float
    condition: Condition       # after the confidence gate
    confidence: float
    affected_fraction: float | None
    is_placeholder: bool

    @property
    def area(self) -> int:
        return max(1, (self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))

    def to_dict(self) -> dict:
        return {"box": list(self.box), "detectorScore": round(self.detector_score, 3),
                "condition": self.condition.value, "label": CONDITION_TO_STATUS_LABEL[self.condition],
                "confidence": round(self.confidence, 4),
                "affectedFraction": None if self.affected_fraction is None else round(self.affected_fraction, 4)}


@dataclass
class PlantSummary:
    condition: Condition
    confidence: float
    affected_fraction: float | None
    severity: Severity
    observed: list[str]
    candidates: list[dict]
    warnings: list[str]


def analyze_leaves(bgr: np.ndarray, leaves, health_classifier) -> tuple[list[LeafResult], int]:
    """(per-leaf results for the largest MAX_LEAVES leaves, number of leaves detected in total)."""
    h, w = bgr.shape[:2]
    usable = [l for l in leaves if l.box[2] - l.box[0] >= MIN_LEAF_PX and l.box[3] - l.box[1] >= MIN_LEAF_PX]
    usable.sort(key=lambda l: (l.box[2] - l.box[0]) * (l.box[3] - l.box[1]), reverse=True)
    results = []
    for leaf in usable[:MAX_LEAVES]:
        x0, y0, x1, y1 = leaf.box
        mx, my = int(0.1 * (x1 - x0)), int(0.1 * (y1 - y0))   # same 10% margin as the training crops
        crop = bgr[max(0, y0 - my):min(h, y1 + my), max(0, x0 - mx):min(w, x1 + mx)]
        pred = health_classifier.classify(crop)
        results.append(LeafResult(leaf.box, leaf.score, apply_confidence_gate(pred.condition, pred.confidence),
                                  pred.confidence, analyze_affected_area(crop).fraction, pred.is_placeholder))
    return results, len(usable)


def _band(value: float, bands: tuple[float, float]) -> Severity:
    return Severity.LOW if value < bands[0] else Severity.MODERATE if value < bands[1] else Severity.HIGH


def summarize(results: list[LeafResult], detected_total: int) -> PlantSummary:
    n = len(results)
    diseased = [r for r in results if r.condition not in (Condition.TOMATO_HEALTHY, Condition.TOMATO_UNCERTAIN)]
    healthy = [r for r in results if r.condition == Condition.TOMATO_HEALTHY]
    uncertain = [r for r in results if r.condition == Condition.TOMATO_UNCERTAIN]

    if diseased:
        weight = defaultdict(float)
        for r in diseased:
            weight[r.condition] += r.confidence
        condition = max(weight, key=weight.get)
        confidence = float(np.mean([r.confidence for r in diseased if r.condition == condition]))
    elif healthy and len(healthy) >= len(uncertain):
        condition, confidence = Condition.TOMATO_HEALTHY, float(np.mean([r.confidence for r in healthy]))
    else:
        condition, confidence = Condition.TOMATO_UNCERTAIN, float(np.mean([r.confidence for r in results]))

    measured = [r for r in results if r.affected_fraction is not None]
    affected = (sum(r.affected_fraction * r.area for r in measured) / sum(r.area for r in measured)) if measured else None
    share = len(diseased) / n

    severity = estimate_severity(condition, affected)
    if condition not in (Condition.TOMATO_HEALTHY,) and diseased:
        leaf_band = _band(share, LEAF_SHARE_BANDS)
        if _RANK[leaf_band] > _RANK[severity]:
            severity = leaf_band

    counts = Counter(r.condition for r in results)
    candidates = [{"condition": c.value, "label": CONDITION_TO_STATUS_LABEL[c], "probability": round(k / n, 4),
                   "leaves": k} for c, k in counts.most_common()]
    observed = [f"{len(diseased)} of {n} analysed leaves show possible problems"]
    if diseased:
        observed.append(f"most common: {CONDITION_TO_STATUS_LABEL[condition]} ({counts[condition]} leaves)")
    if uncertain:
        observed.append(f"{len(uncertain)} leaves could not be assessed reliably")

    warnings = []
    if detected_total > n:
        warnings.append(f"{detected_total} leaves were found; the {n} largest were analysed.")
    distinct = [c for c, k in counts.items()
                if c not in (Condition.TOMATO_HEALTHY, Condition.TOMATO_UNCERTAIN) and (k >= 2 or k / n >= 0.2)]
    if len(distinct) >= 2:
        names = ", ".join(f"{CONDITION_TO_STATUS_LABEL[c]} ({counts[c]})" for c in distinct)
        warnings.append(f"Different leaves point to different conditions: {names}. The plant may have more than "
                        "one problem, or the model is unsure - check the marked leaves.")
    return PlantSummary(condition, confidence, affected, severity, observed, candidates, warnings)
