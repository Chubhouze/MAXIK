"""Bed analysis: a TOP-DOWN photo of a bed -> every plant found and labelled crop / weed -> visible damage
of each crop plant -> a bed summary. Model: models_bed/bed_detector.onnx (scripts/train_leaf_detector.py
--task bed, CropAndWeed), run through the same ONNX + NumPy detector code as the leaf detector.

What it does NOT do: name diseases. The disease model is trained on close-up leaf photos; from a
top-down bed photo it would be guessing. Bed view is visual screening - which plants are discoloured,
how many weeds - and a pointer to which plants deserve a close-up (single-leaf / whole-plant mode).
Damage per plant is the same colour measure as elsewhere (lesion.py), so it works for any crop.
"""
from dataclasses import dataclass, field

import numpy as np

from ..segment import ColorStats, color_stats, plant_mask
from .lesion import analyze_affected_area
from .possible_causes import list_possible_causes
from .schema import Severity

DAMAGED = 0.10            # a crop plant with >= 10% visibly affected area counts as "showing damage"
MIN_PLANT_PX = 24


def _band(affected: float | None) -> Severity:
    if affected is None:
        return Severity.UNKNOWN
    return Severity.LOW if affected < 0.10 else Severity.MODERATE if affected < 0.30 else Severity.HIGH


@dataclass
class BedPlant:
    index: int
    box: tuple[int, int, int, int]
    kind: str                      # "crop" | "weed"
    score: float
    affected_fraction: float | None = None
    severity: Severity = Severity.UNKNOWN

    def to_dict(self) -> dict:
        return {"index": self.index, "box": list(self.box), "kind": self.kind, "score": round(self.score, 3),
                "affectedFraction": None if self.affected_fraction is None else round(self.affected_fraction, 4),
                "severity": self.severity.value}


@dataclass
class BedReport:
    plants: list[BedPlant]
    summary: str
    warnings: list[str] = field(default_factory=list)
    possible_causes: list[dict] = field(default_factory=list)
    weed_cover: float = 0.0

    @property
    def crops(self) -> list[BedPlant]:
        return [p for p in self.plants if p.kind == "crop"]

    @property
    def weeds(self) -> list[BedPlant]:
        return [p for p in self.plants if p.kind == "weed"]

    def to_dict(self) -> dict:
        measured = [p.affected_fraction for p in self.crops if p.affected_fraction is not None]
        return {"crops": len(self.crops), "weeds": len(self.weeds), "weedCover": round(self.weed_cover, 4),
                "damagedCrops": sum(1 for p in self.crops if (p.affected_fraction or 0) >= DAMAGED),
                "meanCropDamage": round(float(np.mean(measured)), 4) if measured else None,
                "plants": [p.to_dict() for p in self.plants], "summary": self.summary,
                "warnings": self.warnings, "possibleCauses": self.possible_causes}


def analyze_bed(bgr: np.ndarray, detector, sensors: dict | None = None) -> BedReport:
    boxes, scores, labels = detector.detect_raw(bgr)
    h, w = bgr.shape[:2]
    found = []
    for b, s, l in zip(boxes, scores, labels):
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        if min(x1 - x0, y1 - y0) >= MIN_PLANT_PX:
            found.append(((x0, y0, x1, y1), float(s), detector.classes[int(l) - 1]))
    found.sort(key=lambda t: (t[0][0], t[0][1]))   # numbered left to right across the photo

    plants, stats, areas = [], [], []
    for i, (box, score, kind) in enumerate(found, 1):
        plant = BedPlant(i, box, kind, score)
        if kind == "crop":
            x0, y0, x1, y1 = box
            crop = bgr[y0:y1, x0:x1]
            plant.affected_fraction = analyze_affected_area(crop).fraction
            plant.severity = _band(plant.affected_fraction)
            m = plant_mask(crop)
            if (m > 0).any():
                stats.append(color_stats(crop, m))
                areas.append((x1 - x0) * (y1 - y0))
        plants.append(plant)

    weed_mask = np.zeros((h, w), bool)
    for p in plants:
        if p.kind == "weed":
            x0, y0, x1, y1 = p.box
            weed_mask[y0:y1, x0:x1] = True
    report = BedReport(plants, "", weed_cover=float(weed_mask.mean()))

    if stats:  # colour-based possible causes over the crop plants (area-weighted), plus sensor readings
        wts = np.array(areas, float) / sum(areas)
        avg = ColorStats(*[float(np.dot(wts, [getattr(s, f) for s in stats])) for f in ColorStats.__dataclass_fields__])
        report.possible_causes = [c.to_dict() for c in list_possible_causes(avg, sensors)]

    crops, weeds = report.crops, report.weeds
    damaged = [p for p in crops if (p.affected_fraction or 0) >= DAMAGED]
    parts = [f"{len(crops)} crop plant{'s' if len(crops) != 1 else ''} and {len(weeds)} weed{'s' if len(weeds) != 1 else ''} found."]
    if crops:
        if damaged:
            worst = sorted(damaged, key=lambda p: -p.affected_fraction)[:3]
            names = ", ".join(f"#{p.index} ({p.affected_fraction:.0%})" for p in worst)
            parts.append(f"{len(damaged)} of {len(crops)} crop plants show visible damage - worst: {names}.")
        else:
            parts.append("No crop plant shows notable visible damage.")
    if weeds:
        parts.append(f"Weeds cover about {report.weed_cover:.0%} of the photo.")
    report.summary = " ".join(parts)

    report.warnings.append("Bed view is visual screening: it finds plants and weeds and measures visible "
                           "discolouration, but does not name diseases - take a close-up of the marked plants for that.")
    if len(found) >= getattr(detector, "max_detections", 10 ** 9):
        report.warnings.append("The detection limit was reached; some plants may not be counted.")
    return report
