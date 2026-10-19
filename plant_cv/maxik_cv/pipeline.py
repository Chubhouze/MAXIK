"""Full inspection pipeline: image (+ sensors + history) -> plant-health report.

    segmentation -> colour indicators ----\
    full frame   -> CNN (ONNX) ------------+-> fusion -> label / "unknown" + reasons
    sensors, previous inspection ---------/

The CNN sees the full frame, exactly as in training. Inspection points have a fixed camera distance,
so plant size in frame carries information (weak growth) that a tight crop would destroy.
"""
from pathlib import Path

import cv2
import numpy as np

from . import CLASSES, UNKNOWN
from .rules import rule_indicators
from .segment import color_stats, plant_mask

SEGMENT_MAX_SIDE = 1024

RECOMMENDATIONS = {
    "healthy": "No visible problems.",
    "dryness": "Possible dryness: check soil moisture and watering.",
    "yellowing": "Yellowing/discolouration: check light, overwatering and nutrients.",
    "weak_damage": "Weak growth or damage: inspect the plant for pests or physical damage.",
    UNKNOWN: "Unclear result: please check the plant manually.",
}


class PlantInspector:
    def __init__(self, model_path: str | Path | None = None, cnn_weight: float = 0.75,
                 unknown_threshold: float = 0.55, gate_model_path: str | Path | None = None):
        self.clf = None
        if model_path and Path(model_path).exists():
            from .classifier import PlantClassifier
            self.clf = PlantClassifier(model_path)
        # Without a trained model the rules carry all the weight.
        self.cnn_weight = cnn_weight if self.clf else 0.0
        self.unknown_threshold = unknown_threshold

        self.gate = None
        if gate_model_path and Path(gate_model_path).exists():
            from .plant_gate import PlantGate
            self.gate = PlantGate(gate_model_path)

    def inspect(self, bgr: np.ndarray, sensors: dict | None = None, prev_stats: dict | None = None) -> dict:
        gate_dict = {}
        if self.gate:
            gate_result = self.gate.check(bgr)
            gate_dict = {"gate": gate_result.to_dict()}
            if not gate_result.is_plant:
                return {"color_stats": {}, "sensors": sensors or {}, **gate_dict,
                        "reasons": ["not a plant (general-image classifier gate)"],
                        "label": UNKNOWN, "confidence": 0.0, "probs": {},
                        "recommendation": RECOMMENDATIONS[UNKNOWN]}

        # Colour ratios don't depend on resolution; segmenting a ~1000px frame is ~5x faster on the Pi.
        small = _downscale(bgr, SEGMENT_MAX_SIDE)
        mask = plant_mask(small)
        stats = color_stats(small, mask)
        rules = rule_indicators(stats, sensors, prev_stats)

        report = {
            "color_stats": stats.to_dict(),
            "sensors": sensors or {},
            **gate_dict,
            "reasons": rules["reasons"],
        }
        if not rules["valid"]:
            return {**report, "label": UNKNOWN, "confidence": 0.0, "probs": {},
                    "recommendation": RECOMMENDATIONS[UNKNOWN]}

        rule_p = _normalise({c: rules["scores"].get(c, 0.0) for c in CLASSES})
        probs = rule_p
        if self.clf:
            cnn_p = self.clf.predict(small)
            cnn_p = {c: cnn_p.get(c, 0.0) for c in CLASSES}
            probs = {c: self.cnn_weight * cnn_p[c] + (1 - self.cnn_weight) * rule_p[c] for c in CLASSES}
            report["cnn_probs"] = {c: round(v, 4) for c, v in cnn_p.items()}

        label = max(probs, key=probs.get)
        conf = probs[label]
        if conf < self.unknown_threshold:
            report["reasons"].append(f"low confidence ({conf:.0%}), best guess: {label}")
            label = UNKNOWN

        return {**report, "label": label, "confidence": round(conf, 4),
                "probs": {c: round(v, 4) for c, v in probs.items()},
                "recommendation": RECOMMENDATIONS[label]}


def _downscale(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr


def _normalise(d: dict) -> dict:
    s = sum(d.values()) or 1.0
    return {k: float(v) / s for k, v in d.items()}
