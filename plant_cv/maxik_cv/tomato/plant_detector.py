"""PlantDetector: thin adapter over the existing trained plant/not-plant gate
(maxik_cv.plant_gate.PlantGate, models_synthetic/plant_gate_trained.onnx - MobileNetV3-Small
fine-tuned on real photos, ~100% val/test accuracy on its source domains). Reused as-is rather
than re-implemented, per the "is a plant visible?" step of the pipeline.
"""
from pathlib import Path

import numpy as np

from ..plant_gate import PlantGate
from .schema import DetectionResult


class PlantDetector:
    def __init__(self, gate: PlantGate):
        self.gate = gate

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> "PlantDetector":
        return cls(PlantGate(model_path))

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        result = self.gate.check(bgr)
        return DetectionResult(is_plant=result.is_plant, confidence=result.plant_score)
