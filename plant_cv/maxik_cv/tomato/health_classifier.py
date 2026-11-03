"""TomatoHealthClassifier: the actual disease/health call (spec sections 3-4). No tomato-disease
model has been trained in this project yet (unlike the plant/not-plant gate, which is real) - so
PlaceholderTomatoHealthClassifier is used until one exists. It always returns TOMATO_UNCERTAIN
with confidence 0.0, explicitly flagged is_placeholder=True, rather than inventing a plausible-
looking disease guess. See ../../DATA_COLLECTION_GUIDE.md for how to collect training photos, and
class_mapping.DATASET_CLASS_NAMES for the label set a real model should train against.

OnnxTomatoHealthClassifier is a complete, ready-to-use backend for when that model exists -
it follows the exact same ONNX + sidecar-JSON convention as maxik_cv.classifier.PlantClassifier
(image_size, class list and temperature read from a `<model>.json` file next to the `.onnx`).
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..preprocess import center_crop_input
from .class_mapping import DATASET_LABEL_TO_CONDITION
from .schema import Condition, HealthPrediction


class TomatoHealthClassifier(ABC):
    @abstractmethod
    def classify(self, bgr: np.ndarray) -> HealthPrediction: ...


class PlaceholderTomatoHealthClassifier(TomatoHealthClassifier):
    """No trained tomato-disease model exists yet. Always reports TOMATO_UNCERTAIN with zero
    confidence rather than faking a diagnosis - see module docstring."""

    def classify(self, bgr: np.ndarray) -> HealthPrediction:
        return HealthPrediction(condition=Condition.TOMATO_UNCERTAIN, confidence=0.0,
                                 observed_features=[], is_placeholder=True)


class OnnxTomatoHealthClassifier(TomatoHealthClassifier):
    """Real backend for a trained model, once one exists at the given path. Expects a sidecar
    `<model>.json` with {"classes": [...DATASET_CLASS_NAMES order...], "img_size": int,
    "temperature": float} - the same convention scripts/train.py already writes."""

    def __init__(self, model_path: str | Path, threads: int = 4):
        import onnxruntime as ort

        model_path = Path(model_path)
        meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf8"))
        self.classes: list[str] = meta["classes"]
        self.size: int = meta["img_size"]
        self.temperature: float = meta.get("temperature", 1.0)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, bgr: np.ndarray) -> HealthPrediction:
        x = center_crop_input(bgr, self.size)
        logits = self.session.run(None, {self.input_name: x})[0][0] / self.temperature
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        order = np.argsort(probs)[::-1]
        ranked = [(DATASET_LABEL_TO_CONDITION.get(self.classes[i], Condition.TOMATO_UNCERTAIN), float(probs[i]))
                  for i in order[:3]]
        top_idx = int(order[0])
        return HealthPrediction(condition=ranked[0][0], confidence=float(probs[top_idx]),
                                 observed_features=[], is_placeholder=False, ranked=ranked)
