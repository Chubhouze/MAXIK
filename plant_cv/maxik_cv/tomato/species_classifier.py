"""PlantSpeciesClassifier: decides whether the plant is a supported species (spec section 14 -
Tomato / Potato / Cucumber / ... under one Plant hierarchy).

- OnnxSpeciesClassifier: a 15-crop model (scripts/train_species.py: lab leaves, real-world leaf crops
  and farm photos). A non-tomato verdict is only acted on when it is confident; otherwise the plant is
  analysed as a tomato and the pipeline warns that the species is uncertain - rejecting a real tomato
  as "unsupported" is the worse mistake for a tomato-monitoring robot.
- PlaceholderSpeciesClassifier: used when no species model is present. It does not fabricate a
  confidence: every confirmed plant is assumed to be a tomato, confidence=None, is_placeholder=True.
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..preprocess import center_crop_input
from .schema import SpeciesPrediction

SUPPORTED_SPECIES = ["tomato"]  # extend as real per-species disease models are added
UNSUPPORTED_MIN_CONFIDENCE = 0.70


class PlantSpeciesClassifier(ABC):
    @abstractmethod
    def classify(self, bgr: np.ndarray) -> SpeciesPrediction: ...


class PlaceholderSpeciesClassifier(PlantSpeciesClassifier):
    """MVP stand-in: no trained species model exists, so every confirmed plant is assumed to be the
    one supported species (tomato) rather than guessing."""

    def classify(self, bgr: np.ndarray) -> SpeciesPrediction:
        return SpeciesPrediction(species="tomato", confidence=None, is_placeholder=True)


class OnnxSpeciesClassifier(PlantSpeciesClassifier):
    def __init__(self, model_path: str | Path, threads: int = 4):
        import onnxruntime as ort

        model_path = Path(model_path)
        meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf8"))
        self.classes = [c.lower() for c in meta["classes"]]
        self.size, self.temperature = meta["img_size"], meta.get("temperature", 1.0)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def probabilities(self, bgr: np.ndarray) -> dict[str, float]:
        logits = self.session.run(None, {self.input_name: center_crop_input(bgr, self.size)})[0][0] / self.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return dict(zip(self.classes, p.tolist()))

    def classify(self, bgr: np.ndarray) -> SpeciesPrediction:
        probs = self.probabilities(bgr)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top, p_top = ranked[0]
        if top not in SUPPORTED_SPECIES and p_top >= UNSUPPORTED_MIN_CONFIDENCE:
            return SpeciesPrediction(species=top, confidence=p_top, ranked=ranked[:3])
        return SpeciesPrediction(species="tomato", confidence=probs.get("tomato", 0.0), ranked=ranked[:3])
