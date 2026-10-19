"""PlantSpeciesClassifier: decides which supported species pipeline to run (spec section 14 -
Tomato / Potato / Cucumber / ... under one Plant hierarchy). No species model has been trained
yet, so PlaceholderSpeciesClassifier does NOT fabricate a per-image confidence score: it states
plainly that species detection isn't implemented and, per the MVP scope ("focus on tomato"),
assumes tomato for any confirmed plant so the rest of the pipeline is exercisable end-to-end.
confidence=None (not a made-up number) is how callers detect this is a placeholder.

Swap in a real classifier later by implementing PlantSpeciesClassifier.classify() the same way
maxik_cv.classifier.PlantClassifier loads an ONNX model, and constructing TomatoInspector with it.
"""
from abc import ABC, abstractmethod

import numpy as np

from .schema import SpeciesPrediction

SUPPORTED_SPECIES = ["tomato"]  # extend as real per-species classifiers/models are added


class PlantSpeciesClassifier(ABC):
    @abstractmethod
    def classify(self, bgr: np.ndarray) -> SpeciesPrediction: ...


class PlaceholderSpeciesClassifier(PlantSpeciesClassifier):
    """MVP stand-in: no trained species model exists, so every confirmed plant is assumed to be
    the one supported species (tomato) rather than guessing. Replace with a real classifier to
    support additional species without changing anything downstream."""

    def classify(self, bgr: np.ndarray) -> SpeciesPrediction:
        return SpeciesPrediction(species="tomato", confidence=None, is_placeholder=True)
