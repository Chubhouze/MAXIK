"""Tomato-specific health pipeline (MVP). Separate from the generic maxik_cv.pipeline
(healthy/dryness/yellowing/weak_damage, any plant) - this package implements the disease-specific
taxonomy (NOT_PLANT / UNSUPPORTED_PLANT / TOMATO_HEALTHY / TOMATO_EARLY_BLIGHT / ...).
"""
from .class_mapping import DATASET_CLASS_NAMES
from .health_classifier import OnnxTomatoHealthClassifier, PlaceholderTomatoHealthClassifier, TomatoHealthClassifier
from .pipeline import TomatoInspector
from .plant_detector import PlantDetector
from .schema import AnalysisResult, Condition, ConfidenceLevel, Severity
from .species_classifier import PlaceholderSpeciesClassifier, PlantSpeciesClassifier

__all__ = [
    "TomatoInspector", "PlantDetector",
    "TomatoHealthClassifier", "PlaceholderTomatoHealthClassifier", "OnnxTomatoHealthClassifier",
    "PlantSpeciesClassifier", "PlaceholderSpeciesClassifier",
    "AnalysisResult", "Condition", "ConfidenceLevel", "Severity",
    "DATASET_CLASS_NAMES",
]
