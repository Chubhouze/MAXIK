"""ConfidenceEvaluator: turns a raw score into a level, and decides whether a disease
prediction is trustworthy enough to show as-is or must fall back to TOMATO_UNCERTAIN.
Pure function, no ML - kept separate so the threshold policy is one obvious place to tune.
"""
from .schema import Condition, ConfidenceLevel

HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.60


def confidence_level(score: float) -> ConfidenceLevel:
    if score >= HIGH_THRESHOLD:
        return ConfidenceLevel.HIGH
    if score >= MEDIUM_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def apply_confidence_gate(condition: Condition, confidence: float) -> Condition:
    """Below MEDIUM_THRESHOLD, never show a specific diagnosis - report uncertainty instead."""
    if condition in (Condition.TOMATO_HEALTHY, Condition.TOMATO_UNCERTAIN):
        return condition
    if confidence_level(confidence) == ConfidenceLevel.LOW:
        return Condition.TOMATO_UNCERTAIN
    return condition
