"""TomatoInspector: orchestrates the full pipeline from spec section 1/13.

    Image -> ImageQualityAnalyzer -> PlantDetector -> PlantSpeciesClassifier
          -> TomatoHealthClassifier -> ConfidenceEvaluator -> SeverityEstimator
          -> RecommendationEngine -> ResultFormatter -> AnalysisResult

Each stage is an injected component (see the other modules in this package), so any one of them -
most likely PlantSpeciesClassifier or TomatoHealthClassifier, currently placeholders - can be
swapped for a real trained model later without touching this orchestration or the UI.
"""
import numpy as np

from ..segment import color_stats, plant_mask
from .class_mapping import CONDITION_TO_STATUS_LABEL
from .confidence import apply_confidence_gate
from .lesion import analyze_affected_area
from .health_classifier import PlaceholderTomatoHealthClassifier, TomatoHealthClassifier
from .image_quality import SHARPNESS_SOFT, analyze_basic_quality, flag_plant_too_small
from .plant_detector import PlantDetector
from .possible_causes import list_possible_causes
from .recommendations import get_recommendation
from .schema import AnalysisResult, Condition, ImageQualityIssue, QualityReport, Severity
from .severity import estimate_severity
from .species_classifier import PlaceholderSpeciesClassifier, PlantSpeciesClassifier

_QUALITY_MESSAGES = {
    ImageQualityIssue.IMAGE_TOO_DARK: "Image is too dark for reliable analysis. Please retake it in better lighting.",
    ImageQualityIssue.IMAGE_TOO_BRIGHT: "Image is too bright/overexposed. Please retake it out of direct glare.",
    ImageQualityIssue.IMAGE_BLURRY: "Image is too blurry for reliable analysis. Please hold the camera steady and retake it.",
    ImageQualityIssue.PLANT_TOO_SMALL:
        "The plant is too small in the frame. Please move closer and retake the photo.",
}


class TomatoInspector:
    def __init__(self, plant_detector: PlantDetector,
                 species_classifier: PlantSpeciesClassifier | None = None,
                 health_classifier: TomatoHealthClassifier | None = None):
        self.plant_detector = plant_detector
        self.species_classifier = species_classifier or PlaceholderSpeciesClassifier()
        self.health_classifier = health_classifier or PlaceholderTomatoHealthClassifier()

    def inspect(self, bgr: np.ndarray, sensors: dict | None = None) -> AnalysisResult:
        quality = analyze_basic_quality(bgr)
        if not quality.ok:
            return self._quality_failure(quality)

        detection = self.plant_detector.detect(bgr)
        if not detection.is_plant:
            return self._result(
                condition=Condition.NOT_PLANT, quality=quality,
                plant_detected=False, plant_type=None, plant_confidence=detection.confidence,
                condition_confidence=None, observed_features=[], severity=Severity.UNKNOWN,
            )

        quality = flag_plant_too_small(bgr, quality)
        if not quality.ok:
            return self._quality_failure(quality, plant_detected=True)

        # Numeric colour breakdown (green/yellow/brown ratio, saturation, vigour index) and the
        # rule-based "possible causes" advisory both run off this, independent of species/health
        # classification below - useful diagnostic context even while those are placeholders.
        stats = color_stats(bgr, plant_mask(bgr))
        causes = [c.to_dict() for c in list_possible_causes(stats, sensors)]

        species = self.species_classifier.classify(bgr)
        if species.species != "tomato":
            return self._result(
                condition=Condition.UNSUPPORTED_PLANT, quality=quality,
                plant_detected=True, plant_type="Unknown / Unsupported", plant_confidence=species.confidence,
                condition_confidence=None, observed_features=[], severity=Severity.UNKNOWN,
                color_stats=stats.to_dict(), possible_causes=causes,
            )

        health = self.health_classifier.classify(bgr)
        condition = apply_confidence_gate(health.condition, health.confidence)
        lesion = analyze_affected_area(bgr)
        severity = estimate_severity(condition, lesion.fraction)

        warnings = _cross_checks(health, condition, severity, lesion.fraction)
        if quality.sharpness < SHARPNESS_SOFT:
            warnings.append("The photo is slightly soft. Blur can make the classifier confidently wrong, "
                            "so treat this result with extra caution or retake the photo.")
        candidates = [{"condition": c.value, "label": CONDITION_TO_STATUS_LABEL[c], "probability": round(p, 4)}
                      for c, p in health.ranked]

        return self._result(
            condition=condition, quality=quality,
            plant_detected=True, plant_type="Tomato", plant_confidence=species.confidence,
            condition_confidence=health.confidence, observed_features=health.observed_features,
            severity=severity, color_stats=stats.to_dict(), possible_causes=causes,
            health_model_is_placeholder=health.is_placeholder,
            affected_fraction=lesion.fraction, warnings=warnings, candidates=candidates, lesion=lesion,
        )

    def _quality_failure(self, quality: QualityReport, plant_detected: bool | None = None) -> AnalysisResult:
        message = " ".join(_QUALITY_MESSAGES[i] for i in quality.issues)
        return AnalysisResult(
            plant_detected=bool(plant_detected) if plant_detected is not None else False,
            plant_type=None, plant_confidence=None,
            health_status="NOT_APPLICABLE", condition=Condition.TOMATO_UNCERTAIN, condition_confidence=None,
            severity=Severity.UNKNOWN, observed_features=[],
            status_label="IMAGE QUALITY ISSUE", message=message, recommendations=[], quality=quality,
        )

    def _result(self, *, condition: Condition, quality: QualityReport, plant_detected: bool,
                plant_type: str | None, plant_confidence: float | None,
                condition_confidence: float | None, observed_features: list[str],
                severity: Severity, color_stats: dict | None = None,
                possible_causes: list[dict] | None = None,
                health_model_is_placeholder: bool = False, affected_fraction: float | None = None,
                warnings: list[str] | None = None, candidates: list[dict] | None = None,
                lesion=None) -> AnalysisResult:
        status_label, message, recommendations = get_recommendation(condition, condition_confidence)
        return AnalysisResult(
            plant_detected=plant_detected, plant_type=plant_type, plant_confidence=plant_confidence,
            health_status=_health_status(condition), condition=condition,
            condition_confidence=condition_confidence, severity=severity,
            observed_features=observed_features, status_label=status_label, message=message,
            recommendations=recommendations, quality=quality,
            color_stats=color_stats, possible_causes=possible_causes or [],
            health_model_is_placeholder=health_model_is_placeholder,
            affected_fraction=affected_fraction, warnings=warnings or [], candidates=candidates or [],
            lesion=lesion,
        )


# Conditions whose damage is small or not a colour change, so the colour-based severity under-reads them.
_LOW_COLOUR_SIGNAL = {
    Condition.TOMATO_MOSAIC_VIRUS, Condition.TOMATO_YELLOW_LEAF_CURL_VIRUS, Condition.TOMATO_SPIDER_MITES,
    Condition.TOMATO_TARGET_SPOT, Condition.TOMATO_BACTERIAL_SPOT,
}
_HEALTHY_BUT_DAMAGED = 0.20


def _cross_checks(health, condition: Condition, severity: Severity, affected: float | None) -> list[str]:
    """Places where the independent signals (classifier vs colour measurement) disagree or the
    severity number can't be trusted. Surfaced to the user instead of silently picking a side."""
    notes = []
    if health.condition != condition and health.ranked:
        notes.append(f"The model's best guess was \"{CONDITION_TO_STATUS_LABEL[health.condition]}\" "
                     f"({health.confidence:.0%}), below the confidence needed to report it.")
    if condition == Condition.TOMATO_HEALTHY and affected is not None and affected >= _HEALTHY_BUT_DAMAGED:
        notes.append(f"Classified healthy, but about {affected:.0%} of the visible leaf area is not green. "
                     "Check the leaf manually.")
    if condition in _LOW_COLOUR_SIGNAL and severity in (Severity.LOW, Severity.UNKNOWN):
        notes.append("Severity is estimated from visible colour damage only. This condition often shows "
                     "little colour change, so a low severity does not mean it is mild.")
    return notes


def _health_status(condition: Condition) -> str:
    if condition in (Condition.NOT_PLANT, Condition.UNSUPPORTED_PLANT):
        return "NOT_APPLICABLE"
    if condition == Condition.TOMATO_HEALTHY:
        return "HEALTHY"
    if condition == Condition.TOMATO_UNCERTAIN:
        return "UNCERTAIN"
    return "UNHEALTHY"
