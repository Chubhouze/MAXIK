"""Shared enums and dataclasses for the tomato pipeline. No ML/IO here on purpose -
every other module in this package imports from here, never from each other's internals,
so the prediction contract stays stable while individual stages (detector, classifier, ...)
get swapped out.
"""
from dataclasses import dataclass, field
from enum import Enum


class Condition(str, Enum):
    NOT_PLANT = "NOT_PLANT"
    UNSUPPORTED_PLANT = "UNSUPPORTED_PLANT"
    TOMATO_HEALTHY = "TOMATO_HEALTHY"
    TOMATO_EARLY_BLIGHT = "TOMATO_EARLY_BLIGHT"
    TOMATO_LATE_BLIGHT = "TOMATO_LATE_BLIGHT"
    TOMATO_LEAF_MOLD = "TOMATO_LEAF_MOLD"
    TOMATO_BACTERIAL_SPOT = "TOMATO_BACTERIAL_SPOT"
    TOMATO_SEPTORIA_LEAF_SPOT = "TOMATO_SEPTORIA_LEAF_SPOT"
    TOMATO_SPIDER_MITES = "TOMATO_SPIDER_MITES"
    TOMATO_TARGET_SPOT = "TOMATO_TARGET_SPOT"
    TOMATO_YELLOW_LEAF_CURL_VIRUS = "TOMATO_YELLOW_LEAF_CURL_VIRUS"
    TOMATO_MOSAIC_VIRUS = "TOMATO_MOSAIC_VIRUS"
    TOMATO_UNCERTAIN = "TOMATO_UNCERTAIN"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Severity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class ImageQualityIssue(str, Enum):
    IMAGE_TOO_DARK = "IMAGE_TOO_DARK"
    IMAGE_TOO_BRIGHT = "IMAGE_TOO_BRIGHT"
    IMAGE_BLURRY = "IMAGE_BLURRY"
    PLANT_TOO_SMALL = "PLANT_TOO_SMALL"


@dataclass
class QualityReport:
    ok: bool
    issues: list[ImageQualityIssue] = field(default_factory=list)
    brightness: float = 0.0       # mean V channel, 0..1
    sharpness: float = 0.0        # image_quality.sharpness_score, higher = sharper
    plant_area_fraction: float | None = None  # coarse colour-based estimate, pre-detection


@dataclass
class DetectionResult:
    is_plant: bool
    confidence: float


@dataclass
class SpeciesPrediction:
    species: str | None          # e.g. "tomato", or None if undetermined
    confidence: float | None     # None when the classifier is a placeholder (no real estimate)
    is_placeholder: bool = False


@dataclass
class HealthPrediction:
    condition: Condition
    confidence: float
    observed_features: list[str] = field(default_factory=list)
    is_placeholder: bool = False
    ranked: list[tuple[Condition, float]] = field(default_factory=list)  # model's top guesses, best first


@dataclass
class AnalysisResult:
    """Mirrors the JSON contract the recommendation/UI layer consumes (spec section 6)."""
    plant_detected: bool
    plant_type: str | None
    plant_confidence: float | None
    health_status: str  # "HEALTHY" | "UNHEALTHY" | "UNCERTAIN" | "NOT_APPLICABLE"
    condition: Condition
    condition_confidence: float | None
    severity: Severity
    observed_features: list[str]
    status_label: str
    message: str
    recommendations: list[str]
    quality: QualityReport
    color_stats: dict | None = None       # from maxik_cv.segment.color_stats, when a plant was analyzed
    possible_causes: list[dict] = field(default_factory=list)  # from possible_causes.py - not an ML output
    health_model_is_placeholder: bool = False  # True until a real TomatoHealthClassifier backend is wired in
    affected_fraction: float | None = None     # share of leaf area visibly not-green (lesion.py), 0..1
    warnings: list[str] = field(default_factory=list)  # e.g. classifier and colour analysis disagree
    candidates: list[dict] = field(default_factory=list)  # model's top guesses incl. runners-up
    trend: dict | None = None                  # change vs the previous inspection of the same point
    sensors: dict = field(default_factory=dict)  # robot sensor readings that came with the photo
    point_id: str | None = None                # which plant/point of the route this inspection belongs to
    summary: str = ""                          # one-paragraph plain-language result (recommendations.build_summary)
    lesion: object | None = field(default=None, repr=False, compare=False)  # masks for drawing, not serialized

    def to_dict(self) -> dict:
        return {
            "plantDetected": self.plant_detected,
            "plantType": self.plant_type,
            "plantConfidence": self.plant_confidence,
            "healthStatus": self.health_status,
            "condition": self.condition.value,
            "conditionConfidence": self.condition_confidence,
            "severity": self.severity.value,
            "observedFeatures": self.observed_features,
            "statusLabel": self.status_label,
            "message": self.message,
            "recommendations": self.recommendations,
            "imageQuality": {
                "ok": self.quality.ok,
                "issues": [i.value for i in self.quality.issues],
                "brightness": round(self.quality.brightness, 3),
                "sharpness": round(self.quality.sharpness, 1),
                "plantAreaFraction": self.quality.plant_area_fraction,
            },
            "colorStats": self.color_stats,
            "possibleCauses": self.possible_causes,
            "healthModelIsPlaceholder": self.health_model_is_placeholder,
            "affectedFraction": None if self.affected_fraction is None else round(self.affected_fraction, 4),
            "warnings": self.warnings,
            "candidates": self.candidates,
            "trend": self.trend,
            "sensors": self.sensors,
            "pointId": self.point_id,
            "summary": self.summary,
        }
