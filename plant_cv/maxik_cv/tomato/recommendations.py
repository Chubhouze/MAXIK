"""RecommendationEngine: converts a (condition, confidence) prediction into user-facing text.
Deliberately has zero ML/IO imports - it only knows about the Condition enum, so the wording
can be edited (or translated) without touching any prediction code, and it's trivially unit-
testable on its own.

Phrasing follows spec section 16: "possible" / "suspected" / "visual characteristics associated
with", never "your plant definitely has..." - image classification alone isn't a lab diagnosis.
"""
from .class_mapping import CONDITION_TO_STATUS_LABEL
from .schema import AnalysisResult, Condition, Severity

_RECOMMENDATIONS: dict[Condition, list[str]] = {
    Condition.TOMATO_HEALTHY: [
        "Continue normal care",
        "Maintain appropriate watering",
        "Continue monitoring the plant",
    ],
    Condition.TOMATO_EARLY_BLIGHT: [
        "Inspect affected leaves",
        "Remove heavily affected dead/damaged foliage where appropriate",
        "Avoid unnecessarily wetting foliage",
        "Improve airflow around the plant",
        "Continue monitoring",
        "If symptoms spread, consult an appropriate plant disease reference or specialist",
    ],
    Condition.TOMATO_LATE_BLIGHT: [
        "Inspect the entire plant",
        "Separate it from other susceptible plants if practical",
        "Avoid prolonged leaf wetness",
        "Remove severely affected material where appropriate",
        "Monitor progression closely",
        "Recommend further confirmation - visual classification alone is not definitive",
    ],
    Condition.TOMATO_LEAF_MOLD: [
        "Improve ventilation",
        "Reduce excessive humidity where possible",
        "Avoid keeping foliage wet",
        "Inspect both sides of leaves",
        "Remove heavily affected foliage if appropriate",
        "Continue monitoring",
    ],
    Condition.TOMATO_BACTERIAL_SPOT: [
        "Avoid overhead watering - use drip or furrow irrigation instead",
        "Remove and dispose of affected leaves/material rather than composting them",
        "Avoid working with plants while foliage is wet",
        "Improve airflow around the plant",
        "If symptoms spread, consult an appropriate plant disease reference or specialist",
    ],
    Condition.TOMATO_SEPTORIA_LEAF_SPOT: [
        "Avoid overhead watering - use drip or furrow irrigation instead",
        "Remove affected lower leaves where appropriate",
        "Improve airflow around the plant",
        "Avoid working with plants while foliage is wet",
        "Continue monitoring",
    ],
    Condition.TOMATO_SPIDER_MITES: [
        "Inspect the undersides of leaves closely for fine stippling or webbing",
        "Spider mites spread quickly - isolate the plant from others if practical",
        "Consider insecticidal soap or a miticide labelled for tomatoes",
        "Increase humidity around the plant where practical (mites favour hot, dry conditions)",
        "Continue monitoring closely",
    ],
    Condition.TOMATO_TARGET_SPOT: [
        "Inspect affected leaves for round, dark, concentric-ring lesions",
        "Remove heavily affected foliage where appropriate",
        "Improve airflow around the plant",
        "Avoid unnecessarily wetting foliage",
        "If symptoms spread, consult an appropriate plant disease reference or specialist",
    ],
    Condition.TOMATO_YELLOW_LEAF_CURL_VIRUS: [
        "This is a viral disease (commonly whitefly-transmitted) with no direct cure",
        "Check for and control whiteflies on and around the plant",
        "Isolate or remove severely affected plants to limit spread to healthy ones",
        "Recommend further confirmation - visual classification alone is not definitive",
    ],
    Condition.TOMATO_MOSAIC_VIRUS: [
        "This is a viral disease with no direct cure",
        "Wash hands and tools after handling the plant - this virus spreads easily by contact",
        "Avoid handling tobacco products before touching plants (cross-infection risk)",
        "Isolate or remove severely affected plants to limit spread",
        "Recommend further confirmation - visual classification alone is not definitive",
    ],
    Condition.TOMATO_UNCERTAIN: [
        "Take another photo in good lighting, showing the affected leaf clearly from a closer distance",
    ],
}

_MESSAGES: dict[Condition, str] = {
    Condition.NOT_PLANT: "Please point the camera at a plant and try again.",
    Condition.UNSUPPORTED_PLANT: "This version of M.A.X.I.K. currently supports tomato health analysis.",
    Condition.TOMATO_HEALTHY: "No major visible health problems were detected.",
    Condition.TOMATO_EARLY_BLIGHT: "The model detected visual characteristics associated with early blight.",
    Condition.TOMATO_LATE_BLIGHT: "The model detected visual characteristics associated with late blight.",
    Condition.TOMATO_LEAF_MOLD: "The model detected visual characteristics associated with leaf mold.",
    Condition.TOMATO_BACTERIAL_SPOT: "The model detected visual characteristics associated with bacterial spot.",
    Condition.TOMATO_SEPTORIA_LEAF_SPOT: "The model detected visual characteristics associated with septoria leaf spot.",
    Condition.TOMATO_SPIDER_MITES: "The model detected visual characteristics associated with spider mite damage.",
    Condition.TOMATO_TARGET_SPOT: "The model detected visual characteristics associated with target spot.",
    Condition.TOMATO_YELLOW_LEAF_CURL_VIRUS: "The model detected visual characteristics associated with yellow leaf curl virus.",
    Condition.TOMATO_MOSAIC_VIRUS: "The model detected visual characteristics associated with mosaic virus.",
    Condition.TOMATO_UNCERTAIN:
        "Some visible symptoms may indicate a health problem, but the model is not confident enough to classify it.",
}


def build_summary(r: AnalysisResult) -> str:
    """The one-paragraph answer a person (or the robot's report) reads first: what it looks like, how bad,
    and whether it's getting better or worse. Built only from fields already in the result."""
    if not r.plant_detected or r.status_label == "IMAGE QUALITY ISSUE":
        return r.message
    head = r.status_label + (f" ({r.condition_confidence:.0%} model confidence)" if r.condition_confidence else "")
    parts = [head + "."]
    if r.affected_fraction is not None:
        if r.severity != Severity.UNKNOWN:
            parts.append(f"About {r.affected_fraction:.0%} of the visible leaf area is damaged "
                         f"({r.severity.value.lower()} severity).")
        else:
            parts.append(f"About {r.affected_fraction:.0%} of the visible leaf area is not green.")
    t = r.trend
    if t and t["direction"] != "unknown":
        when = t["previousTs"][:16].replace("T", " ") + " UTC"
        if t["direction"] == "stable":
            parts.append(f"Compared with the previous check ({when}) it is about the same.")
        else:
            word = "worse" if t["direction"] == "worsening" else "better"
            parts.append(f"Compared with the previous check ({when}) it is {word} "
                         f"({t['deltaAffected'] * 100:+.0f} percentage points of damaged area).")
    if t and t["conditionChanged"]:
        prev = CONDITION_TO_STATUS_LABEL.get(Condition(t["previousCondition"]), t["previousCondition"])
        parts.append(f"The assessed condition changed (previously: {prev}).")
    return " ".join(parts)


def get_recommendation(condition: Condition, confidence: float | None) -> tuple[str, str, list[str]]:
    """Returns (status_label, message, recommendations) for the given already-gated condition.
    Confidence gating (deciding whether to downgrade to TOMATO_UNCERTAIN) happens upstream in
    confidence.apply_confidence_gate - by the time this runs, `condition` is final."""
    status_label = CONDITION_TO_STATUS_LABEL[condition]
    message = _MESSAGES.get(condition, "")
    recommendations = list(_RECOMMENDATIONS.get(condition, []))
    return status_label, message, recommendations
