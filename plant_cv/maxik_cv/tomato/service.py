"""The robot-facing entry point: photo (+ point id + sensors) in, finished analysis out, history updated.
Both the JSON API and the browser UI call this, so they can never disagree about what an inspection is.
"""
import numpy as np

from .history import TomatoHistory
from .pipeline import TomatoInspector
from .recommendations import build_summary
from .schema import AnalysisResult

SENSOR_KEYS = ("soil_moisture", "temperature", "humidity")


def clean_sensors(raw: dict) -> dict:
    """Keep only known sensor keys with numeric values; anything else is ignored, not trusted."""
    out = {}
    for k in SENSOR_KEYS:
        try:
            if raw.get(k) not in (None, ""):
                out[k] = float(raw[k])
        except (TypeError, ValueError):
            pass
    return out


def analyze(inspector: TomatoInspector, history: TomatoHistory | None, bgr: np.ndarray,
            point_id: str | None = None, sensors: dict | None = None) -> tuple[AnalysisResult, int | None]:
    """Returns (result, inspection_id). Only real analyses (plant found, image usable) are recorded:
    a rejected frame says nothing about the plant's condition and must not distort its trend."""
    sensors = clean_sensors(sensors or {})
    result = inspector.inspect(bgr, sensors)
    result.sensors, result.point_id = sensors, point_id

    recorded = result.plant_detected and result.status_label != "IMAGE QUALITY ISSUE"
    if point_id and history is not None and recorded:
        result.trend = history.trend(point_id, result.condition.value, result.affected_fraction)
        result.summary = build_summary(result)
        return result, history.add(point_id, result.to_dict(), bgr, sensors)
    result.summary = build_summary(result)
    return result, None
