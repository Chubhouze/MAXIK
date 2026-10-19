"""Rule-based indicators: work from day 1 without a dataset, and later act as a sanity check for the CNN.

Thresholds are starting points - calibrate them on your own Pi-camera photos (scripts/evaluate.py --rules).
"""
from .segment import ColorStats

THRESHOLDS = {
    "min_plant_fraction": 0.005,  # less plant in frame -> nothing to judge -> unknown
    "yellow_ratio": 0.15,
    "brown_ratio": 0.12,
    "low_saturation": 0.30,
    "soil_moisture_dry": 0.30,    # normalised 0..1 from the soil sensor
    "area_drop": 0.20,            # -20% plant area vs previous inspection at the same point
    "yellow_increase": 0.08,      # +8 p.p. yellow share vs previous inspection
}


def rule_indicators(stats: ColorStats, sensors: dict | None = None, prev: dict | None = None) -> dict:
    """Returns {indicator: score 0..1} plus human-readable reasons."""
    t = THRESHOLDS
    sensors = sensors or {}
    scores, reasons = {}, []

    if stats.plant_fraction < t["min_plant_fraction"]:
        # A plant that was there last time and has (almost) vanished is itself a strong damage signal.
        if prev and prev.get("plant_fraction", 0) >= t["min_plant_fraction"]:
            return {"scores": {"weak_damage": 0.9, "healthy": 0.1, "dryness": 0.0, "yellowing": 0.0},
                    "reasons": ["plant almost disappeared from frame since last check"], "valid": True}
        return {"scores": {}, "reasons": ["plant not found in frame"], "valid": False}

    y = min(1.0, stats.yellow_ratio / t["yellow_ratio"] / 2)
    scores["yellowing"] = y
    if stats.yellow_ratio >= t["yellow_ratio"]:
        reasons.append(f"yellow leaf area {stats.yellow_ratio:.0%}")

    d = min(1.0, stats.brown_ratio / t["brown_ratio"] / 2)
    if stats.mean_saturation < t["low_saturation"]:
        d = max(d, 0.5)
        reasons.append("leaves look faded (low saturation)")
    soil = sensors.get("soil_moisture")
    if soil is not None and soil < t["soil_moisture_dry"]:
        d = min(1.0, d + 0.3)
        reasons.append(f"soil moisture low ({soil:.2f})")
    if stats.brown_ratio >= t["brown_ratio"]:
        reasons.append(f"brown/dry tissue {stats.brown_ratio:.0%}")
    scores["dryness"] = d

    w = 0.0
    if prev:
        pa = prev.get("plant_fraction", 0)
        drop = (pa - stats.plant_fraction) / pa if pa > 0 else 0.0
        if drop >= t["area_drop"]:
            w = min(1.0, 0.5 + drop)
            # A large drop dominates weaker colour cues on the (now small) remaining plant.
            scores["dryness"] *= 1 - drop
            scores["yellowing"] *= 1 - drop
            reasons.append(f"plant area dropped {drop:.0%} since last check")
        if stats.yellow_ratio - prev.get("yellow_ratio", 0) >= t["yellow_increase"]:
            scores["yellowing"] = max(scores["yellowing"], 0.7)
            reasons.append("yellowing increased since last check")
    scores["weak_damage"] = w

    scores["healthy"] = max(0.0, 1.0 - max(scores.values()))
    return {"scores": scores, "reasons": reasons, "valid": True}
