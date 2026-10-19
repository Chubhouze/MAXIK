"""PossibleCausesAdvisor: a separate, rule-based, purely numeric overlay - NOT the ML disease
classifier - that surfaces plausible general reasons a plant might look unhealthy, grounded in
general horticultural references. Runs off the same colour statistics maxik_cv.segment already
computes (green/yellow/brown ratio, saturation, vigour index), independent of whether the ML
condition classifier is a placeholder or a real trained model, and always phrased as "possible" /
"may indicate" - never a diagnosis (spec section 16).

Categories and thresholds are grounded in:
- Colorado State University Extension, "Recognizing Common Tomato Pests, Pathogens and Problems"
  https://extension.colostate.edu/resource/recognizing-common-tomato-pests-pathogens-and-problems/
  (nutrient deficiency / disease / pest / environmental / physical / watering categories)
- "Yellow Leaves on Tomato Plants: Causes & How to Fix", The Gardening Cook
  https://thegardeningcook.com/yellow-leaves-on-tomato-plants/
  (overwatering -> soft/dull yellowing; underwatering -> dry/brown edges; nitrogen deficiency ->
  yellowing starting on older/lower leaves)
"""
from dataclasses import dataclass

from ..segment import ColorStats

YELLOW_THRESHOLD = 0.15
BROWN_THRESHOLD = 0.12
LOW_SATURATION_THRESHOLD = 0.30
LOW_GREEN_THRESHOLD = 0.50
LOW_VIGOUR_THRESHOLD = 0.05


@dataclass
class PossibleCause:
    category: str  # "Watering" | "Nutrients" | "Light" | "Pests" | "Disease" | "Physical damage" | "Temperature"
    text: str
    source: str = "image"  # "image" (colour measurements) or "sensor" (robot sensor reading backs it up)

    def to_dict(self) -> dict:
        return {"category": self.category, "text": self.text, "source": self.source}


# Sensor thresholds are starting points (soil moisture normalised 0..1 like maxik_cv.rules, humidity
# as a 0..1 fraction, temperature in degC); calibrate them against the robot's real sensors.
SOIL_DRY = 0.30
SOIL_WET = 0.80
HUMIDITY_HIGH = 0.85
HUMIDITY_LOW = 0.35
TEMP_HOT = 28.0
TEMP_HEAT_STRESS = 35.0
TEMP_COLD = 10.0


def list_possible_causes(stats: ColorStats, sensors: dict | None = None) -> list[PossibleCause]:
    causes: list[PossibleCause] = []
    sensors = sensors or {}

    if stats.yellow_ratio >= YELLOW_THRESHOLD:
        causes.append(PossibleCause(
            "Watering",
            f"Yellow leaf area {stats.yellow_ratio:.0%} — overwatering (waterlogged, oxygen-starved "
            "roots) commonly yellows leaves with a soft, dull look."))
        causes.append(PossibleCause(
            "Nutrients",
            f"Yellow leaf area {stats.yellow_ratio:.0%} — nitrogen deficiency is a common cause of "
            "yellowing that starts on older/lower leaves and moves upward."))

    if stats.brown_ratio >= BROWN_THRESHOLD:
        causes.append(PossibleCause(
            "Watering",
            f"Brown/dry tissue {stats.brown_ratio:.0%} — underwatering typically shows as dry, "
            "browning leaf edges rather than soft yellowing."))
        causes.append(PossibleCause(
            "Disease",
            f"Brown/dry tissue {stats.brown_ratio:.0%} — dark brown-black spots/lesions are also "
            "characteristic of fungal blight; a close, well-lit photo of a single affected leaf "
            "helps tell watering stress and disease apart."))

    if stats.mean_saturation < LOW_SATURATION_THRESHOLD:
        causes.append(PossibleCause(
            "Light",
            f"Low colour saturation ({stats.mean_saturation:.0%}) — pale, washed-out leaves can "
            "indicate insufficient light."))

    if stats.green_ratio < LOW_GREEN_THRESHOLD and stats.plant_fraction > 0.03:
        causes.append(PossibleCause(
            "Pests",
            "Green leaf coverage is lower than expected without strong yellow/brown discolouration "
            "— worth checking leaf undersides for aphids, whiteflies, or spider mites, which are "
            "easy to miss in a single overview photo."))

    if stats.exg_mean < LOW_VIGOUR_THRESHOLD:
        causes.append(PossibleCause(
            "Physical damage",
            "Low vigour index — could reflect wilting, recent transplant stress, or physical "
            "damage; comparing with an earlier photo of the same plant helps confirm this."))

    causes.extend(_sensor_causes(stats, sensors))
    return causes


def _sensor_causes(stats: ColorStats, s: dict) -> list[PossibleCause]:
    """Readings that support (or are the only sign of) a cause. They never override the image: a sensor
    value only adds a cause when it's consistent with what the photo shows, except the temperature
    and humidity notes, which are environmental risk factors on their own."""
    out: list[PossibleCause] = []
    soil, temp, hum = s.get("soil_moisture"), s.get("temperature"), s.get("humidity")
    visible_damage = stats.yellow_ratio >= YELLOW_THRESHOLD or stats.brown_ratio >= BROWN_THRESHOLD

    if soil is not None and soil < SOIL_DRY and visible_damage:
        out.append(PossibleCause(
            "Watering", f"Soil moisture is low ({soil:.2f}) and the leaf shows yellow/brown areas — supports "
            "underwatering as a cause.", "sensor"))
    if soil is not None and soil > SOIL_WET and stats.yellow_ratio >= YELLOW_THRESHOLD:
        out.append(PossibleCause(
            "Watering", f"Soil moisture is very high ({soil:.2f}) and the leaf is yellowing — supports "
            "overwatering / poor drainage as a cause.", "sensor"))
    if hum is not None and hum > HUMIDITY_HIGH:
        out.append(PossibleCause(
            "Disease", f"Air humidity is high ({hum:.0%}) — fungal diseases such as blight and leaf mold are "
            "favoured by wet, humid conditions and poor ventilation.", "sensor"))
    if hum is not None and temp is not None and hum < HUMIDITY_LOW and temp > TEMP_HOT:
        out.append(PossibleCause(
            "Pests", f"Hot ({temp:.0f} °C) and dry ({hum:.0%}) air favours spider mites — check the undersides "
            "of the leaves.", "sensor"))
    if temp is not None and temp >= TEMP_HEAT_STRESS:
        out.append(PossibleCause(
            "Temperature", f"Temperature is very high ({temp:.0f} °C) — heat stress can damage leaves and "
            "flowers.", "sensor"))
    if temp is not None and temp <= TEMP_COLD:
        out.append(PossibleCause(
            "Temperature", f"Temperature is low ({temp:.0f} °C) — cold stress slows growth and nutrient uptake "
            "in tomatoes.", "sensor"))
    return out
