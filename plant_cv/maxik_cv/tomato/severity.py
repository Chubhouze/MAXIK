"""SeverityEstimator (spec section 7): LOW / MODERATE / HIGH from the share of leaf area that is
visibly affected (lesion.py), UNKNOWN when that measurement isn't available or the condition isn't a
problem. Thresholds are the spec's approximate visual bands, not biological measurements.

Applies to the named diseases AND to TOMATO_UNCERTAIN: when the classifier can't say what is wrong
but a large share of the leaf is visibly damaged, "something is clearly wrong, severity HIGH" is
still the useful answer. A healthy classification gets no severity (the raw affected share is still
reported separately).

Known blind spot, repeated here because it matters: severity is derived from colour damage only.
Diseases that don't show as lesions (mosaic virus, mites, leaf curl) can read LOW while serious.
"""
from .schema import Condition, Severity

LOW_MAX = 0.10
MODERATE_MAX = 0.30

_SCORED = {
    Condition.TOMATO_EARLY_BLIGHT, Condition.TOMATO_LATE_BLIGHT, Condition.TOMATO_LEAF_MOLD,
    Condition.TOMATO_BACTERIAL_SPOT, Condition.TOMATO_SEPTORIA_LEAF_SPOT, Condition.TOMATO_SPIDER_MITES,
    Condition.TOMATO_TARGET_SPOT, Condition.TOMATO_YELLOW_LEAF_CURL_VIRUS, Condition.TOMATO_MOSAIC_VIRUS,
    Condition.TOMATO_UNCERTAIN,
}


def estimate_severity(condition: Condition, affected_fraction: float | None) -> Severity:
    if condition not in _SCORED or affected_fraction is None:
        return Severity.UNKNOWN
    if affected_fraction < LOW_MAX:
        return Severity.LOW
    if affected_fraction < MODERATE_MAX:
        return Severity.MODERATE
    return Severity.HIGH
