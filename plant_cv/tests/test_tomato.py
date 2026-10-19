"""Logic tests for the tomato pipeline. Standard library only:

    python -m unittest discover -s tests -v          (run from plant_cv/)

They pin down the decisions that were calibrated on data (image-quality thresholds, severity bands,
trend noise band, confidence gate) and the API contract, using synthetic images and a temp database,
so they run in seconds and need no dataset. API tests are skipped if the ONNX models are missing.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = tempfile.mkdtemp()
os.environ["MAXIK_DB_PATH"] = str(Path(TMP) / "api.db")

from maxik_cv.segment import ColorStats  # noqa: E402
from maxik_cv.tomato.confidence import apply_confidence_gate  # noqa: E402
from maxik_cv.tomato.history import STABLE_BAND, TomatoHistory  # noqa: E402
from maxik_cv.tomato.image_quality import analyze_basic_quality  # noqa: E402
from maxik_cv.tomato.possible_causes import list_possible_causes  # noqa: E402
from maxik_cv.tomato.recommendations import build_summary  # noqa: E402
from maxik_cv.tomato.schema import AnalysisResult, Condition, ImageQualityIssue, QualityReport, Severity  # noqa: E402
from maxik_cv.tomato.service import clean_sensors  # noqa: E402
from maxik_cv.tomato.severity import estimate_severity  # noqa: E402


def textured(seed=0, side=300):
    return np.random.default_rng(seed).integers(0, 256, (side, side, 3), dtype=np.uint8)


class SeverityTests(unittest.TestCase):
    def test_bands_follow_the_spec(self):
        c = Condition.TOMATO_EARLY_BLIGHT
        self.assertEqual(estimate_severity(c, 0.05), Severity.LOW)
        self.assertEqual(estimate_severity(c, 0.10), Severity.MODERATE)
        self.assertEqual(estimate_severity(c, 0.29), Severity.MODERATE)
        self.assertEqual(estimate_severity(c, 0.30), Severity.HIGH)

    def test_unknown_when_nothing_to_score(self):
        self.assertEqual(estimate_severity(Condition.TOMATO_EARLY_BLIGHT, None), Severity.UNKNOWN)
        self.assertEqual(estimate_severity(Condition.TOMATO_HEALTHY, 0.5), Severity.UNKNOWN)
        self.assertEqual(estimate_severity(Condition.NOT_PLANT, 0.5), Severity.UNKNOWN)

    def test_uncertain_still_gets_a_severity(self):
        self.assertEqual(estimate_severity(Condition.TOMATO_UNCERTAIN, 0.6), Severity.HIGH)


class ConfidenceGateTests(unittest.TestCase):
    def test_low_confidence_disease_becomes_uncertain(self):
        self.assertEqual(apply_confidence_gate(Condition.TOMATO_LATE_BLIGHT, 0.48), Condition.TOMATO_UNCERTAIN)

    def test_confident_disease_and_healthy_pass_through(self):
        self.assertEqual(apply_confidence_gate(Condition.TOMATO_LATE_BLIGHT, 0.9), Condition.TOMATO_LATE_BLIGHT)
        self.assertEqual(apply_confidence_gate(Condition.TOMATO_HEALTHY, 0.3), Condition.TOMATO_HEALTHY)


class ImageQualityTests(unittest.TestCase):
    def issues(self, img):
        return set(analyze_basic_quality(img).issues)

    def test_textured_frame_is_fine(self):
        self.assertEqual(self.issues(textured()), set())

    def test_black_frame_is_too_dark(self):
        self.assertIn(ImageQualityIssue.IMAGE_TOO_DARK, self.issues(np.zeros((300, 300, 3), np.uint8)))

    def test_washed_out_frame_is_too_bright(self):
        self.assertIn(ImageQualityIssue.IMAGE_TOO_BRIGHT, self.issues(np.full((300, 300, 3), 245, np.uint8)))

    def test_heavy_blur_is_rejected(self):
        self.assertIn(ImageQualityIssue.IMAGE_BLURRY, self.issues(cv2.GaussianBlur(textured(), (0, 0), 4)))

    def test_leaf_on_black_background_is_not_called_dark(self):
        # regression: judged on the frame MEAN this was rejected as "too dark" although it is a normal photo
        img = np.zeros((300, 300, 3), np.uint8)
        img[60:240, 60:240] = textured(1, 180)
        self.assertNotIn(ImageQualityIssue.IMAGE_TOO_DARK, self.issues(img))


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.h = TomatoHistory(Path(TMP) / f"{self.id()}.db")
        self.img = textured()

    def add(self, point, affected, condition="TOMATO_EARLY_BLIGHT"):
        self.h.add(point, {"condition": condition, "conditionConfidence": 0.9, "severity": "LOW",
                           "affectedFraction": affected, "statusLabel": "x"}, self.img)

    def test_first_visit_has_no_trend(self):
        self.assertIsNone(self.h.trend("p1", "TOMATO_HEALTHY", 0.02))

    def test_worsening_improving_stable(self):
        self.add("p1", 0.20)
        self.assertEqual(self.h.trend("p1", "TOMATO_EARLY_BLIGHT", 0.20 + STABLE_BAND + 0.01)["direction"], "worsening")
        self.assertEqual(self.h.trend("p1", "TOMATO_EARLY_BLIGHT", 0.20 - STABLE_BAND - 0.01)["direction"], "improving")
        self.assertEqual(self.h.trend("p1", "TOMATO_EARLY_BLIGHT", 0.20 + STABLE_BAND - 0.02)["direction"], "stable")

    def test_points_are_independent(self):
        self.add("p1", 0.5)
        self.assertIsNone(self.h.trend("p2", "TOMATO_HEALTHY", 0.0))

    def test_condition_change_is_reported(self):
        self.add("p1", 0.2, "TOMATO_EARLY_BLIGHT")
        self.assertTrue(self.h.trend("p1", "TOMATO_HEALTHY", 0.2)["conditionChanged"])

    def test_series_and_thumbnail_roundtrip(self):
        self.add("p1", 0.1)
        self.add("p1", 0.3)
        s = self.h.series("p1")
        self.assertEqual([x["affected_fraction"] for x in s], [0.1, 0.3])
        self.assertTrue(self.h.thumb(s[0]["id"]).startswith(b"\xff\xd8"))


class PossibleCausesTests(unittest.TestCase):
    YELLOW = ColorStats(0.4, 0.5, 0.30, 0.02, 0.5, 0.2)

    def test_dry_soil_supports_underwatering_only_with_visible_damage(self):
        with_damage = list_possible_causes(self.YELLOW, {"soil_moisture": 0.2})
        self.assertTrue(any(c.source == "sensor" and c.category == "Watering" for c in with_damage))
        healthy = ColorStats(0.4, 0.9, 0.02, 0.01, 0.6, 0.3)
        self.assertFalse(any(c.source == "sensor" for c in list_possible_causes(healthy, {"soil_moisture": 0.2})))

    def test_no_sensors_no_sensor_causes(self):
        self.assertFalse(any(c.source == "sensor" for c in list_possible_causes(self.YELLOW)))

    def test_clean_sensors_drops_garbage(self):
        self.assertEqual(clean_sensors({"soil_moisture": "0.4", "temperature": "abc", "evil": "1", "humidity": ""}),
                         {"soil_moisture": 0.4})


class SummaryTests(unittest.TestCase):
    def result(self, **kw):
        base = dict(plant_detected=True, plant_type="Tomato", plant_confidence=None, health_status="UNHEALTHY",
                    condition=Condition.TOMATO_LATE_BLIGHT, condition_confidence=0.93, severity=Severity.HIGH,
                    observed_features=[], status_label="Possible Late Blight", message="m", recommendations=[],
                    quality=QualityReport(ok=True), affected_fraction=0.42)
        base.update(kw)
        return AnalysisResult(**base)

    def test_mentions_condition_damage_and_severity(self):
        s = build_summary(self.result())
        self.assertIn("Possible Late Blight", s)
        self.assertIn("42%", s)
        self.assertIn("high severity", s)

    def test_mentions_trend_when_present(self):
        t = {"direction": "worsening", "deltaAffected": 0.2, "previousTs": "2026-01-02T10:00:00+00:00",
             "conditionChanged": False, "previousCondition": "TOMATO_LATE_BLIGHT"}
        self.assertIn("worse", build_summary(self.result(trend=t)))

    def test_rejected_photo_summary_is_the_retake_message(self):
        s = build_summary(self.result(plant_detected=False, message="Please retake it."))
        self.assertEqual(s, "Please retake it.")


MODELS = ROOT / "models_synthetic" / "plant_gate_trained.onnx"


@unittest.skipUnless(MODELS.exists(), "ONNX models not present")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("MAXIK_API_TOKEN", None)
        os.environ.pop("MAXIK_UI_PASSWORD", None)
        import webapp
        cls.client = webapp.app.test_client()

    def test_health_endpoint(self):
        self.assertEqual(self.client.get("/api/health").get_json()["status"], "ok")

    def test_undecodable_image_is_400(self):
        self.assertEqual(self.client.post("/api/tomato", data=b"not an image").status_code, 400)

    def test_bad_point_id_is_400(self):
        ok, buf = cv2.imencode(".jpg", textured())
        r = self.client.post("/api/tomato?point=../../etc", data=buf.tobytes(), headers={"Content-Type": "image/jpeg"})
        self.assertEqual(r.status_code, 400)

    def test_non_plant_is_reported_and_not_recorded(self):
        ok, buf = cv2.imencode(".jpg", textured(3))
        r = self.client.post("/api/tomato?point=test-point", data=buf.tobytes(), headers={"Content-Type": "image/jpeg"})
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertFalse(body["plantDetected"])
        self.assertIsNone(body["inspectionId"])
        self.assertEqual(self.client.get("/point/test-point").status_code, 404)

    def test_api_token_is_enforced_when_set(self):
        os.environ["MAXIK_API_TOKEN"] = "secret"
        try:
            self.assertEqual(self.client.post("/api/tomato", data=b"x").status_code, 401)
            self.assertEqual(self.client.post("/api/tomato", data=b"x", headers={"Authorization": "Bearer secret"}).status_code, 400)
        finally:
            os.environ.pop("MAXIK_API_TOKEN")

    def test_ui_password_protects_pages_but_not_health(self):
        os.environ["MAXIK_UI_PASSWORD"] = "pw"
        try:
            self.assertEqual(self.client.get("/dashboard").status_code, 401)
            self.assertEqual(self.client.get("/api/health").status_code, 200)
        finally:
            os.environ.pop("MAXIK_UI_PASSWORD")


if __name__ == "__main__":
    unittest.main()
