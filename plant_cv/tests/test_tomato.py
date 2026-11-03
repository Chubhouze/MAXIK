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


class WholePlantTests(unittest.TestCase):
    def leaf(self, condition, conf=0.9, affected=0.05, side=100, x=0):
        from maxik_cv.tomato.whole_plant import LeafResult
        return LeafResult((x, 0, x + side, side), 0.9, condition, conf, affected, False)

    def summarize(self, leaves, total=None):
        from maxik_cv.tomato.whole_plant import summarize
        return summarize(leaves, total if total is not None else len(leaves))

    def test_all_healthy_plant(self):
        s = self.summarize([self.leaf(Condition.TOMATO_HEALTHY, affected=0.02) for _ in range(5)])
        self.assertEqual(s.condition, Condition.TOMATO_HEALTHY)
        self.assertEqual(s.severity, Severity.UNKNOWN)
        self.assertIn("0 of 5", s.observed[0])

    def test_most_confident_disease_wins_and_leaf_share_raises_severity(self):
        leaves = [self.leaf(Condition.TOMATO_SEPTORIA_LEAF_SPOT, 0.9, 0.05) for _ in range(4)] + \
                 [self.leaf(Condition.TOMATO_EARLY_BLIGHT, 0.6, 0.05), self.leaf(Condition.TOMATO_HEALTHY)]
        s = self.summarize(leaves)
        self.assertEqual(s.condition, Condition.TOMATO_SEPTORIA_LEAF_SPOT)
        # damage only 5% (LOW by area) but 5 of 6 leaves affected -> HIGH by leaf share
        self.assertEqual(s.severity, Severity.HIGH)

    def test_damage_is_area_weighted(self):
        big = self.leaf(Condition.TOMATO_LATE_BLIGHT, affected=0.4, side=200)
        small = self.leaf(Condition.TOMATO_HEALTHY, affected=0.0, side=100, x=300)
        s = self.summarize([big, small])
        self.assertAlmostEqual(s.affected_fraction, 0.4 * 4 / 5, places=3)

    def test_mixed_conditions_are_flagged(self):
        leaves = [self.leaf(Condition.TOMATO_SEPTORIA_LEAF_SPOT) for _ in range(2)] + \
                 [self.leaf(Condition.TOMATO_MOSAIC_VIRUS) for _ in range(2)]
        self.assertTrue(any("different conditions" in w for w in self.summarize(leaves).warnings))

    def test_only_uncertain_leaves_give_uncertain_plant(self):
        s = self.summarize([self.leaf(Condition.TOMATO_UNCERTAIN, 0.4) for _ in range(3)])
        self.assertEqual(s.condition, Condition.TOMATO_UNCERTAIN)

    def test_cap_on_analysed_leaves_is_reported(self):
        s = self.summarize([self.leaf(Condition.TOMATO_HEALTHY) for _ in range(3)], total=40)
        self.assertTrue(any("40 leaves were found" in w for w in s.warnings))


class SpeciesPolicyTests(unittest.TestCase):
    """OnnxSpeciesClassifier's decision rule, with the network replaced by fixed probabilities."""

    def classifier(self, probs):
        from maxik_cv.tomato.species_classifier import OnnxSpeciesClassifier
        clf = OnnxSpeciesClassifier.__new__(OnnxSpeciesClassifier)
        clf.probabilities = lambda bgr: probs
        return clf

    def test_confident_other_crop_is_unsupported(self):
        p = self.classifier({"tomato": 0.05, "corn": 0.9, "potato": 0.05}).classify(None)
        self.assertEqual((p.species, round(p.confidence, 2)), ("corn", 0.9))

    def test_unsure_other_crop_is_still_analysed_as_tomato(self):
        p = self.classifier({"tomato": 0.35, "potato": 0.55, "corn": 0.10}).classify(None)
        self.assertEqual(p.species, "tomato")
        self.assertAlmostEqual(p.confidence, 0.35)
        self.assertEqual(p.ranked[0][0], "potato")


class _FakeGate:
    def detect(self, bgr):
        from maxik_cv.tomato.schema import DetectionResult
        return DetectionResult(True, 0.99)


class _FakeLeafDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, bgr):
        from maxik_cv.tomato.leaf_detector import Leaf
        return [Leaf(b, 0.9) for b in self.boxes]


class _FakeHealth:
    """Says "septoria" for crops from the left half of the photo, "healthy" for the right half."""

    def __init__(self, width):
        self.width = width
        self.calls = []

    def classify(self, crop):
        from maxik_cv.tomato.schema import HealthPrediction
        self.calls.append(crop.shape)
        cond = Condition.TOMATO_SEPTORIA_LEAF_SPOT if crop.mean() < 128 else Condition.TOMATO_HEALTHY
        return HealthPrediction(cond, 0.9, ranked=[(cond, 0.9)])


class WholePlantPipelineTests(unittest.TestCase):
    def photo(self):
        img = textured(5, 400)
        img[:, :200] = img[:, :200] // 3            # left half darker -> fake "septoria" leaves
        img[:, 200:] = 255 - (255 - img[:, 200:]) // 3  # right half brighter -> fake "healthy" leaves
        return img

    def inspect(self, boxes):
        from maxik_cv.tomato.pipeline import TomatoInspector
        health = _FakeHealth(400)
        insp = TomatoInspector(_FakeGate(), health_classifier=health, leaf_detector=_FakeLeafDetector(boxes))
        import maxik_cv.tomato.pipeline as pl
        orig = pl.flag_plant_too_small
        pl.flag_plant_too_small = lambda bgr, q: q   # synthetic texture isn't plant-coloured
        try:
            return insp.inspect(self.photo()), health
        finally:
            pl.flag_plant_too_small = orig

    def test_several_leaves_switch_to_whole_plant_mode(self):
        boxes = [(10, 10, 90, 90), (100, 100, 180, 190), (220, 20, 300, 100), (250, 200, 390, 380)]
        r, health = self.inspect(boxes)
        self.assertEqual(r.mode, "whole_plant")
        self.assertEqual(len(r.leaves), 4)
        self.assertEqual(len(health.calls), 4)        # every leaf diagnosed separately
        self.assertEqual(r.condition, Condition.TOMATO_SEPTORIA_LEAF_SPOT)
        self.assertIn("2 of 4", r.observed_features[0])
        self.assertTrue(r.to_dict()["leaves"][0]["box"])

    def test_one_leaf_keeps_single_leaf_mode(self):
        r, health = self.inspect([(10, 10, 390, 390)])
        self.assertEqual(r.mode, "single_leaf")
        self.assertEqual(r.leaves, [])


class _FakeBedDetector:
    classes = ["crop", "weed"]
    max_detections = 300

    def __init__(self, dets):
        self.dets = dets  # [(box, score, label)]

    def detect_raw(self, bgr):
        if not self.dets:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        b, s, l = zip(*self.dets)
        return np.array(b, np.float32), np.array(s, np.float32), np.array(l, np.int64)


class BedAnalysisTests(unittest.TestCase):
    def bed(self):
        img = np.full((300, 600, 3), (60, 80, 110), np.uint8)              # brown soil
        cv2.circle(img, (80, 150), 45, (40, 160, 40), -1)                   # healthy green plant
        cv2.circle(img, (300, 150), 45, (40, 160, 40), -1)                  # plant with a yellow-brown half
        img[105:195, 300:345] = (30, 180, 220)
        cv2.circle(img, (520, 80), 20, (50, 150, 60), -1)                   # a weed
        return img

    def test_counts_order_and_damage(self):
        from maxik_cv.tomato.bed import analyze_bed
        det = _FakeBedDetector([((255, 105, 345, 195), 0.9, 1), ((35, 105, 125, 195), 0.95, 1),
                                ((500, 60, 540, 100), 0.8, 2), ((10, 10, 20, 20), 0.9, 2)])  # last one too small
        r = analyze_bed(self.bed(), det)
        d = r.to_dict()
        self.assertEqual((d["crops"], d["weeds"]), (2, 1))
        self.assertEqual([p["index"] for p in d["plants"]], [1, 2, 3])           # left to right
        self.assertEqual(d["plants"][0]["kind"], "crop")
        healthy, damaged = d["plants"][0]["affectedFraction"], d["plants"][1]["affectedFraction"]
        self.assertLess(healthy, 0.1)
        self.assertGreater(damaged, healthy)
        self.assertGreater(d["weedCover"], 0)
        self.assertIn("2 crop plants and 1 weed found", d["summary"])

    def test_empty_bed(self):
        from maxik_cv.tomato.bed import analyze_bed
        d = analyze_bed(self.bed(), _FakeBedDetector([])).to_dict()
        self.assertEqual((d["crops"], d["weeds"]), (0, 0))
        self.assertIn("0 crop plants and 0 weeds", d["summary"])


class LeafDetectorPostprocessTests(unittest.TestCase):
    def test_decode_identity_and_nms(self):
        from maxik_cv.tomato.leaf_detector import decode, nms
        props = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 140, 160]], np.float32)
        self.assertTrue(np.allclose(decode(np.zeros((3, 4), np.float32), props), props, atol=1e-4))
        keep = nms(props, np.array([0.9, 0.8, 0.7], np.float32), 0.5)
        self.assertEqual(sorted(keep.tolist()), [0, 2])

    def test_resize_hits_torchvision_target_exactly(self):
        from maxik_cv.tomato.leaf_detector import MAX_SIZE, MIN_SIZE, resize_for_detector
        for h, w in [(300, 400), (1200, 900), (100, 2000), (2000, 100), (512, 512)]:
            rh, rw = resize_for_detector(np.zeros((h, w, 3), np.uint8)).shape[:2]
            self.assertTrue(min(rh, rw) == MIN_SIZE or max(rh, rw) == MAX_SIZE, (h, w, rh, rw))
            self.assertLessEqual(max(rh, rw), MAX_SIZE)
            self.assertLessEqual(min(rh, rw), MIN_SIZE)


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

    def test_whole_plant_summary_explains_severity_source(self):
        leaves = [{"condition": "TOMATO_SEPTORIA_LEAF_SPOT"}] * 5 + [{"condition": "TOMATO_HEALTHY"}]
        s = build_summary(self.result(mode="whole_plant", leaves=leaves, affected_fraction=0.03))
        self.assertIn("6 leaves analysed, 5 show possible problems", s)
        self.assertIn("share of affected leaves", s)
        self.assertNotIn("is damaged (high", s)   # no "3% damaged (high severity)" contradiction

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

    @unittest.skipUnless((ROOT / "samples" / "healthy.jpg").exists(), "samples folder not present")
    def test_bundled_samples_work_and_nothing_else_is_served(self):
        ok = self.client.get("/tomato/sample/healthy.jpg")
        self.assertEqual(ok.status_code, 200)
        self.assertIn("Healthy", ok.get_data(as_text=True))
        self.assertIn("/tomato/sample/early_blight.jpg", self.client.get("/tomato").get_data(as_text=True))
        for bad in ("nope.jpg", "..%2Fwebapp.py", "..%5C..%5Cwebapp.py", "healthy.jpg%00"):
            self.assertEqual(self.client.get(f"/tomato/sample/{bad}").status_code, 404, bad)

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
