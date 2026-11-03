"""Check an installed robot: load every model and analyse the bundled sample photos on this machine,
with timings. Needs no camera and no network; exits with an error if something is missing or broken.

    python3 scripts/selftest.py
"""
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from send_to_server import build_local_inspector  # noqa: E402


def main():
    import onnxruntime as ort
    print(f"Python {sys.version.split()[0]}, OpenCV {cv2.__version__}, onnxruntime {ort.__version__}")

    t = time.perf_counter()
    inspector = build_local_inspector()
    print(f"models loaded in {time.perf_counter() - t:.1f} s "
          f"(whole-plant mode {'on' if inspector.leaf_detector else 'off'})")

    samples = sorted((ROOT / "samples").glob("*.jpg"))
    if not samples:
        sys.exit("no samples/*.jpg to test with")
    times = []
    for path in samples:
        img = cv2.imread(str(path))
        t = time.perf_counter()
        r = inspector.inspect(img)
        times.append(time.perf_counter() - t)
        conf = f" ({r.condition_confidence:.0%})" if r.condition_confidence is not None else ""
        verdict = f"{r.condition.value}{conf}, severity {r.severity.value}" if r.plant_detected else "no plant"
        print(f"  {path.name:<20} {verdict:<45} {times[-1]:.2f} s")

    from maxik_cv.tomato.bed import analyze_bed
    from maxik_cv.tomato.leaf_detector import LeafDetector
    bed_model = ROOT / "models_bed" / "bed_detector.onnx"
    if bed_model.exists():
        det = LeafDetector(bed_model)
        img = cv2.imread(str(samples[0]))
        analyze_bed(img, det, {})  # first run includes one-off setup
        t = time.perf_counter()
        analyze_bed(img, det, {})
        print(f"  bed detector: {time.perf_counter() - t:.2f} s per frame")
    else:
        print("  bed detector: not installed")

    print(f"OK: {len(samples)} photos analysed, {sum(times) / len(times):.2f} s per photo on average")


if __name__ == "__main__":
    main()
