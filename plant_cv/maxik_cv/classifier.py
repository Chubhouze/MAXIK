"""ONNX classifier wrapper. Only needs onnxruntime + numpy + opencv, so it runs on Raspberry Pi 4 (CPU)."""
import json
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantClassifier:
    def __init__(self, model_path: str | Path, threads: int = 4):
        import onnxruntime as ort

        model_path = Path(model_path)
        meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf8"))
        self.classes: list[str] = meta["classes"]
        self.size: int = meta["img_size"]
        self.temperature: float = meta.get("temperature", 1.0)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        # Must match eval transform in scripts/train.py: Resize(short side = size*1.14) + CenterCrop(size).
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = int(self.size * 1.14) / min(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        y0, x0 = (h - self.size) // 2, (w - self.size) // 2
        rgb = rgb[y0:y0 + self.size, x0:x0 + self.size]
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return x.transpose(2, 0, 1)[None]

    def predict(self, bgr: np.ndarray, tta: bool = True) -> dict[str, float]:
        x = self.preprocess(bgr)
        if tta:  # horizontal flip test-time augmentation: cheap and more stable
            x = np.concatenate([x, x[..., ::-1].copy()])
        logits = self.session.run(None, {self.input_name: x})[0].mean(axis=0) / self.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}
