"""Plant / not-plant pre-filter (ONNX, no PyTorch needed on Pi).

Two interchangeable models, auto-detected from the ONNX output shape:

- Trained binary gate (models_synthetic/plant_gate_trained.onnx): MobileNetV3-Small fine-tuned on
  Oxford Flowers102 (plant) vs Imagenette (not_plant) by scripts/train_plant_gate.py. 2-way softmax,
  class 1 = plant. Got 100% val/test accuracy - but that reflects how visually distinct those two
  source datasets are (studio flower photos vs random ImageNet objects), not real-world difficulty.
  It has never seen a genuinely ambiguous negative (a green wall, a leaf-patterned fabric, a salad on
  a plate) or the robot's actual camera/lighting conditions, so treat it as a good starting filter to
  be re-tested (and likely re-thresholded, maybe retrained with harder negatives) on real photos.
- Zero-shot ImageNet-1k gate (models_synthetic/plant_gate.onnx): plain pretrained MobileNetV2,
  1000-way softmax. We sum probability mass on ~37 plant/produce/fungus classes as a proxy. No
  training, works immediately, but coarser and was superseded once real training data existed.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .preprocess import whole_frame_input

# Indices into torchvision's ImageNet-1k category list (MobileNet_V2_Weights.IMAGENET1K_V2.meta["categories"]).
PLANT_CLASS_IDS = [
    936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954,
    955, 956, 957, 958,  # head cabbage ... hay (vegetables, fruit-on-plant, hay)
    984, 985, 986, 987, 988, 989, 990,  # rapeseed, daisy, lady's slipper, corn, acorn, hip, buckeye
    991, 992, 993, 994, 995, 996, 997,  # coral fungus ... bolete (fungi, common in garden shots)
]

ZERO_SHOT_THRESHOLD = 0.10  # summed probability on plant-related ImageNet classes needed to pass
TRAINED_THRESHOLD = 0.5     # softmax probability of the "plant" class needed to pass


@dataclass
class GateResult:
    is_plant: bool
    plant_score: float   # summed softmax prob over PLANT_CLASS_IDS
    top_class_id: int
    top_prob: float

    def to_dict(self):
        return {"is_plant": self.is_plant, "plant_score": round(float(self.plant_score), 4),
                "top_class_id": int(self.top_class_id), "top_prob": round(float(self.top_prob), 4)}


class PlantGate:
    def __init__(self, model_path: str | Path, threshold: float | None = None):
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        n_out = self.sess.get_outputs()[0].shape[-1]
        self.trained = n_out == 2  # 2-way binary head vs 1000-way ImageNet head
        self.threshold = threshold if threshold is not None else (
            TRAINED_THRESHOLD if self.trained else ZERO_SHOT_THRESHOLD)

    def check(self, bgr: np.ndarray) -> GateResult:
        # Whole frame, exactly as scripts/train_plant_gate.py evaluates (torchvision Resize((224, 224))).
        x = whole_frame_input(bgr, 224)
        logits = self.sess.run(None, {self.input_name: x})[0][0]
        probs = _softmax(logits)
        if self.trained:
            plant_score = float(probs[1])  # classes = ["not_plant", "plant"]
        else:
            plant_score = float(probs[PLANT_CLASS_IDS].sum())
        top_id = int(np.argmax(probs))
        return GateResult(is_plant=plant_score >= self.threshold, plant_score=plant_score,
                           top_class_id=top_id, top_prob=float(probs[top_id]))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
