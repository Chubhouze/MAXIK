"""LeafDetector: finds the individual leaves in a whole-plant photo, so each one can be diagnosed
separately (scripts/train_leaf_detector.py: Faster R-CNN MobileNetV3-FPN on PlantDoc, one "leaf"
class for all crops).

The ONNX file holds only the network (backbone, region proposals, RoI features, class and box
outputs). Both ends are done here in NumPy, mirroring torchvision exactly:
- before: resize (short side 512, long side <= 853), ImageNet normalisation, zero-pad to a multiple
  of 32 - inside the ONNX graph the resize would be frozen to the size of the export's dummy image;
- after: decode box offsets, softmax, score threshold, NMS, scale back to the photo - torch's ONNX
  exporters cannot export torchvision's detection post-processing (data-dependent output size).
scripts/train_leaf_detector.py checks that torch and ONNX+NumPy produce the same boxes.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)        # torchvision RoIHeads box coder
BBOX_XFORM_CLIP = math.log(1000.0 / 16)
NMS_IOU = 0.5
MIN_BOX = 1e-2
MIN_SIZE, MAX_SIZE, DIVISIBLE = 512, 853, 32  # must match train_leaf_detector.build_model
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def resize_for_detector(bgr: np.ndarray, min_size: int = MIN_SIZE, max_size: int = MAX_SIZE) -> np.ndarray:
    """Short side to min_size unless the long side would exceed max_size (torchvision's rule). The
    result is exactly at that target, so torch's own transform leaves it unchanged (scale 1.0)."""
    h, w = bgr.shape[:2]
    by_short, by_long = min_size / min(h, w), max_size / max(h, w)
    s = min(by_short, by_long)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    if by_short <= by_long:   # short side is the limit -> it becomes exactly min_size
        nh, nw = (min_size, nw) if h <= w else (nh, min_size)
    else:                     # long side is the limit -> it becomes exactly max_size
        nh, nw = (max_size, nw) if h >= w else (nh, max_size)
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)


def prepare(resized_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (rgb - MEAN) / STD
    h, w = x.shape[1:]
    ph, pw = -(-h // DIVISIBLE) * DIVISIBLE, -(-w // DIVISIBLE) * DIVISIBLE
    out = np.zeros((1, 3, ph, pw), np.float32)
    out[0, :, :h, :w] = x
    return out


@dataclass
class Leaf:
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 in original-photo pixels
    score: float


def decode(deltas: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    wx, wy, ww, wh = BOX_WEIGHTS
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    cx = proposals[:, 0] + 0.5 * widths
    cy = proposals[:, 1] + 0.5 * heights
    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw = np.minimum(deltas[:, 2] / ww, BBOX_XFORM_CLIP)
    dh = np.minimum(deltas[:, 3] / wh, BBOX_XFORM_CLIP)
    pcx, pcy = dx * widths + cx, dy * heights + cy
    pw, ph = np.exp(dw) * widths, np.exp(dh) * heights
    return np.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        overlap = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][overlap <= iou]
    return np.array(keep, dtype=np.int64)


def postprocess(proposals, class_logits, box_regression, resized_hw, original_hw,
                score_threshold: float, max_detections: int):
    """(boxes [N,4] in original pixels, scores [N], labels [N]) from the raw network outputs. Class 0 is
    background; NMS runs per class, like torchvision's batched_nms."""
    logits = class_logits - class_logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    n_classes = probs.shape[1]
    per_class = box_regression.reshape(len(proposals), n_classes, 4)
    h, w = resized_hw
    out_b, out_s, out_l = [], [], []
    for c in range(1, n_classes):
        boxes = decode(per_class[:, c], proposals)
        scores = probs[:, c]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = (scores > score_threshold) & ((boxes[:, 2] - boxes[:, 0]) >= MIN_BOX) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_BOX)
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes):
            k = nms(boxes, scores, NMS_IOU)
            out_b.append(boxes[k]), out_s.append(scores[k]), out_l.append(np.full(len(k), c))
    if not out_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    boxes, scores, labels = np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_l)
    order = scores.argsort()[::-1][:max_detections]
    ry, rx = original_hw[0] / h, original_hw[1] / w
    return boxes[order] * np.array([rx, ry, rx, ry], dtype=np.float32), scores[order], labels[order]


class LeafDetector:
    def __init__(self, model_path: str | Path, score_threshold: float | None = None, max_detections: int = 60):
        import onnxruntime as ort

        model_path = Path(model_path)
        meta_path = model_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf8")) if meta_path.exists() else {}
        self.score_threshold = score_threshold if score_threshold is not None else meta.get("score_threshold", 0.5)
        self.max_detections = max_detections if max_detections != 60 else meta.get("max_detections", 60)
        self.min_size, self.max_size = meta.get("min_size", MIN_SIZE), meta.get("max_size", MAX_SIZE)
        self.classes = meta.get("classes", ["leaf"])
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def raw(self, bgr: np.ndarray):
        resized = resize_for_detector(bgr, self.min_size, self.max_size)
        hw = resized.shape[:2]
        proposals, class_logits, box_regression = self.session.run(
            None, {"image": prepare(resized), "image_hw": np.array(hw, np.int64)})
        return proposals, class_logits, box_regression, hw

    def detect_raw(self, bgr: np.ndarray, score_threshold: float | None = None):
        """(boxes, scores, labels) arrays in original-photo pixels; labels start at 1 (0 = background)."""
        proposals, logits, regression, resized_hw = self.raw(bgr)
        return postprocess(proposals, logits, regression, resized_hw, bgr.shape[:2],
                           self.score_threshold if score_threshold is None else score_threshold, self.max_detections)

    def detect(self, bgr: np.ndarray, score_threshold: float | None = None) -> list[Leaf]:
        boxes, scores, _ = self.detect_raw(bgr, score_threshold)
        return [Leaf(tuple(int(round(v)) for v in b), float(s)) for b, s in zip(boxes, scores)]
