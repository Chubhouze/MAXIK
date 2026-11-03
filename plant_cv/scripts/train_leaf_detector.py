"""Train an object detector (Faster R-CNN, MobileNetV3-Large FPN backbone, COCO-pretrained) and export it
to ONNX for maxik_cv.tomato.leaf_detector. Two presets:

  --task leaf  every leaf in whole-plant photos, one "leaf" class, PlantDoc (13 crops, CC BY 4.0;
               prepare_plantdoc.py). Each detected leaf is then diagnosed by the leaf classifier.
  --task bed   every plant in TOP-DOWN photos of beds/fields, classes "crop" and "weed", CropAndWeed
               (74 species, non-commercial licence; prepare_cropandweed.py). Plants are small (median
               ~22 px after resizing), hence smaller anchors, a larger input and more detections.

    python scripts/train_leaf_detector.py --task leaf --epochs 24
    python scripts/train_leaf_detector.py --task bed  --epochs 12
    python scripts/train_leaf_detector.py --task bed  --export-only

Reports AP@0.5 per class and recall/precision@0.5 on the held-out test split, exports <out>/<name>.onnx
and checks that ONNX + the NumPy post-processing reproduce the torch model's boxes.
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
PRESETS = {
    "leaf": dict(root=DATA / "plantdoc_od", classes=["leaf"], min_size=512, max_size=853, detections=60,
                 small_anchors=False, out=ROOT / "models_leaf", name="leaf_detector", epochs=24,
                 trained_on="PlantDoc object detection, 13 crops (CC BY 4.0)"),
    "bed": dict(root=DATA / "cropandweed_prepared", classes=["crop", "weed"], min_size=720, max_size=1280,
                detections=300, small_anchors=True, out=ROOT / "models_bed", name="bed_detector", epochs=12,
                trained_on="CropAndWeed, 74 species mapped to crop/weed (non-commercial licence)"),
}
# kept for scripts that import them (the leaf preset)
MIN_SIZE, MAX_SIZE = PRESETS["leaf"]["min_size"], PRESETS["leaf"]["max_size"]


def load_split(split: str, root: Path = PRESETS["leaf"]["root"], classes: list[str] | None = None):
    """[(photo path, boxes [N,4], labels [N])]. With one class every box is that class (label 1);
    otherwise the 'class' column must be one of `classes` (labels 1..K)."""
    classes = classes or ["leaf"]
    per = defaultdict(list)
    for r in csv.DictReader(open(root / "split_boxes.csv", encoding="utf8")):
        if r["split"] != split:
            continue
        label = 1 if len(classes) == 1 else classes.index(r["class"]) + 1 if r["class"] in classes else None
        per[r["file"]]  # photos with no (kept) box are still listed: they teach "nothing here"
        if label is not None and r["xmin"] != "":
            per[r["file"]].append([float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")] + [label])
    out = []
    for f, rows in sorted(per.items()):
        a = np.array(rows, np.float32).reshape(-1, 5)
        out.append((root / f, a[:, :4], a[:, 4].astype(np.int64)))
    return out


class LeafBoxes(Dataset):
    def __init__(self, items, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes, labels = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        boxes = boxes.copy()
        if self.train:
            if random.random() < 0.5:  # horizontal flip
                img = img[:, ::-1].copy()
                w = img.shape[1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.8:  # light/colour, no hue shift
                a, b = random.uniform(0.7, 1.3), random.uniform(-25, 25)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.2:
                img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.5))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {"boxes": torch.from_numpy(boxes).reshape(-1, 4), "labels": torch.from_numpy(labels)}
        return t, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained: bool = True, num_classes: int = 2, min_size: int = MIN_SIZE, max_size: int = MAX_SIZE,
                detections: int = 60, small_anchors: bool = False):
    """num_classes includes background."""
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT" if pretrained else None, min_size=min_size,
                                              max_size=max_size, box_detections_per_img=detections)
    if small_anchors:  # same 15 anchors per location as the pretrained RPN head expects, just half the size
        model.rpn.anchor_generator = AnchorGenerator(((16, 32, 64, 128, 256),) * 3, ((0.5, 1.0, 2.0),) * 3)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return model


def _ap(scored, n_gt) -> float:
    scored.sort(key=lambda x: -x[0])
    tp = np.cumsum([h for _, h in scored])
    fp = np.cumsum([not h for _, h in scored])
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], recall, [recall[-1] if len(recall) else 0]])
    mpre = np.concatenate([[1], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = 2, iou_thr=0.5, score_thr=0.5):
    """AP@0.5 per class (a detection only counts if its class is right) and their mean, plus overall
    recall/precision at score_thr."""
    model.eval()
    scored = defaultdict(list)
    n_gt, tp_at, fp_at = defaultdict(int), 0, 0
    for imgs, targets in loader:
        outs = model([im.to(device) for im in imgs])
        for out, tgt in zip(outs, targets):
            for c in range(1, num_classes):
                gt = tgt["boxes"][tgt["labels"] == c].to(device)
                n_gt[c] += len(gt)
                sel = out["labels"] == c
                order = out["scores"][sel].argsort(descending=True)
                boxes, scores = out["boxes"][sel][order], out["scores"][sel][order]
                matched = torch.zeros(len(gt), dtype=torch.bool, device=device)
                ious = box_iou(boxes, gt) if len(gt) and len(boxes) else None
                for k in range(len(boxes)):
                    hit = False
                    if ious is not None:
                        iou, j = ious[k].max(0)
                        if iou >= iou_thr and not matched[j]:
                            matched[j], hit = True, True
                    scored[c].append((float(scores[k]), hit))
                    if scores[k] >= score_thr:
                        tp_at += hit
                        fp_at += not hit
    per_class = {c: _ap(scored[c], n_gt[c]) for c in range(1, num_classes)}
    total = sum(n_gt.values())
    return {"ap50": float(np.mean(list(per_class.values()))), "ap50_per_class": per_class,
            "recall": tp_at / max(total, 1), "precision": tp_at / max(tp_at + fp_at, 1), "gt": total}


class ExportWrapper(torch.nn.Module):
    """Only the network, for ONNX: normalised, padded image [1,3,H,W] + its unpadded size -> proposals,
    class logits, box regression. Resize/normalise/pad and the final box post-processing are done in
    NumPy by maxik_cv.tomato.leaf_detector (inside the graph the resize gets frozen to the dummy image's
    size, and the post-processing can't be exported at all); check_export() verifies the result."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, image, image_hw):
        from torchvision.models.detection.image_list import ImageList
        images = ImageList(image, [(image_hw[0], image_hw[1])])
        features = self.m.backbone(image)
        proposals, _ = self.m.rpn(images, features)
        roi = self.m.roi_heads
        box_features = roi.box_head(roi.box_roi_pool(features, proposals, images.image_sizes))
        class_logits, box_regression = roi.box_predictor(box_features)
        return proposals[0], class_logits, box_regression


def check_export(model, onnx_path: Path, items, n: int = 8) -> float:
    """Max box-coordinate difference (px) between torch and ONNX+NumPy on n held-out photos (inf if the
    number or classes of the boxes differ). Both get the same resized photo, so torch's internal resize
    is a no-op and only the network is compared."""
    sys.path.insert(0, str(ROOT))
    from maxik_cv.tomato.leaf_detector import LeafDetector, resize_for_detector
    det = LeafDetector(onnx_path, score_threshold=model.roi_heads.score_thresh,
                       max_detections=model.roi_heads.detections_per_img)
    worst = 0.0
    for item in items[:n]:
        bgr = resize_for_detector(cv2.imread(str(item[0])), det.min_size, det.max_size)
        t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            ref = model([t])[0]
        boxes, _, labels = det.detect_raw(bgr)
        ref_b, ref_l = ref["boxes"].numpy(), ref["labels"].numpy()
        if len(ref_b) != len(boxes):
            return float("inf")
        if len(ref_b):
            # same set of detections; compare each torch box with its nearest ONNX box of the same class
            for b, l in zip(ref_b, ref_l):
                same = labels == l
                if not same.any():
                    return float("inf")
                worst = max(worst, float(np.abs(boxes[same] - b).max(axis=1).min()))
    return worst


def export_onnx(model, onnx_path: Path):
    wrapper = ExportWrapper(model).eval()
    dummy = (torch.rand(1, 3, 512, 704), torch.tensor([512, 683]))
    torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, opset_version=17,
                      input_names=["image", "image_hw"], output_names=["proposals", "class_logits", "box_regression"],
                      dynamic_axes={"image": {2: "h", 3: "w"}, "proposals": {0: "n"}, "class_logits": {0: "n"},
                                    "box_regression": {0: "n"}})
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(PRESETS), default="leaf")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    cfg = PRESETS[args.task]
    epochs = args.epochs or cfg["epochs"]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    split = {s: load_split(s, cfg["root"], cfg["classes"]) for s in ("train", "val", "test")}
    tr = DataLoader(LeafBoxes(split["train"], True), args.bs, shuffle=True, num_workers=4,
                    collate_fn=collate, persistent_workers=True)
    va = DataLoader(LeafBoxes(split["val"], False), 4, shuffle=False, num_workers=2, collate_fn=collate)
    te = DataLoader(LeafBoxes(split["test"], False), 4, shuffle=False, num_workers=2, collate_fn=collate)
    n_cls = len(cfg["classes"]) + 1
    print(f"task={args.task} device={device} classes={cfg['classes']} photos train {len(tr.dataset)}, "
          f"val {len(va.dataset)}, test {len(te.dataset)}", flush=True)

    model = build_model(True, n_cls, cfg["min_size"], cfg["max_size"], cfg["detections"], cfg["small_anchors"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(epochs * len(tr), 1), pct_start=0.1)
    ckpt = out / f"{cfg['name']}_best.pt"
    best = -1.0
    for ep in range(1, 0 if args.export_only else epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for imgs, targets in tr:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        m = evaluate(model, va, device, n_cls)
        per = "  ".join(f"{cfg['classes'][c - 1]} {v:.3f}" for c, v in m["ap50_per_class"].items()) if n_cls > 2 else ""
        print(f"epoch {ep:2d} loss {tot / n:.3f}  val AP50 {m['ap50']:.3f} {per}  recall {m['recall']:.3f}  "
              f"precision {m['precision']:.3f}  {time.time() - t0:.0f}s", flush=True)
        if m["ap50"] > best:
            best = m["ap50"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    val_m, test_m = evaluate(model, va, device, n_cls), evaluate(model, te, device, n_cls)
    print(f"best val AP50 {val_m['ap50']:.3f} | TEST AP50 {test_m['ap50']:.3f} "
          f"{ {cfg['classes'][c - 1]: round(v, 3) for c, v in test_m['ap50_per_class'].items()} } "
          f"recall {test_m['recall']:.3f} precision {test_m['precision']:.3f} ({test_m['gt']} objects)", flush=True)

    model.cpu().eval()
    onnx_path = out / f"{cfg['name']}.onnx"
    export_onnx(model, onnx_path)
    onnx_path.with_suffix(".json").write_text(json.dumps(
        {"classes": cfg["classes"], "min_size": cfg["min_size"], "max_size": cfg["max_size"], "score_threshold": 0.5,
         "max_detections": cfg["detections"], "val": val_m, "test": test_m, "trained_on": cfg["trained_on"]},
        indent=2, default=str), encoding="utf8")
    model.roi_heads.score_thresh = 0.5
    diff = check_export(model, onnx_path, split["test"])
    print(f"exported {onnx_path}; torch vs ONNX+NumPy max box difference on test photos: {diff:.3f} px", flush=True)
    if diff > 1.0:
        sys.exit("ONNX export does not reproduce the torch model - do not use it")


if __name__ == "__main__":
    main()
