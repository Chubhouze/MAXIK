"""Fine-tune MobileNetV3-Large on the tomato disease split (data/tomato_split, built by
scripts/prepare_tomato_dataset.py) and export to ONNX for maxik_cv.tomato.OnnxTomatoHealthClassifier.

Same architecture/training conventions as scripts/train.py (the generic health model), so results
are comparable and the ONNX+sidecar-JSON format matches what OnnxTomatoHealthClassifier expects.

    python scripts/train_tomato.py --epochs 15
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class RandomMotionBlur:
    """A robot photographing while moving: blur along one random direction. Without this the model
    stays confidently wrong on motion-blurred frames (measured: 40-50% of moderately blurred test
    frames were answered with a wrong diagnosis at high confidence)."""

    def __init__(self, p: float = 0.25, max_len: int = 17):
        self.p, self.max_len = p, max_len

    def __call__(self, img):
        if random.random() > self.p:
            return img
        k = random.randrange(3, self.max_len + 1, 2)
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1.0)
        ker = cv2.warpAffine(ker, rot, (k, k))
        ker /= max(ker.sum(), 1e-6)
        return Image.fromarray(cv2.filter2D(np.asarray(img), -1, ker))


def loaders(data: Path, size: int, bs: int):
    # No hue jitter: leaf colour (yellow/brown/necrotic) is part of the diagnostic signal.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.1, hue=0.0),
        transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.3, 2.5))], p=0.3),
        RandomMotionBlur(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tr = datasets.ImageFolder(data / "train", train_tf)
    va = datasets.ImageFolder(data / "val", eval_tf)
    te = datasets.ImageFolder(data / "test", eval_tf)
    counts = np.bincount(tr.targets, minlength=len(tr.classes))
    w = (1.0 / np.maximum(counts, 1))[tr.targets]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), replacement=True)
    nw = 4
    return (DataLoader(tr, bs, sampler=sampler, num_workers=nw, pin_memory=True, persistent_workers=True),
            DataLoader(va, bs, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=True),
            DataLoader(te, bs, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=True),
            tr.classes, counts)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        targets.append(y)
    logits, targets = torch.cat(logits), torch.cat(targets)
    return (logits.argmax(1) == targets).float().mean().item(), logits, targets


def confusion(logits, targets, classes):
    preds = logits.argmax(1)
    n = len(classes)
    m = torch.zeros(n, n, dtype=torch.long)
    for p, t in zip(preds.tolist(), targets.tolist()):
        m[t, p] += 1
    return m


def fit_temperature(logits, targets) -> float:
    nll = nn.CrossEntropyLoss()
    grid = torch.linspace(0.3, 5.0, 95)
    losses = torch.stack([nll(logits / t, targets) for t in grid])
    return float(grid[losses.argmin()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\chub\Desktop\MAXIK\data\tomato_split")
    ap.add_argument("--out", default="models_tomato")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--export-only", action="store_true", help="skip training, evaluate + export the existing best checkpoint")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_dl, va_dl, te_dl, classes, counts = loaders(data, args.size, args.bs)
    print(f"device={device} classes={dict(zip(classes, counts.tolist()))}")

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr_dl))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, bad = -1.0, 0
    ckpt = out / "tomato_health_best.pt"
    for ep in range(1, 0 if args.export_only else args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        acc, _, _ = evaluate(model, va_dl, device)
        print(f"epoch {ep:3d}  loss {total / n:.4f}  val_acc {acc:.3f}  {time.time() - t0:.0f}s")
        if acc > best:
            best, bad = acc, 0
            torch.save({"model": model.state_dict(), "classes": classes, "size": args.size}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    val_acc, logits, targets = evaluate(model, va_dl, device)
    temp = fit_temperature(logits, targets)
    test_acc, test_logits, test_targets = evaluate(model, te_dl, device)
    cm = confusion(test_logits, test_targets, classes)
    print(f"best val_acc {val_acc:.3f}  temperature {temp:.2f}")
    print(f"held-out test_acc {test_acc:.3f}")
    print("classes:", classes)
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    model.cpu().eval()
    onnx_path = out / "tomato_health.onnx"
    torch.onnx.export(model, torch.randn(1, 3, args.size, args.size), onnx_path,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
    import onnx
    data_file = onnx_path.with_name(onnx_path.name + ".data")
    onnx.save(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    data_file.unlink(missing_ok=True)
    onnx_path.with_suffix(".json").write_text(json.dumps(
        {"classes": classes, "img_size": args.size, "temperature": temp,
         "val_acc": val_acc, "test_acc": test_acc,
         "confusion_matrix": cm.tolist(),
         "caveat": "PlantVillage may contain near-duplicate photos across splits; treat test_acc "
                    "as optimistic until validated on independently captured photos."},
        indent=2), encoding="utf8")
    print("exported", onnx_path)


if __name__ == "__main__":
    main()
