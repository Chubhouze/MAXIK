"""Fine-tune MobileNetV3-Small as a binary plant / not_plant gate and export to ONNX.

Trained on data/plant_gate_split (Oxford Flowers102 = plant, Imagenette = not_plant), built by
scripts/prepare_plant_gate_dataset.py. This is a coarse "is there a plant in frame at all" filter
that runs before the 4-class health pipeline, not a health/species classifier.

    python scripts/train_plant_gate.py --epochs 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def loaders(data: Path, size: int, bs: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # Whole frame, not a centre crop: "is there a plant anywhere in the frame?" needs the edges. Must match
    # maxik_cv.preprocess.whole_frame_input, which the app uses.
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tr = datasets.ImageFolder(data / "train", train_tf)
    va = datasets.ImageFolder(data / "val", eval_tf)
    te = datasets.ImageFolder(data / "test", eval_tf)
    assert tr.classes == ["not_plant", "plant"], tr.classes  # ImageFolder sorts alphabetically

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
    preds = logits.argmax(1)
    acc = (preds == targets).float().mean().item()
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return acc, precision, recall, logits, targets


def fit_temperature(logits, targets) -> float:
    nll = nn.CrossEntropyLoss()
    grid = torch.linspace(0.3, 5.0, 95)
    losses = torch.stack([nll(logits / t, targets) for t in grid])
    return float(grid[losses.argmin()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\chub\Desktop\MAXIK\data\plant_gate_split")
    ap.add_argument("--out", default="models_synthetic")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_dl, va_dl, te_dl, classes, counts = loaders(data, args.size, args.bs)
    print(f"device={device} classes={dict(zip(classes, counts.tolist()))}")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr_dl))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, bad = -1.0, 0
    ckpt = out / "plant_gate_trained_best.pt"
    for ep in range(1, args.epochs + 1):
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
        acc, prec, rec, _, _ = evaluate(model, va_dl, device)
        print(f"epoch {ep:2d}  loss {total / n:.4f}  val_acc {acc:.3f}  P {prec:.3f}  R {rec:.3f}  {time.time() - t0:.0f}s")
        if acc > best:
            best, bad = acc, 0
            torch.save({"model": model.state_dict(), "classes": classes, "size": args.size}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    val_acc, val_p, val_r, logits, targets = evaluate(model, va_dl, device)
    temp = fit_temperature(logits, targets)
    test_acc, test_p, test_r, _, _ = evaluate(model, te_dl, device)
    print(f"best val_acc {val_acc:.3f} P {val_p:.3f} R {val_r:.3f}  temperature {temp:.2f}")
    print(f"held-out test_acc {test_acc:.3f} P {test_p:.3f} R {test_r:.3f}")

    model.cpu().eval()
    onnx_path = out / "plant_gate_trained.onnx"
    torch.onnx.export(model, torch.randn(1, 3, args.size, args.size), onnx_path,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
    import onnx
    data_file = onnx_path.with_name(onnx_path.name + ".data")
    onnx.save(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    data_file.unlink(missing_ok=True)
    onnx_path.with_suffix(".json").write_text(json.dumps(
        {"classes": classes, "img_size": args.size, "temperature": temp,
         "val_acc": val_acc, "val_precision": val_p, "val_recall": val_r,
         "test_acc": test_acc, "test_precision": test_p, "test_recall": test_r},
        indent=2), encoding="utf8")
    print("exported", onnx_path)


if __name__ == "__main__":
    main()
