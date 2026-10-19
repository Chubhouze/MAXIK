"""Fine-tune a lightweight CNN (MobileNetV3 / EfficientNet-B0) on data/split and export to ONNX.

Runs on the GTX 1660 Ti (6 GB): batch 32 at 224px fits comfortably.
    python scripts/train.py --data data/split --arch mobilenet_v3_large --epochs 30
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(arch: str, n_classes: int) -> nn.Module:
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


def loaders(data: Path, size: int, bs: int):
    # NOTE: no hue jitter - colour (yellow/brown) IS the signal we classify.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.1, hue=0.0),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
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
    # Balance classes: diseased/dry examples are usually rarer than healthy ones.
    counts = np.bincount(tr.targets, minlength=len(tr.classes))
    w = (1.0 / np.maximum(counts, 1))[tr.targets]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), replacement=True)
    nw = 4
    return (DataLoader(tr, bs, sampler=sampler, num_workers=nw, pin_memory=True, persistent_workers=True),
            DataLoader(va, bs, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=True),
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


def fit_temperature(logits, targets) -> float:
    """Temperature scaling so that confidence ~ real accuracy (needed for a meaningful 'unknown' threshold).
    Grid search instead of gradient descent: stable on tiny validation sets."""
    nll = nn.CrossEntropyLoss()
    grid = torch.linspace(0.3, 5.0, 95)
    losses = torch.stack([nll(logits / t, targets) for t in grid])
    return float(grid[losses.argmin()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/split")
    ap.add_argument("--out", default="models")
    ap.add_argument("--arch", default="mobilenet_v3_large", choices=["mobilenet_v3_large", "efficientnet_b0"])
    ap.add_argument("--init", help="checkpoint to start from (e.g. pretrained on PlantVillage/PlantDoc)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--export-only", action="store_true", help="skip training, export existing best checkpoint")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # torch.onnx prints emoji; Windows console is cp1252

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_dl, va_dl, classes, counts = loaders(data, args.size, args.bs)
    print(f"device={device} classes={dict(zip(classes, counts.tolist()))}")

    model = build_model(args.arch, len(classes))
    if args.init:
        sd = torch.load(args.init, map_location="cpu")["model"]
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}  # new head for our classes
        print("init:", model.load_state_dict(sd, strict=False))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr_dl))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, bad = -1.0, 0
    ckpt = out / f"{args.arch}_best.pt"
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
            torch.save({"model": model.state_dict(), "classes": classes, "arch": args.arch, "size": args.size}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    acc, logits, targets = evaluate(model, va_dl, device)
    temp = fit_temperature(logits, targets)
    print(f"best val_acc {acc:.3f}, temperature {temp:.2f}")

    model.cpu().eval()
    onnx_path = out / "plant_health.onnx"
    torch.onnx.export(model, torch.randn(1, 3, args.size, args.size), onnx_path,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
    import onnx  # re-save as a single self-contained file (easier to copy to the Pi)
    data_file = onnx_path.with_name(onnx_path.name + ".data")
    onnx.save(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    data_file.unlink(missing_ok=True)
    onnx_path.with_suffix(".json").write_text(json.dumps(
        {"classes": classes, "img_size": args.size, "arch": args.arch, "temperature": temp, "val_acc": acc},
        indent=2), encoding="utf8")
    print("exported", onnx_path)


if __name__ == "__main__":
    main()
