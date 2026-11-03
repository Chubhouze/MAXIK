"""Retrain the 10-class tomato disease model on lab AND real-world photos, so it works on photos of
actual plants, not only on PlantVillage's single leaves on a plain background.

Three training domains, sampled equally (each class balanced within its domain):
  lab         PlantVillage leaves             data/tomato_split          10 classes
  field_leaf  PlantDoc leaf boxes, cropped    data/plantdoc_leaves       8 classes (+2 mite photos)
  field_photo CCMT farm photos (Ghana)        data/ccmt_split            3 mappable classes:
              healthy, septoria leaf spot, leaf curl (-> yellow leaf curl virus, the leaf-curl disease
              of tomato; an assumption). CCMT's "leaf blight" (early or late? unknown) and "verticillium
              wilt" (not one of our classes) are left out.
Starts from the current lab model (models_tomato/tomato_health_best.pt). The FINAL (fully annealed)
epoch is exported: picking the "best" epoch on the small field validation sets mostly picks noise -
field-leaf val swings by +-3 points between epochs, and the first run picked epoch 2 (val 72.9%) whose
real-world test accuracy was 57.8%. Reports each domain's held-out test accuracy separately - the field
numbers are the ones that matter for a robot.

    python scripts/train_tomato_field.py --epochs 12
"""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tomato import MEAN, STD, RandomMotionBlur, fit_temperature  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
CLASSES = sorted([
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
])
PLANTDOC_DIRS = {
    "Tomato_leaf": "Tomato___healthy",
    "Tomato_Early_blight_leaf": "Tomato___Early_blight",
    "Tomato_leaf_late_blight": "Tomato___Late_blight",
    "Tomato_mold_leaf": "Tomato___Leaf_Mold",
    "Tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_two_spotted_spider_mites_leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato_leaf_yellow_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_leaf_mosaic_virus": "Tomato___Tomato_mosaic_virus",
}
CCMT_DIRS = {
    "Tomato___healthy": "Tomato___healthy",
    "Tomato___septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato___leaf curl": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}
DOMAINS = ("lab", "field_leaf", "field_photo")


def collect(split: str) -> list[tuple[Path, int, str]]:
    idx = {c: i for i, c in enumerate(CLASSES)}
    items = []
    for c in CLASSES:
        items += [(p, idx[c], "lab") for p in (DATA / "tomato_split" / split / c).glob("*")]
    for d, c in PLANTDOC_DIRS.items():
        items += [(p, idx[c], "field_leaf") for p in (DATA / "plantdoc_leaves" / split / d).glob("*")]
    for d, c in CCMT_DIRS.items():
        items += [(p, idx[c], "field_photo") for p in (DATA / "ccmt_split" / split / d).glob("*")]
    return items


class Items(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label, i


def balanced_weights(items) -> torch.Tensor:
    """Each domain gets 1/3 of the samples; classes are balanced inside a domain. Counts are floored at
    20 so the 2 spider-mite field crops aren't repeated hundreds of times per epoch."""
    count = collections.Counter((d, y) for _, y, d in items)
    classes_in = collections.Counter(d for d, _ in count)
    return torch.tensor([1.0 / len(DOMAINS) / classes_in[d] / max(count[(d, y)], 20) for _, y, d in items],
                        dtype=torch.double)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels, idx = [], [], []
    for x, y, i in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
        idx.append(i)
    return torch.cat(logits), torch.cat(labels), torch.cat(idx)


def per_domain_acc(logits, labels, idx, items) -> dict:
    pred = logits.argmax(1)
    out = {}
    for d in DOMAINS:
        m = torch.tensor([items[i][2] == d for i in idx.tolist()])
        if m.any():
            out[d] = {"acc": float((pred[m] == labels[m]).float().mean()), "n": int(m.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default=str(ROOT / "models_tomato" / "tomato_health_best.pt"))
    ap.add_argument("--out", default=str(ROOT / "models_tomato_field"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    random.seed(0)
    torch.manual_seed(0)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.0),
        transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.3, 2.5))], p=0.3),
        RandomMotionBlur(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                                  transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr_items, va_items, te_items = collect("train"), collect("val"), collect("test")
    for name, its in (("train", tr_items), ("val", va_items), ("test", te_items)):
        print(name, dict(collections.Counter(d for _, _, d in its)), flush=True)

    sampler = WeightedRandomSampler(balanced_weights(tr_items), len(tr_items), replacement=True)
    tr = DataLoader(Items(tr_items, train_tf), args.bs, sampler=sampler, num_workers=4, pin_memory=True,
                    persistent_workers=True)
    va = DataLoader(Items(va_items, eval_tf), 64, num_workers=3, pin_memory=True, persistent_workers=True)
    te = DataLoader(Items(te_items, eval_tf), 64, num_workers=3, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    init = torch.load(args.init, map_location="cpu")
    assert init["classes"] == CLASSES, "init checkpoint has a different class order"
    model.load_state_dict(init["model"])
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt, best = out / "tomato_health_best.pt", -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, _ in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        acc = per_domain_acc(*predict(model, va, device), va_items)
        field = np.mean([acc[d]["acc"] for d in ("field_leaf", "field_photo")])
        print(f"epoch {ep:2d} loss {tot / n:.3f}  val lab {acc['lab']['acc']:.3f}  field_leaf "
              f"{acc['field_leaf']['acc']:.3f}  field_photo {acc['field_photo']['acc']:.3f}  {time.time() - t0:.0f}s",
              flush=True)
        best = max(best, field)
        torch.save({"model": model.state_dict(), "classes": CLASSES, "size": 224}, ckpt)  # final epoch wins

    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    vl, vy, vi = predict(model, va, device)
    field_mask = torch.tensor([va_items[i][2] != "lab" for i in vi.tolist()])
    temp = fit_temperature(vl[field_mask], vy[field_mask])  # calibrate on field photos, where it is used
    test = per_domain_acc(*predict(model, te, device), te_items)
    print("TEST:", {d: f"{v['acc']:.3f} (n={v['n']})" for d, v in test.items()}, f"temperature {temp:.2f}", flush=True)

    model.cpu().eval()
    onnx_path = out / "tomato_health.onnx"
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), onnx_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
    import onnx
    onnx.save(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    onnx_path.with_name(onnx_path.name + ".data").unlink(missing_ok=True)
    onnx_path.with_suffix(".json").write_text(json.dumps(
        {"classes": CLASSES, "img_size": 224, "temperature": temp, "test": test,
         "test_acc": test["lab"]["acc"], "field_test_acc": {d: test[d]["acc"] for d in ("field_leaf", "field_photo")},
         "trained_on": "PlantVillage + PlantDoc (CC BY 4.0) + CCMT (CC BY 4.0)"}, indent=2), encoding="utf8")
    print("exported", onnx_path, flush=True)


if __name__ == "__main__":
    main()
