"""Train the crop (species) classifier that replaces the "every plant is a tomato" placeholder:
15 crops from lab leaves, real-world leaf crops and farm photos.

  lab         PlantVillage: tomato (10 folders) + healthy apple, corn, grape, potato, pepper, strawberry
  field_leaf  PlantDoc leaf boxes: 13 crops
  field_photo CCMT farm photos: cashew, cassava, maize (corn), tomato
Domains sampled equally, crops balanced within each domain (train_tomato_field.balanced_weights).

    python scripts/train_species.py --epochs 10      -> models_species/species.onnx (+ .json)
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_split import dedup_split  # noqa: E402
from train_tomato import MEAN, STD, RandomMotionBlur, fit_temperature  # noqa: E402
from train_tomato_field import DOMAINS, Items, balanced_weights, per_domain_acc, predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
CROPS = ["Apple", "Blueberry", "Cashew", "Cassava", "Cherry", "Corn", "Grape", "Peach", "Pepper", "Potato",
         "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"]
PV_OTHER = {"Apple___healthy": "Apple", "Corn_(maize)___healthy": "Corn", "Grape___healthy": "Grape",
            "Potato___healthy": "Potato", "Pepper,_bell___healthy": "Pepper", "Strawberry___healthy": "Strawberry"}
PLANTDOC_PREFIX = {"Apple": "Apple", "Bell_pepper": "Pepper", "Blueberry": "Blueberry", "Cherry": "Cherry",
                   "Corn": "Corn", "Peach": "Peach", "Potato": "Potato", "Raspberry": "Raspberry",
                   "Soyabean": "Soybean", "Squash": "Squash", "Strawberry": "Strawberry", "Tomato": "Tomato",
                   "grape": "Grape"}
CCMT_CROP = {"Cashew": "Cashew", "Cassava": "Cassava", "Maize": "Corn", "Tomato": "Tomato"}


def collect(split: str) -> list[tuple[Path, int, str]]:
    idx = {c: i for i, c in enumerate(CROPS)}
    items = [(p, idx["Tomato"], "lab") for p in (DATA / "tomato_split" / split).glob("*/*")]
    pos = {"train": 0, "val": 1, "test": 2}[split]
    for folder, crop in PV_OTHER.items():  # same seed as the plant-gate split, so the splits agree
        files = sorted((DATA / "PlantVillage-Dataset" / "raw" / "color" / folder).glob("*"))
        items += [(p, idx[crop], "lab") for p in dedup_split(files, seed=0)[pos]]
    for d in (DATA / "plantdoc_leaves" / split).iterdir():
        crop = PLANTDOC_PREFIX[d.name.split("_")[0] if not d.name.startswith("Bell_pepper") else "Bell_pepper"]
        items += [(p, idx[crop], "field_leaf") for p in d.glob("*")]
    for d in (DATA / "ccmt_split" / split).iterdir():
        items += [(p, idx[CCMT_CROP[d.name.split("___")[0]]], "field_photo") for p in d.glob("*")]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(ROOT / "models_species"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    random.seed(0)
    torch.manual_seed(0)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.0),
        transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.3, 2.5))], p=0.3),
        RandomMotionBlur(), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                                  transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr_items, va_items, te_items = collect("train"), collect("val"), collect("test")
    for name, its in (("train", tr_items), ("val", va_items), ("test", te_items)):
        print(name, len(its), dict(collections.Counter(d for _, _, d in its)),
              dict(sorted(collections.Counter(CROPS[y] for _, y, _ in its).items())), flush=True)

    sampler = WeightedRandomSampler(balanced_weights(tr_items), len(tr_items), replacement=True)
    tr = DataLoader(Items(tr_items, train_tf), args.bs, sampler=sampler, num_workers=4, pin_memory=True,
                    persistent_workers=True)
    va = DataLoader(Items(va_items, eval_tf), 64, num_workers=3, pin_memory=True, persistent_workers=True)
    te = DataLoader(Items(te_items, eval_tf), 64, num_workers=3, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CROPS))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt, best = out / "species_best.pt", -1.0

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
        mean = np.mean([v["acc"] for v in acc.values()])
        print(f"epoch {ep:2d} loss {tot / n:.3f}  val " + "  ".join(f"{d} {v['acc']:.3f}" for d, v in acc.items())
              + f"  {time.time() - t0:.0f}s", flush=True)
        if mean > best:
            best = mean
            torch.save({"model": model.state_dict(), "classes": CROPS, "size": 224}, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    vl, vy, _ = predict(model, va, device)
    temp = fit_temperature(vl, vy)
    tl, ty, ti = predict(model, te, device)
    test = per_domain_acc(tl, ty, ti, te_items)
    tom = CROPS.index("Tomato")
    field = torch.tensor([te_items[i][2] != "lab" for i in ti.tolist()])
    is_tom, said_tom = ty[field] == tom, tl[field].argmax(1) == tom
    binary = float((is_tom == said_tom).float().mean())
    print("TEST:", {d: f"{v['acc']:.3f} (n={v['n']})" for d, v in test.items()},
          f"| field photos tomato-vs-other {binary:.3f} | temperature {temp:.2f}", flush=True)

    model.cpu().eval()
    onnx_path = out / "species.onnx"
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), onnx_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
    import onnx
    onnx.save(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    onnx_path.with_name(onnx_path.name + ".data").unlink(missing_ok=True)
    onnx_path.with_suffix(".json").write_text(json.dumps(
        {"classes": CROPS, "img_size": 224, "temperature": temp, "test": test, "field_tomato_vs_other": binary,
         "trained_on": "PlantVillage + PlantDoc (CC BY 4.0) + CCMT (CC BY 4.0)"}, indent=2), encoding="utf8")
    print("exported", onnx_path, flush=True)


if __name__ == "__main__":
    main()
