"""Export a pretrained (ImageNet) MobileNetV2 to ONNX for the plant/not-plant gate.

One-time, on the training PC. No plant dataset needed - this model already knows about
plants (rapeseed, corn, cabbage, mushrooms, ...) from ImageNet, we just read its output.

    python scripts/export_plant_gate.py
"""
from pathlib import Path

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

OUT = Path(__file__).resolve().parents[1] / "models_synthetic" / "plant_gate.onnx"


def main():
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(OUT),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print("categories:", len(weights.meta["categories"]))


if __name__ == "__main__":
    main()
