"""Model input preparation that reproduces the training scripts' evaluation transform EXACTLY:
torchvision Resize(int(size*1.14)) + CenterCrop(size) on a PIL image, then ImageNet normalisation.

It has to be the same resize, not just a similar one: with OpenCV's INTER_AREA instead of PIL's
bilinear, the field-trained tomato model scored 51.6% instead of 57.8% on the same real-world leaf
crops. (On PlantVillage's 256x256 photos the resize is nearly a no-op, which is why it went unnoticed.)
"""
import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def whole_frame_input(bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """BGR image -> float32 [1, 3, size, size] showing the WHOLE frame (aspect ratio not kept), identical
    to torchvision Resize((size, size)) on a PIL image. Used by the plant gate: "is there a plant in the
    frame" needs the frame's edges, which a centre crop would cut off."""
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((size, size), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def center_crop_input(bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """BGR image -> float32 [1, 3, size, size], identical to the torchvision eval transform."""
    resize_to = int(size * 1.14)
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    w, h = img.size
    if w <= h:   # torchvision: short side -> resize_to, long side truncated, not rounded
        ow, oh = resize_to, int(resize_to * h / w)
    else:
        oh, ow = resize_to, int(resize_to * w / h)
    img = img.resize((ow, oh), Image.BILINEAR)
    top, left = int(round((oh - size) / 2.0)), int(round((ow - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)
