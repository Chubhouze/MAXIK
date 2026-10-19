"""Dataset labels <-> internal Condition enum <-> user-facing text.

Keeping this in one place means the training dataset's class names (e.g. PlantVillage-style
"Tomato___Early_blight") can change, or a model can be retrained with different folder names,
without touching the pipeline or the UI - only this mapping needs updating.

    Tomato___Early_blight  ->  Condition.TOMATO_EARLY_BLIGHT  ->  "Possible Early Blight"
"""
from .schema import Condition

# Expected training dataset class names (folder names / ImageFolder classes), in the order a
# future ONNX model's softmax output should use. Update this list to match the real dataset.
DATASET_CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
]

DATASET_LABEL_TO_CONDITION: dict[str, Condition] = {
    "Tomato___healthy": Condition.TOMATO_HEALTHY,
    "Tomato___Early_blight": Condition.TOMATO_EARLY_BLIGHT,
    "Tomato___Late_blight": Condition.TOMATO_LATE_BLIGHT,
    "Tomato___Leaf_Mold": Condition.TOMATO_LEAF_MOLD,
    "Tomato___Bacterial_spot": Condition.TOMATO_BACTERIAL_SPOT,
    "Tomato___Septoria_leaf_spot": Condition.TOMATO_SEPTORIA_LEAF_SPOT,
    "Tomato___Spider_mites Two-spotted_spider_mite": Condition.TOMATO_SPIDER_MITES,
    "Tomato___Target_Spot": Condition.TOMATO_TARGET_SPOT,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": Condition.TOMATO_YELLOW_LEAF_CURL_VIRUS,
    "Tomato___Tomato_mosaic_virus": Condition.TOMATO_MOSAIC_VIRUS,
}

# Conditions a trained model can output directly. TOMATO_UNCERTAIN is never a model class -
# it's assigned by the confidence gate (see confidence.py) when the model's own prediction
# isn't trustworthy enough to show.
MODEL_CONDITIONS = list(DATASET_LABEL_TO_CONDITION.values())

CONDITION_TO_STATUS_LABEL: dict[Condition, str] = {
    Condition.NOT_PLANT: "NO PLANT DETECTED",
    Condition.UNSUPPORTED_PLANT: "PLANT DETECTED",
    Condition.TOMATO_HEALTHY: "Healthy",
    Condition.TOMATO_EARLY_BLIGHT: "Possible Early Blight",
    Condition.TOMATO_LATE_BLIGHT: "Possible Late Blight",
    Condition.TOMATO_LEAF_MOLD: "Possible Leaf Mold",
    Condition.TOMATO_BACTERIAL_SPOT: "Possible Bacterial Spot",
    Condition.TOMATO_SEPTORIA_LEAF_SPOT: "Possible Septoria Leaf Spot",
    Condition.TOMATO_SPIDER_MITES: "Possible Spider Mite Damage",
    Condition.TOMATO_TARGET_SPOT: "Possible Target Spot",
    Condition.TOMATO_YELLOW_LEAF_CURL_VIRUS: "Possible Yellow Leaf Curl Virus",
    Condition.TOMATO_MOSAIC_VIRUS: "Possible Mosaic Virus",
    Condition.TOMATO_UNCERTAIN: "Condition Uncertain",
}
