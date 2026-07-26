"""
inference.py
------------
Loads the trained model once (module-level singleton) and exposes a single
`predict_and_explain()` function used by the API layer. Keeping model
loading out of the request path is a basic but important serving best
practice (avoids reloading weights on every HTTP call).
"""

import os
import sys
import uuid
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))
from model_def import load_checkpoint, get_target_layer, IMG_SIZE  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "models/pneumonia_densenet121.pth")
GRADCAM_DIR = os.environ.get("GRADCAM_DIR", "gradcam_outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(GRADCAM_DIR, exist_ok=True)

_PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
_class_names = None


def _get_model():
    global _model, _class_names
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at {MODEL_PATH}. Train it first with "
                f"`python ml/generate_synthetic_data.py && python ml/train.py`."
            )
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        _class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])
        _model = load_checkpoint(MODEL_PATH, device=DEVICE)
    return _model, _class_names


def predict_and_explain(image: Image.Image) -> dict:
    """
    Runs classification + Grad-CAM on a PIL image.
    Returns a dict with predicted_class, confidence, class_probabilities,
    and the path to a saved Grad-CAM overlay PNG.
    """
    model, class_names = _get_model()
    image = image.convert("RGB")
    original_resized = image.resize((IMG_SIZE, IMG_SIZE))
    input_tensor = _PREPROCESS(image).unsqueeze(0).to(DEVICE)

    cam_extractor = GradCAM(model, get_target_layer(model))
    try:
        cam, class_idx, probs = cam_extractor.generate(input_tensor)
    finally:
        cam_extractor.remove_hooks()

    overlay = overlay_heatmap(np.array(original_resized), cam)
    filename = f"gradcam_{uuid.uuid4().hex[:12]}.png"
    out_path = os.path.join(GRADCAM_DIR, filename)
    Image.fromarray(overlay).save(out_path)

    return {
        "predicted_class": class_names[class_idx],
        "confidence": float(probs[class_idx]),
        "class_probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
        "gradcam_path": out_path,
        "gradcam_filename": filename,
    }
