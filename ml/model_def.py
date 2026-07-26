"""
model_def.py
------------
DenseNet-121 (ImageNet pretrained) with a replaced classifier head, fine-tuned
for binary medical image classification (NORMAL vs PNEUMONIA on chest X-rays).

DenseNet-121 is chosen because:
  - It is a well-validated backbone in the medical imaging literature
    (e.g. CheXNet uses this exact architecture for chest X-ray diagnosis).
  - Its dense connectivity improves gradient flow, which also makes it a
    good fit for the Grad-CAM explainability we build on top of it
    (its final conv feature maps in `features.norm5` retain strong spatial
    localization).
"""

import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model = None
    if pretrained:
        try:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception as exc:
            print(f"[warn] Could not download pretrained ImageNet weights ({exc}). "
                  f"Falling back to random initialization. For production training, "
                  f"ensure network access to download.pytorch.org or pre-cache the weights.")
    if model is None:
        model = models.densenet121(weights=None)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_target_layer(model: nn.Module):
    """Returns the last convolutional feature layer used as the Grad-CAM target."""
    return model.features.norm5


def save_checkpoint(model: nn.Module, path: str, extra: dict | None = None):
    payload = {"state_dict": model.state_dict(), "class_names": CLASS_NAMES}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(num_classes=len(checkpoint.get("class_names", CLASS_NAMES)), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
