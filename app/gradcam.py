"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation used to
explain the DenseNet-121 classifier's predictions by highlighting the image
regions that most influenced the predicted class.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # NOTE: we deliberately register a *tensor*-level gradient hook (via
        # output.register_hook inside the forward hook) rather than a
        # module-level register_full_backward_hook. DenseNet applies an
        # in-place ReLU immediately after this layer's output, which is
        # incompatible with module-level full backward hooks (PyTorch raises
        # a "view is being modified inplace" RuntimeError). Tensor-level hooks
        # capture the same gradients without that conflict.
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._tensor_hook_handle = None

    def _save_activation(self, module, input, output):
        self.activations = output
        if self._tensor_hook_handle is not None:
            self._tensor_hook_handle.remove()
        self._tensor_hook_handle = output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        if self._tensor_hook_handle is not None:
            self._tensor_hook_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """
        input_tensor: (1, C, H, W) preprocessed tensor, requires no grad set by caller.
        Returns: (heatmap [H,W] normalized 0-1, predicted_class_idx, probabilities)
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)          # (1, num_classes)
        probs = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients[0]                 # (C, h, w)
        activations = self.activations[0].detach()    # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # global average pool -> (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx, probs.detach().cpu().numpy()[0]


def overlay_heatmap(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    original_rgb: HxWx3 uint8 RGB image
    cam: normalized 0-1 heatmap at feature-map resolution
    Returns: HxWx3 uint8 RGB overlay image
    """
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap.astype(np.float32) * alpha + original_rgb.astype(np.float32) * (1 - alpha))
    return np.clip(overlay, 0, 255).astype(np.uint8)
