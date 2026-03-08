"""
utils/gradcam.py
-----------------
Grad-CAM (Gradient-weighted Class Activation Mapping)

THEORY:
Grad-CAM explains CNN decisions by asking:
"Which spatial regions in the image most influenced this prediction?"

It does this by:
1. Forward pass → get the CNN's prediction
2. Compute gradients of the target class score w.r.t. the last conv layer
3. Average the gradients spatially to get importance weights per channel
4. Weighted sum of feature maps → activation map
5. ReLU + resize → heatmap on original image

This is different from patch localization:
- Patch localization: runs many forward passes on patches
- Grad-CAM: single forward pass, uses gradients internally
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional
import os


class GradCAM:
    """
    Grad-CAM implementation.

    Usage:
        cam = GradCAM(model, target_layer="layer4")
        heatmap = cam(image_tensor, target_class=1)  # 1 = fake
    """

    def __init__(self, model: nn.Module, target_layer: str = "layer4"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward and backward hooks on the target layer.
        Hooks are functions called automatically during forward/backward pass.
        """
        # Find the target layer by name
        layer = self._find_layer(self.target_layer)
        if layer is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in model")

        # Forward hook: saves the feature maps
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Backward hook: saves the gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def _find_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Traverse model to find layer by partial name match."""
        for name, module in self.model.named_modules():
            if layer_name in name:
                return module
        return None

    def __call__(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Class index to explain (None = use predicted class)

        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero existing gradients
        self.model.zero_grad()

        # Backward pass for the target class only
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        # gradients: (1, C, H, W) — gradient of target class w.r.t. feature maps
        # activations: (1, C, H, W) — actual feature maps
        grads  = self.gradients[0]    # (C, H, W)
        acts   = self.activations[0]  # (C, H, W)

        # Global average pooling on gradients → importance weights per channel
        weights = grads.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activation maps
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ReLU: we only care about positive contributions
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def apply_gradcam(model: nn.Module,
                  image_path: str,
                  target_layer: str = "layer4",
                  target_class: int = 1,
                  img_size: int = 224,
                  save_path: str = None,
                  show: bool = True,
                  device: torch.device = None) -> np.ndarray:
    """
    Full Grad-CAM pipeline for a single image.

    Args:
        model:        Trained CNN model
        image_path:   Path to image
        target_layer: Name of the convolutional layer to hook
        target_class: Class to explain (1 = fake)
        img_size:     Input size for model
        save_path:    Save visualization
        show:         Display plot

    Returns:
        heatmap: (H, W) numpy array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess image
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    original = np.array(Image.open(image_path).convert("RGB"))
    transformed = transform(image=original)
    input_tensor = transformed["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    model = model.to(device)
    cam = GradCAM(model, target_layer)
    heatmap = cam(input_tensor, target_class)

    # Resize heatmap to original image size
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply color map
    hm_uint8   = (heatmap_resized * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    overlay    = cv2.addWeighted(original, 0.5, hm_colored, 0.5, 0)

    # Prediction info
    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()

    verdict = f"{'FAKE' if pred == 1 else 'REAL'} ({conf:.1%} confidence)"

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original);    axes[0].set_title("Original");         axes[0].axis("off")
    axes[1].imshow(hm_colored);  axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay);     axes[2].set_title("Overlay");          axes[2].axis("off")
    fig.suptitle(f"Grad-CAM: {verdict}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Grad-CAM] Saved to {save_path}")
    if show:
        plt.show()
    plt.close()

    return heatmap_resized
