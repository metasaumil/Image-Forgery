"""
deep_learning/models.py
-----------------------
CNN architectures for image forgery detection.

Includes:
1. CustomCNN  — built from scratch, great for learning
2. ResNet18   — transfer learning with ResNet18
3. MobileNetV2 — lightweight transfer learning

Why CNNs work for forgery detection:
- Convolutional layers learn local patterns (edges, textures, noise patterns)
- Forged regions often have different noise characteristics than authentic ones
- Deep layers capture semantic inconsistencies
- Transfer learning from ImageNet gives us strong low-level feature detectors for free
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


# ─────────────────────────────────────────────
#  1. Custom CNN (from scratch)
# ─────────────────────────────────────────────

class CustomCNN(nn.Module):
    """
    A simple CNN built from scratch.
    Good for learning how CNNs work before using pretrained models.

    Architecture:
        Conv → BN → ReLU → MaxPool  (x3)
        Flatten → FC → Dropout → FC
    """

    def __init__(self, num_classes: int = 2, img_size: int = 224):
        super().__init__()

        # Each block: Conv2d + BatchNorm + ReLU + MaxPool
        # Output size after 3 blocks of MaxPool(2x2): img_size / 8
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 → 112

            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 → 56

            # Block 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 → 28

            # Block 4: 128 → 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # Always output 7x7
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
#  2. ResNet18 (transfer learning)
# ─────────────────────────────────────────────

class ResNet18Forgery(nn.Module):
    """
    ResNet18 pretrained on ImageNet, fine-tuned for forgery detection.

    Strategy: Replace the last fully-connected layer with our own.
    Optionally freeze early layers to train faster.

    Why ResNet18?
    - Residual connections help avoid vanishing gradients
    - Pretrained weights give us excellent feature extractors for free
    - Small enough to train on a laptop GPU
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Optionally freeze all backbone layers
        # (useful when you have very little data)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the last layer
        # Original: Linear(512, 1000) for ImageNet
        # Ours:     Linear(512, num_classes)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze(self, from_layer: str = "layer3"):
        """Gradually unfreeze layers during training (gradual unfreezing)."""
        unfreeze = False
        for name, param in self.backbone.named_parameters():
            if from_layer in name:
                unfreeze = True
            if unfreeze:
                param.requires_grad = True


# ─────────────────────────────────────────────
#  3. MobileNetV2 (lightweight)
# ─────────────────────────────────────────────

class MobileNetV2Forgery(nn.Module):
    """
    MobileNetV2 — optimized for mobile/edge deployment.

    ~3x fewer parameters than ResNet18.
    Great if you need a fast, deployable model.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─────────────────────────────────────────────
#  Model Factory
# ─────────────────────────────────────────────

def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory function: get model by name.

    Args:
        model_name:  "custom", "resnet18", or "mobilenetv2"
        num_classes: Number of output classes (2 for binary: real/fake)
        pretrained:  Whether to use ImageNet pretrained weights

    Returns:
        nn.Module
    """
    models_dict = {
        "custom":      CustomCNN(num_classes),
        "resnet18":    ResNet18Forgery(num_classes, pretrained),
        "mobilenetv2": MobileNetV2Forgery(num_classes, pretrained),
    }

    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}")

    model = models_dict[model_name]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name}: {n_params:,} trainable parameters")
    return model


# ─────────────────────────────────────────────
#  Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing all models with a dummy batch...\n")
    dummy_input = torch.randn(4, 3, 224, 224)  # batch=4, RGB, 224x224

    for name in ["custom", "resnet18", "mobilenetv2"]:
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  {name}: output shape = {output.shape}")  # Should be (4, 2)
    print("\nAll models OK!")
