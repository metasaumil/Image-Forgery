"""
patch_localization/localizer.py
---------------------------------
Forgery Localization via Patch Classification.

IDEA:
Instead of classifying the whole image, split it into patches (small tiles)
and classify each patch independently. Patches classified as "fake" form
the forgery heatmap.

PIPELINE:
1. Divide image into N×N overlapping patches
2. Run each patch through the CNN classifier
3. Record the "fake" probability for each patch position
4. Reconstruct a probability map (same size as image)
5. Overlay the heatmap on the original for visualization
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
import os


# ─────────────────────────────────────────────
#  Transform for patches
# ─────────────────────────────────────────────

def patch_transform(size: int = 64):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ─────────────────────────────────────────────
#  Patch Extraction
# ─────────────────────────────────────────────

def extract_patches(image: np.ndarray,
                    patch_size: int = 64,
                    stride: int = 32) -> Tuple[list, list]:
    """
    Extract overlapping patches from an image.

    Args:
        image:       RGB numpy array (H, W, 3)
        patch_size:  Size of each patch in pixels
        stride:      How many pixels to move between patches
                     stride < patch_size creates overlapping patches (better results)

    Returns:
        patches:    List of (patch array, (row, col, row_end, col_end))
        positions:  List of (center_row, center_col) for each patch
    """
    H, W = image.shape[:2]
    patches   = []
    positions = []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = image[r:r+patch_size, c:c+patch_size]
            patches.append((patch, (r, c, r+patch_size, c+patch_size)))
            positions.append((r + patch_size//2, c + patch_size//2))

    return patches, positions


# ─────────────────────────────────────────────
#  Batch Patch Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def classify_patches(model: nn.Module,
                     patches: list,
                     patch_size: int = 64,
                     batch_size: int = 64,
                     device: torch.device = None) -> np.ndarray:
    """
    Classify all patches and return fake probabilities.

    Args:
        model:      Trained CNN classifier
        patches:    List of (patch, bbox) from extract_patches
        patch_size: Resize target for each patch
        batch_size: How many patches to process at once
        device:     CPU or CUDA

    Returns:
        fake_probs: numpy array of shape (N,) with fake probabilities
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    transform = patch_transform(patch_size)

    fake_probs = []

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i+batch_size]
        tensors = []

        for patch_arr, _ in batch_patches:
            augmented = transform(image=patch_arr)
            tensors.append(augmented["image"])

        batch_tensor = torch.stack(tensors).to(device)
        outputs = model(batch_tensor)

        # Convert logits to probabilities
        probs = torch.softmax(outputs, dim=1)
        fake_prob = probs[:, 1].cpu().numpy()  # index 1 = fake class
        fake_probs.extend(fake_prob.tolist())

    return np.array(fake_probs)


# ─────────────────────────────────────────────
#  Heatmap Reconstruction
# ─────────────────────────────────────────────

def reconstruct_heatmap(image_shape: Tuple[int, int],
                        patches: list,
                        fake_probs: np.ndarray) -> np.ndarray:
    """
    Build a heatmap by averaging patch probabilities at each pixel.

    Args:
        image_shape: (H, W) of the original image
        patches:     List of (patch, (r, c, r_end, c_end))
        fake_probs:  Per-patch fake probabilities

    Returns:
        heatmap: numpy float array of shape (H, W), values in [0, 1]
    """
    H, W     = image_shape
    accum    = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)

    for (_, bbox), prob in zip(patches, fake_probs):
        r, c, r_end, c_end = bbox
        accum[r:r_end, c:c_end] += prob
        count[r:r_end, c:c_end] += 1

    # Avoid division by zero
    count   = np.maximum(count, 1)
    heatmap = accum / count
    return heatmap


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def visualize_localization(original: np.ndarray,
                           heatmap: np.ndarray,
                           threshold: float = 0.5,
                           save_path: str = None,
                           show: bool = True) -> np.ndarray:
    """
    Overlay the heatmap on the original image.

    Args:
        original:   RGB numpy array
        heatmap:    Float array (H, W) with values in [0, 1]
        threshold:  Pixels above this probability get highlighted
        save_path:  Save visualization here
        show:       Call plt.show()

    Returns:
        overlay_img: RGB visualization
    """
    H, W = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Convert to color heatmap (blue = low, red = high)
    hm_uint8   = (heatmap_resized * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)

    # Blend with original
    overlay = cv2.addWeighted(original, 0.5, hm_colored, 0.5, 0)

    # Binary mask for threshold
    mask      = (heatmap_resized > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = overlay.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 255, 0), 2)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original);      axes[0].set_title("Original");          axes[0].axis("off")
    axes[1].imshow(hm_colored);    axes[1].set_title("Fake Probability");  axes[1].axis("off")
    im = axes[2].imshow(heatmap_resized, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title("Probability Map"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(contour_img);   axes[3].set_title("Detected Regions"); axes[3].axis("off")

    plt.suptitle("Patch-based Forgery Localization", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Localize] Saved to {save_path}")

    if show:
        plt.show()

    plt.close()
    return overlay


# ─────────────────────────────────────────────
#  Full Pipeline
# ─────────────────────────────────────────────

def localize_forgery(model: nn.Module,
                     image_path: str,
                     patch_size: int = 64,
                     stride: int = 32,
                     threshold: float = 0.5,
                     save_path: str = None,
                     show: bool = True,
                     device: torch.device = None) -> Tuple[np.ndarray, float]:
    """
    Full forgery localization pipeline.

    Returns:
        (heatmap, mean_score)
        mean_score > threshold → image is likely forged
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    H, W  = image.shape[:2]
    print(f"[Localize] Image size: {W}×{H}")

    # Extract patches
    patches, positions = extract_patches(image, patch_size, stride)
    print(f"[Localize] Extracted {len(patches)} patches (size={patch_size}, stride={stride})")

    # Classify patches
    fake_probs = classify_patches(model, patches, patch_size, device=device)
    mean_score = float(fake_probs.mean())
    print(f"[Localize] Mean fake probability: {mean_score:.3f}")

    # Reconstruct heatmap
    heatmap = reconstruct_heatmap((H, W), patches, fake_probs)

    # Visualize
    visualize_localization(image, heatmap, threshold, save_path, show)

    return heatmap, mean_score


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from deep_learning.models import get_model

    if len(sys.argv) < 3:
        print("Usage: python localizer.py <image_path> <checkpoint_path>")
    else:
        model = get_model("resnet18")
        model.load_state_dict(torch.load(sys.argv[2], map_location="cpu"))
        heatmap, score = localize_forgery(model, sys.argv[1], save_path="localization.png")
        print(f"\nMean fake score: {score:.3f}")
