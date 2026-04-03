"""
classical_methods/ela.py
------------------------
Error Level Analysis (ELA)

THEORY:
When a JPEG image is saved, it's compressed. If you save it again at a known
quality, authentic regions will have uniform error levels, but forged/edited
regions (which may have been saved at a different compression history) will
show higher or inconsistent error levels.

HOW IT WORKS:
1. Take the input image
2. Re-save it as JPEG at a known quality (e.g., 90%)
3. Compute the difference: original - re-saved
4. Amplify the difference for visibility
5. Bright regions = high error = potentially forged

LIMITATIONS:
- Works best on JPEG images (not PNG)
- Lossless edits may not show up
- Multiple re-saves can mask forgeries
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import io
import os


def compute_ela(image_path: str, quality: int = 90, amplify: int = 20) -> np.ndarray:
    """
    Computes the Error Level Analysis map of an image.

    Args:
        image_path: Path to the image file
        quality:    JPEG re-save quality (lower = more artifacts visible)
        amplify:    Multiplier to make differences visible

    Returns:
        ela_image: numpy array (H, W, 3) — the ELA map
    """
    # Load original image
    original = Image.open(image_path).convert("RGB")

    # Re-save at known quality using an in-memory buffer
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    # Convert to numpy for math
    orig_arr    = np.array(original, dtype=np.float32)
    resaved_arr = np.array(resaved, dtype=np.float32)

    # Compute absolute difference and amplify
    ela = np.abs(orig_arr - resaved_arr) * amplify
    ela = np.clip(ela, 0, 255).astype(np.uint8)

    return ela


def ela_score(ela_map: np.ndarray) -> float:
    """
    Converts an ELA map to a single suspicion score.
    Higher score = more likely to be forged.
    
    Simple heuristic: high variance in ELA suggests tampering.
    """
    # Normalize ELA to [0, 1]
    normalized = ela_map.astype(np.float32) / 255.0
    # Score = weighted combination of mean and std
    score = 0.5 * normalized.mean() + 0.5 * normalized.std()
    return float(score)


def analyze_image(image_path: str, quality: int = 90, amplify: int = 20,
                  save_path: str = None, show: bool = True):
    """
    Full ELA analysis pipeline: compute, visualize, and score.

    Args:
        image_path: Path to input image
        quality:    JPEG re-save quality
        amplify:    ELA amplification factor
        save_path:  If given, saves the visualization here
        show:       Whether to call plt.show()

    Returns:
        (ela_map, score)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = np.array(Image.open(image_path).convert("RGB"))
    ela_map  = compute_ela(image_path, quality, amplify)
    score    = ela_score(ela_map)

    # ── Visualization ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(ela_map)
    axes[1].set_title(f"ELA Map (quality={quality}, amp={amplify})", fontsize=13)
    axes[1].axis("off")

    # Heatmap overlay
    ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY)
    heatmap  = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay  = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay", fontsize=13)
    axes[2].axis("off")

    verdict = "⚠ POSSIBLY FORGED" if score > 0.05 else "✓ LIKELY AUTHENTIC"
    fig.suptitle(f"ELA Analysis — Score: {score:.4f} — {verdict}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ELA] Saved to {save_path}")

    if show:
        plt.show()

    plt.close()
    return ela_map, score


# ─────────────────────────────────────────────
#  Batch Analysis
# ─────────────────────────────────────────────

def batch_ela(image_dir: str, output_dir: str = None, quality: int = 90):
    """
    Run ELA on all images in a directory and return scores.
    """
    results = {}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() not in valid_ext:
            continue
        path = os.path.join(image_dir, fname)
        try:
            save_path = os.path.join(output_dir, f"ela_{fname}") if output_dir else None
            _, score  = analyze_image(path, quality=quality, save_path=save_path, show=False)
            results[fname] = score
            print(f"  {fname}: score={score:.4f}")
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ela.py <image_path>")
        print("Example: python ela.py data/raw/test.jpg")
    else:
        ela_map, score = analyze_image(sys.argv[1], save_path="ela_result.png")
        print(f"\nELA Score: {score:.4f}")
        print("High score (>0.05) = suspicious. Low score = likely authentic.")
