"""
inference.py
------------
Single-image inference pipeline.

Run ANY image through:
1. Classical methods (ELA, metadata analysis)
2. CNN-based detection
3. Patch-based localization

Outputs:
- Real/Fake prediction with confidence
- ELA map
- Localization heatmap
- Metadata report
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classical_methods.ela      import compute_ela, ela_score
from classical_methods.metadata import analyze_metadata
from deep_learning.models       import get_model
from patch_localization.localizer import extract_patches, classify_patches, reconstruct_heatmap


# ─────────────────────────────────────────────
#  CNN Prediction
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_cnn(model: torch.nn.Module, image_path: str,
                img_size: int = 224,
                device: torch.device = None) -> tuple:
    """
    Run a single image through the CNN.

    Returns:
        (label, confidence, probabilities)
        label: 0=real, 1=fake
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    image  = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    model.eval()
    model.to(device)
    output = model(tensor)
    probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    label = int(probs.argmax())
    return label, float(probs[label]), probs


# ─────────────────────────────────────────────
#  Composite Visualization
# ─────────────────────────────────────────────

def create_report(image_path: str,
                  cnn_label: int,
                  cnn_conf: float,
                  ela_map: np.ndarray,
                  ela_sc: float,
                  heatmap: np.ndarray,
                  meta_report: dict,
                  save_path: str = None,
                  show: bool = True):
    """Create a composite visualization report."""
    original = np.array(Image.open(image_path).convert("RGB"))
    H, W     = original.shape[:2]

    import cv2
    hm_resized = cv2.resize(heatmap, (W, H))
    hm_uint8   = (hm_resized * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    overlay    = cv2.addWeighted(original, 0.5, hm_colored, 0.5, 0)

    # Build the figure
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0]);  ax0.imshow(original);   ax0.set_title("Original"); ax0.axis("off")
    ax1 = fig.add_subplot(gs[0, 1]);  ax1.imshow(ela_map);     ax1.set_title(f"ELA Map (score={ela_sc:.3f})"); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 2]);  ax2.imshow(hm_colored);  ax2.set_title("Patch Heatmap"); ax2.axis("off")
    ax3 = fig.add_subplot(gs[0, 3]);  ax3.imshow(overlay);     ax3.set_title("Heatmap Overlay"); ax3.axis("off")

    # Summary text box
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
    flags_str = "\n".join(f"  • {f}" for f in meta_report.get("flags", [])) or "  None"
    cnn_verdict  = "🔴 FORGED" if cnn_label == 1 else "🟢 AUTHENTIC"
    ela_verdict  = "⚠ Suspicious" if ela_sc > 0.05 else "✓ OK"
    meta_verdict = "⚠ Suspicious" if meta_report.get("suspicion_score", 0) > 0.3 else "✓ OK"

    summary = (
        f"FILE: {os.path.basename(image_path)}\n\n"
        f"CNN PREDICTION:   {cnn_verdict}  (confidence: {cnn_conf:.1%})\n"
        f"ELA ANALYSIS:     {ela_verdict}  (score: {ela_sc:.4f})\n"
        f"METADATA:         {meta_verdict}  (score: {meta_report.get('suspicion_score', 0):.2f})\n\n"
        f"METADATA FLAGS:\n{flags_str}"
    )
    ax4.text(0.01, 0.95, summary, transform=ax4.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"))

    title_color = "red" if cnn_label == 1 else "green"
    fig.suptitle(f"Image Forgery Analysis Report — {cnn_verdict}",
                 fontsize=15, fontweight="bold", color=title_color)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[Report] Saved to {save_path}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  Full Inference Pipeline
# ─────────────────────────────────────────────

def run_inference(image_path: str,
                  checkpoint_path: str = None,
                  model_name: str = "resnet18",
                  patch_size: int = 64,
                  stride: int = 32,
                  output_dir: str = "outputs",
                  show: bool = True,
                  device: torch.device = None) -> dict:
    """
    Full inference on a single image.

    Args:
        image_path:      Path to the image to analyze
        checkpoint_path: Path to trained model .pth file (optional)
        model_name:      Architecture to use
        patch_size:      Patch size for localization
        stride:          Stride for patch extraction
        output_dir:      Where to save results
        show:            Show visualizations
        device:          CPU or CUDA

    Returns:
        Dictionary with all analysis results
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  ANALYZING: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # ── 1. Classical: ELA ──────────────────────────────────
    print("\n[1/4] Running Error Level Analysis...")
    ela_map = compute_ela(image_path)
    ela_sc  = ela_score(ela_map)

    # ── 2. Classical: Metadata ─────────────────────────────
    print("[2/4] Analyzing metadata...")
    meta_report = analyze_metadata(image_path, verbose=False)

    # ── 3. CNN Detection ───────────────────────────────────
    print("[3/4] Running CNN detection...")
    model = get_model(model_name, pretrained=checkpoint_path is None)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print("  [Warning] No checkpoint loaded — using random weights!")
        print("  Train a model first with: python training/train.py")

    cnn_label, cnn_conf, probs = predict_cnn(model, image_path, device=device)

    # ── 4. Patch Localization ─────────────────────────────
    print("[4/4] Running patch-based localization...")
    image_arr = np.array(Image.open(image_path).convert("RGB"))
    patches, _ = extract_patches(image_arr, patch_size, stride)
    fake_probs  = classify_patches(model, patches, patch_size, device=device)
    heatmap     = reconstruct_heatmap(image_arr.shape[:2], patches, fake_probs)

    # ── Report ────────────────────────────────────────────
    base_name  = os.path.splitext(os.path.basename(image_path))[0]
    report_path = os.path.join(output_dir, f"report_{base_name}.png")

    create_report(image_path, cnn_label, cnn_conf,
                  ela_map, ela_sc, heatmap, meta_report,
                  save_path=report_path, show=show)

    results = {
        "file":          image_path,
        "cnn_prediction": "fake" if cnn_label == 1 else "real",
        "cnn_confidence": cnn_conf,
        "cnn_probs":      {"real": float(probs[0]), "fake": float(probs[1])},
        "ela_score":      ela_sc,
        "metadata":       meta_report,
        "heatmap":        heatmap,
        "report_path":    report_path,
    }

    print(f"\n{'='*60}")
    print(f"  VERDICT: {'🔴 FORGED' if cnn_label == 1 else '🟢 AUTHENTIC'}")
    print(f"  CNN Confidence: {cnn_conf:.1%}")
    print(f"  ELA Score:      {ela_sc:.4f}")
    print(f"  Meta Score:     {meta_report.get('suspicion_score', 0):.2f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Forgery Detection Inference")
    parser.add_argument("image",       help="Path to image file")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--model",     default="resnet18", choices=["custom", "resnet18", "mobilenetv2"])
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--no_show",   action="store_true")
    args = parser.parse_args()

    results = run_inference(
        args.image,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        output_dir=args.output_dir,
        show=not args.no_show
    )
