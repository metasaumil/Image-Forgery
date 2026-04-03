"""
deployment/streamlit_app/app.py
--------------------------------
Streamlit Web App for Image Forgery Detection.

Run: streamlit run deployment/streamlit_app/app.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit

from PIL import Image
import cv2
import io
import os
import sys
import tempfile

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from classical_methods.ela      import compute_ela, ela_score
from classical_methods.metadata import analyze_metadata
from deep_learning.models       import get_model
from patch_localization.localizer import (
    extract_patches, classify_patches, reconstruct_heatmap
)


# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Image Forgery Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .verdict-fake { color: #ff4444; font-size: 24px; font-weight: bold; }
    .verdict-real { color: #44bb44; font-size: 24px; font-weight: bold; }
    .metric-box { background: #f0f2f6; padding: 15px; border-radius: 10px; }
    .stProgress > div > div { background-color: #ff4444; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Model Loading (cached so it only loads once)
# ─────────────────────────────────────────────

@st.cache_resource
def load_model(model_name: str, checkpoint_path: str = None):
    """Load model once and cache it."""
    device = torch.device("cpu")  # Always CPU for web app
    model  = get_model(model_name, pretrained=True)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        st.sidebar.success(f"✓ Checkpoint loaded: {os.path.basename(checkpoint_path)}")
    else:
        st.sidebar.warning("⚠ Using pretrained (untrained for forgery). Train for best results.")

    model.eval()
    return model, device


# ─────────────────────────────────────────────
#  Inference Helpers
# ─────────────────────────────────────────────

import albumentations as A
from albumentations.pytorch import ToTensorV2

@torch.no_grad()
def predict(model, image_np, device):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
    output = model(tensor)
    probs  = torch.softmax(output, dim=1)[0].numpy()
    return int(probs.argmax()), float(probs.max()), probs


def get_heatmap(model, image_np, device, patch_size=64, stride=32):
    patches, _ = extract_patches(image_np, patch_size, stride)
    if not patches:
        return np.zeros(image_np.shape[:2])
    fake_probs  = classify_patches(model, patches, patch_size, device=device)
    return reconstruct_heatmap(image_np.shape[:2], patches, fake_probs)


def overlay_heatmap(image_np, heatmap):
    H, W       = image_np.shape[:2]
    hm_resized = cv2.resize(heatmap, (W, H))
    hm_uint8   = (hm_resized * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_np, 0.55, hm_colored, 0.45, 0)


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/300x80?text=Forgery+Detector", use_column_width=True)
    st.title("⚙️ Settings")

    model_choice = st.selectbox(
        "CNN Model",
        ["resnet18", "mobilenetv2", "custom"],
        index=0,
        help="ResNet18 is recommended. MobileNetV2 is faster. Custom is from scratch."
    )

    checkpoint_path = st.text_input(
        "Checkpoint path (optional)",
        placeholder="models/best_resnet18.pth",
        help="Leave empty to use pretrained ImageNet weights (untrained for forgery)"
    )

    ela_quality  = st.slider("ELA Quality", 50, 99, 90,
                              help="Lower quality = more compression artifacts visible")
    ela_amplify  = st.slider("ELA Amplify", 5, 50, 20,
                              help="Multiplier to make ELA differences visible")
    patch_size   = st.slider("Patch Size", 32, 128, 64, step=16,
                              help="Smaller = more detailed heatmap but slower")
    patch_stride = st.slider("Patch Stride", 16, 64, 32, step=8)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Detects image forgeries using ELA, EXIF analysis, and deep learning.")
    st.markdown("[GitHub](https://github.com/your-repo) | [Docs](https://docs.your-site.com)")


# ─────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────

st.title("🔍 Image Forgery Detection System")
st.markdown("Upload an image to check if it's authentic or tampered.")

# Load model
model, device = load_model(model_choice, checkpoint_path if checkpoint_path else None)

# Upload
uploaded = st.file_uploader(
    "Upload an image (JPG, PNG, BMP)",
    type=["jpg", "jpeg", "png", "bmp", "tif"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.info("👆 Upload an image to begin analysis.")
    st.stop()

# Load image
image_pil = Image.open(uploaded).convert("RGB")
image_np  = np.array(image_pil)

# Save to temp file for classical methods
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    image_pil.save(tmp.name, format="JPEG", quality=95)
    tmp_path = tmp.name

# ── Run Analysis ──────────────────────────────────────────

col_spinner, _ = st.columns([1, 3])
with col_spinner:
    with st.spinner("Analyzing image..."):

        # CNN
        cnn_label, cnn_conf, probs = predict(model, image_np, device)

        # ELA
        ela_map = compute_ela(tmp_path, quality=ela_quality, amplify=ela_amplify)
        ela_sc  = ela_score(ela_map)

        # Metadata
        meta = analyze_metadata(tmp_path, verbose=False)

        # Heatmap
        heatmap = get_heatmap(model, image_np, device, patch_size, patch_stride)
        overlay = overlay_heatmap(image_np, heatmap)

os.unlink(tmp_path)  # Clean up temp file

# ── Display Verdict ──────────────────────────────────────

st.markdown("---")
c1, c2, c3 = st.columns(3)

verdict_class = "verdict-fake" if cnn_label == 1 else "verdict-real"
verdict_text  = "🔴 FORGED" if cnn_label == 1 else "🟢 AUTHENTIC"

with c1:
    st.markdown(f'<p class="{verdict_class}">{verdict_text}</p>', unsafe_allow_html=True)
    st.markdown(f"**CNN Confidence:** {cnn_conf:.1%}")
    st.progress(cnn_conf)

with c2:
    ela_verdict = "⚠ Suspicious" if ela_sc > 0.05 else "✓ OK"
    st.metric("ELA Score", f"{ela_sc:.4f}", delta=ela_verdict, delta_color="inverse")

with c3:
    m_score = meta.get("suspicion_score", 0)
    m_verdict = "⚠ Suspicious" if m_score > 0.3 else "✓ OK"
    st.metric("Metadata Score", f"{m_score:.2f}", delta=m_verdict, delta_color="inverse")

# ── Visualizations ───────────────────────────────────────

st.markdown("---")
st.subheader("📊 Analysis Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Original + Heatmap", "ELA Analysis", "Metadata", "Probability"])

with tab1:
    cols = st.columns(2)
    cols[0].image(image_np, caption="Original Image", use_column_width=True)
    cols[1].image(overlay,  caption="Forgery Heatmap (red = suspicious)", use_column_width=True)

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(ela_map);            axes[0].set_title("ELA Map");           axes[0].axis("off")
    ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY)
    hm_ela   = cv2.applyColorMap(ela_gray, cv2.COLORMAP_HOT)
    hm_ela   = cv2.cvtColor(hm_ela, cv2.COLOR_BGR2RGB)
    ela_ov   = cv2.addWeighted(image_np, 0.5, hm_ela, 0.5, 0)
    axes[1].imshow(ela_ov);             axes[1].set_title("ELA Overlay");       axes[1].axis("off")
    plt.tight_layout()
    st.image(fig_to_image(fig), use_column_width=True)
    plt.close()

with tab3:
    st.markdown("**EXIF Metadata Report**")
    meta_display = {k: v for k, v in meta.items() if k not in ("raw_exif", "flags")}
    st.json(meta_display)
    if meta.get("flags"):
        st.warning("**Suspicious Flags:**")
        for flag in meta["flags"]:
            st.warning(f"→ {flag}")
    else:
        st.success("No suspicious metadata flags found.")

with tab4:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Real", "Fake"]
    colors = ["#44bb44", "#ff4444"]
    bars   = ax.bar(labels, [probs[0]*100, probs[1]*100], color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Probability (%)")
    ax.set_title("CNN Class Probabilities")
    ax.set_ylim([0, 100])
    for bar, val in zip(bars, [probs[0]*100, probs[1]*100]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f"{val:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    st.image(fig_to_image(fig), use_column_width=False)
    plt.close()

# Footer
st.markdown("---")
st.caption("Image Forgery Detection System | Built with PyTorch + Streamlit | Educational Project")