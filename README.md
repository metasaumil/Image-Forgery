# 🔍 Image Forgery Detection System

A complete end-to-end system for detecting image forgeries using classical computer vision and deep learning.

**Built for:** Learning, internships, hackathons, and GitHub portfolios.

---

## 📸 What it detects

| Forgery Type | Method Used |
|---|---|
| Copy-move (copy-paste within image) | ORB/SIFT feature matching |
| Splicing (pasting from another image) | CNN, ELA, patch localization |
| Image retouching | ELA + metadata analysis |
| Deepfakes | CNN classification |

---

## ✨ Features

- **Classical CV Methods:** Error Level Analysis (ELA), ORB/SIFT copy-move detection, EXIF metadata forensics
- **Deep Learning:** Custom CNN, ResNet18, MobileNetV2 (transfer learning)
- **Localization:** Patch-based heatmap showing *where* the forgery is
- **Grad-CAM:** Neural network explanation visualization
- **Streamlit Web App:** Upload → Analyze → Visualize, all in your browser
- **Docker-ready** deployment

---

## 🗂 Project Structure

```
image-forgery-detection/
│
├── preprocessing/
│   ├── dataset.py          ← PyTorch Dataset class + DataLoaders
│   └── prepare_data.py     ← Download/organize datasets
│
├── classical_methods/
│   ├── ela.py              ← Error Level Analysis
│   ├── copy_move.py        ← ORB/SIFT copy-move detection
│   └── metadata.py         ← EXIF forensic analysis
│
├── deep_learning/
│   └── models.py           ← CustomCNN, ResNet18, MobileNetV2
│
├── training/
│   ├── train.py            ← Main training script
│   └── trainer.py          ← Training loop, early stopping
│
├── patch_localization/
│   └── localizer.py        ← Patch-based forgery heatmap
│
├── evaluation/
│   └── evaluate.py         ← Metrics, ROC, confusion matrix
│
├── utils/
│   └── gradcam.py          ← Grad-CAM visualization
│
├── deployment/
│   └── streamlit_app/
│       └── app.py          ← Web interface
│
├── inference.py            ← Single-image analysis pipeline
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/your-username/image-forgery-detection
cd image-forgery-detection
pip install -r requirements.txt
```

### 2. Generate dummy data and train a model

```bash
# Creates a small dummy dataset for testing
python training/train.py --dummy --model resnet18 --epochs 5
```

### 3. Run the web app

```bash
streamlit run deployment/streamlit_app/app.py
```

### 4. Analyze a single image from the command line

```bash
python inference.py path/to/image.jpg --checkpoint models/best_resnet18.pth
```

---

## 📦 Dataset Setup

### CASIA v2.0 (Recommended)
1. Request access at [https://forensics.idealtest.org/](https://forensics.idealtest.org/)
2. Download `CASIA2.0_revised.zip` and extract to `data/raw/CASIA2.0/`
3. Organize: `python preprocessing/prepare_data.py --mode casia`

### For testing without real data
```bash
python preprocessing/prepare_data.py --mode dummy
```

---

## 🏋️ Training

```bash
# Basic training with ResNet18
python training/train.py \
    --model resnet18 \
    --data_dir data/processed \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3

# Lightweight model (faster)
python training/train.py --model mobilenetv2 --epochs 20

# Train from scratch (no pretrained weights)
python training/train.py --model custom --epochs 50
```

Training automatically:
- Saves the best checkpoint to `models/best_<model>.pth`
- Plots training curves
- Runs evaluation on test set

---

## 🌐 Web App Deployment

### Local
```bash
streamlit run deployment/streamlit_app/app.py
```

### Docker
```bash
docker build -t forgery-detector .
docker run -p 8501:8501 forgery-detector
```

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select `deployment/streamlit_app/app.py` as the main file

### HuggingFace Spaces
1. Create a new Space (Streamlit SDK)
2. Push your code
3. Add `requirements.txt` at the root

---

## 🧪 Classical Methods (standalone)

```python
from classical_methods.ela import analyze_image
from classical_methods.copy_move import detect_copy_move
from classical_methods.metadata import analyze_metadata

# ELA
ela_map, score = analyze_image("image.jpg")

# Copy-move
is_forged, n_matches, vis = detect_copy_move("image.jpg", method="ORB")

# Metadata
report = analyze_metadata("image.jpg")
```

---

## 📊 Expected Performance

When trained on CASIA v2.0 with ResNet18:

| Metric | Expected |
|---|---|
| Accuracy | ~85-92% |
| AUC | ~0.90-0.95 |
| F1 Score | ~0.85-0.91 |

> Performance depends heavily on dataset size and training duration.

---

## 🧠 Theory: How It Works

### Error Level Analysis (ELA)
JPEG images lose quality each time they're saved. Forged regions (pasted from another image) have a *different compression history* than the rest. ELA re-saves the image and amplifies differences — forged regions appear brighter.

### Copy-Move Detection
ORB extracts "keypoints" — distinctive visual features. If the same feature appears in two different locations in the same image, it was likely copied and pasted.

### CNN Detection
ResNet18 learns to distinguish authentic vs. forged images by detecting subtle noise patterns, lighting inconsistencies, and texture anomalies that are invisible to humans but consistent enough for a neural network.

### Patch Localization
Instead of "is the whole image fake?", we ask "is this 64×64 region fake?" for hundreds of patches. The answers form a heatmap showing exactly where the forgery is.

---

## 🚀 Future Improvements

- [ ] Transformer-based model (ViT, DeiT)
- [ ] GAN-based data augmentation for rare forgery types
- [ ] Real-time webcam analysis
- [ ] Deepfake video detection
- [ ] Cross-dataset evaluation (train on CASIA, test on Columbia)
- [ ] REST API with FastAPI
- [ ] Mobile app export (ONNX → TFLite)

---

## 📄 License

MIT License. Free for educational and commercial use.

---

*Built with ❤️ for learning. If this helped you, star the repo!*
