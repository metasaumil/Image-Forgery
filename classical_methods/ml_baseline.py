"""
classical_methods/ml_baseline.py
---------------------------------
Classical ML baseline for forgery detection.

Feature extraction:
- LBP (Local Binary Patterns) — texture features
- Color histograms
- Gabor filter responses

Models:
- SVM (Support Vector Machine)
- Random Forest

Good as a baseline to compare against your CNN.
"""

import numpy as np
import cv2
import os
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[Warning] scikit-image not installed. LBP disabled.")


# ─────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────

def extract_lbp(image_gray: np.ndarray, n_points: int = 24, radius: int = 3) -> np.ndarray:
    """
    Local Binary Pattern features.
    Captures local texture by comparing each pixel to its neighbors.

    Returns a normalized histogram of LBP codes.
    """
    if not SKIMAGE_AVAILABLE:
        return np.zeros(26)

    lbp = local_binary_pattern(image_gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_color_histogram(image_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Color histogram in HSV space.
    Forged regions may have different color distributions.

    Returns concatenated H, S, V histograms.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    hist   = np.concatenate([hist_h, hist_s, hist_v])
    # Normalize
    return hist / (hist.sum() + 1e-7)


def extract_ela_features(ela_map: np.ndarray) -> np.ndarray:
    """
    Statistical features from the ELA map.
    Forged regions show higher ELA values.
    """
    ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.array([
        ela_gray.mean(),
        ela_gray.std(),
        ela_gray.max(),
        np.percentile(ela_gray, 75),
        np.percentile(ela_gray, 90),
        np.percentile(ela_gray, 99),
    ])


def extract_features(image_path: str, img_size: int = 128) -> np.ndarray:
    """
    Extract all features from a single image.

    Returns a 1D feature vector combining all feature types.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None

    image_bgr = cv2.resize(image_bgr, (img_size, img_size))
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # LBP features (texture)
    lbp_feat = extract_lbp(image_gray)

    # Color histogram
    color_feat = extract_color_histogram(image_bgr)

    # ELA features
    from ela import compute_ela
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(image_rgb)
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    resaved = np.array(Image.open(buf).convert("RGB"))
    ela_map = np.abs(image_rgb.astype(np.float32) - resaved.astype(np.float32)) * 20
    ela_map = np.clip(ela_map, 0, 255).astype(np.uint8)
    ela_feat = extract_ela_features(ela_map)

    return np.concatenate([lbp_feat, color_feat, ela_feat])


# ─────────────────────────────────────────────
#  Dataset Feature Extraction
# ─────────────────────────────────────────────

def build_feature_dataset(data_dir: str, split: str = "train",
                           save_path: str = None) -> tuple:
    """
    Extract features for all images in a split.

    Returns:
        (X, y) arrays
    """
    features = []
    labels   = []
    failed   = 0

    for label, class_name in enumerate(["real", "fake"]):
        class_dir = os.path.join(data_dir, split, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = [f for f in Path(class_dir).glob("*")
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        for img_path in tqdm(image_files, desc=f"  {split}/{class_name}"):
            feat = extract_features(str(img_path))
            if feat is not None:
                features.append(feat)
                labels.append(label)
            else:
                failed += 1

    if failed:
        print(f"[Warning] {failed} images failed to load")

    X = np.array(features)
    y = np.array(labels)

    if save_path:
        np.savez(save_path, X=X, y=y)
        print(f"[Saved] Features to {save_path}")

    return X, y


# ─────────────────────────────────────────────
#  Train + Evaluate
# ─────────────────────────────────────────────

def train_ml_baseline(data_dir: str, model_type: str = "svm",
                      save_dir: str = "models"):
    """
    Train SVM or Random Forest on extracted features.
    """
    print(f"\n[ML Baseline] Training {model_type.upper()} on {data_dir}")

    # Extract features
    print("Extracting training features...")
    X_train, y_train = build_feature_dataset(data_dir, "train")
    print("Extracting test features...")
    X_test, y_test   = build_feature_dataset(data_dir, "test")

    if len(X_train) == 0:
        print("[Error] No training data found.")
        return

    # Normalize
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train
    if model_type == "svm":
        clf = SVC(kernel="rbf", C=10, probability=True, random_state=42)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    print(f"\nFitting {model_type.upper()}...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f"{model_type.upper()} Confusion Matrix"); plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{model_type}.png"), dpi=150)
    plt.show()

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path  = os.path.join(save_dir, f"{model_type}_classifier.pkl")
    scaler_path = os.path.join(save_dir, f"{model_type}_scaler.pkl")
    with open(model_path,  "wb") as f: pickle.dump(clf,    f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    print(f"\n[Saved] Model: {model_path}")

    return clf, scaler


if __name__ == "__main__":
    import sys
    data_dir   = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    model_type = sys.argv[2] if len(sys.argv) > 2 else "svm"
    train_ml_baseline(data_dir, model_type)
