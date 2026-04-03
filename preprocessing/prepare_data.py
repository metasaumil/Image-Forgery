"""
preprocessing/prepare_data.py
------------------------------
Downloads and organizes datasets into the expected folder structure:
    data/processed/
        train/real/, train/fake/
        val/real/,   val/fake/
        test/real/,  test/fake/
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
#  Download Instructions (manual steps)
# ─────────────────────────────────────────────

DATASET_INFO = """
=== DATASET DOWNLOAD INSTRUCTIONS ===

1. CASIA v2.0 (recommended for forgery detection)
   → Google: "CASIA Image Tampering Detection Evaluation Database"
   → Request access at: https://forensics.idealtest.org/
   → Download: CASIA2.0_revised.zip
   → Contains ~12,000 images (real and tampered)

2. Columbia Uncompressed Image Splicing Dataset
   → Direct link: https://www.ee.columbia.edu/ln/dvmm/downloads/
   → Small but high-quality splicing dataset

3. FaceForensics++ (Deepfakes)
   → https://github.com/ondyari/FaceForensics
   → Requires request for access

After downloading, place raw datasets in: data/raw/
Then run this script to organize them.
"""

print(DATASET_INFO)


# ─────────────────────────────────────────────
#  Organizer Function
# ─────────────────────────────────────────────

def organize_casia(raw_dir: str, output_dir: str, split: tuple = (0.7, 0.15, 0.15), seed: int = 42):
    """
    Organizes CASIA v2.0 into train/val/test splits.

    CASIA v2.0 structure:
        Au/    ← authentic images
        Tp/    ← tampered images

    Args:
        raw_dir:    Path to raw CASIA folder (contains Au/ and Tp/)
        output_dir: Where to save the organized data
        split:      (train, val, test) fractions
        seed:       For reproducibility
    """
    random.seed(seed)

    au_dir = os.path.join(raw_dir, "Au")
    tp_dir = os.path.join(raw_dir, "Tp")

    if not os.path.isdir(au_dir) or not os.path.isdir(tp_dir):
        print(f"[ERROR] Expected 'Au' and 'Tp' folders in: {raw_dir}")
        return

    real_images = [f for f in Path(au_dir).glob("**/*") if f.suffix.lower() in {".jpg", ".png", ".bmp", ".tif"}]
    fake_images = [f for f in Path(tp_dir).glob("**/*") if f.suffix.lower() in {".jpg", ".png", ".bmp", ".tif"}]

    print(f"Found {len(real_images)} real, {len(fake_images)} fake images")

    def split_list(lst):
        random.shuffle(lst)
        n = len(lst)
        n_train = int(n * split[0])
        n_val   = int(n * split[1])
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    real_train, real_val, real_test = split_list(real_images)
    fake_train, fake_val, fake_test = split_list(fake_images)

    splits = {
        "train": (real_train, fake_train),
        "val":   (real_val,   fake_val),
        "test":  (real_test,  fake_test),
    }

    for split_name, (reals, fakes) in splits.items():
        for label, imgs in [("real", reals), ("fake", fakes)]:
            dest = os.path.join(output_dir, split_name, label)
            os.makedirs(dest, exist_ok=True)
            for src_path in tqdm(imgs, desc=f"Copying {split_name}/{label}"):
                shutil.copy2(src_path, os.path.join(dest, src_path.name))

    print(f"\n[Done] Dataset organized in: {output_dir}")
    _print_stats(output_dir)


def _print_stats(output_dir: str):
    """Print how many images are in each split."""
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            d = os.path.join(output_dir, split, label)
            count = len(list(Path(d).glob("*"))) if os.path.isdir(d) else 0
            print(f"  {split}/{label}: {count} images")


# ─────────────────────────────────────────────
#  Create Dummy Dataset for Testing
# ─────────────────────────────────────────────

def create_dummy_dataset(output_dir: str = "data/processed", n_per_class: int = 50):
    """
    Creates a tiny dummy dataset with random noise images.
    Useful for testing your pipeline without real data.
    """
    import numpy as np
    from PIL import Image

    print(f"Creating dummy dataset with {n_per_class} images per class per split...")

    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            out = os.path.join(output_dir, split, label)
            os.makedirs(out, exist_ok=True)
            for i in range(n_per_class):
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(out, f"{label}_{i:04d}.jpg"))

    print(f"[Done] Dummy dataset created at: {output_dir}")
    _print_stats(output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["casia", "dummy"], default="dummy",
                        help="'casia' to organize real data, 'dummy' to generate test data")
    parser.add_argument("--raw_dir",    default="data/raw/CASIA2.0")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    if args.mode == "casia":
        organize_casia(args.raw_dir, args.output_dir)
    else:
        create_dummy_dataset(args.output_dir)
